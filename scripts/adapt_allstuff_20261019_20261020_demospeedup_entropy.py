#!/usr/bin/env python3
"""Index action-entropy npz files for the merged all-stuff + sheet train set.

The 900-episode bucket is not the dataset either entropy run was computed on.
Sheet episodes 0-178 keep the new sheet npz, whose episode_index already
matches. The 721 all-stuff episodes reuse the fittings npz, but
input_episode_index points into the all-stuff subset, not the fittings train
that produced those files. Each copied npz stores the merged episode_index,
which is what DemoSpeedup checks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

MERGED = Path("/vePFS/data/linglong/linglong_allstuff_20261019_20261020_eef_train_15hz")
ALLSTUFF_SUBSET = Path(
    "/vePFS/data/linglong/linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable/"
    "data/linglong_0923_1010_1012_1017_fittings_eef_gtstate_allstuff_train_15hz"
)
FITTINGS = Path(
    "/vePFS/data/linglong/linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable/"
    "data/linglong_0923_1010_1012_1017_fittings_eef_gtstate_train_15hz"
)
SHEET_ENTROPY = Path(
    "/vePFS/zhuoyu/OpenWAM/outputs/action_entropy_20261019_20261020_openwam_train_step21830_stride2"
)
FITTINGS_ENTROPY = Path("/vePFS/zhuoyu/OpenWAM/outputs/action_entropy_fittings_eef_step18185_stride2")
OUTPUT = Path("/vePFS/zhuoyu/OpenWAM/outputs/action_entropy_allstuff_20261019_20261020_eef_step18185_stride2")


def _load_mapping(path: Path) -> list[dict]:
    rows = json.loads((path / "source_mapping.json").read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} source_mapping.json is empty")
    return rows


def _identity(row: dict) -> tuple[str, int, str]:
    episode = row.get("source_episode")
    if episode is None:
        raise ValueError(f"missing source_episode in episode {row.get('episode_index')}")
    return (str(episode), int(row["length"]), str(row.get("collection")))


def _episodes(dataset: Path) -> list[dict]:
    path = dataset / "meta" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for local, row in enumerate(rows):
        if int(row["episode_index"]) != local:
            raise ValueError(f"{path} row {local} has episode_index {row['episode_index']}")
    return rows


def _fittings_index(merged_rows: list[dict], subset_rows: list[dict], fittings_rows: list[dict]) -> dict[int, int]:
    subset = {int(row["episode_index"]): row for row in subset_rows}
    fittings: dict[tuple[str, int, str], int] = {}
    for row in fittings_rows:
        key = _identity(row)
        if key in fittings:
            raise ValueError(f"duplicate fittings identity {key}")
        fittings[key] = int(row["episode_index"])
    mapped: dict[int, int] = {}
    for row in merged_rows:
        if row.get("group") != "allstuff":
            continue
        source = subset[int(row["input_episode_index"])]
        key = _identity(source)
        if key not in fittings:
            raise ValueError(f"merged episode {row['episode_index']} has no fittings entropy: {key}")
        fittings_index = fittings[key]
        if int(source["length"]) != int(row["length"]):
            raise ValueError(f"length mismatch at merged episode {row['episode_index']}")
        source_sha = source.get("source_sha256")
        fittings_sha = next(
            item.get("source_sha256") for item in fittings_rows if int(item["episode_index"]) == fittings_index
        )
        if source_sha and fittings_sha and source_sha != fittings_sha:
            raise ValueError(f"sha mismatch at merged episode {row['episode_index']}")
        mapped[int(row["episode_index"])] = fittings_index
    if len(mapped) != 721:
        raise ValueError(f"expected 721 all-stuff episodes, mapped {len(mapped)}")
    return mapped


def _same_payload(path: Path, episode_index: int, source: Path) -> bool:
    if not path.is_file():
        return False
    with np.load(path) as saved, np.load(source) as original:
        if int(saved["episode_index"]) != episode_index:
            return False
        if int(saved["frame_stride"]) != 2 or int(saved["noises"]) != 8:
            return False
        return np.array_equal(saved["frame_index"], original["frame_index"]) and np.array_equal(
            saved["entropy"], original["entropy"]
        )


def _rewrite(source: Path, destination: Path, episode_index: int) -> None:
    with np.load(source) as payload:
        arrays = {key: np.array(payload[key]) for key in payload.files}
    if int(arrays["frame_stride"]) != 2 or int(arrays["noises"]) != 8:
        raise ValueError(f"{source} is not stride-2, 8-noise entropy")
    arrays["episode_index"] = np.int64(episode_index)
    temporary = destination.with_name(f".{destination.stem}.partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)


def _link_sheet(source: Path, destination: Path, episode_index: int) -> None:
    with np.load(source) as payload:
        if int(payload["episode_index"]) != episode_index:
            raise ValueError(f"{source} episode_index {int(payload['episode_index'])} != {episode_index}")
        if int(payload["frame_stride"]) != 2 or int(payload["noises"]) != 8:
            raise ValueError(f"{source} is not stride-2, 8-noise entropy")
    if destination.is_file() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        _rewrite(source, destination, episode_index)


def main() -> None:
    merged = sorted(_load_mapping(MERGED), key=lambda row: int(row["episode_index"]))
    episodes = _episodes(MERGED)
    if len(merged) != len(episodes):
        raise ValueError(f"source_mapping has {len(merged)} rows, episodes.jsonl has {len(episodes)}")
    fittings_index = _fittings_index(merged, _load_mapping(ALLSTUFF_SUBSET), _load_mapping(FITTINGS))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = []
    for row, episode in zip(merged, episodes):
        index = int(row["episode_index"])
        if index != int(episode["episode_index"]) or int(row["length"]) != int(episode["length"]):
            raise ValueError(f"source_mapping and episodes.jsonl disagree at {index}")
        destination = OUTPUT / f"episode_{index:06d}.npz"
        if row["group"] == "sheet":
            source_index = int(row["input_episode_index"])
            if source_index != index:
                raise ValueError(f"sheet episode {index} input index is {source_index}")
            source = SHEET_ENTROPY / f"episode_{source_index:06d}.npz"
            if not _same_payload(destination, index, source):
                _link_sheet(source, destination, index)
            group = "sheet"
        elif row["group"] == "allstuff":
            source_index = fittings_index[index]
            source = FITTINGS_ENTROPY / f"episode_{source_index:06d}.npz"
            if not _same_payload(destination, index, source):
                _rewrite(source, destination, index)
            group = "allstuff"
        else:
            raise ValueError(f"unknown group {row['group']} at episode {index}")
        with np.load(destination) as saved:
            if int(saved["episode_index"]) != index:
                raise ValueError(f"{destination} episode_index is {int(saved['episode_index'])}")
            if int(saved["frame_index"][-1]) >= int(episode["length"]):
                raise ValueError(f"{destination} frame_index exceeds episode length {episode['length']}")
        provenance.append(
            {
                "episode_index": index,
                "group": group,
                "source_episode_index": source_index,
                "source": str(source),
            }
        )
    written = sorted(OUTPUT.glob("episode_*.npz"))
    if len(written) != len(episodes):
        raise ValueError(f"wrote {len(written)} npz files for {len(episodes)} episodes")
    (OUTPUT / "source_index.json").write_text(json.dumps(provenance))
    print(f"entropy files: {len(written)} -> {OUTPUT}")


if __name__ == "__main__":
    main()
