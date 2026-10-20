"""Decode the 2026-10-19 sheet-pickup bags for convert_linglong.py.

Episodes 122 and 123 are both-hand takes and are left out. Episodes 21-71 are
right-hand and 72-121 are left-hand. Each hand keeps its last five episodes as
the test split, matching the ten-episode holdout used for the 0923 and 1010
collections. The decoded layout is the processed train/test tree that
``convert_linglong.py`` already reads.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import tarfile
from pathlib import Path

import numpy as np


RIGHT_PROMPT = "Use the right arm to pick up the sheet workpiece from the box."
LEFT_PROMPT = "Use the left arm to pick up the sheet workpiece from the box."
TEST_NUMBERS = {67, 68, 69, 70, 71, 117, 118, 119, 120, 121}
EXCLUDED_NUMBERS = {122, 123}
LIMB_WIDTHS = (7, 7, 4, 2, 2, 2)
ALIGN_NS = 40_000_000


def _stamp_ns(blob: bytes) -> int:
    sec = int(np.frombuffer(blob[4:8], dtype="<i4")[0])
    nsec = int(np.frombuffer(blob[8:12], dtype="<u4")[0])
    return sec * 1_000_000_000 + nsec


def _jpeg(blob: bytes) -> bytes:
    start = blob.find(b"\xff\xd8\xff")
    if start < 0:
        raise ValueError("compressed image has no JPEG payload")
    end = blob.rfind(b"\xff\xd9")
    if end < start:
        raise ValueError("compressed image JPEG is truncated")
    return blob[start : end + 2]


def _joints(blob: bytes) -> tuple[int, np.ndarray, np.ndarray]:
    words = np.frombuffer(blob[4:], dtype="<u4")
    if int(words[4]) != len(LIMB_WIDTHS):
        raise ValueError(f"expected {len(LIMB_WIDTHS)} limbs, got {int(words[4])}")
    cursor = 5
    measured, command = [], []
    for width in LIMB_WIDTHS:
        count = int(words[cursor])
        cursor += 1
        if count != width:
            raise ValueError(f"limb width {count} does not match {LIMB_WIDTHS}")
        block = words[cursor : cursor + count * 6].view("<f4").reshape(count, 6)
        measured.append(block[:, 0].copy())
        command.append(block[:, 1].copy())
        cursor += count * 6
    if cursor != len(words):
        raise ValueError("driver_pvt payload was not fully consumed")
    q = np.zeros(24, dtype=np.float32)
    q_exp = np.zeros(24, dtype=np.float32)
    # Bag limbs are left arm, right arm, waist, head, grippers, base.
    # OpenWAM order inserts each gripper after its arm.
    for target, source in ((q, measured), (q_exp, command)):
        target[0:7] = source[0]
        target[7] = source[4][0]
        target[8:15] = source[1]
        target[15] = source[4][1]
        target[16:20] = source[2]
        target[20:22] = source[3]
        target[22:24] = source[5]
    return _stamp_ns(blob), q, q_exp


def _nearest(stamps: np.ndarray, target: int) -> int:
    position = int(np.searchsorted(stamps, target, side="left"))
    candidates = [index for index in (position - 1, position) if 0 <= index < len(stamps)]
    best = min(candidates, key=lambda index: abs(int(stamps[index]) - target))
    if abs(int(stamps[best]) - target) > ALIGN_NS:
        raise ValueError(f"nearest sample is {abs(int(stamps[best]) - target) / 1e6:.1f} ms away")
    return best


def _load_topic(connection: sqlite3.Connection, name: str) -> list[bytes]:
    topic_id = connection.execute("SELECT id FROM topics WHERE name=?", (name,)).fetchone()
    if topic_id is None:
        raise ValueError(f"missing topic {name}")
    return [blob for (blob,) in connection.execute(
        "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id[0],)
    )]


def _decode_episode(db_path: Path, episode_dir: Path, hand: str) -> dict:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        driver = [_joints(blob) for blob in _load_topic(connection, "/driver_pvt")]
        end_pos = np.asarray([_stamp_ns(blob) for blob in _load_topic(connection, "/end_pos")], dtype=np.int64)
        cameras = {}
        for camera in ("head", "left", "right"):
            blobs = _load_topic(connection, f"/camera/{camera}/image_raw/compressed")
            cameras[camera] = (np.asarray([_stamp_ns(blob) for blob in blobs], dtype=np.int64), [_jpeg(blob) for blob in blobs])
    driver_stamps = np.asarray([item[0] for item in driver], dtype=np.int64)
    head_stamps, head_jpegs = cameras["head"]
    if len(head_stamps) < 2 or np.any(np.diff(head_stamps) <= 0) or np.any(np.diff(driver_stamps) <= 0):
        raise ValueError(f"{episode_dir.name}: non-monotonic head or driver timestamps")

    rows = []
    alignments = []
    max_error_ns = 0
    for index, head_stamp in enumerate(head_stamps):
        driver_index = _nearest(driver_stamps, int(head_stamp))
        end_index = _nearest(end_pos, int(head_stamp))
        _, q, q_exp = driver[driver_index]
        if not np.isfinite(q).all() or not np.isfinite(q_exp).all():
            raise ValueError(f"{episode_dir.name}: non-finite joint sample")
        images = {"head": f"head/{index:06d}.jpg"}
        source_timestamps = {
            "head": int(head_stamp),
            "driver": int(driver_stamps[driver_index]),
            "end_pos": int(end_pos[end_index]),
        }
        (episode_dir / "head").mkdir(parents=True, exist_ok=True)
        (episode_dir / "head" / images["head"].split("/", 1)[1]).write_bytes(head_jpegs[index])
        for camera in ("left", "right"):
            stamps, jpegs = cameras[camera]
            chosen = _nearest(stamps, int(head_stamp))
            images[camera] = f"{camera}/{index:06d}.jpg"
            source_timestamps[camera] = int(stamps[chosen])
            (episode_dir / camera).mkdir(parents=True, exist_ok=True)
            (episode_dir / camera / f"{index:06d}.jpg").write_bytes(jpegs[chosen])
            max_error_ns = max(max_error_ns, abs(int(stamps[chosen]) - int(head_stamp)))
        max_error_ns = max(
            max_error_ns,
            abs(int(driver_stamps[driver_index]) - int(head_stamp)),
            abs(int(end_pos[end_index]) - int(head_stamp)),
        )
        row = {"timestamp_ns": int(head_stamp)}
        for joint in range(24):
            row[f"q_{joint}"] = float(q[joint])
            row[f"q_exp_{joint}"] = float(q_exp[joint])
        rows.append(row)
        alignments.append({"images": images, "source_timestamps": source_timestamps})

    q_values = np.asarray([[row[f"q_{joint}"] for joint in range(24)] for row in rows], dtype=np.float32)
    active = 15 if hand == "right" else 7
    inactive = 7 if hand == "right" else 15
    if np.ptp(q_values[:, active]) < 0.5 or np.ptp(q_values[:, inactive]) > 0.05:
        raise ValueError(
            f"{episode_dir.name}: gripper motion does not match {hand} "
            f"(active {np.ptp(q_values[:, active]):.3f}, inactive {np.ptp(q_values[:, inactive]):.3f})"
        )
    fieldnames = ["timestamp_ns"] + [f"q_{joint}" for joint in range(24)] + [f"q_exp_{joint}" for joint in range(24)]
    with (episode_dir / "aligned.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (episode_dir / "alignment.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in alignments)
    )
    return {"rows": len(rows), "max_align_ms": max_error_ns / 1e6}


def prepare(archive: Path, processed_root: Path) -> dict:
    if processed_root.exists():
        raise FileExistsError(f"processed output already exists: {processed_root}")
    inventory = {"train": [], "test": []}
    summary = []
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            if not (member.isfile() and member.name.endswith(".tar")):
                continue
            outer_name = Path(member.name).parent.name
            inner = tarfile.open(fileobj=io.BytesIO(bundle.extractfile(member).read()))
            metadata = json.loads(inner.extractfile("__loongdata_metadata.json").read())
            number = int(metadata["episode"])
            if number in EXCLUDED_NUMBERS:
                print(f"skip {number}", flush=True)
                continue
            if not 21 <= number <= 121:
                raise ValueError(f"unexpected episode number {number}")
            filename_number = int(Path(member.name).name.rsplit("_", 1)[-1].removesuffix(".tar"))
            if filename_number != number or metadata.get("taskName") != "拿板子1019":
                raise ValueError(f"metadata does not match {member.name}")
            hand = "right" if number <= 71 else "left"
            split = "test" if number in TEST_NUMBERS else "train"
            episode_id = outer_name
            episode_dir = processed_root / split / "decoded" / episode_id
            db_member = next(item for item in inner.getmembers() if item.name.endswith("bag_0.db3"))
            db_path = processed_root / f".tmp_{number}.db3"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with db_path.open("wb") as stream:
                stream.write(inner.extractfile(db_member).read())
            try:
                decoded = _decode_episode(db_path, episode_dir, hand)
            finally:
                db_path.unlink(missing_ok=True)
            inventory[split].append({
                "episode": episode_id,
                "source_number": number,
                "hand": hand,
                "task": RIGHT_PROMPT if hand == "right" else LEFT_PROMPT,
            })
            summary.append({"episode": number, "id": episode_id, "split": split, "hand": hand, **decoded})
            print(f"decoded {number:3d} {split:5s} {hand:5s} rows={decoded['rows']} align={decoded['max_align_ms']:.1f} ms", flush=True)
    if len(summary) != 101:
        raise ValueError(f"expected 101 decoded episodes, got {len(summary)}")
    for split, records in inventory.items():
        records.sort(key=lambda record: int(record["source_number"]))
        target = processed_root / split / "inventory.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    numbers = sorted(item["episode"] for item in summary)
    if numbers != list(range(21, 122)):
        raise ValueError(f"decoded episode numbers are {numbers}")
    (processed_root / "split.json").write_text(json.dumps({
        "excluded": sorted(EXCLUDED_NUMBERS),
        "right": [21, 71],
        "left": [72, 121],
        "test_numbers": sorted(TEST_NUMBERS),
        "holdout": "last five episodes of each hand",
        "prompts": {"right": RIGHT_PROMPT, "left": LEFT_PROMPT},
    }, ensure_ascii=False, indent=2))
    return {"episodes": len(summary), "train": len(inventory["train"]), "test": len(inventory["test"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("/vePFS/data/linglong/linglong_2026-10-19.tar.gz"))
    parser.add_argument("--processed-root", type=Path, default=Path("/vePFS/data/linglong/linglong_20261019_processed"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.processed_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
