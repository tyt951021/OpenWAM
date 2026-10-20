"""Merge the 2026-10-19 sheet-workpiece train bucket into the portable fittings train.

Copies joint and EEF values verbatim. Episode indices, the global frame index,
and 20261019 image paths are rewritten so one source_root resolves every camera
file. The portable train directory is not modified. 20261019 test stays separate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PORTABLE = Path(
    "/vePFS/data/linglong/linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable/data/"
    "linglong_0923_1010_1012_1017_fittings_eef_gtstate_train_15hz"
)
ADDITION = Path("/vePFS/zhuoyu/OpenWAM/data/linglong_20261019_eef_train_15hz")
PROCESSED = Path("/vePFS/data/linglong/linglong_20261019_processed/train")
OUTPUT = Path(
    "/vePFS/data/linglong/linglong_0923_1010_1012_1017_fittings_20261019_eef_gtstate_train_15hz"
)
OPENWAM_LINK = Path(
    "/vePFS/zhuoyu/OpenWAM/data/linglong_0923_1010_1012_1017_fittings_20261019_eef_gtstate_train_15hz"
)
HTML = Path("/vePFS/data/linglong/linglong_datasets.html")
COLLECTION = "20261019"
RIGHT = "Use the right arm to pick up the sheet workpiece from the box."
LEFT = "Use the left arm to pick up the sheet workpiece from the box."
NEW_GLOSS = {RIGHT: "右臂，从盒子取出薄片工件", LEFT: "左臂，从盒子取出薄片工件"}
EXPECTED_NUMBERS = set(range(21, 67)) | set(range(72, 117))
NUMERIC = ("observation.state", "action", "observation.state_eef", "action_eef")
CAMERAS = ("head", "left", "right")
COLUMNS = (
    "observation.state", "action", "observation.state_eef", "action_eef", "timestamp",
    "frame_index", "episode_index", "observation.images.head", "observation.images.left",
    "observation.images.right", "index", "task_index",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stats(values, ddof):
    values = np.concatenate(values).astype(np.float64)
    return {
        "count": int(len(values)),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0, ddof=ddof).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def load_bucket(root):
    root = root.resolve(strict=True)
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    mapping = read_json(root / "source_mapping.json")
    tasks = read_jsonl(root / "meta/tasks.jsonl") if (root / "meta/tasks.jsonl").exists() else []
    info = read_json(root / "meta/info.json")
    manifest = read_json(root / "manifest.json")
    assert manifest["status"] == "complete" and info["fps"] == 15
    assert len(episodes) == len(mapping) == info["total_episodes"]
    assert sum(episode["length"] for episode in episodes) == info["total_frames"]
    by_index = {row["episode_index"]: row for row in mapping}
    assert set(by_index) == set(range(len(episodes)))
    episodes = sorted(episodes, key=lambda episode: episode["episode_index"])
    for episode in episodes:
        row = by_index[episode["episode_index"]]
        assert row["length"] == episode["length"] >= 33
        assert len(episode["tasks"]) == 1 and episode["tasks"][0].strip()
    files = sorted((root / "data").rglob("episode_*.parquet"))
    assert len(files) == len(episodes)
    return {"root": root, "episodes": episodes, "mapping": by_index, "tasks": tasks,
            "info": info, "manifest": manifest, "files": files}


def check_frame(table, label):
    timestamp = np.asarray(table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
    frame_index = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False))
    if not np.allclose(timestamp, frame_index / 15.0, atol=1e-5, rtol=0):
        raise ValueError(f"{label} timestamp is not frame_index/15")
    for key in ("observation.state_eef", "action_eef"):
        values = np.asarray(table[key].to_pylist(), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{label} {key} is not finite")
        for start in (3, 11):
            norm = np.linalg.norm(values[:, start:start + 4], axis=1)
            if np.max(np.abs(norm - 1.0)) > 1e-3:
                raise ValueError(f"{label} {key} quaternion is not unit")


def rewrite(source, episode_index, index_start, task_index, schema, image_prefix):
    count = source.num_rows
    columns = []
    for name in COLUMNS:
        if name == "episode_index":
            columns.append(pa.array(np.full(count, episode_index, dtype=np.int64), type=pa.int64()))
        elif name == "index":
            columns.append(pa.array(np.arange(index_start, index_start + count, dtype=np.int64), type=pa.int64()))
        elif name == "task_index":
            columns.append(pa.array(np.full(count, task_index, dtype=np.int64), type=pa.int64()))
        elif name.startswith("observation.images.") and image_prefix:
            paths = source[name].to_pylist()
            if any(not path.startswith("decoded/") for path in paths):
                raise ValueError(f"unexpected 20261019 image path {paths[0]}")
            columns.append(pa.array([image_prefix + path for path in paths], type=pa.string()))
        else:
            columns.append(source[name])
    return pa.Table.from_arrays(columns, schema=schema)


def merge(portable, addition, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    identities = set()
    for episode in portable["episodes"]:
        row = portable["mapping"][episode["episode_index"]]
        identity = (row["collection"], str(row["source_episode"]))
        if identity in identities:
            raise ValueError(f"duplicate portable identity {identity}")
        identities.add(identity)
    numbers = []
    for episode in addition["episodes"]:
        row = addition["mapping"][episode["episode_index"]]
        identity = (COLLECTION, str(row["source_episode"]))
        if identity in identities:
            raise ValueError(f"duplicate identity {identity}")
        identities.add(identity)
        if row.get("source_number") in (122, 123):
            raise ValueError("excluded episodes 122/123 are present")
        numbers.append(row["source_number"])
    if set(numbers) != EXPECTED_NUMBERS:
        raise ValueError(f"20261019 train source numbers {sorted(set(numbers) ^ EXPECTED_NUMBERS)}")

    task_rows = portable["tasks"]
    task_ids = {}
    for row in task_rows:
        if row["task"] in task_ids or row["task_index"] != len(task_ids):
            raise ValueError(f"portable task order changed at {row}")
        task_ids[row["task"]] = row["task_index"]
    for task in (RIGHT, LEFT):
        if task in task_ids:
            raise ValueError(f"{task} already exists in the portable tasks")
        task_ids[task] = len(task_ids)

    output.mkdir()
    write_json(output / "manifest.json", {"status": "incomplete", "split": "train"})
    images = output / "images"
    images.mkdir()
    for name in ("0923", "1010", "1012", "1017", "1017_fittings"):
        (images / name).symlink_to((portable["root"] / "images" / name).resolve(strict=True))
    (images / COLLECTION).symlink_to(PROCESSED.resolve(strict=True))
    chunk = output / "data/chunk-000"
    chunk.mkdir(parents=True)
    schema = pq.read_schema(portable["files"][0])
    if list(schema.names) != list(COLUMNS):
        raise ValueError(f"unexpected portable columns {schema.names}")

    episodes, mappings, image_refs = [], [], []
    arrays = {key: [] for key in NUMERIC}
    offset = 0
    sources = [(portable, None), (addition, f"images/{COLLECTION}/")]
    index = 0
    for bucket, image_prefix in sources:
        for episode in bucket["episodes"]:
            old = episode["episode_index"]
            source_path = bucket["files"][old]
            table = pq.read_table(source_path)
            if table.num_rows != episode["length"]:
                raise ValueError(f"{source_path} has {table.num_rows} rows, episodes.jsonl says {episode['length']}")
            check_frame(table, source_path)
            task = episode["tasks"][0]
            if task not in task_ids:
                raise ValueError(f"unknown task {task}")
            if image_prefix is None:
                stored = set(table["task_index"].to_pylist())
                if stored != {task_ids[task]}:
                    raise ValueError(f"{source_path} task_index {stored} != {task_ids[task]}")
            rewritten = rewrite(table, index, offset, task_ids[task], schema, image_prefix)
            destination = chunk / f"episode_{index:06d}.parquet"
            pq.write_table(rewritten, destination, compression="zstd")
            saved = pq.read_table(destination)
            for key in NUMERIC + ("timestamp", "frame_index"):
                if not saved[key].equals(table[key]):
                    raise ValueError(f"{destination} changed {key}")
            if set(saved["episode_index"].to_pylist()) != {index}:
                raise ValueError(f"{destination} episode_index")
            if saved["index"][0].as_py() != offset or saved["index"][-1].as_py() != offset + table.num_rows - 1:
                raise ValueError(f"{destination} global index")
            for camera in CAMERAS:
                image_refs.extend(saved[f"observation.images.{camera}"].to_pylist())
            row = bucket["mapping"][old]
            episode_id = episode["episode_id"] if image_prefix is None else f"{COLLECTION}/{row['source_episode']}"
            episodes.append({"episode_index": index, "episode_id": episode_id, "length": episode["length"], "tasks": [task]})
            mappings.append({
                **row,
                "episode_index": index,
                "collection": row.get("collection", COLLECTION),
                "input_dataset": str(bucket["root"]),
                "input_episode_index": old,
                "split": "train",
            })
            for key in NUMERIC:
                arrays[key].append(np.asarray(table[key].to_pylist(), dtype=np.float32))
            offset += episode["length"]
            index += 1
            if index % 100 == 0:
                print(f"rewrote {index} episodes, {offset} frames", flush=True)

    missing = []

    def exists(path):
        return os.path.isfile(output / path)

    with ThreadPoolExecutor(max_workers=32) as pool:
        for path, ok in zip(image_refs, pool.map(exists, image_refs)):
            if not ok:
                missing.append(path)
                if len(missing) >= 5:
                    break
    if missing:
        raise FileNotFoundError(missing)

    collections = {}
    for mapping in mappings:
        entry = collections.setdefault(mapping["collection"], {"episodes": 0, "frames": 0})
        entry["episodes"] += 1
        entry["frames"] += mapping["length"]
    write_jsonl(output / "meta/episodes.jsonl", episodes)
    write_jsonl(output / "meta/episodes_stats.jsonl", [
        {"episode_index": episode["episode_index"], "length": episode["length"]} for episode in episodes
    ])
    write_jsonl(output / "meta/tasks.jsonl", [{"task_index": task_index, "task": task} for task, task_index in task_ids.items()])
    write_json(output / "source_mapping.json", mappings)
    write_json(output / "train_episodes.json", list(range(len(episodes))))

    joint_stats = {label: stats(arrays[key], 0) for label, key in (("state", "observation.state"), ("action", "action"))}
    write_json(output / "meta/normalization_stats.json", joint_stats)
    np.save(output / "meta/normalization_stats.npy", {
        label: {name: np.asarray(values[name], dtype=np.float32) for name in ("min", "max", "mean", "std", "q01", "q99")}
        for label, values in (("joint", joint_stats["action"]), ("joint_state", joint_stats["state"]))
    })
    for label, key in (("state", "observation.state_eef"), ("action", "action_eef")):
        values = stats(arrays[key], 1)
        write_json(output / f"meta/{label}_eef_stats.json", {
            "source_column": key,
            "statistics_source": "merged train frames once; 20261019 test is not included",
            "num_episodes": len(episodes),
            "num_transition": offset,
            "std_ddof": 1,
            "zero_std_indices": np.flatnonzero(np.asarray(values["std"]) == 0).tolist(),
            label: {"default": {f"global_{name}": value for name, value in values.items() if name != "count"}},
        })

    info = {
        **portable["info"],
        "total_episodes": len(episodes),
        "total_frames": offset,
        "total_tasks": len(task_ids),
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(episodes)}"},
    }
    write_json(output / "meta/info.json", info)
    base = portable["manifest"]
    manifest = {key: base[key] for key in (
        "format", "fps", "state_dim", "action_dim", "action_semantics", "state_semantics", "layout",
        "num_frames", "video_stride", "num_video_frames", "prompt_source",
    )}
    manifest["prompt"] = next(iter(task_ids))
    manifest["prompts"] = list(task_ids)
    manifest["action_eef"] = {
        **base["action_eef"],
        "source": "FK(q_exp); portable values preserved, 20261019 generated with wrist TCP [0.0813, 0, 0.0081] m",
    }
    manifest["observation_state_eef"] = {
        **base["observation_state_eef"],
        "source": "mixed: recorded /end_pos for 0923, 1010, 1012, 1017, 1017_fittings; FK(measured q) for 20261019",
        "kinematics_used": False,
        "kinematics_by_collection": {name: False for name in collections if name != COLLECTION} | {COLLECTION: True},
    }
    windows = sum(episode["length"] - 32 for episode in episodes)
    manifest.update(
        status="complete",
        split="train",
        source_root=str(output),
        total_episodes=len(episodes),
        total_frames=offset,
        total_windows=windows,
        collections=collections,
        image_storage="symlinks to the portable image trees and linglong_20261019_processed/train",
        normalization="recomputed on this merged train; 20261019 test is not included",
        normalization_dataset_dir=str(output),
        holdout="portable train assignments preserved; 20261019 train is episodes 21-66 and 72-116; 67-71 and 117-121 stay in the 20261019 test bucket; 122 and 123 excluded",
        input_buckets=[str(portable["root"]), str(addition["root"])],
        training_state_key="observation.state_eef",
        training_action_key="action_eef",
    )
    task_counts = Counter(episode["tasks"][0] for episode in episodes)
    frame_counts = Counter()
    for episode in episodes:
        frame_counts[episode["tasks"][0]] += episode["length"]
    report = {
        "status": "complete",
        "split": "train",
        "episodes": len(episodes),
        "frames": offset,
        "windows": windows,
        "collections": collections,
        "image_references_checked": len(image_refs),
        "numeric_columns_preserved": True,
        "state_provenance": manifest["observation_state_eef"]["source"],
        "action_provenance": manifest["action_eef"]["source"],
        "excluded_source_numbers": [122, 123],
        "held_out_test": "/vePFS/zhuoyu/OpenWAM/data/linglong_20261019_eef_test_15hz",
        "task_counts": {task: {"episodes": task_counts[task], "frames": frame_counts[task]} for task in task_ids},
    }
    write_json(output / "merge_report.json", report)
    write_json(output / "manifest.json", manifest)
    if not OPENWAM_LINK.exists():
        OPENWAM_LINK.symlink_to(output)
    return report


def glosses(html):
    found = dict(re.findall(r"<div>([^<]+)</div><div class=\"prompt\">([^<]+)</div>", html))
    gloss = {english: chinese for chinese, english in found.items()}
    gloss.update(NEW_GLOSS)
    return gloss


def prompt_items(task_counts, gloss, with_counts):
    parts = []
    search = []
    for task, counts in task_counts.items():
        chinese = gloss[task]
        search.append(f"{task} {chinese}")
        muted = f'<div class="muted">{counts["episodes"]} 条 / {counts["frames"]:,} 帧</div>' if with_counts else ""
        parts.append(f'<div class="prompt-item"><div>{chinese}</div><div class="prompt">{task}</div>{muted}</div>')
    return "".join(parts), " ".join(search)


def row(name, path, tag, kind, episodes, frames, inventory, prompts, search_extra=""):
    episode_cell = "—" if episodes is None else f"{episodes:,}"
    frame_cell = "—" if frames is None else f"{frames:,}"
    muted = f'<div class="muted">{path}</div>' if path else ""
    extra = f'<div class="muted">{search_extra}</div>' if search_extra else ""
    return (
        f'<tr><td id="{name}" data-text="{name} {prompts[1]}">'
        f"<code>{name}</code>{muted}{extra}</td>"
        f'<td><span class="tag">{tag}</span></td><td>{kind}</td>'
        f'<td class="num">{episode_cell}</td><td class="num">{frame_cell}</td>'
        f"<td>{inventory}</td><td>{prompts[0]}</td></tr>"
    )


def totals(counts):
    return sum(item["episodes"] for item in counts.values()), sum(item["frames"] for item in counts.values())


def counts_for(root):
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    counts = {}
    for episode in episodes:
        task = episode["tasks"][0]
        entry = counts.setdefault(task, {"episodes": 0, "frames": 0})
        entry["episodes"] += 1
        entry["frames"] += episode["length"]
    return counts


def update_html(report):
    html = HTML.read_text()
    gloss = glosses(html)
    sheet = {RIGHT: report["task_counts"][RIGHT], LEFT: report["task_counts"][LEFT]}
    # Sheet counts in the merged report include only the added episodes for those two new prompts.
    train_counts = counts_for(ADDITION)
    test_counts = counts_for(Path("/vePFS/zhuoyu/OpenWAM/data/linglong_20261019_eef_test_15hz"))
    openwam_train = counts_for(Path("/vePFS/zhuoyu/OpenWAM/data/linglong_20261019_openwam_train_15hz"))
    openwam_test = counts_for(Path("/vePFS/zhuoyu/OpenWAM/data/linglong_20261019_openwam_test_15hz"))
    if train_counts != sheet or train_counts != openwam_train or test_counts != openwam_test:
        raise ValueError("20261019 prompt counts differ across joint and EEF buckets")
    if totals(train_counts) != (91, 15957) or totals(test_counts) != (10, 1501):
        raise ValueError(f"unexpected 20261019 totals {totals(train_counts)} {totals(test_counts)}")
    merged_counts = report["task_counts"]
    for task in merged_counts:
        if task not in gloss:
            raise KeyError(task)

    def items(counts, with_counts):
        return prompt_items(counts, gloss, with_counts)

    section = (
        '<h2 id="g-20261019">20261019 薄片工件</h2><table><thead><tr>'
        '<th>数据集</th><th>用途</th><th>内容</th><th class="num">条数</th><th class="num">帧数</th>'
        "<th>清单</th><th>prompt</th></tr></thead><tbody>"
        + row(
            "linglong_20261019_processed",
            "/vePFS/data/linglong/linglong_20261019_processed",
            "解码中间结果", "解码中间结果", None, None, "—",
            items({task: {"episodes": 0, "frames": 0} for task in (RIGHT, LEFT)}, False),
            "训练源 91 · 测试源 10 · 不含 122、123",
        )
        + row(
            "linglong_20261019_openwam_train_15hz",
            "/vePFS/zhuoyu/OpenWAM/data/linglong_20261019_openwam_train_15hz",
            "OpenWAM 导出", "15 Hz · 关节 24 维", *totals(openwam_train), "训练清单 91",
            items(openwam_train, True),
        )
        + row(
            "linglong_20261019_openwam_test_15hz",
            "/vePFS/zhuoyu/OpenWAM/data/linglong_20261019_openwam_test_15hz",
            "OpenWAM 导出", "15 Hz · 关节 24 维", *totals(openwam_test), "测试清单 10",
            items(openwam_test, True),
        )
        + row(
            "linglong_20261019_eef_train_15hz",
            str(ADDITION),
            "训练", "15 Hz · 关节 24 维 + 末端 16 维", *totals(train_counts), "训练清单 91",
            items(train_counts, True),
            "action_eef 与 state_eef 都是 FK",
        )
        + row(
            "linglong_20261019_eef_test_15hz",
            "/vePFS/zhuoyu/OpenWAM/data/linglong_20261019_eef_test_15hz",
            "测试", "15 Hz · 关节 24 维 + 末端 16 维", *totals(test_counts), "测试清单 10",
            items(test_counts, True),
            "每只手最后 5 条；归一化沿用训练集",
        )
        + "</tbody></table>"
        '<h2 id="g-0923-1010-1012-1017-20261019">0923+1010+1012+1017+20261019</h2><table><thead><tr>'
        '<th>数据集</th><th>用途</th><th>内容</th><th class="num">条数</th><th class="num">帧数</th>'
        "<th>清单</th><th>prompt</th></tr></thead><tbody>"
        + row(
            OUTPUT.name,
            str(OUTPUT),
            "合并", "15 Hz · 关节 24 维 + 末端 16 维",
            report["episodes"], report["frames"], f"训练清单 {report['episodes']}",
            items(merged_counts, True),
            "20261019 只并入训练集。其 state_eef 来自 FK(measured q)，其余采集的 state_eef 来自录制 /end_pos。不含 122、123。",
        )
        + "</tbody></table>"
    )
    if 'id="g-20261019"' in html or f'id="{OUTPUT.name}"' in html:
        raise ValueError("catalog already contains the 20261019 section")
    html = html.replace("快照时间 2026-10-18。", "快照时间 2026-10-20。", 1)
    html = html.replace("<b>31</b><span>个 LeRobot 目录</span>", "<b>36</b><span>个 LeRobot 目录</span>", 1)
    html = html.replace("<b>2</b><span>个解码中间目录</span>", "<b>3</b><span>个解码中间目录</span>", 1)
    html = html.replace(
        "</nav>",
        '<a href="#g-20261019">20261019 薄片工件（5）</a>'
        '<a href="#g-0923-1010-1012-1017-20261019">0923+1010+1012+1017+20261019（1）</a></nav>',
        1,
    )
    html = html.replace("<script>", section + "<script>", 1)
    HTML.write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    portable = load_bucket(PORTABLE)
    addition = load_bucket(ADDITION)
    report = merge(portable, addition, args.output.resolve())
    update_html(report)
    print(json.dumps({key: report[key] for key in ("episodes", "frames", "windows", "collections")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
