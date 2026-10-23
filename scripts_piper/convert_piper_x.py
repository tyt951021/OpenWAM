"""Convert Piper X segments into Linglong-compatible 16D EEF buckets.

Each arm is expressed in its own URDF ``base_link``. State comes from the
recorded pose. Action comes from forward kinematics of the recorded joint
command, using the same tool point. Measured joints are checked against the
recorded pose before any bucket is written. Gripper metres map to ``[0, 1]``
with 0 closed. The 16D layout is left ``xyz + wxyz + gripper``, then right.

One episode cannot be held out without emptying the training bucket. With the
default test fraction, that episode is written to both buckets and the test
manifest records ``single_episode_reused_for_test``. Two or more episodes use
the Linglong episode holdout and are not copied across splits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_piper.kinematics import PiperXKinematics, make_quaternion_continuous


FPS = 15
NUM_FRAMES = 33
EEF_DIM = 16
CAMERAS = (("front", "head"), ("left_wrist", "left"), ("right_wrist", "right"))
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
SIDES = ("left", "right")


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_list(values, width):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError(f"expected finite (*, {width}) array, got {values.shape}")
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), width)


def _stats(arrays, ddof=0):
    values = np.concatenate([np.asarray(item, dtype=np.float64) for item in arrays])
    return {
        "count": int(len(values)),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0, ddof=ddof).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def discover_segments(input_path: Path) -> list[Path]:
    input_path = Path(input_path).expanduser().resolve(strict=True)
    if (input_path / "samples.jsonl").is_file():
        return [input_path]
    segments = sorted(
        path for path in input_path.iterdir()
        if path.is_dir() and (path / "samples.jsonl").is_file()
    )
    if not segments:
        raise ValueError(f"no segment with samples.jsonl under {input_path}")
    return segments


def _load_segment(segment: Path):
    manifest_path = segment / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{segment} is missing manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("fps", 0)) != FPS:
        raise ValueError(f"{segment.name}: manifest fps must be {FPS}")
    rows = [json.loads(line) for line in (segment / "samples.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) < NUM_FRAMES:
        raise ValueError(f"{segment.name}: {len(rows)} frames; OpenWAM requires at least {NUM_FRAMES}")
    timestamps = np.array([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{segment.name}: timestamps must increase")
    median_dt = float(np.median(np.diff(timestamps)) / 1e9)
    if abs(median_dt - 1.0 / FPS) > 0.002:
        raise ValueError(f"{segment.name}: median frame interval {median_dt:.6f}s is not {FPS} Hz")
    return manifest, rows


def _column(rows, side):
    names = rows[0]["arms"][side]["joints"]["name"]
    if names != JOINT_NAMES:
        raise ValueError(f"{side} joint names must be {JOINT_NAMES}, got {names}")
    if rows[0]["actions"][side]["name"] != JOINT_NAMES:
        raise ValueError(f"{side} action names must be {JOINT_NAMES}")
    measured = np.asarray([row["arms"][side]["joints"]["position"] for row in rows], dtype=np.float64)
    command = np.asarray([row["actions"][side]["position"] for row in rows], dtype=np.float64)
    if measured.shape[1] != 7 or command.shape != measured.shape or not np.isfinite(measured).all():
        raise ValueError(f"{side} joints must be finite (frames, 7)")
    if not np.isfinite(command).all():
        raise ValueError(f"{side} commands must be finite")
    position = np.asarray([row["arms"][side]["pose"]["position"] for row in rows], dtype=np.float64)
    orientation = np.asarray([row["arms"][side]["pose"]["orientation_xyzw"] for row in rows], dtype=np.float64)
    width = np.asarray([row["arms"][side]["gripper"]["width"] for row in rows], dtype=np.float64)
    return measured, command, position, orientation, width


def _image_sources(segment: Path, rows):
    sources = {dest: [] for _, dest in CAMERAS}
    names = []
    for row in rows:
        sample_id = str(row["sample_id"])
        names.append(sample_id)
        for source_name, dest in CAMERAS:
            path = segment / row["images"][source_name]
            if not path.is_file():
                raise FileNotFoundError(path)
            sources[dest].append(path)
    return names, sources


def build_episode(segment: Path, fk: PiperXKinematics, prompt: str | None, entry: dict):
    manifest, rows = _load_segment(segment)
    task = entry.get("prompt", prompt if prompt is not None else manifest.get("task"))
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"{segment.name}: pass --prompt or a non-empty manifest task")
    split = entry.get("split")
    if split not in (None, "train", "test"):
        raise ValueError(f"{segment.name}: split must be train or test")
    state = np.empty((len(rows), EEF_DIM), dtype=np.float64)
    action = np.empty((len(rows), EEF_DIM), dtype=np.float64)
    fk_report = {}
    for offset, side in ((0, "left"), (8, "right")):
        measured, command, position, orientation, width = _column(rows, side)
        fk_report[side] = fk.assert_matches_state(measured[:, :6], position, orientation, label=f"{segment.name} {side}")
        commanded_xyz, commanded_wxyz = fk.poses(command[:, :6])
        state[:, offset:offset + 3] = position
        state[:, offset + 3:offset + 7] = orientation[:, [3, 0, 1, 2]]
        state[:, offset + 7] = fk.gripper_aperture(width)
        action[:, offset:offset + 3] = commanded_xyz
        action[:, offset + 3:offset + 7] = commanded_wxyz
        action[:, offset + 7] = fk.gripper_aperture(command[:, 6])
    state = make_quaternion_continuous(state)
    action = make_quaternion_continuous(action)
    for name, values in (("state", state), ("action", action)):
        for start in (3, 11):
            norms = np.linalg.norm(values[:, start:start + 4], axis=1)
            if not np.allclose(norms, 1.0, atol=1e-6):
                raise ValueError(f"{segment.name}: {name} quaternion left unit norm")
    names, sources = _image_sources(segment, rows)
    origins = {}
    if rows and isinstance(rows[0].get("action_sources"), dict):
        for side in SIDES:
            counts = {}
            for row in rows:
                origin = row["action_sources"][side]["gripper"]["origin"]
                counts[origin] = counts.get(origin, 0) + 1
            origins[side] = counts
    return {
        "segment": segment.name,
        "source_episode": str(manifest.get("episode", segment.name)),
        "source_id": str(segment),
        "prompt": task.strip(),
        "split": split,
        "sha256": _sha256(segment / "samples.jsonl"),
        "state": state.astype(np.float32),
        "action": action.astype(np.float32),
        "sample_ids": names,
        "image_sources": sources,
        "fk_report": fk_report,
        "gripper_command_origins": origins,
        "frames": len(rows),
    }


def assign_splits(records, test_fraction, seed):
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1)")
    unassigned = sorted((record for record in records if record["split"] is None), key=lambda record: record["sha256"])
    count = (min(len(unassigned) - 1, max(1, round(len(unassigned) * test_fraction)))
             if test_fraction and len(unassigned) > 1 else 0)
    order = np.random.default_rng(seed).permutation(len(unassigned))
    test_hashes = {unassigned[int(index)]["sha256"] for index in order[:count]}
    reused = False
    for record in unassigned:
        record["split"] = "test" if record["sha256"] in test_hashes else "train"
    train = [record for record in records if record["split"] == "train"]
    test = [record for record in records if record["split"] == "test"]
    if not train:
        raise ValueError("at least one training episode is required")
    if test_fraction and not test and len(train) == 1 and len(records) == 1:
        test = list(train)
        reused = True
    test_ids = {id(record) for record in test}
    for record in records:
        record["reuse_for_test"] = reused and id(record) in test_ids
    return train, test, reused


def _contracts(fk: PiperXKinematics):
    common = {
        "shape": [EEF_DIM],
        "dtype": "float32",
        "reference": "base_link",
        "frame": "per_arm_base_link",
        "position_unit": "metre",
        "quaternion_order": "wxyz",
        "layout": "left_xyz3_quat_wxyz4_gripper,right_xyz3_quat_wxyz4_gripper",
        "gripper_convention": "0_closed_1_open",
        "gripper_open_m": fk.gripper_open_m,
        "normalization_split": "train",
        "quaternion_sign": "continuous per hand per episode",
        "supervised_arm": "both",
    }
    return {
        "action_eef": {
            **common,
            "column": "action_eef",
            "source": "FK of recorded absolute joint commands through tcp_link",
            "kinematics_used": True,
            "urdf": "scripts_piper/urdf/piper_x_kinematics.urdf",
        },
        "observation_state_eef": {
            **common,
            "column": "observation.state_eef",
            "source": "recorded pose; gripper width mapped by the URDF upper limit",
            "kinematics_used": False,
            "state_check": "FK of measured joints through tcp_link must match the recorded pose",
        },
    }


def _place_image(pair):
    source, dest = pair
    try:
        os.link(source, dest)
    except OSError:
        shutil.copyfile(source, dest)


def _write_bucket(root: Path, split: str, records, tasks, train_stats, contracts, provenance, reused):
    root.mkdir(parents=True)
    episodes, mapping, offset = [], [], 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        for index, record in enumerate(records):
            image_paths = {}
            for camera, sources in record["image_sources"].items():
                dest_dir = root / "images" / f"episode_{index:06d}" / camera
                dest_dir.mkdir(parents=True, exist_ok=True)
                pairs = []
                relative = []
                for sample_id, source in zip(record["sample_ids"], sources):
                    dest = dest_dir / f"{sample_id}.jpg"
                    pairs.append((source, dest))
                    relative.append(dest.relative_to(root).as_posix())
                list(pool.map(_place_image, pairs))
                image_paths[camera] = relative
            print(f"linked {split} episode {index}: {record['frames']} frames", flush=True)
            n = record["frames"]
            task_index = tasks.index(record["prompt"])
            values = {
                "observation.state_eef": _fixed_list(record["state"], EEF_DIM),
                "action_eef": _fixed_list(record["action"], EEF_DIM),
                "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "episode_index": pa.array(np.full(n, index, dtype=np.int64)),
                "index": pa.array(np.arange(offset, offset + n, dtype=np.int64)),
                "task_index": pa.array(np.full(n, task_index, dtype=np.int64)),
                **{f"observation.images.{camera}": pa.array(image_paths[camera]) for camera in image_paths},
            }
            path = root / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(values), path, compression="zstd")
            episodes.append({
                "episode_index": index,
                "episode_id": record["source_episode"],
                "length": n,
                "tasks": [record["prompt"]],
                "task_index": task_index,
            })
            mapping.append({
                "episode_index": index,
                "source_id": record["source_id"],
                "source_episode": record["source_episode"],
                "segment": record["segment"],
                "sha256": record["sha256"],
                "frames": n,
                "fk_report": record["fk_report"],
                "gripper_command_origins": record["gripper_command_origins"],
                "reuse_for_test": record["reuse_for_test"],
            })
            offset += n
    _write_jsonl(root / "meta/episodes.jsonl", episodes)
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": i, "task": task} for i, task in enumerate(tasks)])
    _write_jsonl(root / "meta/episodes_stats.jsonl", [
        {"episode_index": episode["episode_index"], "length": episode["length"]} for episode in episodes
    ])
    features = {
        "observation.state_eef": {"dtype": "float32", "shape": [EEF_DIM]},
        "action_eef": {"dtype": "float32", "shape": [EEF_DIM]},
        **{f"observation.images.{camera}": {"dtype": "image"} for _, camera in CAMERAS},
    }
    _write_json(root / "meta/info.json", {
        "codebase_version": "openwam_piper_x_eef_v1",
        "robot_type": "piper_x",
        "total_episodes": len(records),
        "total_frames": offset,
        "total_tasks": len(tasks),
        "fps": FPS,
        "splits": {split: f"0:{len(records)}"},
        "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    })
    _write_json(root / "meta/normalization_stats.json", {
        "state": train_stats["state_flat"],
        "action": train_stats["action_flat"],
    })
    _write_json(root / "meta/action_eef_stats.json", train_stats["action_eef"])
    _write_json(root / "meta/state_eef_stats.json", train_stats["state_eef"])
    _write_json(root / "source_mapping.json", mapping)
    manifest = {
        "format": "openwam_piper_x_eef_v1",
        "status": "complete",
        "split": split,
        "source_root": ".",
        "prompt": tasks[0],
        "fps": FPS,
        "num_frames": NUM_FRAMES,
        "video_stride": 4,
        "total_episodes": len(records),
        "total_frames": offset,
        "total_windows": sum(record["frames"] - NUM_FRAMES + 1 for record in records),
        "image_storage": "JPEG files inside this bucket; paths relative to the bucket",
        "normalization": "train frames once; test reuses train statistics",
        "single_episode_reused_for_test": bool(reused and split == "test"),
        "active_raw_action_dims": list(range(EEF_DIM)),
        "active_raw_state_dims": list(range(EEF_DIM)),
        **contracts,
        **provenance,
    }
    _write_json(root / "manifest.json", manifest)
    return {"episodes": len(records), "frames": offset, "windows": manifest["total_windows"]}


def _build_episode_job(job):
    segment, prompt, entry = job
    return build_episode(Path(segment), PiperXKinematics(), prompt, entry)


def convert_piper_x(input_path, output_root, *, prompt=None, episode_config=None, test_fraction=0.1, seed=0):
    input_path = Path(input_path).expanduser().resolve(strict=True)
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"choose a fresh output directory: {output_root}")
    if output_root == input_path or input_path in output_root.parents or output_root in input_path.parents:
        raise ValueError("output must be outside the input directory")
    config = {} if episode_config is None else episode_config
    if not isinstance(config, dict):
        raise ValueError("episode config must be an object")
    segments = discover_segments(input_path)
    jobs = []
    used = set()
    for segment in segments:
        if segment.name in config:
            key = segment.name
        else:
            manifest = json.loads((segment / "manifest.json").read_text())
            key = str(manifest.get("episode", segment.name))
        entry = config.get(key, {})
        if key in config:
            used.add(key)
        if not isinstance(entry, dict):
            raise ValueError(f"episode config entry must be an object: {key}")
        jobs.append((str(segment), prompt, entry))
    records = [None] * len(jobs)
    workers = min(len(jobs), 32)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_episode_job, job): index for index, job in enumerate(jobs)}
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            records[index] = future.result()
            print(f"decoded {done}/{len(jobs)} {records[index]['segment']}: {records[index]['frames']} frames", flush=True)
    if set(config) - used:
        raise ValueError(f"unmatched episode-config keys: {sorted(set(config) - used)}")
    train, test, reused = assign_splits(records, test_fraction, seed)
    output_root.mkdir(parents=True)
    state_flat = _stats([record["state"] for record in train])
    action_flat = _stats([record["action"] for record in train])
    state_eef = _stats([record["state"] for record in train], ddof=1)
    action_eef = _stats([record["action"] for record in train], ddof=1)
    train_stats = {
        "state_flat": state_flat,
        "action_flat": action_flat,
        "state_eef": {
            "source_column": "observation.state_eef",
            "statistics_source": "train frames once, no window weighting",
            "num_episodes": len(train),
            "num_transition": state_eef["count"],
            "std_ddof": 1,
            "state": {"default": {f"global_{key}": value for key, value in state_eef.items() if key != "count"}},
        },
        "action_eef": {
            "source_column": "action_eef",
            "statistics_source": "train frames once, no window weighting",
            "num_episodes": len(train),
            "num_transition": action_eef["count"],
            "std_ddof": 1,
            "action": {"default": {f"global_{key}": value for key, value in action_eef.items() if key != "count"}},
        },
    }
    tasks = sorted({record["prompt"] for record in records})
    fk = PiperXKinematics()
    provenance = {
        "raw_input": str(input_path),
        "urdf_sha256": _sha256(fk.urdf_path),
        "split_method": "explicit episode config; otherwise seeded shuffle of samples.jsonl hashes",
        "split_seed": seed,
        "test_fraction": test_fraction,
        "single_episode_reused_for_test": reused,
    }
    contracts = _contracts(fk)
    results = {}
    for split, rows in (("train", train), ("test", test)):
        if not rows:
            continue
        bucket = output_root / f"piper_x_openwam_{split}_15hz"
        results[split] = _write_bucket(bucket, split, rows, tasks, train_stats, contracts, provenance, reused)
    report = {"status": "complete", "input": str(input_path), "output": str(output_root), "splits": results,
              "single_episode_reused_for_test": reused, "fk": {record["segment"]: record["fk_report"] for record in records}}
    _write_json(output_root / "conversion.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="segment directory or a directory of segments")
    parser.add_argument("--output-root", type=Path, required=True, help="fresh directory for the train and test buckets")
    parser.add_argument("--prompt", help="task text used when a segment manifest has no task")
    parser.add_argument("--episode-config", type=Path, help='JSON {"segment or episode id": {"prompt": "...", "split": "train|test"}}')
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.episode_config.read_text()) if args.episode_config else None
    convert_piper_x(args.input, args.output_root, prompt=args.prompt, episode_config=config,
                    test_fraction=args.test_fraction, seed=args.seed)


if __name__ == "__main__":
    main()
