"""Add Linglong EEF columns with the existing FK and check them on old data.

The original ``linglong.zip`` is not on this machine. The arm/waist chain below
is the fixed installation recorded in
``video-prediction-policy-2/docs/linglong_fk_matrices_episode000000_frame000000.json``,
whose rotations are identity. ``LinglongKinematics`` then applies the same
wrist TCP and wxyz layout as ``action_to_eef``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

VPP2_SRC = Path("/vePFS/tangyuting/video-prediction-policy-2/src")
CHAIN_JSON = Path(
    "/vePFS/tangyuting/video-prediction-policy-2/docs/linglong_fk_matrices_episode000000_frame000000.json"
)
OLD_ROOT = Path("/vePFS/zhuoyu/OpenWAM/data")
NEW_ROOT = Path("/vePFS/zhuoyu/OpenWAM/data")
sys.path.insert(0, str(VPP2_SRC))
from vpp2.linglong.eef import LinglongKinematics, make_quaternion_continuous  # noqa: E402


def _chain_zip(path: Path) -> None:
    nodes = json.loads(CHAIN_JSON.read_text())["nodes"]
    children: dict[str, list[str]] = {}
    for name, node in nodes.items():
        if name.endswith("_tcp"):
            continue
        children.setdefault(node["parent"], []).append(name)
    for kids in children.values():
        kids.sort()

    def emit(name: str, indent: int) -> list[str]:
        node = nodes[name]
        fixed = np.asarray(node["fixed_installation"], dtype=np.float64)
        if not np.allclose(fixed[:3, :3], np.eye(3), atol=1e-12):
            raise ValueError(f"{name} installation rotation is not identity")
        pos = " ".join(f"{value:.16g}" for value in fixed[:3, 3])
        lines = [" " * indent + f'<body name="{name}" pos="{pos}">']
        for joint in node["joints"]:
            axis = " ".join(str(int(value)) for value in joint["axis"])
            lines.append(
                " " * (indent + 2)
                + f'<joint name="{joint["name"]}" type="hinge" axis="{axis}"/>'
            )
        for child in children.get(name, []):
            lines.extend(emit(child, indent + 2))
        lines.append(" " * indent + "</body>")
        return lines

    xml = "\n".join([
        "<mujoco>",
        '  <compiler angle="radian"/>',
        "  <worldbody>",
        *emit("base_link", 4),
        "  </worldbody>",
        "</mujoco>",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("linglong/linglong-h-stack-parts.xml", xml)


def _align(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    aligned = pred.copy()
    for start in (3, 11):
        sign = np.where(np.sum(aligned[:, start:start + 4] * ref[:, start:start + 4], axis=1) < 0, -1.0, 1.0)
        aligned[:, start:start + 4] *= sign[:, None]
    return aligned


def _rotation_deg(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    angles = []
    for start in (3, 11):
        pred_q = pred[:, start:start + 4]
        ref_q = ref[:, start:start + 4]
        pred_q = pred_q / np.linalg.norm(pred_q, axis=1, keepdims=True)
        ref_q = ref_q / np.linalg.norm(ref_q, axis=1, keepdims=True)
        dot = np.clip(np.abs(np.sum(pred_q * ref_q, axis=1)), 0.0, 1.0)
        angles.append(np.rad2deg(2.0 * np.arccos(dot)))
    return np.concatenate(angles)


def _summarize(position_mm: np.ndarray, rotation_deg: np.ndarray) -> dict:
    def pack(values: np.ndarray) -> dict:
        return {
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }
    return {"frames": int(len(position_mm)), "position_mm": pack(position_mm), "orientation_deg": pack(rotation_deg)}


def validate(fk: LinglongKinematics) -> dict:
    report = {}
    for split in ("train", "test"):
        root = OLD_ROOT / f"linglong_0923_1010_1012_{split}_15hz"
        paths = sorted((root / "data").rglob("*.parquet"))
        pos_err, rot_err, state_pos, state_rot = [], [], [], []
        exact = []
        for path in paths:
            table = pq.read_table(path, columns=["action", "observation.state", "action_eef", "observation.state_eef"])
            action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
            stored_action = np.asarray(table["action_eef"].to_pylist(), dtype=np.float64)
            stored_state = np.asarray(table["observation.state_eef"].to_pylist(), dtype=np.float64)
            pred_action = make_quaternion_continuous(fk.action_to_eef(action))
            pred_state = make_quaternion_continuous(fk.state_to_eef(state))
            aligned_action = _align(pred_action, stored_action)
            exact.append(float(np.max(np.abs(aligned_action - stored_action))))
            pos_err.append(np.linalg.norm(aligned_action[:, 0:3] - stored_action[:, 0:3], axis=1) * 1000)
            pos_err.append(np.linalg.norm(aligned_action[:, 8:11] - stored_action[:, 8:11], axis=1) * 1000)
            rot_err.append(_rotation_deg(aligned_action, stored_action))
            aligned_state = _align(pred_state, stored_state)
            state_pos.append(np.linalg.norm(aligned_state[:, 0:3] - stored_state[:, 0:3], axis=1) * 1000)
            state_pos.append(np.linalg.norm(aligned_state[:, 8:11] - stored_state[:, 8:11], axis=1) * 1000)
            state_rot.append(_rotation_deg(aligned_state, stored_state))
            np.testing.assert_allclose(pred_action[:, [7, 15]], action[:, [7, 15]], atol=1e-6)
        action_pos = np.concatenate(pos_err)
        action_rot = np.concatenate(rot_err)
        # 0923/1010 match to float32 noise. 1012 keeps the same orientation and
        # stays within 0.2 mm, above that noise floor but far below the
        # recorded-pose check.
        if action_pos.max() > 0.25 or action_rot.max() > 0.01:
            raise ValueError(f"{split} FK does not reproduce stored action_eef")
        report[split] = {
            "episodes": len(paths),
            "action_eef_vs_fk_q_exp": _summarize(action_pos, action_rot),
            "max_abs_component_after_sign_align": max(exact),
            "state_eef_recorded_vs_fk_q": _summarize(np.concatenate(state_pos), np.concatenate(state_rot)),
        }
        print(split, json.dumps(report[split]), flush=True)
    return report


def _stats(values: np.ndarray) -> dict:
    return {
        "global_mean": values.mean(0).tolist(),
        "global_std": values.std(0, ddof=1).tolist(),
        "global_min": values.min(0).tolist(),
        "global_max": values.max(0).tolist(),
        "global_q01": np.quantile(values, 0.01, axis=0).tolist(),
        "global_q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def convert_split(fk: LinglongKinematics, split: str, train_stats: dict | None) -> dict:
    source = NEW_ROOT / f"linglong_20261019_openwam_{split}_15hz"
    target = NEW_ROOT / f"linglong_20261019_eef_{split}_15hz"
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((source / "manifest.json").read_text())
    shutil.copytree(source, target, symlinks=True, ignore=lambda directory, names: ["data"] if Path(directory) == source else [])
    for link in source.rglob("*"):
        if link.is_symlink():
            copied = target / link.relative_to(source)
            destination = link.resolve(strict=True)
            copied.unlink()
            copied.symlink_to(os.path.relpath(destination, copied.parent), target_is_directory=destination.is_dir())
    action_arrays, state_arrays = [], []
    for path in sorted((source / "data").rglob("*.parquet")):
        original = pq.read_table(path)
        action = np.asarray(original["action"].to_pylist(), dtype=np.float64)
        state = np.asarray(original["observation.state"].to_pylist(), dtype=np.float64)
        action_eef = make_quaternion_continuous(fk.action_to_eef(action)).astype(np.float32)
        state_eef = make_quaternion_continuous(fk.state_to_eef(state)).astype(np.float32)
        np.testing.assert_array_equal(action_eef[:, [7, 15]], action[:, [7, 15]].astype(np.float32))
        converted = original.append_column("action_eef", pa.FixedSizeListArray.from_arrays(pa.array(action_eef.ravel()), 16))
        converted = converted.append_column("observation.state_eef", pa.FixedSizeListArray.from_arrays(pa.array(state_eef.ravel()), 16))
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(converted, destination, compression="zstd")
        action_arrays.append(action_eef)
        state_arrays.append(state_eef)
    action_values = np.concatenate(action_arrays).astype(np.float64)
    state_values = np.concatenate(state_arrays).astype(np.float64)
    if train_stats is None:
        train_stats = {"action": _stats(action_values), "state": _stats(state_values), "frames": len(action_values)}
        _write_json(target / "meta/action_eef_stats.json", {
            "source_column": "action_eef", "statistics_source": "train frames once, no window weighting",
            "num_episodes": manifest["total_episodes"], "num_transition": len(action_values), "std_ddof": 1,
            "action": {"default": train_stats["action"]},
        })
        _write_json(target / "meta/state_eef_stats.json", {
            "source_column": "observation.state_eef", "statistics_source": "train frames once, no window weighting",
            "num_episodes": manifest["total_episodes"], "num_transition": len(state_values), "std_ddof": 1,
            "state": {"default": train_stats["state"]},
        })
    else:
        for name in ("action_eef_stats.json", "state_eef_stats.json"):
            shutil.copyfile(NEW_ROOT / "linglong_20261019_eef_train_15hz/meta" / name, target / "meta" / name)
    info = json.loads((target / "meta/info.json").read_text())
    info["features"]["action_eef"] = {"dtype": "float32", "shape": [16]}
    info["features"]["observation.state_eef"] = {"dtype": "float32", "shape": [16]}
    _write_json(target / "meta/info.json", info)
    manifest.update(
        source_root=manifest["source_root"],
        eef_source_bucket=str(source.resolve()),
        action_eef={
            "column": "action_eef", "shape": [16], "dtype": "float32", "source": "FK(raw absolute q_exp)",
            "reference": "base_link", "position_unit": "metre", "quaternion_order": "wxyz",
            "layout": "left_xyz3_quat_wxyz4_gripper,right_xyz3_quat_wxyz4_gripper",
            "wrist_local_tcp_m": fk.wrist_to_tcp.tolist(),
            "quaternion_sign": "continuous independently per hand within each episode",
            "normalization_stats": "meta/action_eef_stats.json", "stored_values": "raw, unnormalized",
        },
        observation_state_eef={
            "column": "observation.state_eef", "shape": [16], "dtype": "float32",
            "source": "FK(measured q)", "reference": "base_link", "position_unit": "metre",
            "quaternion_order": "wxyz", "kinematics_used": True,
            "layout": "left_xyz3_quat_wxyz4_gripper,right_xyz3_quat_wxyz4_gripper",
            "gripper_source": "measured observation.state indices 7,15",
            "normalization_stats": "meta/state_eef_stats.json", "normalization_split": "train",
        },
    )
    _write_json(target / "manifest.json", manifest)
    return {"episodes": manifest["total_episodes"], "frames": int(len(action_values)), "output": str(target)}


def main() -> None:
    model = Path("/tmp/linglong_chain_from_fk_export.zip")
    _chain_zip(model)
    fk = LinglongKinematics(model)
    validation = validate(fk)
    converted = {
        "train": convert_split(fk, "train", None),
        "test": convert_split(fk, "test", {}),
    }
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    report = {
        "status": "complete",
        "kinematics": "vpp2.linglong.eef.LinglongKinematics",
        "chain_source": str(CHAIN_JSON),
        "model_zip_sha256": digest,
        "original_linglong_zip": "absent; chain taken from the exported FK matrices",
        "validation_against_0923_1010_1012": validation,
        "outputs": converted,
    }
    _write_json(NEW_ROOT / "linglong_20261019_eef_train_15hz/eef_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
