import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts_piper.convert_piper_x import convert_piper_x
from scripts_piper.kinematics import PiperXKinematics


ZERO_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ZERO_POSITION = [0.22640408890517927, 7.930022094730974e-18, 0.22813690377085494]
ZERO_XYZW = [-0.4777553610960826, 0.4777553610960825, -0.5212962832631284, 0.5212962832631285]
FRAME_JOINTS = [
    0.6384239870870058, 2.265315196041, -1.9497422139879055,
    0.8126951128986396, 0.2054950661298124, 0.2805267706730486,
]
FRAME_POSITION = [0.5023168470512395, 0.3307783435054392, 0.07093152297548508]
FRAME_XYZW = [-0.7735537098281366, 0.5692413578527393, -0.17713544994582525, 0.21494642796191446]


def test_recorded_poses_match_tcp():
    fk = PiperXKinematics()
    assert fk.gripper_open_m == 0.1
    for joints, position, orientation in (
        (ZERO_JOINTS, ZERO_POSITION, ZERO_XYZW),
        (FRAME_JOINTS, FRAME_POSITION, FRAME_XYZW),
    ):
        report = fk.assert_matches_state(np.asarray([joints]), np.asarray([position]), np.asarray([orientation]))
        assert report["position_error_mm_max"] < 5
        assert report["orientation_error_deg_max"] < 1.5


def test_gripper_aperture_uses_urdf_stroke():
    fk = PiperXKinematics()
    assert fk.gripper_aperture(0.0) == 0.0
    assert fk.gripper_aperture(0.1) == 1.0
    assert abs(float(fk.gripper_aperture(0.0003)) - 0.003) < 1e-9
    try:
        fk.gripper_aperture(0.2)
    except ValueError as exc:
        assert "0.1" in str(exc)
    else:
        raise AssertionError("opening past the URDF limit must fail")


def _segment(root: Path, name: str, frames: int, command_scale: float):
    fk = PiperXKinematics()
    segment = root / name
    for camera in ("front", "left_wrist", "right_wrist"):
        (segment / "images" / camera).mkdir(parents=True)
    rows = []
    start = 1_000_000_000
    step = round(1e9 / 15)
    joints = np.zeros(6)
    command = joints + command_scale
    xyz, wxyz = fk.poses(joints)
    xyzw = np.asarray(wxyz)[[1, 2, 3, 0]]
    for index in range(frames):
        sample = f"{index:06d}"
        for camera in ("front", "left_wrist", "right_wrist"):
            Image.new("RGB", (8, 8), (index, 0, 0)).save(segment / "images" / camera / f"{sample}.jpg")
        arm = {
            "joints": {"name": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
                       "position": [*joints.tolist(), 0.0]},
            "pose": {"position": xyz.tolist(), "orientation_xyzw": xyzw.tolist()},
            "gripper": {"width": 0.0},
        }
        rows.append({
            "sample_id": sample,
            "timestamp_ns": start + index * step,
            "arms": {"left": arm, "right": arm},
            "actions": {
                "left": {"name": arm["joints"]["name"], "position": [*command.tolist(), 0.0]},
                "right": {"name": arm["joints"]["name"], "position": [*command.tolist(), 0.1]},
            },
            "images": {camera: f"images/{camera}/{sample}.jpg" for camera in ("front", "left_wrist", "right_wrist")},
            "action_sources": {
                "left": {"gripper": {"origin": "recorded_command_hold"}},
                "right": {"gripper": {"origin": "recorded_command_hold"}},
            },
        })
    (segment / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (segment / "manifest.json").write_text(json.dumps({"task": "pick the block", "fps": 15, "episode": name}))
    return segment


def test_conversion_writes_train_and_reused_test(tmp_path):
    raw = tmp_path / "raw"
    _segment(raw, "segment_000", 33, 0.0)
    report = convert_piper_x(raw, tmp_path / "out")
    assert report["single_episode_reused_for_test"] is True
    train = tmp_path / "out" / "piper_x_openwam_train_15hz"
    test = tmp_path / "out" / "piper_x_openwam_test_15hz"
    train_manifest = json.loads((train / "manifest.json").read_text())
    test_manifest = json.loads((test / "manifest.json").read_text())
    assert train_manifest["split"] == "train"
    assert train_manifest["observation_state_eef"]["reference"] == "base_link"
    assert train_manifest["observation_state_eef"]["quaternion_order"] == "wxyz"
    assert test_manifest["single_episode_reused_for_test"] is True
    assert test_manifest["active_raw_action_dims"] == list(range(16))
    stats = json.loads((train / "meta" / "action_eef_stats.json").read_text())
    assert stats["source_column"] == "action_eef"
    assert len(stats["action"]["default"]["global_mean"]) == 16


def test_two_episodes_are_held_out_without_copying(tmp_path):
    raw = tmp_path / "raw"
    _segment(raw, "segment_000", 33, 0.0)
    _segment(raw, "segment_001", 33, 0.2)
    report = convert_piper_x(raw, tmp_path / "out", test_fraction=0.5)
    assert report["single_episode_reused_for_test"] is False
    assert report["splits"]["train"]["episodes"] == 1
    assert report["splits"]["test"]["episodes"] == 1


def test_rejects_pose_that_disagrees_with_fk(tmp_path):
    raw = tmp_path / "raw"
    segment = _segment(raw, "segment_000", 33, 0.0)
    rows = [json.loads(line) for line in (segment / "samples.jsonl").read_text().splitlines()]
    rows[0]["arms"]["right"]["pose"]["position"][0] += 0.02
    (segment / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    try:
        convert_piper_x(raw, tmp_path / "out")
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("a 20 mm pose error must stop conversion")
