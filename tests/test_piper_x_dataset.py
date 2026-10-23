import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from hydra import compose, initialize_config_dir

from omegaconf import OmegaConf

from openwam.dataloader.piper_x import PiperXDataset, _pack_eef20
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20
from openwam.dataloader.utils.relative_eef import absolute_to_relative, relative_to_absolute, restore_absolute_command
from openwam.deploy.engine import _restore_relative_piper_command
from openwam.dataloader.registry import build_dataset
from scripts_piper.convert_piper_x import convert_piper_x
from tests.test_piper_x import _segment


def _buckets(tmp_path: Path):
    raw = tmp_path / "raw"
    _segment(raw, "segment_000", 33, 0.0)
    convert_piper_x(raw, tmp_path / "out")
    return tmp_path / "out" / "piper_x_openwam_train_15hz", tmp_path / "out" / "piper_x_openwam_test_15hz"


def test_right_arm_window_uses_piper_bucket(tmp_path):
    train, test = _buckets(tmp_path)
    dataset = PiperXDataset(str(train), split="train")
    assert len(dataset) == 1
    assert dataset.action_dim == 80
    item = dataset[0]
    assert item["action"].shape == (32, 80)
    assert item["proprio"].shape == (1, 80)
    assert len(item["video"]) == 9
    assert item["video"][0].size == (320, 384)
    assert item["prompt"] == "pick the block"
    assert int(item["action_mask"][0].sum()) == 16
    assert bool(item["action_mask"][0, 0:7].all())
    assert bool(item["action_mask"][0, 9])
    assert bool(item["action_mask"][0, 34:41].all())
    assert bool(item["action_mask"][0, 43])
    evaluation = PiperXDataset(str(test), split="test", normalization_dataset_dir=str(train))
    assert len(evaluation) == 1
    assert evaluation[0]["episode_index"] == 0


def test_rotation_6d_uses_official_eef20_slots(tmp_path):
    train, _test = _buckets(tmp_path)
    dataset = PiperXDataset(str(train), split="train", action_mode="eef")
    item = dataset[0]
    assert int(item["action_mask"][0].sum()) == 20
    assert bool(item["action_mask"][0, 0:10].all())
    assert bool(item["action_mask"][0, 34:44].all())
    table = pq.read_table(train / "data/chunk-000/episode_000000.parquet", columns=["action_eef"])
    stored = np.asarray(table["action_eef"].to_pylist(), dtype=np.float32)[:32]
    packed = _pack_eef20(stored)
    np.testing.assert_allclose(item["action"][:, 3:9].numpy(), packed[:, 3:9], atol=1e-6)
    np.testing.assert_allclose(item["action"][:, 37:43].numpy(), packed[:, 13:19], atol=1e-6)
    rot6d = list(ROT6D_DIMS_EEF20)
    assert np.allclose(dataset._action_stats["min"][rot6d], -1.0)
    assert np.allclose(dataset._action_stats["max"][rot6d], 1.0)
    assert dataset.normalization_stats_payload["eef"] is dataset._action_stats


def _arm(position, quaternion, gripper):
    return [*position, *quaternion, gripper]


def _row(left, right):
    return np.asarray([*left, *right], dtype=np.float32)


def test_relative_pose_uses_the_input_state_frame():
    still = _arm((0.2, -0.1, 0.4), (1.0, 0.0, 0.0, 0.0), 0.25)
    identity = _row(still, still)
    same = absolute_to_relative(identity, identity)
    arms = same.reshape(2, 8)
    np.testing.assert_allclose(arms[:, :3], np.zeros((2, 3)), atol=1e-6)
    np.testing.assert_allclose(arms[:, 3:7], np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)), atol=1e-6)
    np.testing.assert_allclose(arms[:, 7], np.full(2, 0.25), atol=1e-6)
    quarter = _arm((0.2, -0.1, 0.4), (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)), 1.0)
    turned = _row(quarter, still)
    moved = turned.copy()
    moved[:3] += (1.0, 0.0, 0.0)
    relative = absolute_to_relative(moved, turned)
    np.testing.assert_allclose(relative[:3], (0.0, -1.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(relative[3:7], (1.0, 0.0, 0.0, 0.0), atol=1e-6)
    assert relative[7] == 1.0
    np.testing.assert_allclose(relative[8:11], 0.0, atol=1e-6)
    np.testing.assert_allclose(relative[11:15], (1.0, 0.0, 0.0, 0.0), atol=1e-6)
    assert relative[15] == 0.25
    np.testing.assert_allclose(relative_to_absolute(relative, turned), moved, atol=1e-6)


def test_relative_window_keeps_state_prompt_and_absolute_command(tmp_path):
    train, _test = _buckets(tmp_path)
    absolute = PiperXDataset(str(train), split="train", normalize_mode=None)
    relative = PiperXDataset(str(train), split="train", normalize_mode=None, relative_to_state=True)
    table = pq.read_table(
        train / "data/chunk-000/episode_000000.parquet",
        columns=["observation.state_eef", "action_eef"],
    )
    state = np.asarray(table["observation.state_eef"][0].as_py(), dtype=np.float32)
    action = np.asarray(table["action_eef"].to_pylist(), dtype=np.float32)[:32]
    expected = absolute_to_relative(action, state)
    item = relative[0]
    assert item["prompt"] == absolute[0]["prompt"]
    np.testing.assert_allclose(item["proprio"].numpy(), absolute[0]["proprio"].numpy())
    np.testing.assert_allclose(item["action"][:, 0:7].numpy(), expected[:, 0:7], atol=1e-6)
    np.testing.assert_allclose(item["action"][:, 9].numpy(), expected[:, 7], atol=1e-6)
    np.testing.assert_allclose(item["action"][:, 34:41].numpy(), expected[:, 8:15], atol=1e-6)
    np.testing.assert_allclose(item["action"][:, 43].numpy(), expected[:, 15], atol=1e-6)
    restored = restore_absolute_command(expected, state, "eef_quat")
    np.testing.assert_allclose(restored, action, atol=1e-5)
    cfg = OmegaConf.create({
        "dataloader": {"type": "piper_x", "relative_to_state": True, "action_mode": "eef_quat"},
    })
    output = _restore_relative_piper_command(cfg, state, {"actions": expected, "video": None})
    np.testing.assert_allclose(output["actions"], action, atol=1e-5)
    untouched = _restore_relative_piper_command(
        OmegaConf.create({"dataloader": {"type": "piper_x", "relative_to_state": False}}),
        state,
        {"actions": expected},
    )
    assert untouched["actions"] is expected


def test_relative_rot6d_restores_the_absolute_command(tmp_path):
    train, _test = _buckets(tmp_path)
    dataset = PiperXDataset(str(train), split="train", action_mode="eef", normalize_mode=None, relative_to_state=True)
    table = pq.read_table(
        train / "data/chunk-000/episode_000000.parquet",
        columns=["observation.state_eef", "action_eef"],
    )
    state = np.asarray(table["observation.state_eef"][0].as_py(), dtype=np.float32)
    action = np.asarray(table["action_eef"].to_pylist(), dtype=np.float32)[:32]
    packed = _pack_eef20(absolute_to_relative(action, state))
    item = dataset[0]
    np.testing.assert_allclose(item["action"][:, 3:9].numpy(), packed[:, 3:9], atol=1e-5)
    np.testing.assert_allclose(item["action"][:, 37:43].numpy(), packed[:, 13:19], atol=1e-5)
    restored = restore_absolute_command(packed, state, "eef")
    np.testing.assert_allclose(restored[:, 0:3], action[:, 0:3], atol=1e-5)
    np.testing.assert_allclose(restored[:, 8:11], action[:, 8:11], atol=1e-5)
    np.testing.assert_allclose(restored[:, 7], action[:, 7], atol=1e-6)
    np.testing.assert_allclose(restored[:, 15], action[:, 15], atol=1e-6)
    for start in (3, 11):
        dots = np.abs(np.sum(restored[:, start:start + 4] * action[:, start:start + 4], axis=-1))
        np.testing.assert_allclose(dots, 1.0, atol=1e-5)


def test_unknown_action_mode_is_rejected(tmp_path):
    train, _test = _buckets(tmp_path)
    try:
        PiperXDataset(str(train), split="train", action_mode="joint")
    except ValueError as exc:
        assert "action_mode" in str(exc)
    else:
        raise AssertionError("Piper X must reject an unknown action_mode")


def test_registry_and_test_sibling(tmp_path):
    train, _test = _buckets(tmp_path)
    config = {
        "type": "piper_x",
        "dataset_dir": str(train),
        "action_mode": "eef_quat",
        "normalize_mode": "min-max",
    }
    dataset = build_dataset(config, split="test")
    assert isinstance(dataset, PiperXDataset)
    assert dataset.split == "test"
    assert dataset.dataset_dir == train.with_name("piper_x_openwam_test_15hz")


def test_rejects_a_non_piper_bucket(tmp_path):
    train, _test = _buckets(tmp_path)
    manifest_path = train / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format"] = "openwam_linglong_raw_v2"
    manifest_path.write_text(json.dumps(manifest))
    try:
        PiperXDataset(str(train), split="train")
    except ValueError as exc:
        assert "openwam_piper_x_eef_v1" in str(exc)
    else:
        raise AssertionError("a Linglong bucket must not open as Piper X")


def test_train_config_selects_piper_dataset():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_piper_x_eef_8gpu")
    assert cfg.dataloader.type == "piper_x"
    assert cfg.dataloader.action_mode == "eef_quat"
    assert list(cfg.dataloader.active_raw_action_dims) == list(range(16))
    assert list(cfg.dataloader.unify_action_map) == ["0-6", 9, "34-40", 43]
    assert cfg.model.architecture.action_dim == 80
    assert str(cfg.training.finetune_ckpt_path).endswith("OpenWAM-Alpha-Pretrain-Foundation-Model")
    assert cfg.training.resume_ckpt_path is None
