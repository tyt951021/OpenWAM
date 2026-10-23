import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from hydra import compose, initialize_config_dir

from openwam.dataloader.piper_x import PiperXDataset, _pack_eef20
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20
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
