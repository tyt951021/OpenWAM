import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from openwam.dataloader.piper_x import PiperXDataset
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
