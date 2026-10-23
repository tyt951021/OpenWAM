import numpy as np
import pytest

from openwam.dataloader.piper_x import PiperXDataset
from scripts_piper.eval_piper_x import closed_loop_proprio, episode_starts, require_contiguous
from scripts_piper.convert_piper_x import convert_piper_x
from tests.test_piper_x import _segment


def test_closed_loop_proprio_renormalizes_both_hands():
    action = np.zeros((32, 16), dtype=np.float32)
    action[-1, 3:7] = [2, 0, 0, 0]
    action[-1, 11:15] = [0, 3, 0, 0]
    proprio = closed_loop_proprio(action)
    assert proprio.shape == (16,)
    np.testing.assert_allclose(proprio[3:7], [1, 0, 0, 0])
    np.testing.assert_allclose(proprio[11:15], [0, 1, 0, 0])
    action[-1, 3:7] = 0
    with pytest.raises(ValueError, match="degenerate"):
        closed_loop_proprio(action)


def test_stride_32_windows_are_contiguous(tmp_path):
    raw = tmp_path / "raw"
    _segment(raw, "segment_000", 65, 0.0)
    convert_piper_x(raw, tmp_path / "out")
    dataset = PiperXDataset(str(tmp_path / "out" / "piper_x_openwam_train_15hz"), split="train", window_stride=32)
    starts = episode_starts(dataset._windows, 0, 32, 1000)
    assert starts == [0, 32]
    require_contiguous(starts, 32, 0)
    table = dataset._read_table(0).slice(0, dataset.num_frames)
    state = np.asarray(table[dataset.state_key][0].as_py(), dtype=np.float32)
    action = np.asarray(table[dataset.action_key].to_pylist(), dtype=np.float32)[: dataset.num_frames - 1]
    assert state.shape == (16,)
    assert action.shape == (32, 16)
    assert dataset.state_key == "observation.state_eef"
    assert dataset.action_key == "action_eef"
