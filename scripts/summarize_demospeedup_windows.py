"""Print DemoSpeedup window counts and check one training sample.

Does not start training. Builds the fittings training dataset with the online
keep-tables, then reads the first window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openwam.dataloader.demospeedup import (  # noqa: E402
    expand_entropy,
    gripper_precision_mask,
    keep_indices,
    median_filter,
    two_means_threshold,
)
from openwam.dataloader.linglong import LinglongDataset  # noqa: E402

DATASET_DIR = (
    "/vePFS/data/linglong/linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable/"
    "data/linglong_0923_1010_1012_1017_fittings_eef_gtstate_train_15hz"
)
ENTROPY_DIR = "/vePFS/zhuoyu/OpenWAM/outputs/action_entropy_fittings_eef_step18185_stride2"


def _check_rules() -> None:
    casual = np.array([False, False, False, False, True, True, True], dtype=bool)
    assert keep_indices(casual, 0, 2, 3).tolist() == [0, 2, 4]
    assert keep_indices(casual, 1, 2, 3).tolist() == [1, 3, 6]
    assert keep_indices(casual, 2, 2, 3).tolist() == [0, 2, 5]
    filled = expand_entropy(np.array([0, 2, 4]), np.array([1.0, 3.0, 5.0]), 5)
    assert filled.tolist() == [1.0, 1.0, 3.0, 3.0, 5.0]
    grip = np.zeros((8, 2))
    grip[4:, 0] = 1.0
    forced = gripper_precision_mask(grip, 0.25, 3)
    assert forced.tolist() == [False, True, True, True, True, True, True, True]
    smoothed = median_filter(np.array([0.0, 5.0, 0.0, 0.0, 9.0]), 3)
    assert smoothed.tolist() == [0.0, 0.0, 0.0, 0.0, 9.0]
    threshold, low, high = two_means_threshold(np.array([0.0, 0.1, 10.0, 11.0]))
    assert low < threshold < high
    print("[demospeedup] rule checks passed", flush=True)


def main() -> None:
    _check_rules()
    dataset = LinglongDataset(
        DATASET_DIR,
        split="train",
        action_key="action_eef",
        state_key="observation.state_eef",
        action_mode="eef_quat",
        unify_action_map=["0-6", 9, "34-40", 43],
        unify_state_map=["0-6", 9, "34-40", 43],
        active_raw_action_dims=list(range(16)),
        active_raw_state_dims=list(range(16)),
        normalization_dataset_dir=DATASET_DIR,
        demospeedup_enabled=True,
        demospeedup_entropy_dir=ENTROPY_DIR,
    )
    summary = dataset.demospeedup_summary
    sample = dataset[0]
    action = sample["action"].numpy()
    kept = dataset._kept[0][0]
    gaps = np.diff(kept[:40])
    print(
        "[demospeedup] sample "
        f"video={len(sample['video'])} action={tuple(sample['action'].shape)} "
        f"proprio={tuple(sample['proprio'].shape)} "
        f"start_frame={sample['start_frame']} episode={sample['episode_index']} "
        f"first_phase_gaps={gaps.tolist()}",
        flush=True,
    )
    if len(sample["video"]) != 9 or action.shape[0] != 32 or not np.isfinite(action).all():
        raise SystemExit("first DemoSpeedup sample has an unexpected shape")
    if summary["windows"] <= 0 or summary["dropped_phases"] < 0:
        raise SystemExit(f"unexpected summary: {summary}")
    episode_length = int(dataset._episodes[0]["length"])
    if len(kept) >= episode_length or np.any(np.diff(kept) < 1) or float(np.mean(gaps)) <= 1.0:
        raise SystemExit(f"first phase was not subsampled: length={len(kept)}/{episode_length} gaps={gaps.tolist()}")
    plain = LinglongDataset(
        DATASET_DIR,
        split="train",
        action_key="action_eef",
        state_key="observation.state_eef",
        action_mode="eef_quat",
        unify_action_map=["0-6", 9, "34-40", 43],
        unify_state_map=["0-6", 9, "34-40", 43],
        active_raw_action_dims=list(range(16)),
        active_raw_state_dims=list(range(16)),
        normalization_dataset_dir=DATASET_DIR,
        demospeedup_enabled=False,
    )
    if len(plain) != summary["source_windows"] or len(plain._windows[0]) != 2:
        raise SystemExit("disabling DemoSpeedup did not restore the contiguous windows")
    print(f"[demospeedup] disabled windows={len(plain)}", flush=True)


if __name__ == "__main__":
    main()
