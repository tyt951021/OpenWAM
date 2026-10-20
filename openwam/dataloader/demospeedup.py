"""Online DemoSpeedup keep-tables for Linglong training.

Labels are rebuilt from the action-entropy npz files every time the training
dataset is constructed. Odd frames inherit the nearest sampled-frame entropy,
one two-means threshold is fit on every smoothed frame, and gripper motion
forces a short neighborhood into the precision rate. Each phase offset then
keeps a different residue inside every constant-label run. No new parquet
bucket is written.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

GRIPPER_COLUMNS = (7, 15)


@dataclass(frozen=True)
class KeepPlan:
    """Per-episode keep-tables plus the counts printed before training."""

    tables: list[list[np.ndarray]]
    summary: dict


def median_filter(values: np.ndarray, width: int) -> np.ndarray:
    """Odd-width median filter. Edges repeat the endpoint value."""
    values = np.asarray(values, dtype=np.float64)
    if width <= 1:
        return values.copy()
    if width % 2 == 0:
        raise ValueError(f"smooth width must be odd, got {width}")
    pad = width // 2
    padded = np.pad(values, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=-1)


def two_means_threshold(values: np.ndarray) -> tuple[float, float, float]:
    """Split one pooled series into a low and a high cluster.

    Returns ``(threshold, low_center, high_center)``. Values at or below the
    threshold belong to the low cluster. A constant series has no split and
    the threshold sits on that value, so every frame is low.
    """
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("entropy values must be non-empty and finite")
    low = float(samples.min())
    high = float(samples.max())
    if high <= low:
        return low, low, high
    for _ in range(32):
        mid = 0.5 * (low + high)
        low_mask = samples <= mid
        if not low_mask.any() or low_mask.all():
            break
        next_low = float(samples[low_mask].mean())
        next_high = float(samples[~low_mask].mean())
        if next_low == low and next_high == high:
            break
        low, high = next_low, next_high
    return 0.5 * (low + high), low, high


def expand_entropy(frame_index: np.ndarray, entropy: np.ndarray, length: int) -> np.ndarray:
    """Fill every source frame from the nearest sampled entropy.

    Equal distance keeps the earlier sample, so an odd frame between two
    stride-2 samples inherits the preceding even frame.
    """
    source = np.asarray(frame_index, dtype=np.int64)
    values = np.asarray(entropy, dtype=np.float64)
    if source.ndim != 1 or values.shape != source.shape or source.size == 0:
        raise ValueError("entropy npz frame_index and entropy must be the same non-empty 1-D arrays")
    if np.any(np.diff(source) <= 0) or source[0] < 0 or source[-1] >= length:
        raise ValueError("entropy frame_index must be strictly increasing and inside the episode")
    targets = np.arange(length, dtype=np.int64)
    right = np.searchsorted(source, targets, side="left")
    right = np.clip(right, 0, source.size - 1)
    left = np.clip(right - 1, 0, source.size - 1)
    use_left = np.abs(source[left] - targets) <= np.abs(source[right] - targets)
    chosen = np.where(use_left, left, right)
    return values[chosen]


def gripper_precision_mask(grip: np.ndarray, delta: float, radius: int) -> np.ndarray:
    """Mark frames whose gripper moves, expanded by ``radius`` on both sides."""
    grip = np.asarray(grip, dtype=np.float64)
    if grip.ndim != 2 or grip.shape[1] != 2:
        raise ValueError(f"gripper series must have shape (T, 2), got {grip.shape}")
    change = np.zeros(grip.shape[0], dtype=bool)
    if grip.shape[0] > 1:
        moved = np.any(np.abs(np.diff(grip, axis=0)) > delta, axis=1)
        change[1:] = moved
    if radius < 0:
        raise ValueError(f"gripper radius must be >= 0, got {radius}")
    if radius == 0 or not change.any():
        return change
    dilated = change.copy()
    for center in np.flatnonzero(change):
        dilated[max(0, center - radius) : min(grip.shape[0], center + radius + 1)] = True
    return dilated


def keep_indices(casual: np.ndarray, offset: int, r_precision: int, r_casual: int) -> np.ndarray:
    """Keep one residue class inside each constant-label run.

    ``offset`` is the phase. A precision run uses ``r_precision`` and a casual
    run uses ``r_casual``. Frame ``t`` of a run is kept when
    ``(t - run_start + offset) % r == 0``.
    """
    casual = np.asarray(casual, dtype=bool)
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    kept: list[int] = []
    start = 0
    length = int(casual.size)
    while start < length:
        end = start + 1
        while end < length and bool(casual[end]) == bool(casual[start]):
            end += 1
        rate = r_casual if bool(casual[start]) else r_precision
        for frame in range(start, end):
            if (frame - start + offset) % rate == 0:
                kept.append(frame)
        start = end
    return np.asarray(kept, dtype=np.int64)


def _load_episode_signal(episode: dict, table_path: Path, entropy_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    episode_index = int(episode["episode_index"])
    length = int(episode["length"])
    path = entropy_dir / f"episode_{episode_index:06d}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing action entropy for episode {episode_index}: {path}")
    with np.load(path) as payload:
        entropy = expand_entropy(payload["frame_index"], payload["entropy"], length)
        if int(payload["episode_index"]) != episode_index:
            raise ValueError(f"{path} episode_index does not match the dataset row")
    grip_table = pq.read_table(table_path, columns=["action_eef"])
    grip = np.asarray(grip_table.column("action_eef").to_pylist(), dtype=np.float32)
    if grip.shape != (length, 16):
        raise ValueError(f"{table_path} action_eef shape {grip.shape} != ({length}, 16)")
    return entropy, grip[:, GRIPPER_COLUMNS]


def build_keep_plan(
    episodes: list[dict],
    table_paths: list[Path],
    entropy_dir: Path,
    *,
    r_precision: int,
    r_casual: int,
    num_offsets: int | None,
    gripper_radius: int,
    gripper_delta: float,
    smooth_width: int,
    min_frames: int,
) -> KeepPlan:
    """Build one keep-table per phase for every episode."""
    if r_precision < 1 or r_casual < 1:
        raise ValueError(f"keep rates must be >= 1, got precision={r_precision} casual={r_casual}")
    if gripper_delta < 0:
        raise ValueError(f"gripper delta must be >= 0, got {gripper_delta}")
    offsets = max(r_precision, r_casual) if num_offsets is None else int(num_offsets)
    if offsets < 1:
        raise ValueError(f"num_offsets must be >= 1, got {offsets}")
    entropy_dir = Path(entropy_dir)
    if not entropy_dir.is_dir():
        raise FileNotFoundError(f"action entropy directory does not exist: {entropy_dir}")

    smoothed = []
    grips = []
    for episode, table_path in zip(episodes, table_paths):
        entropy, grip = _load_episode_signal(episode, table_path, entropy_dir)
        smoothed.append(median_filter(entropy, smooth_width))
        grips.append(grip)
    threshold, low_center, high_center = two_means_threshold(np.concatenate(smoothed))

    tables: list[list[np.ndarray]] = []
    kept_frames = 0
    source_frames = 0
    precision_frames = 0
    dropped_phases = 0
    for episode, entropy, grip in zip(episodes, smoothed, grips):
        casual = entropy > threshold
        forced = gripper_precision_mask(grip, gripper_delta, gripper_radius)
        casual[forced] = False
        precision_frames += int((~casual).sum())
        source_frames += int(casual.size)
        phase_tables = [
            keep_indices(casual, offset, r_precision, r_casual)
            for offset in range(offsets)
        ]
        dropped_phases += sum(len(table) < min_frames for table in phase_tables)
        kept_frames += sum(len(table) for table in phase_tables)
        tables.append(phase_tables)

    phases = len(episodes) * offsets
    summary = {
        "episodes": len(episodes),
        "num_offsets": offsets,
        "r_precision": int(r_precision),
        "r_casual": int(r_casual),
        "gripper_radius": int(gripper_radius),
        "gripper_delta": float(gripper_delta),
        "smooth_width": int(smooth_width),
        "threshold": float(threshold),
        "low_center": float(low_center),
        "high_center": float(high_center),
        "source_frames": int(source_frames),
        "precision_frames": int(precision_frames),
        "kept_frames": int(kept_frames),
        "mean_keep_ratio": float(kept_frames / (source_frames * offsets)) if source_frames else 0.0,
        "phases": int(phases),
        "dropped_phases": int(dropped_phases),
    }
    return KeepPlan(tables=tables, summary=summary)
