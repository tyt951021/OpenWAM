"""Piper X absolute EEF reader.

Buckets come from ``scripts_piper/convert_piper_x.py``. Each arm is stored in
its own base as 16D ``left xyz + wxyz + gripper, right xyz + wxyz + gripper``.
Gripper aperture is 0 closed and 1 open. Both arms are supervised. Values
are normalized in that 16D space, then scattered into OpenWAM's
80D EEF slots: left ``0-6`` and ``9``, right ``34-40`` and ``43``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.normalization import apply_normalization
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec


EEF_DIM = 16
NUM_FRAMES = 33
VIDEO_STRIDE = 4
VIDEO_INDICES = np.arange(0, NUM_FRAMES, VIDEO_STRIDE, dtype=np.int64)
ALL_ARM_DIMS = list(range(EEF_DIM))
UNIFY_MAP = ["0-6", 9, "34-40", 43]
BUCKET_FORMAT = "openwam_piper_x_eef_v1"
STATE_COLUMN = "observation.state_eef"
ACTION_COLUMN = "action_eef"


class PiperXDataset(BaseDataset):
    """One Piper X train or test bucket."""

    @classmethod
    def from_config(cls, config, split: str = "train"):
        dataset_dir = Path(get_cfg(config, "dataset_dir"))
        if split in ("test", "val") and dataset_dir.name.endswith("_train_15hz"):
            candidate = dataset_dir.with_name(dataset_dir.name.replace("_train_15hz", "_test_15hz"))
            if candidate.is_dir():
                dataset_dir = candidate
        normalization_dataset_dir = get_cfg(config, "normalization_dataset_dir", None)
        if split in ("test", "val") and not normalization_dataset_dir:
            if not dataset_dir.name.endswith("_test_15hz"):
                raise ValueError("Piper X evaluation needs normalization_dataset_dir pointing at the training bucket")
            normalization_dataset_dir = dataset_dir.with_name(
                dataset_dir.name.replace("_test_15hz", "_train_15hz")
            )
        return cls(
            dataset_dir=str(dataset_dir),
            split=split,
            num_frames=int(get_cfg(config, "num_frames", NUM_FRAMES)),
            video_stride=int(get_cfg(config, "video_stride", VIDEO_STRIDE)),
            window_stride=int(get_cfg(config, "window_stride", 1)),
            height=int(get_cfg(config, "height", 384)),
            width=int(get_cfg(config, "width", 320)),
            normalize_mode=get_cfg(config, "normalize_mode", "min-max"),
            camera_layout=list(get_cfg(config, "camera_layout", ["head", "left", "right"])),
            multiview=bool(get_cfg(config, "multiview", True)),
            overfit_single=bool(get_cfg(config, "overfit_single", False)),
            overfit_episode=int(get_cfg(config, "overfit_episode", 0)),
            overfit_repeats=int(get_cfg(config, "overfit_repeats", 1)),
            unify_action=bool(get_cfg(config, "unify_action", True)),
            unify_action_map=get_cfg(config, "unify_action_map", UNIFY_MAP),
            unify_state_map=get_cfg(config, "unify_state_map", None),
            action_mode=get_cfg(config, "action_mode", "eef_quat"),
            normalization_dataset_dir=normalization_dataset_dir,
            active_raw_action_dims=get_cfg(config, "active_raw_action_dims", None),
            active_raw_state_dims=get_cfg(config, "active_raw_state_dims", None),
        )

    def __init__(
        self,
        dataset_dir: str,
        *,
        split: str = "train",
        num_frames: int = NUM_FRAMES,
        video_stride: int = VIDEO_STRIDE,
        window_stride: int = 1,
        height: int = 384,
        width: int = 320,
        normalize_mode: Optional[str] = "min-max",
        camera_layout: Optional[list[str]] = None,
        multiview: bool = True,
        overfit_single: bool = False,
        overfit_episode: int = 0,
        overfit_repeats: int = 1,
        unify_action: bool = True,
        unify_action_map=None,
        unify_state_map=None,
        action_mode: str = "eef_quat",
        normalization_dataset_dir=None,
        active_raw_action_dims=None,
        active_raw_state_dims=None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = str(split)
        self.action_mode = str(action_mode)
        if self.action_mode != "eef_quat":
            raise ValueError(f"Piper X only reads absolute EEF quaternions, got action_mode={self.action_mode!r}")
        self.num_frames = int(num_frames)
        self.video_stride = int(video_stride)
        self.window_stride = int(window_stride)
        if self.num_frames != NUM_FRAMES or self.video_stride != VIDEO_STRIDE:
            raise ValueError(
                f"Piper X video windows are {NUM_FRAMES} frames at stride {VIDEO_STRIDE}, "
                f"got num_frames={self.num_frames} video_stride={self.video_stride}"
            )
        if self.window_stride < 1:
            raise ValueError("window_stride must be >= 1")
        self.height = int(height)
        self.width = int(width)
        self.normalize_mode = None if normalize_mode in (None, "", "none", "null") else str(normalize_mode)
        if self.normalize_mode not in (None, "min-max", "z-score", "quantile"):
            raise ValueError(f"unsupported normalize_mode={self.normalize_mode!r}")
        self.camera_layout = list(camera_layout or ["head", "left", "right"])
        self.multiview = bool(multiview)
        if self.multiview and self.camera_layout != ["head", "left", "right"]:
            raise ValueError("Piper X multiview cameras are head, left, right")
        self.overfit_single = bool(overfit_single)
        self.overfit_episode = int(overfit_episode)
        self.overfit_repeats = int(overfit_repeats)
        if self.overfit_single and self.overfit_repeats < 1:
            raise ValueError("overfit_repeats must be >= 1")

        manifest = self._read_manifest(self.dataset_dir, self.split)
        self._source_root = self.dataset_dir / str(manifest.get("source_root", "."))
        if not self._source_root.is_dir():
            raise FileNotFoundError(f"Piper X images are missing under {self._source_root}")
        self._prompt = str(manifest["prompt"])
        self._episodes = [
            json.loads(line)
            for line in (self.dataset_dir / "meta" / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self._tables = [
            self.dataset_dir / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
            for index in range(len(self._episodes))
        ]
        if any(not path.is_file() for path in self._tables):
            raise FileNotFoundError(f"Piper X episode parquet is missing under {self.dataset_dir / 'data'}")

        action_dims = active_raw_action_dims
        state_dims = active_raw_state_dims
        if action_dims is None:
            action_dims = manifest.get("active_raw_action_dims", ALL_ARM_DIMS)
        if state_dims is None:
            state_dims = manifest.get("active_raw_state_dims", ALL_ARM_DIMS)
        self.raw_action_mask = self._raw_mask(action_dims)
        self.raw_state_mask = self._raw_mask(state_dims)

        self.unify_action = bool(unify_action)
        if self.unify_action:
            action_map = UNIFY_MAP if unify_action_map is None else unify_action_map
            self._action_dst_index = parse_unify_spec(action_map, UNIFY_DIM)
            state_map = action_map if unify_state_map is None else unify_state_map
            self._state_dst_index = parse_unify_spec(state_map, UNIFY_DIM)
            if self._action_dst_index.shape != (EEF_DIM,) or self._state_dst_index.shape != (EEF_DIM,):
                raise ValueError(f"Piper X unify maps must cover {EEF_DIM} EEF dimensions")
            self._action_dim = UNIFY_DIM
            self._state_dim = UNIFY_DIM
        else:
            self._action_dst_index = None
            self._state_dst_index = None
            self._action_dim = EEF_DIM
            self._state_dim = EEF_DIM

        self._windows = []
        for local, episode in enumerate(self._episodes):
            max_start = int(episode["length"]) - self.num_frames
            if max_start >= 0:
                self._windows.extend((local, start) for start in range(0, max_start + 1, self.window_stride))
        if not self._windows:
            raise ValueError(f"Piper X split={self.split!r} has no {self.num_frames}-frame windows")
        if self.overfit_single:
            if self.overfit_episode < 0 or self.overfit_episode >= len(self._episodes):
                raise ValueError(f"overfit_episode={self.overfit_episode} is outside the bucket")
            episode_windows = [item for item in self._windows if item[0] == self.overfit_episode]
            if not episode_windows:
                raise ValueError(f"overfit_episode={self.overfit_episode} has no full window")
            self._windows = episode_windows * self.overfit_repeats

        stats_dir = self.dataset_dir if normalization_dataset_dir is None else Path(normalization_dataset_dir)
        if self.split in ("test", "val"):
            train_manifest = self._read_manifest(stats_dir, "train")
            if train_manifest.get("split") != "train":
                raise ValueError("Piper X normalization statistics must come from the training bucket")
        elif manifest.get("split") != "train":
            raise ValueError("Piper X training statistics require a train bucket")
        self._action_stats = self._load_stats(stats_dir / "meta" / "action_eef_stats.json", ACTION_COLUMN, "action")
        self._state_stats = self._load_stats(stats_dir / "meta" / "state_eef_stats.json", STATE_COLUMN, "state")
        self.normalization_stats_payload = {
            "eef_quat": self._action_stats,
            "eef_quat_state": self._state_stats,
        }
        self.normalization_stats_path = None

    @staticmethod
    def _read_manifest(bucket: Path, split: str) -> dict:
        path = Path(bucket) / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Piper X manifest is missing: {path}")
        manifest = json.loads(path.read_text())
        if manifest.get("format") != BUCKET_FORMAT:
            raise ValueError(f"{path} format must be {BUCKET_FORMAT}")
        expected_split = "test" if split == "val" else split
        if manifest.get("split") != expected_split:
            raise ValueError(f"{path} split is {manifest.get('split')!r}, expected {expected_split!r}")
        for key, column in (("observation_state_eef", STATE_COLUMN), ("action_eef", ACTION_COLUMN)):
            contract = manifest.get(key, {})
            if (contract.get("column") != column or contract.get("shape") != [EEF_DIM]
                    or contract.get("quaternion_order") != "wxyz" or contract.get("reference") != "base_link"):
                raise ValueError(f"{path} {key} must be 16D base_link wxyz EEF")
        return manifest

    @staticmethod
    def _load_stats(path: Path, column: str, stream: str) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"Piper X statistics are missing: {path}")
        payload = json.loads(path.read_text())
        if payload.get("source_column") != column:
            raise ValueError(f"{path} must describe {column}")
        block = payload[stream]["default"]
        stats = {
            key: np.asarray(block[f"global_{key}"], dtype=np.float32)
            for key in ("mean", "std", "min", "max", "q01", "q99")
        }
        for key, values in stats.items():
            if values.shape != (EEF_DIM,) or not np.isfinite(values).all():
                raise ValueError(f"{path} statistic {key} must be {EEF_DIM} finite values")
        return stats

    @staticmethod
    def _raw_mask(dims) -> np.ndarray:
        indices = np.asarray(list(dims), dtype=np.int64)
        if (indices.size == 0 or len(np.unique(indices)) != len(indices)
                or indices.min() < 0 or indices.max() >= EEF_DIM):
            raise ValueError(f"active EEF dimensions must be unique indices inside [0, {EEF_DIM})")
        mask = np.zeros(EEF_DIM, dtype=bool)
        mask[indices] = True
        return mask

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def __len__(self) -> int:
        return len(self._windows)

    @lru_cache(maxsize=128)
    def _read_table(self, local: int):
        return pq.read_table(
            self._tables[local],
            columns=[
                STATE_COLUMN,
                ACTION_COLUMN,
                "observation.images.head",
                "observation.images.left",
                "observation.images.right",
            ],
        )

    def _image(self, relative: str) -> Image.Image:
        with Image.open(self._source_root / relative) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int) -> dict:
        local, start = self._windows[index]
        frames_idx = np.arange(start, start + self.num_frames, dtype=np.int64)
        table = self._read_table(local).take(pa.array(frames_idx))
        state = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
        action = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)
        if state.shape != (self.num_frames, EEF_DIM) or action.shape != (self.num_frames, EEF_DIM):
            raise ValueError(f"Piper X row shape mismatch: state={state.shape}, action={action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Piper X EEF contains a non-finite value")
        for values, label in ((state, "state"), (action, "action")):
            for quat_start in (3, 11):
                norms = np.linalg.norm(values[:, quat_start:quat_start + 4], axis=-1)
                if not np.allclose(norms, 1.0, atol=1e-3):
                    raise ValueError(f"Piper X {label} quaternions must be unit wxyz")

        images = []
        for frame in VIDEO_INDICES:
            cameras = {
                "head": self._image(table["observation.images.head"][int(frame)].as_py()),
                "left": self._image(table["observation.images.left"][int(frame)].as_py()),
                "right": self._image(table["observation.images.right"][int(frame)].as_py()),
            }
            if self.multiview:
                images.append(assemble_multiview_layout(cameras, self.camera_layout, self.height, self.width))
            else:
                images.append(cameras["head"].resize((self.width, self.height), Image.BILINEAR))

        raw_action = np.asarray(apply_normalization(action[:self.num_frames - 1], self._action_stats, self.normalize_mode), dtype=np.float32)
        raw_state = np.asarray(apply_normalization(state[:1], self._state_stats, self.normalize_mode), dtype=np.float32)
        raw_action[..., ~self.raw_action_mask] = 0.0
        raw_state[..., ~self.raw_state_mask] = 0.0
        if self.unify_action:
            action_out, action_dim_mask = map_to_unify(raw_action, self._action_dst_index, UNIFY_DIM)
            proprio_out, proprio_dim_mask = map_to_unify(raw_state, self._state_dst_index, UNIFY_DIM)
            action_dim_mask = action_dim_mask & self._scatter_mask(self._action_dst_index, self.raw_action_mask)
            proprio_dim_mask = proprio_dim_mask & self._scatter_mask(self._state_dst_index, self.raw_state_mask)
        else:
            action_out, proprio_out = raw_action, raw_state
            action_dim_mask = self.raw_action_mask
            proprio_dim_mask = self.raw_state_mask
        return {
            "video": images,
            "vace_video": None,
            "first_frame_image": [images[0]],
            "action": torch.from_numpy(np.ascontiguousarray(action_out)),
            "action_mask": torch.from_numpy(np.broadcast_to(action_dim_mask, action_out.shape).copy()),
            "video_mask": torch.ones(len(images), dtype=torch.bool),
            "proprio": torch.from_numpy(np.ascontiguousarray(proprio_out)),
            "proprio_mask": torch.from_numpy(proprio_dim_mask[None, :].copy()),
            "prompt": (self._episodes[local].get("tasks") or [self._prompt])[0],
            "episode_index": int(self._episodes[local]["episode_index"]),
            "start_frame": int(start),
        }

    @staticmethod
    def _scatter_mask(dst_index: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
        mask = np.zeros(UNIFY_DIM, dtype=bool)
        mask[dst_index[raw_mask]] = True
        return mask


__all__ = ["PiperXDataset"]
