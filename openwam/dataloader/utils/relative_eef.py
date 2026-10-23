"""Relative Piper X EEF, expressed in the input state's frame.

Stored rows stay absolute ``xyz + wxyz + gripper`` per arm. A relative action
is the commanded pose of each arm written in the initial measured state:

- position: ``R_state.T @ (p_action - p_state)``
- rotation: ``q_state^{-1} ⊗ q_action``
- gripper: the absolute command, unchanged

``relative_to_absolute`` is the inverse, so a model that predicts the relative
pose can be turned back into the original absolute command.
"""

from __future__ import annotations

import numpy as np

from openwam.dataloader.utils.poses import eef20_to_arms


EEF16_DIM = 16
_ARM_BLOCKS = (0, 8)


def _unit_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("relative EEF quaternion is degenerate")
    return quaternion / norm


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(_unit_quaternion(quaternion), -1, 0)
    return np.stack(
        (
            np.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)), axis=-1),
            np.stack((2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)), axis=-1),
            np.stack((2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)), axis=-1),
        ),
        axis=-2,
    )


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _broadcast_state(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    if action.shape[-1] != EEF16_DIM or state.shape[-1] != EEF16_DIM:
        raise ValueError(
            f"relative EEF expects 16D absolute rows, got action {action.shape} and state {state.shape}"
        )
    try:
        return np.broadcast_to(state, action.shape)
    except ValueError as error:
        raise ValueError(
            f"state shape {state.shape} cannot broadcast to action shape {action.shape}"
        ) from error


def absolute_to_relative(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Express absolute 16D actions in the frame of ``state``."""
    action = np.asarray(action, dtype=np.float64)
    state = _broadcast_state(action, np.asarray(state, dtype=np.float64))
    relative = np.empty(action.shape, dtype=np.float64)
    for start in _ARM_BLOCKS:
        rotation = _quaternion_to_matrix(state[..., start + 3:start + 7])
        delta = action[..., start:start + 3] - state[..., start:start + 3]
        relative[..., start:start + 3] = np.einsum("...ij,...j->...i", np.swapaxes(rotation, -1, -2), delta)
        relative[..., start + 3:start + 7] = _unit_quaternion(
            _quaternion_multiply(
                _unit_quaternion(state[..., start + 3:start + 7]) * np.array([1.0, -1.0, -1.0, -1.0]),
                _unit_quaternion(action[..., start + 3:start + 7]),
            )
        )
        relative[..., start + 7] = action[..., start + 7]
    return relative.astype(np.float32, copy=False)


def relative_to_absolute(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Invert :func:`absolute_to_relative` back to a 16D absolute command."""
    action = np.asarray(action, dtype=np.float64)
    state = _broadcast_state(action, np.asarray(state, dtype=np.float64))
    absolute = np.empty(action.shape, dtype=np.float64)
    for start in _ARM_BLOCKS:
        rotation = _quaternion_to_matrix(state[..., start + 3:start + 7])
        absolute[..., start:start + 3] = state[..., start:start + 3] + np.einsum(
            "...ij,...j->...i", rotation, action[..., start:start + 3]
        )
        absolute[..., start + 3:start + 7] = _unit_quaternion(
            _quaternion_multiply(
                _unit_quaternion(state[..., start + 3:start + 7]),
                _unit_quaternion(action[..., start + 3:start + 7]),
            )
        )
        absolute[..., start + 7] = action[..., start + 7]
    return absolute.astype(np.float32, copy=False)


def eef20_to_eef16(action: np.ndarray) -> np.ndarray:
    """Unpack official 20D ``xyz + rot6d + gripper`` back to stored 16D wxyz."""
    left_pose, left_gripper, right_pose, right_gripper = eef20_to_arms(action)
    return np.concatenate((left_pose, left_gripper, right_pose, right_gripper), axis=-1).astype(np.float32, copy=False)


def restore_absolute_command(action: np.ndarray, state: np.ndarray, action_mode: str) -> np.ndarray:
    """Turn a relative model command back into the stored 16D absolute action."""
    values = np.asarray(action, dtype=np.float32)
    reference = np.asarray(state, dtype=np.float32).reshape(-1)
    if reference.shape != (EEF16_DIM,):
        raise ValueError(f"relative EEF restore expects one 16D input state, got {np.asarray(state).shape}")
    mode = str(action_mode)
    if mode == "eef":
        if values.shape[-1] != 20:
            raise ValueError(f"rot6d relative command must be 20D, got {values.shape}")
        values = eef20_to_eef16(values)
    elif mode == "eef_quat":
        if values.shape[-1] != EEF16_DIM:
            raise ValueError(f"quaternion relative command must be 16D, got {values.shape}")
    else:
        raise ValueError(f"unsupported relative action_mode {action_mode!r}")
    return relative_to_absolute(values, reference)
