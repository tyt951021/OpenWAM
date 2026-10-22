"""Piper X forward kinematics from the vendored URDF in this directory.

Recorded poses match ``tcp_link``: the arm chain through ``link6``, plus the
fixed tool translation stored on ``tcp_joint``. Gripper width is metres in
``[0, gripper upper]`` and becomes an aperture in ``[0, 1]``, with 0 closed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


URDF_PATH = Path(__file__).resolve().parent / "urdf" / "piper_x_kinematics.urdf"
ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 7))
TCP_LINK = "tcp_link"
POSE_POSITION_TOL_M = 0.005
POSE_ORIENTATION_TOL_RAD = float(np.deg2rad(1.5))


def make_quaternion_continuous(values: np.ndarray) -> np.ndarray:
    """Keep each hand's wxyz quaternion from flipping sign inside one episode."""
    values = np.asarray(values, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"expected shape (frames, 16), got {values.shape}")
    for hand_start in (0, 8):
        quat = values[:, hand_start + 3:hand_start + 7]
        if quat[0, 0] < 0:
            quat[0] *= -1
        for index in range(1, len(quat)):
            if float(np.dot(quat[index - 1], quat[index])) < 0:
                quat[index] *= -1
    return values


class PiperXKinematics:
    """One arm. Left and right Piper X arms use the same URDF in their own base."""

    def __init__(self, urdf_path: Path = URDF_PATH):
        self.urdf_path = Path(urdf_path)
        root = ET.parse(self.urdf_path).getroot()
        self._by_child = {}
        gripper_upper = None
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            axis_element = joint.find("axis")
            axis = (np.fromstring(axis_element.get("xyz"), sep=" ")
                    if axis_element is not None else np.array([0.0, 0.0, 1.0]))
            record = {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "xyz": np.fromstring(origin.get("xyz"), sep=" ") if origin is not None else np.zeros(3),
                "rpy": np.fromstring(origin.get("rpy"), sep=" ") if origin is not None else np.zeros(3),
                "axis": axis / np.linalg.norm(axis),
            }
            self._by_child[joint.find("child").get("link")] = record
            if record["name"] == "gripper":
                gripper_upper = float(joint.find("limit").get("upper"))
        if gripper_upper is None or gripper_upper <= 0:
            raise ValueError(f"{self.urdf_path} has no positive gripper upper limit")
        if TCP_LINK not in self._by_child:
            raise ValueError(f"{self.urdf_path} has no {TCP_LINK}")
        self.gripper_open_m = gripper_upper
        chain = []
        link = TCP_LINK
        while link in self._by_child:
            joint = self._by_child[link]
            chain.append(joint)
            link = joint["parent"]
        self._chain = list(reversed(chain))

    def gripper_aperture(self, meters) -> np.ndarray:
        """Map a metre opening to ``[0, 1]``. 0 is closed and 1 is the URDF upper limit."""
        values = np.asarray(meters, dtype=np.float64)
        upper = self.gripper_open_m
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("gripper opening must be finite")
        if np.any(values < -1e-4) or np.any(values > upper + 1e-4):
            raise ValueError(
                f"gripper opening must be within [0, {upper}] metres, got [{values.min()}, {values.max()}]"
            )
        return np.clip(values / upper, 0.0, 1.0)

    def _fk_one(self, joints) -> np.ndarray:
        positions = {name: float(value) for name, value in zip(ARM_JOINTS, joints)}
        transform = np.eye(4)
        for joint in self._chain:
            origin = np.eye(4)
            origin[:3, :3] = Rotation.from_euler("xyz", joint["rpy"]).as_matrix()
            origin[:3, 3] = joint["xyz"]
            motion = np.eye(4)
            angle = positions.get(joint["name"], 0.0)
            if joint["type"] in ("revolute", "continuous"):
                motion[:3, :3] = Rotation.from_rotvec(joint["axis"] * angle).as_matrix()
            elif joint["type"] == "prismatic":
                motion[:3, 3] = joint["axis"] * angle
            elif joint["type"] != "fixed":
                raise ValueError(f"unsupported joint {joint['name']} type {joint['type']}")
            transform = transform @ origin @ motion
        return transform

    def poses(self, joints):
        """Return TCP ``xyz`` and ``wxyz`` for joints shaped ``(6,)`` or ``(frames, 6)``."""
        values = np.asarray(joints, dtype=np.float64)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
            raise ValueError(f"expected finite (frames, 6) joints, got {values.shape}")
        xyz = np.empty((len(values), 3), dtype=np.float64)
        wxyz = np.empty((len(values), 4), dtype=np.float64)
        for index, joints_i in enumerate(values):
            transform = self._fk_one(joints_i)
            xyz[index] = transform[:3, 3]
            xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
            wxyz[index] = xyzw[[3, 0, 1, 2]]
        if single:
            return xyz[0], wxyz[0]
        return xyz, wxyz

    def state_errors(self, joints, position, orientation_xyzw):
        """Position (metres) and orientation (radians) between TCP FK and a recorded pose."""
        xyz, wxyz = self.poses(joints)
        position = np.asarray(position, dtype=np.float64)
        orientation = np.asarray(orientation_xyzw, dtype=np.float64)
        if xyz.ndim == 1:
            xyz = xyz[None, :]
            wxyz = wxyz[None, :]
            position = position[None, :]
            orientation = orientation[None, :]
        if position.shape != xyz.shape or orientation.shape != wxyz.shape:
            raise ValueError(
                f"recorded pose shape {position.shape} / {orientation.shape} "
                f"does not match joints {xyz.shape}"
            )
        if not np.isfinite(position).all() or not np.isfinite(orientation).all():
            raise ValueError("recorded pose must be finite")
        if not np.allclose(np.linalg.norm(orientation, axis=-1), 1.0, atol=1e-4):
            raise ValueError("recorded orientation_xyzw must be a unit quaternion")
        position_error = np.linalg.norm(xyz - position, axis=-1)
        predicted = Rotation.from_quat(np.asarray(wxyz[..., [1, 2, 3, 0]]))
        recorded = Rotation.from_quat(orientation)
        orientation_error = (predicted.inv() * recorded).magnitude()
        return position_error, orientation_error

    def assert_matches_state(self, joints, position, orientation_xyzw, label="state"):
        position_error, orientation_error = self.state_errors(joints, position, orientation_xyzw)
        worst = int(np.argmax(position_error + orientation_error))
        if (float(position_error.max()) > POSE_POSITION_TOL_M
                or float(orientation_error.max()) > POSE_ORIENTATION_TOL_RAD):
            raise ValueError(
                f"{label} FK does not match the recorded pose at frame {worst}: "
                f"position {position_error[worst] * 1000:.2f} mm, "
                f"orientation {np.degrees(orientation_error[worst]):.2f} deg"
            )
        return {
            "frames": int(position_error.shape[0]),
            "position_error_mm_median": float(np.median(position_error) * 1000),
            "position_error_mm_max": float(position_error.max() * 1000),
            "orientation_error_deg_median": float(np.median(np.degrees(orientation_error))),
            "orientation_error_deg_max": float(np.degrees(orientation_error.max())),
        }
