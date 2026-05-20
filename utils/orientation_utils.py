"""Quaternion helpers for end-effector orientation tracking (MuJoCo w,x,y,z)."""

from __future__ import annotations

import numpy as np
from gymnasium_robotics.utils import rotations


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    return q / (n + 1e-12)


def quat_geodesic_distance(q0: np.ndarray, q1: np.ndarray) -> float:
    """Shortest rotation angle in radians between two unit quaternions (double-cover safe)."""
    a = normalize_quat(q0)
    b = normalize_quat(q1)
    d = float(np.clip(np.abs(np.dot(a, b)), 0.0, 1.0))
    return float(2.0 * np.arccos(d))


def quat_from_xy_path_tangent(
    tangent_world: np.ndarray,
    *,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a world-frame rotation (quaternion) whose local +x axis aligns with `tangent_world`.
    Local axes: x along motion, y and z complete a right-handed frame (z ~ up when possible).
    """
    t = np.asarray(tangent_world, dtype=np.float64).reshape(3)
    t = t / (np.linalg.norm(t) + 1e-12)
    up = (
        np.asarray(world_up, dtype=np.float64).reshape(3)
        if world_up is not None
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    up = up / (np.linalg.norm(up) + 1e-12)
    z = np.cross(t, up)
    if float(np.linalg.norm(z)) < 1e-8:
        z = np.cross(t, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    z = z / (np.linalg.norm(z) + 1e-12)
    y = np.cross(z, t)
    y = y / (np.linalg.norm(y) + 1e-12)
    # Columns are body axes expressed in world coordinates
    r = np.stack([t, y, z], axis=1)
    return rotations.mat2quat(r)


def numeric_tangent(
    position_fn,
    t: float,
    eps: float = 1e-4,
) -> np.ndarray:
    """Finite-difference tangent dp/dt for a smooth position(t)."""
    p0 = np.asarray(position_fn(t - eps), dtype=np.float64).reshape(3)
    p1 = np.asarray(position_fn(t + eps), dtype=np.float64).reshape(3)
    v = (p1 - p0) / (2.0 * eps)
    return v / (np.linalg.norm(v) + 1e-12)
