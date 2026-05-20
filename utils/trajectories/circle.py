from __future__ import annotations

import numpy as np

from utils.orientation_utils import quat_from_xy_path_tangent
from utils.trajectories.base import as_array3


class CircleTrajectory:
    """Circular motion in the XY plane at fixed Z."""

    def __init__(
        self,
        center: np.ndarray | None = None,
        radius: float = 0.15,
        speed: float = 1.0,
    ):
        if center is None:
            center = np.array([1.3, 0.75, 0.5], dtype=np.float64)
        self.center = as_array3(center)
        self.radius = float(radius)
        self.speed = float(speed)

    def position(self, t: float) -> np.ndarray:
        x = self.center[0] + self.radius * np.cos(self.speed * t)
        y = self.center[1] + self.radius * np.sin(self.speed * t)
        z = self.center[2]
        return np.array([x, y, z], dtype=np.float64)

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)

    def orientation(self, t: float) -> np.ndarray:
        """EE +x aligns with motion tangent in the circle plane (XY)."""
        s = self.speed * t
        tangent = self.radius * self.speed * np.array(
            [-np.sin(s), np.cos(s), 0.0], dtype=np.float64
        )
        return quat_from_xy_path_tangent(tangent)
