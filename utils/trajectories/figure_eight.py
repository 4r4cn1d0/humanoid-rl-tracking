from __future__ import annotations

import numpy as np

from utils.orientation_utils import quat_from_xy_path_tangent
from utils.trajectories.base import as_array3


class FigureEightTrajectory:
    """
    Lemniscate-style figure-8 in XY: uses param s = speed * t.
    x = cx + a * sin(s), y = cy + a * sin(s) * cos(s)  (scaled variant).
    """

    def __init__(
        self,
        center: np.ndarray | None = None,
        amplitude: float = 0.12,
        speed: float = 0.8,
    ):
        if center is None:
            center = np.array([1.3, 0.75, 0.5], dtype=np.float64)
        self.center = as_array3(center)
        self.amplitude = float(amplitude)
        self.speed = float(speed)

    def position(self, t: float) -> np.ndarray:
        s = self.speed * t
        # Figure-8 in plane: Lissajous-like
        x = self.center[0] + self.amplitude * np.sin(s)
        y = self.center[1] + self.amplitude * np.sin(s) * np.cos(s)
        z = self.center[2]
        return np.array([x, y, z], dtype=np.float64)

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)

    def orientation(self, t: float) -> np.ndarray:
        s = self.speed * t
        sp = self.speed
        tx = self.amplitude * np.cos(s) * sp
        ty = self.amplitude * np.cos(2.0 * s) * sp
        tangent = np.array([tx, ty, 0.0], dtype=np.float64)
        return quat_from_xy_path_tangent(tangent)
