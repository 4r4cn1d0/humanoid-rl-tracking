from __future__ import annotations

import numpy as np

from utils.orientation_utils import numeric_tangent, quat_from_xy_path_tangent
from utils.trajectories.base import as_array3


class WaypointSplineTrajectory:
    """
    Piecewise linear path through 3D waypoints, traversed at constant speed
    along the polyline (parameter t in seconds advances arc length).
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        speed: float = 0.2,
        loop: bool = True,
        rng: np.random.Generator | None = None,
    ):
        wp = np.asarray(waypoints, dtype=np.float64)
        if wp.ndim != 2 or wp.shape[1] != 3:
            raise ValueError("waypoints must be shape (N, 3)")
        if len(wp) < 2:
            raise ValueError("Need at least 2 waypoints")
        self.waypoints = wp
        self.speed = float(speed)
        self.loop = bool(loop)
        self.rng = rng or np.random.default_rng()

        seg = np.diff(self.waypoints, axis=0)
        self.segment_lengths = np.linalg.norm(seg, axis=1)
        self.total_length = float(np.sum(self.segment_lengths))
        if self.total_length <= 0:
            raise ValueError("Waypoint path has zero length")

    @classmethod
    def random_in_box(
        cls,
        center: np.ndarray | None = None,
        half_extent: float = 0.1,
        n_points: int = 5,
        speed: float = 0.2,
        seed: int | None = None,
    ) -> "WaypointSplineTrajectory":
        rng = np.random.default_rng(seed)
        if center is None:
            center = np.array([1.3, 0.75, 0.5], dtype=np.float64)
        center = as_array3(center)
        pts = center + rng.uniform(-half_extent, half_extent, size=(n_points, 3))
        return cls(pts, speed=speed, loop=True, rng=rng)

    def position(self, t: float) -> np.ndarray:
        distance_along = (self.speed * t) % (
            self.total_length if self.loop else max(self.total_length, 1e-9)
        )
        if not self.loop:
            distance_along = min(distance_along, self.total_length - 1e-9)

        acc = 0.0
        for i, seg_len in enumerate(self.segment_lengths):
            if acc + seg_len >= distance_along - 1e-12:
                alpha = (distance_along - acc) / seg_len if seg_len > 1e-12 else 0.0
                p0 = self.waypoints[i]
                p1 = self.waypoints[i + 1]
                return (1.0 - alpha) * p0 + alpha * p1
            acc += seg_len
        return self.waypoints[-1].copy()

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)

    def orientation(self, t: float) -> np.ndarray:
        tan = numeric_tangent(self.position, t)
        return quat_from_xy_path_tangent(tan)
