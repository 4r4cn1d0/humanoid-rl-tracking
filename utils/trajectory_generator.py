"""
Backward-compatible alias: use utils.trajectories.CircleTrajectory or make_trajectory.
"""

from __future__ import annotations

import numpy as np

from utils.trajectories.circle import CircleTrajectory


class TrajectoryGenerator(CircleTrajectory):
    """Legacy name; delegates to circle trajectory with get_target(t) API."""

    def get_target(self, t: float) -> np.ndarray:
        return self.position(t)
