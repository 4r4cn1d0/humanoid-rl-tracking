from __future__ import annotations

from typing import Callable

from utils.trajectories.base import Trajectory
from utils.trajectories.circle import CircleTrajectory
from utils.trajectories.figure_eight import FigureEightTrajectory
from utils.trajectories.spline import WaypointSplineTrajectory

TRAJECTORY_REGISTRY: dict[str, Callable[..., Trajectory]] = {
    "circle": CircleTrajectory,
    "figure8": FigureEightTrajectory,
    "figure_eight": FigureEightTrajectory,
    "spline": WaypointSplineTrajectory,
}


def make_trajectory(
    name: str,
    *,
    seed: int | None = None,
    **kwargs,
) -> Trajectory:
    kwargs = dict(kwargs)
    seed_from_kw = kwargs.pop("seed", None)
    if seed is None:
        seed = seed_from_kw

    key = name.lower().replace("-", "_")
    if key in ("random_spline",):
        key = "spline"
    if key == "spline":
        if kwargs.get("waypoints") is not None:
            return WaypointSplineTrajectory(**kwargs)
        return WaypointSplineTrajectory.random_in_box(seed=seed, **kwargs)
    if key not in TRAJECTORY_REGISTRY:
        raise ValueError(
            f"Unknown trajectory {name!r}. Choose from: {sorted(set(TRAJECTORY_REGISTRY))}"
        )
    cls = TRAJECTORY_REGISTRY[key]
    return cls(**kwargs)


__all__ = [
    "Trajectory",
    "CircleTrajectory",
    "FigureEightTrajectory",
    "WaypointSplineTrajectory",
    "make_trajectory",
    "TRAJECTORY_REGISTRY",
]
