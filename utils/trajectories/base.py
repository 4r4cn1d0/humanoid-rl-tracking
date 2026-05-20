"""Time-parameterized desired target positions for trajectory tracking."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Trajectory(Protocol):
    """Desired workspace position p(t) in R^3."""

    def position(self, t: float) -> np.ndarray:
        """Return target position at time t (seconds)."""


def as_array3(x: np.ndarray | list[float]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {a.shape}")
    return a
