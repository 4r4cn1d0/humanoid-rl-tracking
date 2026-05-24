from __future__ import annotations

"""
Pure SAC trajectory tracking environment — no PD controller.

The policy directly outputs mocap position deltas (same as FetchReach's
native action space). SAC must learn everything from scratch.

Observation (flat Dict):
  achieved_goal    (3,)  — current EE position
  desired_goal     (3,)  — current target position
  observation      (10,) — FetchReach internal obs (joint pos/vel)
  tracking_error   (3,)  — target - EE  (signed, in metres)
  tracking_error_vel (3,)— derivative of tracking error
  target_vel       (3,)  — trajectory velocity at current t
  phase_encoding   (2,)  — [sin(ωt), cos(ωt)]
  prev_action      (4,)  — last applied action

All values cast to float32 for MPS compatibility.
"""

from typing import Any

import numpy as np
from gymnasium import spaces

from envs.tracking_env import (
    TrajectoryTrackingEnv,
    _observation_space_float32,
    _obs_to_float32,
)
from utils.trajectories import Trajectory


class PureSACEnv(TrajectoryTrackingEnv):
    """
    Trajectory tracking with pure SAC — no PD baseline.

    Extends TrajectoryTrackingEnv (which already handles noise, delay,
    reward, and obs augmentation) and adds:
      - target_vel observation
      - phase_encoding observation
      - tighter is_success threshold (1.5 cm)
      - episode-level tracking stats in info
    """

    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        trajectory: str | Trajectory = "circle",
        trajectory_kwargs: dict[str, Any] | None = None,
        control_dt: float = 0.05,
        max_episode_steps: int | None = 200,
        # Reward weights
        w_track: float = 1.0,
        w_smooth: float = 0.05,
        w_velocity: float = 0.0,
        # Robustness
        obs_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
        action_delay: int = 0,
        # Success threshold
        success_threshold: float = 0.015,  # 1.5 cm
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            render_mode=render_mode,
            trajectory=trajectory,
            trajectory_kwargs=trajectory_kwargs,
            control_dt=control_dt,
            max_episode_steps=max_episode_steps,
            w_track=w_track,
            w_smooth=w_smooth,
            w_velocity=w_velocity,
            obs_noise_std=obs_noise_std,
            action_noise_std=action_noise_std,
            action_delay=action_delay,
            rng=rng,
        )

        self.success_threshold = float(success_threshold)
        self._tracking_errors: list[float] = []

        # Extend observation space with target_vel and phase_encoding
        orig = dict(self.observation_space.spaces)
        orig["target_vel"] = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        orig["phase_encoding"] = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = _observation_space_float32(spaces.Dict(orig))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _target_velocity(self, t: float) -> np.ndarray:
        """Finite-difference velocity of the trajectory at time t."""
        dt = 1e-5
        return (self.trajectory.position(t + dt) - self.trajectory.position(t)) / dt

    def _phase_encoding(self, t: float) -> np.ndarray:
        """sin/cos phase for periodic trajectories."""
        omega = getattr(self.trajectory, "speed", 1.0)
        return np.array([np.sin(omega * t), np.cos(omega * t)], dtype=np.float32)

    def _augment_full(self, obs: dict, target: np.ndarray) -> dict:
        """Add target_vel and phase on top of parent augmentation."""
        out = self._augment_obs_dict(obs, target)
        out["target_vel"] = self._target_velocity(self.t).astype(np.float32)
        out["phase_encoding"] = self._phase_encoding(self.t)
        return out

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._tracking_errors = []

        # Re-augment with the extra keys (parent already called _augment_obs_dict)
        target = self.trajectory.position(self.t)
        obs["target_vel"] = self._target_velocity(self.t).astype(np.float32)
        obs["phase_encoding"] = self._phase_encoding(self.t)
        return _obs_to_float32(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = super().step(action)

        # Add extra obs keys
        target = self.trajectory.position(self.t)
        obs["target_vel"] = self._target_velocity(self.t).astype(np.float32)
        obs["phase_encoding"] = self._phase_encoding(self.t)
        obs = _obs_to_float32(obs)

        dist = float(info["tracking_error"])
        self._tracking_errors.append(dist)

        # Override success with tighter threshold
        info["is_success"] = float(dist < self.success_threshold)

        # Episode-level stats
        if self._tracking_errors:
            info["mean_tracking_error"] = float(np.mean(self._tracking_errors))
            info["max_tracking_error"] = float(np.max(self._tracking_errors))
            info["std_tracking_error"] = float(np.std(self._tracking_errors))

        return obs, reward, terminated, truncated, info

    def set_success_threshold(self, threshold: float) -> None:
        self.success_threshold = float(threshold)
