"""Stable-Baselines3 callbacks: rollout diagnostics and curriculum."""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from envs.tracking_env import TrajectoryTrackingEnv


def iter_tracking_envs(model) -> list[TrajectoryTrackingEnv]:
    """All TrajectoryTrackingEnv instances inside a VecEnv (or single env)."""
    vec = model.get_env()
    out: list[TrajectoryTrackingEnv] = []
    for wrapped in vec.envs:
        e: Any = wrapped
        while not isinstance(e, TrajectoryTrackingEnv):
            if not hasattr(e, "env"):
                raise RuntimeError("Expected TrajectoryTrackingEnv inside SB3 wrappers")
            e = e.env
        out.append(e)
    return out


def unwrap_tracking_env(model) -> TrajectoryTrackingEnv:
    """First TrajectoryTrackingEnv (backward compatible)."""
    return iter_tracking_envs(model)[0]


class RewardLoggingCallback(BaseCallback):
    """Log mean tracking diagnostics once per rollout to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._tracking_errors: list[float] = []
        self._reward_track: list[float] = []
        self._reward_smooth: list[float] = []
        self._orient_errors: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            if "tracking_error" in info:
                self._tracking_errors.append(float(info["tracking_error"]))
            if "reward_tracking" in info:
                self._reward_track.append(float(info["reward_tracking"]))
            if "reward_smoothness" in info:
                self._reward_smooth.append(float(info["reward_smoothness"]))
            if "orientation_error" in info:
                self._orient_errors.append(float(info["orientation_error"]))
        return True

    def _on_rollout_end(self) -> None:
        if self._tracking_errors:
            self.logger.record("train/mean_tracking_error", float(np.mean(self._tracking_errors)))
            self._tracking_errors.clear()
        if self._reward_track:
            self.logger.record("train/mean_reward_tracking", float(np.mean(self._reward_track)))
            self._reward_track.clear()
        if self._reward_smooth:
            self.logger.record("train/mean_reward_smoothness", float(np.mean(self._reward_smooth)))
            self._reward_smooth.clear()
        if self._orient_errors:
            self.logger.record("train/mean_orientation_error", float(np.mean(self._orient_errors)))
            self._orient_errors.clear()


class CurriculumCallback(BaseCallback):
    """
    Piecewise schedule keyed by training timestep (see experiments/*.yaml).
    Each stage dict may include: trajectory_speed, obs_noise_std, action_noise_std, action_delay.
    """

    def __init__(self, stages: list[dict[str, Any]], verbose: int = 0):
        super().__init__(verbose)
        self.stages = sorted(stages, key=lambda s: int(s.get("start_timestep", 0)))
        self._last_applied_index = -1

    def _apply_stage(self, index: int) -> None:
        cfg = self.stages[index]
        for env in iter_tracking_envs(self.model):
            if "trajectory_speed" in cfg and hasattr(env.trajectory, "set_speed"):
                env.trajectory.set_speed(float(cfg["trajectory_speed"]))  # type: ignore[attr-defined]
            if "obs_noise_std" in cfg:
                env.set_obs_noise_std(float(cfg["obs_noise_std"]))
            if "action_noise_std" in cfg:
                env.set_action_noise_std(float(cfg["action_noise_std"]))
            if "action_delay" in cfg:
                env.set_action_delay(int(cfg["action_delay"]))
        if self.verbose > 0:
            print(f"[Curriculum] stage {index}: {cfg}")

    def _current_stage_index(self) -> int:
        t = int(self.num_timesteps)
        idx = 0
        for i, s in enumerate(self.stages):
            if t >= int(s.get("start_timestep", 0)):
                idx = i
        return idx

    def _on_training_start(self) -> None:
        self._last_applied_index = -1
        idx = self._current_stage_index()
        self._apply_stage(idx)
        self._last_applied_index = idx

    def _on_step(self) -> bool:
        idx = self._current_stage_index()
        if idx != self._last_applied_index:
            self._apply_stage(idx)
            self._last_applied_index = idx
        return True
