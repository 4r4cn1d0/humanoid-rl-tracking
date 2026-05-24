"""Stable-Baselines3 callbacks: rollout diagnostics and curriculum."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from envs.tracking_env import TrajectoryTrackingEnv

# Import ResidualTrackingEnv if available
try:
    from envs.residual_tracking_env import ResidualTrackingEnv
    _HAS_RESIDUAL = True
except ImportError:
    _HAS_RESIDUAL = False
    ResidualTrackingEnv = None  # type: ignore


class RenderEvalCallback(BaseCallback):
    """
    Every `render_freq` training episodes, pause training and run one full
    episode with render_mode='human' so you can watch the arm live.

    A fresh render env is created each time (avoids MuJoCo window conflicts).
    The render env mirrors the current curriculum stage settings.
    """

    def __init__(self, env_factory, render_freq: int = 10, verbose: int = 1):
        """
        Args:
            env_factory: callable() -> ResidualTrackingEnv (no render_mode set)
            render_freq:  run a render episode every this many training episodes
        """
        super().__init__(verbose)
        self._env_factory = env_factory
        self.render_freq = render_freq
        self._episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones") or []
        for done in dones:
            if done:
                self._episode_count += 1
                if self._episode_count % self.render_freq == 0:
                    self._run_render_episode()
        return True

    def _run_render_episode(self) -> None:
        """Spin up a render env, copy current curriculum state, run one episode."""
        # Build a render env that mirrors the training env's current state
        train_env = iter_tracking_envs(self.model)[0]

        render_env = self._env_factory(render_mode="human")

        # Mirror curriculum settings from training env
        render_env.trajectory = train_env.trajectory
        render_env.obs_noise_std = train_env.obs_noise_std
        render_env.action_noise_std = train_env.action_noise_std
        render_env.action_delay = train_env.action_delay
        # Copy residual alpha only if both envs are residual
        if (_HAS_RESIDUAL
                and isinstance(train_env, ResidualTrackingEnv)
                and isinstance(render_env, ResidualTrackingEnv)):
            render_env.residual_alpha = train_env.residual_alpha

        obs, _ = render_env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        errors = []

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = render_env.step(action)
            total_reward += float(reward)
            steps += 1
            if "tracking_error" in info:
                errors.append(info["tracking_error"])
            done = terminated or truncated

        render_env.close()

        if self.verbose > 0:
            mean_err = float(np.mean(errors)) * 100 if errors else 0.0
            print(f"\n[Render ep {self._episode_count}] "
                  f"steps={steps}, reward={total_reward:.2f}, "
                  f"mean_err={mean_err:.1f}cm  "
                  f"(stage={getattr(self, '_cur_stage', '?')})\n")

def iter_tracking_envs(model) -> list[Union[TrajectoryTrackingEnv, 'ResidualTrackingEnv']]:
    """All TrajectoryTrackingEnv or ResidualTrackingEnv instances inside a VecEnv (or single env)."""
    vec = model.get_env()
    out: list[Union[TrajectoryTrackingEnv, 'ResidualTrackingEnv']] = []
    for wrapped in vec.envs:
        e: Any = wrapped
        while True:
            if isinstance(e, TrajectoryTrackingEnv):
                out.append(e)
                break
            if _HAS_RESIDUAL and isinstance(e, ResidualTrackingEnv):
                out.append(e)
                break
            if not hasattr(e, "env"):
                raise RuntimeError("Expected TrajectoryTrackingEnv or ResidualTrackingEnv inside SB3 wrappers")
            e = e.env
    return out


def unwrap_tracking_env(model) -> Union[TrajectoryTrackingEnv, 'ResidualTrackingEnv']:
    """First TrajectoryTrackingEnv or ResidualTrackingEnv (backward compatible)."""
    return iter_tracking_envs(model)[0]


class RewardLoggingCallback(BaseCallback):
    """Log mean tracking diagnostics once per rollout to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._tracking_errors: list[float] = []
        self._reward_track: list[float] = []
        self._reward_smooth: list[float] = []
        self._orient_errors: list[float] = []
        # Action diagnostics
        self._action_mean_abs: list[float] = []
        self._residual_mean_abs: list[float] = []
        self._pd_mean_abs: list[float] = []
        self._rl_contribution_mean_abs: list[float] = []

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
            # Action diagnostics
            if "action_mean_abs" in info:
                self._action_mean_abs.append(float(info["action_mean_abs"]))
            if "residual_mean_abs" in info:
                self._residual_mean_abs.append(float(info["residual_mean_abs"]))
            if "pd_mean_abs" in info:
                self._pd_mean_abs.append(float(info["pd_mean_abs"]))
            if "rl_contribution_mean_abs" in info:
                self._rl_contribution_mean_abs.append(float(info["rl_contribution_mean_abs"]))
            # Log environment's tracking error statistics directly
            if "mean_tracking_error" in info:
                self.logger.record("train/env_mean_tracking_error", float(info["mean_tracking_error"]))
            if "std_tracking_error" in info:
                self.logger.record("train/env_std_tracking_error", float(info["std_tracking_error"]))
            if "max_tracking_error" in info:
                self.logger.record("train/env_max_tracking_error", float(info["max_tracking_error"]))
            if "is_success" in info:
                self.logger.record("train/success_rate", float(info["is_success"]))
        return True

    def _on_rollout_end(self) -> None:
        if self._tracking_errors:
            errors = np.array(self._tracking_errors)
            self.logger.record("train/mean_tracking_error", float(np.mean(errors)))
            self.logger.record("train/max_tracking_error", float(np.max(errors)))
            self.logger.record("train/std_tracking_error", float(np.std(errors)))
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
        # Action diagnostics
        if self._action_mean_abs:
            self.logger.record("train/action_mean_abs", float(np.mean(self._action_mean_abs)))
            self._action_mean_abs.clear()
        if self._residual_mean_abs:
            self.logger.record("train/residual_mean_abs", float(np.mean(self._residual_mean_abs)))
            self._residual_mean_abs.clear()
        if self._pd_mean_abs:
            self.logger.record("train/pd_mean_abs", float(np.mean(self._pd_mean_abs)))
            self._pd_mean_abs.clear()
        if self._rl_contribution_mean_abs:
            self.logger.record("train/rl_contribution_mean_abs", float(np.mean(self._rl_contribution_mean_abs)))
            self._rl_contribution_mean_abs.clear()


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


class ResidualScalingCallback(BaseCallback):
    """
    Anneal residual scaling factor α during training.
    Gradually increases from small value (PD-dominated) to 1.0 (full RL control).
    """

    def __init__(self, alpha_schedule: list[float], alpha_timesteps: list[int], verbose: int = 0):
        super().__init__(verbose)
        if len(alpha_schedule) != len(alpha_timesteps):
            raise ValueError("alpha_schedule and alpha_timesteps must have same length")
        self.alpha_schedule = alpha_schedule
        self.alpha_timesteps = alpha_timesteps
        self._last_applied_index = -1

    def _apply_alpha(self, index: int) -> None:
        alpha = self.alpha_schedule[index]
        for env in iter_tracking_envs(self.model):
            if _HAS_RESIDUAL and isinstance(env, ResidualTrackingEnv):
                env.set_residual_alpha(alpha)
        if self.verbose > 0:
            print(f"[ResidualScaling] α = {alpha:.3f} at timestep {self.num_timesteps}")

    def _current_alpha_index(self) -> int:
        t = int(self.num_timesteps)
        idx = 0
        for i, ts in enumerate(self.alpha_timesteps):
            if t >= ts:
                idx = i
        return idx

    def _on_training_start(self) -> None:
        self._last_applied_index = -1
        idx = self._current_alpha_index()
        self._apply_alpha(idx)
        self._last_applied_index = idx

    def _on_step(self) -> bool:
        idx = self._current_alpha_index()
        if idx != self._last_applied_index:
            self._apply_alpha(idx)
            self._last_applied_index = idx
        return True


class PerformanceGatedCurriculumCallback(BaseCallback):
    """
    Curriculum with PERFORMANCE-GATED transitions, not timestep-gated.

    Each stage dict includes:
    - trajectory_speed, obs_noise_std, action_noise_std, action_delay
    - advance_if: dict with success_rate_gt and/or mean_tracking_error_lt thresholds

    Advancement to next stage ONLY occurs when BOTH conditions are met:
    - success_rate > success_rate_gt (over recent episodes)
    - mean_tracking_error < mean_tracking_error_lt (over recent episodes)

    This prevents catastrophic curriculum jumps when the policy isn't ready.
    """

    def __init__(self, stages: list[dict[str, Any]], verbose: int = 0):
        super().__init__(verbose)
        self.stages = stages
        self._current_stage = 0
        # Episode-level rolling statistics (one value per episode, not per step)
        self._episode_successes: list[float] = []   # mean success rate per episode
        self._episode_errors: list[float] = []       # mean tracking error per episode
        # Per-episode accumulators
        self._ep_step_successes: list[int] = []
        self._ep_step_errors: list[float] = []
        self._max_episodes = 50        # rolling window size
        self._min_episodes = 20        # minimum episodes before checking gates
        self._min_timesteps_per_stage = 15_000  # ~75 episodes per stage at 200 steps/ep
        self._stage_start_timestep = 0
        # Hysteresis: track how many consecutive episodes gates have been met
        self._consecutive_gate_episodes: int = 0

    def _apply_stage(self, index: int) -> None:
        """Apply stage configuration to all environments."""
        if index >= len(self.stages):
            if self.verbose > 0:
                print(f"[PerformanceCurriculum] Already at final stage {index}")
            return

        cfg = self.stages[index]
        for env in iter_tracking_envs(self.model):
            # Change trajectory type if specified
            if "trajectory" in cfg:
                from utils.trajectories import make_trajectory
                traj_type = cfg["trajectory"]
                traj_kwargs = cfg.get("trajectory_kwargs", {})
                # Preserve center from current trajectory if not specified
                if "center" not in traj_kwargs and hasattr(env.trajectory, "center"):
                    traj_kwargs["center"] = env.trajectory.center
                # Only preserve radius for circle/figure8 trajectories (not spline)
                if traj_type in ["circle", "figure8"]:
                    if "radius" not in traj_kwargs and hasattr(env.trajectory, "radius"):
                        traj_kwargs["radius"] = env.trajectory.radius
                # Set speed if specified
                if "trajectory_speed" in cfg:
                    traj_kwargs["speed"] = float(cfg["trajectory_speed"])
                env.trajectory = make_trajectory(traj_type, **traj_kwargs)
            elif "trajectory_speed" in cfg and hasattr(env.trajectory, "set_speed"):
                # Only change speed if trajectory type not specified
                env.trajectory.set_speed(float(cfg["trajectory_speed"]))
            
            if "obs_noise_std" in cfg:
                env.set_obs_noise_std(float(cfg["obs_noise_std"]))
            if "action_noise_std" in cfg:
                env.set_action_noise_std(float(cfg["action_noise_std"]))
            if "action_delay" in cfg:
                env.set_action_delay(int(cfg["action_delay"]))

        if self.verbose > 0:
            traj_name = cfg.get('trajectory', 'same')
            speed = cfg.get('trajectory_speed', '?')
            print(f"[PerformanceCurriculum] Stage {index}: {traj_name} @ speed={speed}")

    def _check_advancement(self) -> bool:
        """Check if performance gates are met with hysteresis for advancement."""
        # Hard floor: minimum timesteps in this stage
        steps_in_stage = self.num_timesteps - self._stage_start_timestep
        if steps_in_stage < self._min_timesteps_per_stage:
            return False

        # Need enough episodes to compute reliable statistics
        if len(self._episode_errors) < self._min_episodes:
            return False

        if self._current_stage >= len(self.stages) - 1:
            return False  # Already at final stage

        current_stage = self.stages[self._current_stage]
        advance_cfg = current_stage.get("advance_if", {})
        if not advance_cfg:
            return False

        # Hysteresis: how many consecutive episodes must gates be sustained
        sustain_required = int(advance_cfg.get("sustain_episodes", 1))

        # Compute rolling stats over recent episodes
        recent_successes = self._episode_successes[-self._max_episodes:]
        recent_errors = self._episode_errors[-self._max_episodes:]
        success_rate = float(np.mean(recent_successes))
        mean_error = float(np.mean(recent_errors))

        success_gate = advance_cfg.get("success_rate_gt", 0.0)
        error_gate = advance_cfg.get("mean_tracking_error_lt", float('inf'))

        gates_met = (success_rate >= success_gate) and (mean_error <= error_gate)

        # Hysteresis counter: increment when gates met, reset when not
        if gates_met:
            self._consecutive_gate_episodes += 1
        else:
            self._consecutive_gate_episodes = 0

        if self.verbose > 0:
            print(f"[PerformanceCurriculum] Stage {self._current_stage} "
                  f"({steps_in_stage} steps, {len(recent_errors)} eps): "
                  f"success={success_rate:.2%}/{success_gate:.0%}, "
                  f"err={mean_error*100:.1f}cm/{error_gate*100:.1f}cm, "
                  f"sustained={self._consecutive_gate_episodes}/{sustain_required}")

        return gates_met and (self._consecutive_gate_episodes >= sustain_required)

    def _on_step(self) -> bool:
        """Accumulate per-step stats; flush to episode stats on episode end."""
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones") or []

        for i, info in enumerate(infos):
            # Accumulate within the current episode
            if "is_success" in info:
                self._ep_step_successes.append(int(info["is_success"]))
            if "tracking_error" in info:
                self._ep_step_errors.append(float(info["tracking_error"]))

            # Detect episode end
            episode_done = (i < len(dones) and dones[i]) or info.get("TimeLimit.truncated", False)

            if episode_done and self._ep_step_errors:
                # Flush episode stats
                ep_success = float(np.mean(self._ep_step_successes)) if self._ep_step_successes else 0.0
                ep_error = float(np.mean(self._ep_step_errors))
                self._episode_successes.append(ep_success)
                self._episode_errors.append(ep_error)
                self._ep_step_successes.clear()
                self._ep_step_errors.clear()

                # Trim rolling window
                if len(self._episode_errors) > self._max_episodes:
                    self._episode_successes = self._episode_successes[-self._max_episodes:]
                    self._episode_errors = self._episode_errors[-self._max_episodes:]

                # Check advancement
                if self._check_advancement():
                    old_stage = self._current_stage
                    self._current_stage += 1
                    self._stage_start_timestep = self.num_timesteps
                    self._consecutive_gate_episodes = 0  # reset hysteresis
                    self._apply_stage(self._current_stage)
                    self._episode_successes.clear()
                    self._episode_errors.clear()
                    if self.verbose > 0:
                        print(f"[PerformanceCurriculum] *** ADVANCED stage {old_stage} → "
                              f"{self._current_stage} at {self.num_timesteps} steps ***")

        return True

    def _on_training_start(self) -> None:
        """Apply initial stage configuration."""
        self._current_stage = 0
        self._stage_start_timestep = 0
        self._consecutive_gate_episodes = 0
        self._apply_stage(0)
        self._episode_successes.clear()
        self._episode_errors.clear()
        self._ep_step_successes.clear()
        self._ep_step_errors.clear()
