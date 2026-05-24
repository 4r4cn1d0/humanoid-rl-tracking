from __future__ import annotations

import types
from collections import deque
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium import spaces
from gymnasium_robotics.utils import rotations

from envs.tracking_env import TrajectoryTrackingEnv, _DEFAULT_FETCH_QUAT, _observation_space_float32, _obs_to_float32
from utils.orientation_utils import normalize_quat, quat_geodesic_distance
from utils.trajectories import Trajectory, make_trajectory


class ResidualTrackingEnv(TrajectoryTrackingEnv):
    """
    Residual RL environment with operational-space PD controller.
    
    Architecture:
    - PD controller provides baseline tracking: u_pd = Kp * e + Kd * ė
    - SAC policy learns residual correction: u_rl
    - Total action: u_total = u_pd + α * u_rl
    
    Features:
    - Velocity command action space (policy outputs v_cmd, integrated internally)
    - Action repeat (policy at lower frequency than sim)
    - Enhanced observations: target_vel, phase_encoding, future_targets
    - Enhanced rewards: exponential tracking, velocity matching, jerk penalty
    """

    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        trajectory: str | Trajectory = "circle",
        trajectory_kwargs: dict[str, Any] | None = None,
        control_dt: float = 0.03,
        max_episode_steps: int | None = None,
        w_track: float = 1.0,
        w_smooth: float = 0.1,
        w_velocity: float = 0.0,
        w_orient: float = 0.0,
        w_align: float = 0.0,
        track_orientation: bool = False,
        use_squared_error: bool = False,
        obs_noise_std: float = 0.0,
        obs_noise_keys: tuple[str, ...] | None = None,
        action_noise_std: float = 0.0,
        action_delay: int = 0,
        # Residual RL specific parameters
        pd_kp: float = 25.0,
        pd_kd: float = 8.0,
        residual_alpha: float = 0.1,
        action_repeat: int = 2,
        velocity_max: float = 0.5,
        # Enhanced reward parameters
        track_exp_k: float = 10.0,
        vel_match_k: float = 5.0,
        jerk_penalty_weight: float = 0.01,
        # Future waypoints
        future_horizon: int = 3,
        rng: np.random.Generator | None = None,
    ):
        # Initialize parent class
        super().__init__(
            render_mode=render_mode,
            trajectory=trajectory,
            trajectory_kwargs=trajectory_kwargs,
            control_dt=control_dt,
            max_episode_steps=max_episode_steps,
            w_track=w_track,
            w_smooth=w_smooth,
            w_velocity=w_velocity,
            w_orient=w_orient,
            w_align=w_align,
            track_orientation=track_orientation,
            use_squared_error=use_squared_error,
            obs_noise_std=obs_noise_std,
            obs_noise_keys=obs_noise_keys,
            action_noise_std=action_noise_std,
            action_delay=action_delay,
            rng=rng,
        )

        # PD controller gains
        self.pd_kp = float(pd_kp)
        self.pd_kd = float(pd_kd)

        # Action scale for PD output mapping
        self.action_scale = 5.0

        # Residual scaling factor
        self.residual_alpha = float(residual_alpha)

        # Step counter for debug prints
        self._step_count = 0
        
        # Action repeat
        self.action_repeat = int(action_repeat)
        self._action_repeat_counter = 0
        self._current_action = np.zeros(self.action_space.shape, dtype=np.float64)
        
        # Velocity command parameters
        self.velocity_max = float(velocity_max)
        self._integrated_position = np.zeros(3, dtype=np.float64)
        
        # Enhanced reward parameters
        self.track_exp_k = float(track_exp_k)
        self.vel_match_k = float(vel_match_k)
        self.jerk_penalty_weight = float(jerk_penalty_weight)
        
        # Future waypoints
        self.future_horizon = int(future_horizon)
        
        # Action history for jerk penalty
        self._action_history_jerk: deque[np.ndarray] = deque(maxlen=3)
        
        # Velocity command mode flag
        self._use_velocity_commands = False
        
        # Update observation space to include new features
        self._update_observation_space()

    def _update_observation_space(self) -> None:
        """Add enhanced observations: target_vel, phase_encoding, future_targets."""
        orig_spaces = dict(self.observation_space.spaces)

        # Target velocity
        orig_spaces["target_vel"] = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

        # Phase encoding (sin, cos)
        orig_spaces["phase_encoding"] = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Future targets — only add if horizon > 0 to avoid zero-size observation
        if self.future_horizon > 0:
            orig_spaces["future_targets"] = spaces.Box(
                -np.inf, np.inf, shape=(self.future_horizon * 3,), dtype=np.float32
            )

        self.observation_space = spaces.Dict(orig_spaces)
        self.observation_space = _observation_space_float32(self.observation_space)

    def _compute_pd_control(self, target_pos: np.ndarray, achieved_pos: np.ndarray, 
                           target_vel: np.ndarray, achieved_vel: np.ndarray) -> np.ndarray:
        """
        Compute PD operational-space control action.
        
        u_pd = Kp * e + Kd * ė
        where e = target_pos - achieved_pos
              ė = target_vel - achieved_vel
        
        Returns action scaled to [-1, 1] range for FetchReach mocap control.
        """
        error = target_pos - achieved_pos
        error_vel = target_vel - achieved_vel

        u_pd = self.pd_kp * error + self.pd_kd * error_vel
        u_pd = u_pd / self.action_scale  # Softer scaling — avoids saturation at small errors
        return np.clip(u_pd, -1.0, 1.0)

    def _get_trajectory_velocity(self, t: float) -> np.ndarray:
        """Compute trajectory velocity at time t using finite difference."""
        dt = 1e-6
        pos_t = self.trajectory.position(t)
        pos_t_dt = self.trajectory.position(t + dt)
        vel = (pos_t_dt - pos_t) / dt
        return vel

    def _get_phase_encoding(self, t: float) -> np.ndarray:
        """
        Compute phase encoding for periodic trajectories.
        Returns [sin(ωt), cos(ωt)] where ω is trajectory frequency.
        """
        if hasattr(self.trajectory, 'speed'):
            omega = self.trajectory.speed
        else:
            # Estimate frequency from trajectory
            omega = 1.0  # Default fallback
        
        phase = omega * t
        return np.array([np.sin(phase), np.cos(phase)], dtype=np.float64)

    def _get_future_targets(self, t: float) -> np.ndarray:
        """
        Get future target positions for model-predictive tracking.
        Returns flattened array of shape (future_horizon * 3,).
        """
        future_targets = []
        for i in range(1, self.future_horizon + 1):
            future_pos = self.trajectory.position(t + i * self.control_dt)
            future_targets.append(future_pos)
        return np.array(future_targets, dtype=np.float64).flatten()

    def _augment_obs_dict_residual(self, obs: dict, target: np.ndarray) -> dict:
        """Add residual-specific observations to existing obs dict."""
        out = super()._augment_obs_dict(obs, target)
        
        # Add target velocity
        target_vel = self._get_trajectory_velocity(self.t)
        out["target_vel"] = target_vel.copy()
        
        # Add phase encoding
        phase_encoding = self._get_phase_encoding(self.t)
        out["phase_encoding"] = phase_encoding.copy()

        # Add future targets — only when horizon > 0
        if self.future_horizon > 0:
            future_targets = self._get_future_targets(self.t)
            out["future_targets"] = future_targets.copy()

        return out

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        obs, info = super().reset(seed=seed, options=options)
        
        # Reset residual-specific state
        self._action_repeat_counter = 0
        self._current_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self._integrated_position = obs["achieved_goal"].copy()
        self._action_history_jerk.clear()
        self._step_count = 0
        self._tracking_errors = []
        
        # Augment with residual observations
        target = self.trajectory.position(self.t)
        obs = self._augment_obs_dict_residual(obs, target)
        
        return _obs_to_float32(self._maybe_noise_obs(obs)), info

    def step(self, action: np.ndarray):
        # Action repeat logic
        if self._action_repeat_counter == 0:
            self._current_action = np.asarray(action, dtype=np.float64).reshape(self.action_space.shape)
        
        self._action_repeat_counter += 1
        if self._action_repeat_counter >= self.action_repeat:
            self._action_repeat_counter = 0
        
        a_rl = self._current_action  # RL policy output (residual)

        # Get target information for PD controller
        target_position = self.trajectory.position(self.t)
        target_vel = self._get_trajectory_velocity(self.t)

        # Get current EE position from previous observation
        achieved_goal = self._prev_achieved_goal
        ee_vel = (achieved_goal - self._prev_achieved_goal) / self.control_dt if self.t > 0 else np.zeros(3)

        # Action-space residual: u_total = u_pd + α * u_rl
        u_pd = self._compute_pd_control(target_position, achieved_goal, target_vel, ee_vel)
        u_rl = np.tanh(a_rl[:3])
        u_total = u_pd + self.residual_alpha * u_rl
        u_total = np.clip(u_total, -1.0, 1.0)
        u_total_pos = u_total

        # For gripper component (last 1), just use RL action
        u_total_gripper = a_rl[3:]

        a_cmd = np.concatenate([u_total_pos, u_total_gripper])
        
        # Convert velocity command to position command
        if self._use_velocity_commands:
            # Scale velocity command
            v_cmd = a_cmd[:3] * self.velocity_max
            # Integrate position
            self._integrated_position = self._integrated_position + v_cmd * self.control_dt
            # Clip to workspace bounds
            self._integrated_position = np.clip(
                self._integrated_position,
                self.env.action_space.low[:3],
                self.env.action_space.high[:3]
            )
            # Construct full action (position + gripper)
            a_apply = np.concatenate([self._integrated_position, a_cmd[3:]])
        else:
            a_apply = a_cmd
        
        # Apply action delay and noise (from parent)
        a_apply = self._policy_action_to_torque_command(a_apply)
        
        # Set target in environment
        self.env.unwrapped.goal = target_position
        if self.track_orientation:
            self._target_quat_current = normalize_quat(
                np.asarray(self.trajectory.orientation(self.t), dtype=np.float64).reshape(4)
            )
        
        # Step environment
        obs, _base_reward, terminated, truncated, info = self.env.step(a_apply)
        
        # Augment observations
        obs = self._augment_obs_dict_residual(obs, target_position)
        achieved_goal = obs["achieved_goal"]
        
        # Compute velocities for next step
        ee_vel = (achieved_goal - self._prev_achieved_goal) / self.control_dt
        
        # Compute tracking error
        err_vec = achieved_goal - target_position
        dist = float(np.linalg.norm(err_vec))
        
        # Accumulate tracking error for statistics
        self._tracking_errors.append(dist)
        
        # Enhanced reward computation
        reward = self._compute_enhanced_reward(
            achieved_goal, target_position, ee_vel, target_vel, a_cmd, err_vec
        )
        
        # Update state
        self.previous_action = a_cmd.copy()
        self._prev_achieved_goal = achieved_goal.copy()
        self._prev_tracking_error = target_position - achieved_goal
        self.t += self.control_dt
        self._step_count += 1
        
        # Debug prints for controller authority
        if self._step_count % 100 == 0:
            print("\n=== CONTROL DEBUG ===")
            print("Target position:", target_position)
            print("PD output mean abs:", float(np.mean(np.abs(u_pd[:3]))))
            print("RL action mean abs:", float(np.mean(np.abs(a_rl[:3]))))
            print("RL contribution mean abs:", float(np.mean(np.abs(self.residual_alpha * u_rl))))
        
        # Update action history for jerk
        self._action_history_jerk.append(a_cmd.copy())
        
        # Update info
        info = dict(info)
        info["tracking_error"] = dist
        info["target_position"] = target_position.copy()
        info["pd_control"] = u_pd.copy()
        info["residual_alpha"] = self.residual_alpha
        
        # Tracking error statistics
        if self._tracking_errors:
            info["mean_tracking_error"] = float(np.mean(self._tracking_errors))
            info["std_tracking_error"] = float(np.std(self._tracking_errors))
            info["max_tracking_error"] = float(np.max(self._tracking_errors))
        
        # Success metric (3cm threshold — PD alone hits ~2cm, so RL needs to be consistent)
        info["is_success"] = float(dist < 0.03)

        # Action diagnostics
        info["action_mean_abs"] = float(np.mean(np.abs(a_rl)))
        info["residual_mean_abs"] = float(np.mean(np.abs(u_rl)))
        info["pd_mean_abs"] = float(np.mean(np.abs(u_pd)))
        info["rl_contribution_mean_abs"] = float(np.mean(np.abs(self.residual_alpha * u_rl)))
        
        return _obs_to_float32(self._maybe_noise_obs(obs)), reward, terminated, truncated, info

    def _compute_enhanced_reward(
        self,
        achieved_goal: np.ndarray,
        target_position: np.ndarray,
        ee_vel: np.ndarray,
        target_vel: np.ndarray,
        action: np.ndarray,
        error_vec: np.ndarray,
    ) -> float:
        """Minimal reward for debugging control authority."""
        tracking_reward = -np.linalg.norm(error_vec)
        smoothness_penalty = -0.01 * np.linalg.norm(action - self.previous_action)
        return tracking_reward + smoothness_penalty

    # --- Residual-specific setters ---
    def set_residual_alpha(self, alpha: float) -> None:
        """Set residual scaling factor α."""
        self.residual_alpha = float(alpha)

    def set_pd_gains(self, kp: float, kd: float) -> None:
        """Set PD controller gains."""
        self.pd_kp = float(kp)
        self.pd_kd = float(kd)

    def set_action_repeat(self, repeat: int) -> None:
        """Set action repeat factor."""
        self.action_repeat = int(max(1, repeat))
