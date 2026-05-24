"""
Residual RL wrapper for FetchReachDense-v3.

Uses the standard FetchReach task (random goal each episode) but with
residual control architecture: u_total = u_pd + α * u_rl

This is the standard benchmark task, not trajectory tracking.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium import spaces


class ResidualFetchReachEnv(gym.Wrapper):
    """
    Residual RL wrapper for FetchReachDense-v3.
    
    Architecture:
    - PD controller provides baseline: u_pd = Kp * e + Kd * ė
    - SAC policy learns residual: u_rl
    - Total action: u_total = u_pd + α * u_rl
    
    Task: Move gripper to random goal position (standard FetchReach)
    """

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int | None = 50,
        # PD controller parameters
        pd_kp: float = 40.0,
        pd_kd: float = 10.0,
        action_scale: float = 5.0,
        # Residual parameters
        residual_alpha: float = 0.5,
        # Reward parameters
        w_smooth: float = 0.01,
    ):
        gym.register_envs(gymnasium_robotics)
        
        env = gym.make(
            "FetchReachDense-v4",
            render_mode=render_mode,
            max_episode_steps=max_episode_steps or 50,
        )
        super().__init__(env)
        
        # PD controller gains
        self.pd_kp = float(pd_kp)
        self.pd_kd = float(pd_kd)
        self.action_scale = float(action_scale)
        
        # Residual scaling
        self.residual_alpha = float(residual_alpha)
        
        # Reward parameters
        self.w_smooth = float(w_smooth)
        
        # State tracking
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self._prev_achieved_goal = np.zeros(3, dtype=np.float64)
        self._step_count = 0
        self._tracking_errors = []
        
        # Augment observation space with tracking error and previous action
        orig_spaces = dict(self.observation_space.spaces)
        orig_spaces["tracking_error"] = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        orig_spaces["prev_action"] = spaces.Box(
            self.action_space.low, self.action_space.high, dtype=np.float32
        )
        self.observation_space = spaces.Dict(orig_spaces)

    def _compute_pd_control(
        self,
        target_pos: np.ndarray,
        achieved_pos: np.ndarray,
        target_vel: np.ndarray,
        achieved_vel: np.ndarray,
    ) -> np.ndarray:
        """Compute PD operational-space control."""
        error = target_pos - achieved_pos
        error_vel = target_vel - achieved_vel
        
        u_pd = self.pd_kp * error + self.pd_kd * error_vel
        u_pd = u_pd / self.action_scale
        return np.clip(u_pd, -1.0, 1.0)

    def _augment_obs(self, obs: dict) -> dict:
        """Add tracking error and previous action to observation."""
        out = dict(obs)
        
        # Tracking error
        tracking_error = obs["desired_goal"] - obs["achieved_goal"]
        out["tracking_error"] = tracking_error.astype(np.float32)
        
        # Previous action
        out["prev_action"] = self.previous_action.astype(np.float32)
        
        return out

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Reset state
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self._prev_achieved_goal = obs["achieved_goal"].copy()
        self._step_count = 0
        self._tracking_errors = []
        
        return self._augment_obs(obs), info

    def step(self, action: np.ndarray):
        a_rl = np.asarray(action, dtype=np.float64).reshape(self.action_space.shape)
        
        # Get current state
        # Note: We use the previous observation since we don't have current state yet
        # This is standard in RL - action is based on previous observation
        achieved_goal = self._prev_achieved_goal
        
        # For FetchReach, desired_goal is static during episode
        # We'll get it from the step, but for PD we use the stored goal
        # Velocity estimation (finite difference)
        if self._step_count > 0:
            dt = 0.04  # FetchReach control dt (25 Hz)
            ee_vel = (achieved_goal - self._prev_achieved_goal) / dt
        else:
            ee_vel = np.zeros(3, dtype=np.float64)
        
        # Target velocity is zero (static goal)
        target_vel = np.zeros(3, dtype=np.float64)
        
        # We need the target position - get it from the environment's current goal
        target_position = self.env.unwrapped.goal.copy()
        
        # Compute PD control
        u_pd = self._compute_pd_control(target_position, achieved_goal, target_vel, ee_vel)
        
        # Action-space residual: u_total = u_pd + α * u_rl
        u_rl = np.tanh(a_rl[:3])
        u_total = u_pd + self.residual_alpha * u_rl
        u_total = np.clip(u_total, -1.0, 1.0)
        
        # Gripper action (unused in FetchReach but required)
        u_total_gripper = a_rl[3:]
        a_cmd = np.concatenate([u_total, u_total_gripper])
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(a_cmd)
        
        # Compute tracking error
        achieved_goal = obs["achieved_goal"]
        err_vec = achieved_goal - target_position
        dist = float(np.linalg.norm(err_vec))
        self._tracking_errors.append(dist)
        
        # Modified reward: dense tracking + smoothness penalty
        tracking_reward = -dist  # Negative distance (dense reward)
        smoothness_penalty = -self.w_smooth * np.linalg.norm(a_cmd - self.previous_action)
        reward = tracking_reward + smoothness_penalty
        
        # Update state
        self.previous_action = a_cmd.copy()
        self._prev_achieved_goal = achieved_goal.copy()
        self._step_count += 1
        
        # Debug prints
        if self._step_count % 100 == 0:
            print(f"\n=== CONTROL DEBUG (step {self._step_count}) ===")
            print(f"Target: {target_position}")
            print(f"Achieved: {achieved_goal}")
            print(f"Error: {dist:.4f}m")
            print(f"PD mean abs: {float(np.mean(np.abs(u_pd[:3]))):.4f}")
            print(f"RL mean abs: {float(np.mean(np.abs(u_rl))):.4f}")
            print(f"RL contribution: {float(np.mean(np.abs(self.residual_alpha * u_rl))):.4f}")
        
        # Enhanced info
        info["tracking_error"] = dist
        info["pd_control"] = u_pd.copy()
        info["residual_alpha"] = self.residual_alpha
        info["pd_mean_abs"] = float(np.mean(np.abs(u_pd)))
        info["rl_contribution_mean_abs"] = float(np.mean(np.abs(self.residual_alpha * u_rl)))
        
        if self._tracking_errors:
            info["mean_tracking_error"] = float(np.mean(self._tracking_errors))
            info["std_tracking_error"] = float(np.std(self._tracking_errors))
            info["max_tracking_error"] = float(np.max(self._tracking_errors))
        
        # Success uses FetchReach's standard 5cm threshold
        info["is_success"] = float(dist < 0.05)
        
        return self._augment_obs(obs), reward, terminated, truncated, info

    def set_residual_alpha(self, alpha: float) -> None:
        """Set residual scaling factor α."""
        self.residual_alpha = float(alpha)

    def set_pd_gains(self, kp: float, kd: float) -> None:
        """Set PD controller gains."""
        self.pd_kp = float(kp)
        self.pd_kd = float(kd)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Expose the underlying environment's compute_reward for SB3 compatibility."""
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)
