from __future__ import annotations

import types
from collections import deque
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium import spaces
from gymnasium_robotics.utils import rotations

from utils.orientation_utils import normalize_quat, quat_geodesic_distance
from utils.trajectories import Trajectory, make_trajectory

# Default FetchReach mocap quaternion (w, x, y, z) from Gymnasium-Robotics docs / env.
_DEFAULT_FETCH_QUAT = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)


def _observation_space_float32(sp: spaces.Space) -> spaces.Space:
    """MPS/CUDA policies need float32 observations; Gymnasium Fetch uses float64."""
    if isinstance(sp, spaces.Dict):
        return spaces.Dict({k: _observation_space_float32(v) for k, v in sp.spaces.items()})
    if isinstance(sp, spaces.Box):
        low = np.asarray(sp.low, dtype=np.float32)
        high = np.asarray(sp.high, dtype=np.float32)
        return spaces.Box(low, high, dtype=np.float32)
    return sp


def _obs_to_float32(obs: dict[str, Any]) -> dict[str, Any]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}


class TrajectoryTrackingEnv(gym.Env):
    """
    FetchReachDense with a moving goal and custom tracking reward.
    Optional end-effector orientation via mocap quaternion override (FetchReach
    normally fixes orientation in its low-level controller).
    Optional observation noise, action noise, and action delay.
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
        track_orientation: bool = False,
        use_squared_error: bool = False,
        obs_noise_std: float = 0.0,
        obs_noise_keys: tuple[str, ...] | None = None,
        action_noise_std: float = 0.0,
        action_delay: int = 0,
        rng: np.random.Generator | None = None,
    ):
        super().__init__()
        gym.register_envs(gymnasium_robotics)

        make_kw: dict[str, Any] = {}
        if max_episode_steps is not None:
            make_kw["max_episode_steps"] = int(max_episode_steps)

        self.env = gym.make(
            "FetchReachDense-v4",
            render_mode=render_mode,
            **make_kw,
        )

        traj_kw = dict(trajectory_kwargs or {})
        if isinstance(trajectory, str):
            seed = traj_kw.pop("seed", None)
            self.trajectory = make_trajectory(trajectory, seed=seed, **traj_kw)
        else:
            self.trajectory = trajectory

        self.control_dt = float(control_dt)
        self.w_track = float(w_track)
        self.w_smooth = float(w_smooth)
        self.w_velocity = float(w_velocity)
        self.w_orient = float(w_orient)
        self.track_orientation = bool(track_orientation)
        self.use_squared_error = bool(use_squared_error)

        self.obs_noise_std = float(obs_noise_std)
        if obs_noise_keys is None:
            self.obs_noise_keys = (
                "achieved_goal",
                "desired_goal",
                "observation",
                "desired_orientation",
                "achieved_orientation",
            )
        else:
            self.obs_noise_keys = obs_noise_keys
        self.action_noise_std = float(action_noise_std)
        self.action_delay = int(max(0, action_delay))

        self._internal_rng = rng or np.random.default_rng()
        self.t = 0.0
        self.previous_action = np.zeros(self.env.action_space.shape, dtype=np.float64)
        self._action_history: deque[np.ndarray] = deque()
        self._init_action_buffer()

        self._target_quat_current = normalize_quat(_DEFAULT_FETCH_QUAT.copy())
        self._inner_unwrapped: Any = self.env.unwrapped
        self._orig_set_action: Any = None

        if self.track_orientation:
            if not hasattr(self.trajectory, "orientation"):
                raise ValueError("Trajectory must implement orientation(t) when track_orientation=True")
            self._install_orientation_patch()
            orig_spaces = dict(self.env.observation_space.spaces)
            orig_spaces["desired_orientation"] = spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float64
            )
            orig_spaces["achieved_orientation"] = spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float64
            )
            self.observation_space = spaces.Dict(orig_spaces)
        else:
            self.observation_space = self.env.observation_space

        self.observation_space = _observation_space_float32(self.observation_space)
        self.action_space = self.env.action_space

    def _install_orientation_patch(self) -> None:
        inner = self._inner_unwrapped
        if getattr(inner, "_fetch_orientation_patch_installed", False):
            return
        outer = self

        def _set_action_with_orientation(inner_self, action: np.ndarray) -> None:
            action = np.asarray(action, dtype=np.float64).copy()
            assert action.shape == (4,)
            pos_ctrl, gripper_ctrl = action[:3], action[3]
            pos_ctrl = pos_ctrl * 0.05
            rot_ctrl = normalize_quat(outer._target_quat_current)
            gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl], dtype=np.float64)
            if inner_self.block_gripper:
                gripper_ctrl = np.zeros_like(gripper_ctrl)
            full = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
            inner_self._utils.ctrl_set_action(inner_self.model, inner_self.data, full)
            inner_self._utils.mocap_set_action(inner_self.model, inner_self.data, full)

        self._orig_set_action = inner._set_action
        inner._set_action = types.MethodType(_set_action_with_orientation, inner)
        inner._fetch_orientation_patch_installed = True

    def _read_grip_quaternion(self) -> np.ndarray:
        inner = self._inner_unwrapped
        mat = inner._utils.get_site_xmat(inner.model, inner.data, "robot0:grip")
        m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
        return rotations.mat2quat(m)

    # --- Curriculum / robustness setters (called from callbacks) ---
    def set_trajectory_speed(self, speed: float) -> None:
        if hasattr(self.trajectory, "set_speed"):
            self.trajectory.set_speed(speed)  # type: ignore[attr-defined]

    def set_obs_noise_std(self, std: float) -> None:
        self.obs_noise_std = float(std)

    def set_action_noise_std(self, std: float) -> None:
        self.action_noise_std = float(std)

    def set_action_delay(self, delay: int) -> None:
        self.action_delay = int(max(0, delay))
        self._init_action_buffer()

    def set_control_dt(self, dt: float) -> None:
        self.control_dt = float(dt)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._internal_rng = np.random.default_rng(seed)

        obs, info = self.env.reset(seed=seed, options=options)
        self.t = 0.0
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self._init_action_buffer()

        target = self.trajectory.position(self.t)
        self.env.unwrapped.goal = target
        if self.track_orientation:
            self._target_quat_current = normalize_quat(
                np.asarray(self.trajectory.orientation(self.t), dtype=np.float64).reshape(4)
            )
        obs = self._augment_obs_dict(obs, target)

        return _obs_to_float32(self._maybe_noise_obs(obs)), info

    def _augment_obs_dict(self, obs: dict, target: np.ndarray) -> dict:
        """Align desired_goal with moving target; optionally add orientation goals."""
        out = {k: (np.array(v, copy=True) if hasattr(v, "copy") else v) for k, v in obs.items()}
        out["desired_goal"] = np.array(target, dtype=np.float64, copy=True)
        if self.track_orientation:
            q_des = normalize_quat(
                np.asarray(self.trajectory.orientation(self.t), dtype=np.float64).reshape(4)
            )
            q_ach = normalize_quat(self._read_grip_quaternion())
            out["desired_orientation"] = q_des.copy()
            out["achieved_orientation"] = q_ach.copy()
        return out

    def _maybe_noise_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self.obs_noise_std <= 0.0:
            return obs
        noisy = {k: np.array(v, copy=True) for k, v in obs.items()}
        for key in self.obs_noise_keys:
            if key in noisy and isinstance(noisy[key], np.ndarray):
                noisy[key] = noisy[key] + self._internal_rng.normal(
                    0.0, self.obs_noise_std, size=noisy[key].shape
                ).astype(noisy[key].dtype, copy=False)
        return noisy

    def _init_action_buffer(self) -> None:
        maxlen = self.action_delay + 1 if self.action_delay > 0 else 1
        self._action_history = deque(maxlen=maxlen)

    def _policy_action_to_torque_command(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(self.action_space.shape)
        if self.action_noise_std > 0.0:
            a = a + self._internal_rng.normal(0.0, self.action_noise_std, size=a.shape)
            a = np.clip(a, self.action_space.low, self.action_space.high)
        if self.action_delay <= 0:
            return a
        self._action_history.append(a.copy())
        if len(self._action_history) < self.action_delay + 1:
            return np.zeros_like(a)
        return np.array(self._action_history[0], dtype=np.float64, copy=True)

    def step(self, action: np.ndarray):
        target_position = self.trajectory.position(self.t)
        self.env.unwrapped.goal = target_position
        if self.track_orientation:
            self._target_quat_current = normalize_quat(
                np.asarray(self.trajectory.orientation(self.t), dtype=np.float64).reshape(4)
            )

        a_cmd = np.asarray(action, dtype=np.float64).reshape(self.action_space.shape)
        a_apply = self._policy_action_to_torque_command(a_cmd)

        obs, _base_reward, terminated, truncated, info = self.env.step(a_apply)

        obs = self._augment_obs_dict(obs, target_position)
        achieved_goal = obs["achieved_goal"]
        err_vec = achieved_goal - target_position
        dist = float(np.linalg.norm(err_vec))

        if self.use_squared_error:
            tracking_term = self.w_track * (dist**2)
        else:
            tracking_term = self.w_track * dist
        tracking_reward = -tracking_term

        smoothness_penalty = -self.w_smooth * float(
            np.linalg.norm(a_cmd - self.previous_action)
        )

        vel_penalty = 0.0
        qvel_norm = 0.0
        obs_arr = obs.get("observation")
        if isinstance(obs_arr, np.ndarray) and obs_arr.size >= 10:
            qvel = obs_arr[5:10]
            qvel_norm = float(np.linalg.norm(qvel))
            if self.w_velocity > 0.0:
                vel_penalty = -self.w_velocity * qvel_norm

        orient_penalty = 0.0
        orient_err = 0.0
        if self.track_orientation:
            q_des = normalize_quat(
                np.asarray(self.trajectory.orientation(self.t), dtype=np.float64).reshape(4)
            )
            q_ach = normalize_quat(self._read_grip_quaternion())
            orient_err = quat_geodesic_distance(q_des, q_ach)
            orient_penalty = -self.w_orient * float(orient_err)

        reward = tracking_reward + smoothness_penalty + vel_penalty + orient_penalty

        self.previous_action = a_cmd.copy()
        self.t += self.control_dt

        info = dict(info)
        info["tracking_error"] = dist
        info["target_position"] = target_position.copy()
        info["reward_tracking"] = tracking_reward
        info["reward_smoothness"] = smoothness_penalty
        info["reward_velocity"] = vel_penalty
        info["qvel_norm"] = qvel_norm
        info["applied_action"] = a_apply.copy()
        info["orientation_error"] = float(orient_err)
        info["reward_orientation"] = float(orient_penalty)

        return _obs_to_float32(self._maybe_noise_obs(obs)), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        inner = getattr(self, "_inner_unwrapped", None)
        if inner is not None and getattr(inner, "_fetch_orientation_patch_installed", False):
            if self._orig_set_action is not None:
                inner._set_action = self._orig_set_action
            inner._fetch_orientation_patch_installed = False
        self.env.close()
