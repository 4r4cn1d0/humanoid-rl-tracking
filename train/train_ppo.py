#!/usr/bin/env python3
"""
Train PPO on TrajectoryTrackingEnv with optional YAML config, curriculum, TensorBoard,
vectorized envs, and Apple Silicon (MPS) / CUDA device selection.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.tracking_env import TrajectoryTrackingEnv
from train.callbacks import CurriculumCallback, RewardLoggingCallback


def load_yaml(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def resolve_device(name: str) -> str:
    n = (name or "auto").lower()
    if n == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return name


def build_env(cfg: dict, render_mode: str | None = None) -> TrajectoryTrackingEnv:
    env_cfg = cfg.get("env") or {}
    reward = cfg.get("reward") or {}
    rob = cfg.get("robustness") or {}
    traj = env_cfg.get("trajectory", "circle")
    traj_kw = dict(env_cfg.get("trajectory_kwargs") or {})
    max_eps = env_cfg.get("max_episode_steps")
    return TrajectoryTrackingEnv(
        render_mode=render_mode,
        trajectory=traj,
        trajectory_kwargs=traj_kw,
        control_dt=float(env_cfg.get("control_dt", 0.03)),
        max_episode_steps=int(max_eps) if max_eps is not None else None,
        w_track=float(reward.get("w_track", 1.0)),
        w_smooth=float(reward.get("w_smooth", 0.1)),
        w_velocity=float(reward.get("w_velocity", 0.0)),
        w_orient=float(reward.get("w_orient", 0.0)),
        track_orientation=bool(env_cfg.get("track_orientation", False)),
        use_squared_error=bool(reward.get("use_squared_error", False)),
        obs_noise_std=float(rob.get("obs_noise_std", 0.0)),
        action_noise_std=float(rob.get("action_noise_std", 0.0)),
        action_delay=int(rob.get("action_delay", 0)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(_ROOT / "experiments" / "default.yaml"),
        help="YAML experiment config",
    )
    p.add_argument("--skip-check-env", action="store_true")
    p.add_argument("--seed", type=int, default=-1, help="Override config seed if >= 0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    train_cfg = cfg.get("train") or {}
    seed = int(args.seed) if args.seed >= 0 else int(train_cfg.get("seed", 0))
    set_random_seed(seed, using_cuda=torch.cuda.is_available())
    random.seed(seed)
    np.random.seed(seed)

    n_envs = int(train_cfg.get("n_envs", 1))
    device = resolve_device(str(train_cfg.get("device", "auto")))

    def _make() -> TrajectoryTrackingEnv:
        return build_env(cfg, render_mode=None)

    if n_envs <= 1:
        env = _make()
        if not args.skip_check_env:
            check_env(env, warn=True)
        train_env = env
    else:
        train_env = make_vec_env(_make, n_envs=n_envs, seed=seed)
        if not args.skip_check_env:
            w0 = train_env.envs[0]
            while hasattr(w0, "env") and not isinstance(w0, TrajectoryTrackingEnv):
                w0 = w0.env
            check_env(w0, warn=True)

    ppo_cfg = train_cfg.get("ppo") or {}
    tensorboard_log = str(train_cfg.get("tensorboard_log", "./ppo_tracking_tensorboard/"))
    os.makedirs(tensorboard_log, exist_ok=True)

    model = PPO(
        policy=str(ppo_cfg.get("policy", "MultiInputPolicy")),
        env=train_env,
        device=device,
        verbose=int(train_cfg.get("verbose", 1)),
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        tensorboard_log=tensorboard_log,
    )

    callbacks = [RewardLoggingCallback(verbose=0)]
    cur = cfg.get("curriculum") or {}
    if bool(cur.get("enabled", False)):
        stages = cur.get("stages") or []
        if stages:
            callbacks.append(CurriculumCallback(stages, verbose=int(train_cfg.get("verbose", 0))))

    total_timesteps = int(train_cfg.get("total_timesteps", 100_000))
    print(f"Training on device={device!r}, n_envs={n_envs}, total_timesteps={total_timesteps}")
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))

    out_model = str(train_cfg.get("model_save_path", "ppo_tracking_model"))
    model.save(out_model)
    train_env.close()
    print(f"Saved model to {out_model}.zip")


if __name__ == "__main__":
    main()
