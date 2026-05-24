#!/usr/bin/env python3
"""
Train SAC on ResidualTrackingEnv with operational-space PD controller.
Implements residual RL: u_total = u_pd + α * u_rl
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
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.residual_tracking_env import ResidualTrackingEnv
from train.callbacks import CurriculumCallback, RewardLoggingCallback, RenderEvalCallback


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


def build_residual_env(cfg: dict, render_mode: str | None = None) -> ResidualTrackingEnv:
    env_cfg = cfg.get("env") or {}
    reward = cfg.get("reward") or {}
    rob = cfg.get("robustness") or {}
    residual = cfg.get("residual") or {}
    traj = env_cfg.get("trajectory", "circle")
    traj_kw = dict(env_cfg.get("trajectory_kwargs") or {})
    max_eps = env_cfg.get("max_episode_steps")
    
    return ResidualTrackingEnv(
        render_mode=render_mode,
        trajectory=traj,
        trajectory_kwargs=traj_kw,
        control_dt=float(env_cfg.get("control_dt", 0.03)),
        max_episode_steps=int(max_eps) if max_eps is not None else None,
        w_track=float(reward.get("w_track", 1.0)),
        w_smooth=float(reward.get("w_smooth", 0.1)),
        w_velocity=float(reward.get("w_velocity", 0.0)),
        w_orient=float(reward.get("w_orient", 0.0)),
        w_align=float(reward.get("w_align", 0.0)),
        track_orientation=bool(env_cfg.get("track_orientation", False)),
        use_squared_error=bool(reward.get("use_squared_error", False)),
        obs_noise_std=float(rob.get("obs_noise_std", 0.0)),
        action_noise_std=float(rob.get("action_noise_std", 0.0)),
        action_delay=int(rob.get("action_delay", 0)),
        # Residual-specific parameters
        pd_kp=float(residual.get("pd_kp", 25.0)),
        pd_kd=float(residual.get("pd_kd", 8.0)),
        residual_alpha=float(residual.get("residual_alpha", 0.1)),
        action_repeat=int(residual.get("action_repeat", 2)),
        velocity_max=float(residual.get("velocity_max", 0.5)),
        track_exp_k=float(residual.get("track_exp_k", 10.0)),
        vel_match_k=float(residual.get("vel_match_k", 5.0)),
        jerk_penalty_weight=float(residual.get("jerk_penalty_weight", 0.01)),
        future_horizon=int(residual.get("future_horizon", 3)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(_ROOT / "experiments" / "sac_default.yaml"),
        help="YAML experiment config",
    )
    p.add_argument("--skip-check-env", action="store_true")
    p.add_argument("--seed", type=int, default=-1, help="Override config seed if >= 0")
    p.add_argument(
        "--render-freq",
        type=int,
        default=10,
        help="Run a live render episode every N training episodes (0 = disabled)",
    )
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

    def _make() -> ResidualTrackingEnv:
        return build_residual_env(cfg, render_mode=None)

    if n_envs <= 1:
        env = _make()
        if not args.skip_check_env:
            check_env(env, warn=True)
        train_env = DummyVecEnv([lambda: env])
    else:
        train_env = make_vec_env(_make, n_envs=n_envs, seed=seed)
        if not args.skip_check_env:
            w0 = train_env.envs[0]
            while hasattr(w0, "env") and not isinstance(w0, ResidualTrackingEnv):
                w0 = w0.env
            check_env(w0, warn=True)

    sac_cfg = train_cfg.get("sac") or {}
    tensorboard_log = str(train_cfg.get("tensorboard_log", "./sac_residual_tensorboard/"))
    os.makedirs(tensorboard_log, exist_ok=True)

    # Use standard SAC with SiLU activation and custom network architecture
    ent_coef = sac_cfg.get("ent_coef", "auto")
    if ent_coef != "auto":
        ent_coef = float(ent_coef)
    
    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        device=device,
        verbose=int(train_cfg.get("verbose", 1)),
        learning_rate=float(sac_cfg.get("learning_rate", 3e-4)),
        buffer_size=int(sac_cfg.get("buffer_size", 1_000_000)),
        batch_size=int(sac_cfg.get("batch_size", 256)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        tau=float(sac_cfg.get("tau", 0.005)),
        gradient_steps=int(sac_cfg.get("gradient_steps", 1)),
        train_freq=int(sac_cfg.get("train_freq", 1)),
        ent_coef=ent_coef,
        target_entropy=-2.0,  # Higher entropy for better exploration across curriculum stages
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                qf=[256, 256],
            ),
            activation_fn=torch.nn.ReLU,
        ),
        tensorboard_log=tensorboard_log,
    )

    callbacks = [RewardLoggingCallback(verbose=0)]

    # Live render callback — opens a MuJoCo window every N episodes
    if args.render_freq > 0:
        def _make_render(render_mode=None):
            return build_residual_env(cfg, render_mode=render_mode)
        callbacks.append(RenderEvalCallback(
            env_factory=_make_render,
            render_freq=args.render_freq,
            verbose=1,
        ))

    # Save checkpoints every 5k steps so the live viewer can pick them up
    checkpoint_dir = str(train_cfg.get("checkpoint_dir", "./checkpoints/"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks.append(CheckpointCallback(
        save_freq=5000,
        save_path=checkpoint_dir,
        name_prefix="sac_residual",
        verbose=0,
    ))
    
    # Add residual scaling callback
    residual_cfg = cfg.get("residual") or {}
    if bool(residual_cfg.get("anneal_alpha", False)):
        from train.callbacks import ResidualScalingCallback
        alpha_schedule = residual_cfg.get("alpha_schedule", [0.1, 0.3, 0.5, 1.0])
        alpha_timesteps = residual_cfg.get("alpha_timesteps", [0, 50000, 100000, 200000])
        callbacks.append(
            ResidualScalingCallback(alpha_schedule, alpha_timesteps, verbose=int(train_cfg.get("verbose", 0)))
        )
    
    # Add performance-gated curriculum callback
    cur = cfg.get("curriculum") or {}
    if bool(cur.get("enabled", False)):
        stages = cur.get("stages") or []
        if stages:
            from train.callbacks import PerformanceGatedCurriculumCallback
            callbacks.append(PerformanceGatedCurriculumCallback(stages, verbose=int(train_cfg.get("verbose", 0))))

    total_timesteps = int(train_cfg.get("total_timesteps", 500_000))
    print(f"Training SAC on device={device!r}, n_envs={n_envs}, total_timesteps={total_timesteps}")
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))

    out_model = str(train_cfg.get("model_save_path", "sac_residual_model"))
    model.save(out_model)
    train_env.close()
    print(f"Saved SAC model to {out_model}.zip")


if __name__ == "__main__":
    main()
