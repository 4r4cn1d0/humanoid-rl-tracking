#!/usr/bin/env python3
"""
Train SAC on standard FetchReachDense-v3 with residual RL.
Task: Move gripper to random goal (standard benchmark).
Architecture: u_total = u_pd + α * u_rl
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
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.residual_fetchreach_env import ResidualFetchReachEnv
from train.callbacks import RewardLoggingCallback


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


def build_env(cfg: dict, render_mode: str | None = None) -> ResidualFetchReachEnv:
    env_cfg = cfg.get("env") or {}
    residual = cfg.get("residual") or {}
    
    return ResidualFetchReachEnv(
        render_mode=render_mode,
        max_episode_steps=int(env_cfg.get("max_episode_steps", 50)),
        pd_kp=float(residual.get("pd_kp", 40.0)),
        pd_kd=float(residual.get("pd_kd", 10.0)),
        action_scale=float(residual.get("action_scale", 5.0)),
        residual_alpha=float(residual.get("residual_alpha", 0.5)),
        w_smooth=float(residual.get("w_smooth", 0.01)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(_ROOT / "experiments" / "sac_fetchreach.yaml"),
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

    device = resolve_device(str(train_cfg.get("device", "auto")))

    env = build_env(cfg, render_mode=None)
    # Skip env check - we modify the reward function for residual RL
    # if not args.skip_check_env:
    #     check_env(env, warn=True)
    
    train_env = DummyVecEnv([lambda: env])

    sac_cfg = train_cfg.get("sac") or {}
    tensorboard_log = str(train_cfg.get("tensorboard_log", "./sac_fetchreach_tensorboard/"))
    os.makedirs(tensorboard_log, exist_ok=True)

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
        target_entropy=float(sac_cfg.get("target_entropy", -2.0)),
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

    # Save checkpoints for live viewing
    checkpoint_dir = str(train_cfg.get("checkpoint_dir", "./checkpoints_fetchreach/"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks.append(CheckpointCallback(
        save_freq=5000,
        save_path=checkpoint_dir,
        name_prefix="sac_fetchreach",
        verbose=0,
    ))

    total_timesteps = int(train_cfg.get("total_timesteps", 200_000))
    print(f"Training SAC on FetchReachDense-v3 with residual RL")
    print(f"Device: {device}, Total timesteps: {total_timesteps}")
    print(f"PD gains: Kp={env.pd_kp}, Kd={env.pd_kd}")
    print(f"Residual alpha: {env.residual_alpha}")
    
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))

    out_model = str(train_cfg.get("model_save_path", "sac_fetchreach_model"))
    model.save(out_model)
    train_env.close()
    print(f"Saved SAC model to {out_model}.zip")


if __name__ == "__main__":
    main()
