#!/usr/bin/env python3
"""
Pure SAC trajectory tracking — no PD controller.
Policy directly outputs mocap position deltas.
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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.pure_sac_env import PureSACEnv
from train.callbacks import (
    RewardLoggingCallback,
    PerformanceGatedCurriculumCallback,
    RenderEvalCallback,
)


def load_yaml(path: Path) -> dict:
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


def build_env(cfg: dict, render_mode: str | None = None) -> PureSACEnv:
    env_cfg = cfg.get("env") or {}
    reward = cfg.get("reward") or {}
    traj_kw = dict(env_cfg.get("trajectory_kwargs") or {})

    return PureSACEnv(
        render_mode=render_mode,
        trajectory=str(env_cfg.get("trajectory", "circle")),
        trajectory_kwargs=traj_kw,
        control_dt=float(env_cfg.get("control_dt", 0.05)),
        max_episode_steps=int(env_cfg.get("max_episode_steps", 200)),
        w_track=float(reward.get("w_track", 1.0)),
        w_smooth=float(reward.get("w_smooth", 0.05)),
        w_velocity=float(reward.get("w_velocity", 0.0)),
        obs_noise_std=0.0,       # curriculum callback sets this
        action_noise_std=0.0,    # curriculum callback sets this
        action_delay=0,          # curriculum callback sets this
        success_threshold=float(cfg.get("success_threshold", 0.015)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pure SAC trajectory tracking")
    p.add_argument(
        "--config",
        default=str(_ROOT / "experiments" / "pure_sac.yaml"),
        help="YAML experiment config",
    )
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--skip-check-env", action="store_true")
    p.add_argument(
        "--render-freq",
        type=int,
        default=10,
        help="Show live render every N training episodes (0 = off)",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., checkpoints/pure_sac_100000_steps.zip)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config))

    train_cfg = cfg.get("train") or {}
    seed = int(args.seed) if args.seed >= 0 else int(train_cfg.get("seed", 0))
    set_random_seed(seed, using_cuda=torch.cuda.is_available())
    random.seed(seed)
    np.random.seed(seed)

    device = resolve_device(str(train_cfg.get("device", "auto")))

    # --- Build training env ---
    env = build_env(cfg, render_mode=None)
    if not args.skip_check_env:
        check_env(env, warn=True)
    train_env = DummyVecEnv([lambda: env])

    # --- SAC model ---
    sac_cfg = train_cfg.get("sac") or {}
    tensorboard_log = str(train_cfg.get("tensorboard_log", "./tensorboard/"))
    os.makedirs(tensorboard_log, exist_ok=True)

    ent_coef = sac_cfg.get("ent_coef", "auto")
    target_entropy = sac_cfg.get("target_entropy", "auto")
    if ent_coef != "auto":
        ent_coef = float(ent_coef)
    if target_entropy != "auto":
        target_entropy = float(target_entropy)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        model = SAC.load(
            args.resume,
            env=train_env,
            device=device,
            tensorboard_log=tensorboard_log,
        )
        # Update target_entropy if changed in config
        if target_entropy != "auto":
            model.target_entropy = target_entropy
            print(f"Updated target_entropy to {target_entropy}")
    else:
        model = SAC(
            policy="MultiInputPolicy",
            env=train_env,
            device=device,
            verbose=int(train_cfg.get("verbose", 1)),
            learning_rate=float(sac_cfg.get("learning_rate", 3e-4)),
            buffer_size=int(sac_cfg.get("buffer_size", 300_000)),
            batch_size=int(sac_cfg.get("batch_size", 256)),
            gamma=float(sac_cfg.get("gamma", 0.99)),
            tau=float(sac_cfg.get("tau", 0.005)),
            gradient_steps=int(sac_cfg.get("gradient_steps", 1)),
            train_freq=int(sac_cfg.get("train_freq", 1)),
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], qf=[256, 256]),
                activation_fn=torch.nn.ReLU,
            ),
            tensorboard_log=tensorboard_log,
        )

    # --- Callbacks ---
    callbacks = [RewardLoggingCallback(verbose=0)]

    # Checkpoints
    checkpoint_dir = str(train_cfg.get("checkpoint_dir", "./checkpoints/"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks.append(CheckpointCallback(
        save_freq=5_000,
        save_path=checkpoint_dir,
        name_prefix="pure_sac",
        verbose=1,
    ))

    # Performance-gated curriculum
    cur = cfg.get("curriculum") or {}
    if cur.get("enabled", False):
        stages = cur.get("stages") or []
        if stages:
            callbacks.append(
                PerformanceGatedCurriculumCallback(stages, verbose=1)
            )

    # Live render
    if args.render_freq > 0:
        callbacks.append(RenderEvalCallback(
            env_factory=lambda render_mode=None: build_env(cfg, render_mode=render_mode),
            render_freq=args.render_freq,
            verbose=1,
        ))

    # --- Train ---
    total_timesteps = int(train_cfg.get("total_timesteps", 1_000_000))
    print(f"\nPure SAC | device={device} | timesteps={total_timesteps:,}")
    print(f"Trajectory: {cfg['env']['trajectory']} | "
          f"radius={cfg['env']['trajectory_kwargs'].get('radius', '?')}m | "
          f"speed={cfg['env']['trajectory_kwargs'].get('speed', '?')}")
    print(f"Success threshold: {cfg.get('success_threshold', 0.015)*100:.1f} cm\n")

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))

    out = str(train_cfg.get("model_save_path", "pure_sac_model"))
    model.save(out)
    train_env.close()
    print(f"\nSaved model → {out}.zip")


if __name__ == "__main__":
    main()
