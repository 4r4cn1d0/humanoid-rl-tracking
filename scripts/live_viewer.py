#!/usr/bin/env python3
"""
Live viewer — watches the checkpoints/ directory for new SAC checkpoints
and renders the latest policy in the MuJoCo viewer.

Usage (in a separate terminal while training is running):
    python scripts/live_viewer.py

Controls:
    Ctrl+C  — exit
    The viewer auto-refreshes every time a new checkpoint is saved (~every 5k steps).
"""
from __future__ import annotations

import sys
import time
import glob
import os
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import SAC
from envs.residual_tracking_env import ResidualTrackingEnv
import yaml


def load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def build_env(cfg: dict) -> ResidualTrackingEnv:
    env_cfg = cfg.get("env") or {}
    reward = cfg.get("reward") or {}
    rob = cfg.get("robustness") or {}
    residual = cfg.get("residual") or {}
    traj_kw = dict(env_cfg.get("trajectory_kwargs") or {})
    max_eps = env_cfg.get("max_episode_steps")

    return ResidualTrackingEnv(
        render_mode="human",
        trajectory=env_cfg.get("trajectory", "circle"),
        trajectory_kwargs=traj_kw,
        control_dt=float(env_cfg.get("control_dt", 0.03)),
        max_episode_steps=int(max_eps) if max_eps is not None else 400,
        w_track=float(reward.get("w_track", 1.0)),
        w_smooth=float(reward.get("w_smooth", 0.01)),
        pd_kp=float(residual.get("pd_kp", 40.0)),
        pd_kd=float(residual.get("pd_kd", 10.0)),
        residual_alpha=float(residual.get("residual_alpha", 0.5)),
        action_repeat=int(residual.get("action_repeat", 1)),
        future_horizon=int(residual.get("future_horizon", 0)),
    )


def latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Return the most recently modified checkpoint zip."""
    files = glob.glob(os.path.join(checkpoint_dir, "sac_residual_*.zip"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def run_episode(model: SAC, env: ResidualTrackingEnv) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    errors = []
    steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        errors.append(info.get("tracking_error", 0.0))
        steps += 1
        env.render()

        if terminated or truncated:
            break

    return {
        "steps": steps,
        "mean_error": float(np.mean(errors)),
        "min_error": float(np.min(errors)),
        "total_reward": total_reward,
        "success_rate": float(np.mean([e < 0.10 for e in errors])),
    }


def main() -> None:
    cfg_path = _ROOT / "experiments" / "sac_default.yaml"
    checkpoint_dir = str(_ROOT / "checkpoints")

    print("=== SAC Live Viewer ===")
    print(f"Watching: {checkpoint_dir}")
    print("Waiting for first checkpoint (saved every 5k training steps)...")
    print("Press Ctrl+C to exit.\n")

    cfg = load_cfg(cfg_path)
    env = build_env(cfg)

    last_checkpoint = None
    model = None

    try:
        while True:
            ckpt = latest_checkpoint(checkpoint_dir)

            if ckpt is None:
                print("No checkpoint yet — waiting...", end="\r")
                time.sleep(5)
                continue

            if ckpt != last_checkpoint:
                print(f"\nLoading checkpoint: {os.path.basename(ckpt)}")
                try:
                    model = SAC.load(ckpt, env=None, device="cpu")
                    last_checkpoint = ckpt
                    # Extract timestep from filename
                    try:
                        ts = int(os.path.basename(ckpt).split("_")[-2])
                        print(f"  Timestep: {ts:,}")
                    except (ValueError, IndexError):
                        pass
                except Exception as e:
                    print(f"  Failed to load: {e}")
                    time.sleep(3)
                    continue

            if model is not None:
                stats = run_episode(model, env)
                print(
                    f"  Episode: mean_error={stats['mean_error']*100:.1f}cm  "
                    f"min_error={stats['min_error']*100:.1f}cm  "
                    f"success={stats['success_rate']*100:.0f}%  "
                    f"steps={stats['steps']}"
                )

                # Check for newer checkpoint between episodes
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nExiting viewer.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
