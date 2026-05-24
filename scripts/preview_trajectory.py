#!/usr/bin/env python3
"""
Preview the trajectory in the MuJoCo viewer without any policy.
The red ball traces the circle trajectory so you can verify
the center, radius, and reachability before training.

Usage:
    python scripts/preview_trajectory.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.residual_tracking_env import ResidualTrackingEnv

def main() -> None:
    print("=== Trajectory Preview ===")
    print("FetchReach environment with modified table:")
    print("  Table: pos=[0.95, 0.75, 0.2], size=[0.18, 0.25, 0.15] (closer to robot)")
    print("  Trajectory center: [1.34, 0.75, 0.53]")
    print("  Radius: 0.08m  Speed: 0.6 rad/s")
    print("Close the MuJoCo window or press Ctrl+C to exit.\n")

    env = ResidualTrackingEnv(
        render_mode="human",
        trajectory="circle",
        trajectory_kwargs={
            "center": [1.34, 0.75, 0.53],
            "radius": 0.08,
            "speed": 0.6,
        },
        pd_kp=40.0,
        pd_kd=10.0,
        residual_alpha=0.0,
        max_episode_steps=2000,
        future_horizon=0,
    )

    try:
        while True:
            obs, _ = env.reset(seed=0)
            print(f"Arm reset position: {obs['achieved_goal'].round(3)}")
            print(f"First target:       {obs['desired_goal'].round(3)}")
            print(f"Initial distance:   {np.linalg.norm(obs['tracking_error']):.3f}m")
            print("Running episode (zero action — PD only)...")

            for step in range(2000):
                # Zero RL action — just PD tracking
                obs, _, terminated, truncated, info = env.step(np.zeros(4))
                env.render()
                time.sleep(0.01)  # Slow down so it's visible

                if step % 200 == 0:
                    print(f"  step {step:4d}: dist={info['tracking_error']*100:.1f}cm  "
                          f"pd_abs={info['pd_mean_abs']:.3f}")

                if terminated or truncated:
                    break

            print("Episode done. Restarting...\n")

    except KeyboardInterrupt:
        print("\nExiting preview.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
