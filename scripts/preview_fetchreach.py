#!/usr/bin/env python3
"""
Preview the standard FetchReachDense-v3 task with PD controller only.
Each episode, a random goal is sampled and the PD controller attempts to reach it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.residual_fetchreach_env import ResidualFetchReachEnv


def main() -> None:
    print("=== FetchReachDense-v3 Preview (PD only) ===")
    print("Standard FetchReach task:")
    print("  - Random goal each episode")
    print("  - Gripper starts at stow position")
    print("  - Table: pos=[0.95, 0.75, 0.2], size=[0.18, 0.25, 0.15]")
    print("  - PD controller: Kp=40, Kd=10")
    print("  - Residual alpha=0 (PD only, no RL)")
    print("\nClose the MuJoCo window or press Ctrl+C to exit.\n")

    env = ResidualFetchReachEnv(
        render_mode="human",
        max_episode_steps=50,
        pd_kp=40.0,
        pd_kd=10.0,
        residual_alpha=0.0,  # PD only
    )

    try:
        episode = 0
        while True:
            obs, _ = env.reset(seed=episode)
            print(f"\n=== Episode {episode} ===")
            print(f"Gripper start: {obs['achieved_goal']}")
            print(f"Goal position: {obs['desired_goal']}")
            initial_dist = np.linalg.norm(obs['tracking_error'])
            print(f"Initial distance: {initial_dist:.3f}m")

            for step in range(50):
                # Zero RL action — just PD tracking
                obs, reward, terminated, truncated, info = env.step(np.zeros(4))
                env.render()
                time.sleep(0.02)

                if step % 10 == 0:
                    print(f"  step {step:2d}: dist={info['tracking_error']*100:.1f}cm  "
                          f"pd_abs={info['pd_mean_abs']:.3f}  "
                          f"success={int(info['is_success'])}")

                if terminated or truncated:
                    break

            final_dist = info['tracking_error']
            success = info['is_success']
            print(f"Final distance: {final_dist*100:.1f}cm  Success: {success}")
            
            episode += 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nExiting preview.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
