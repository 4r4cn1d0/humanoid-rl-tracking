#!/usr/bin/env python3
"""
Scan checkpoints and find the best one using EXACT training parameters.
Evaluates step-level success @3cm and mean tracking error.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.pure_sac_env import PureSACEnv


# ── Exact training parameters ────────────────────────────────────────────────
CONTROL_DT       = 0.05
MAX_EPISODE_STEPS = 200
SUCCESS_THRESHOLD = 0.03   # 3 cm — matches pure_sac.yaml
TRAJ_KWARGS      = {"center": [1.34, 0.75, 0.53], "radius": 0.15, "speed": 0.6}
N_EPISODES       = 20      # enough for stable estimate, fast enough to scan
SEED             = 42

# Checkpoints to scan: every 25k from 100k to 700k
STEPS_TO_SCAN = list(range(100_000, 705_000, 25_000))


def make_env() -> PureSACEnv:
    return PureSACEnv(
        trajectory="circle",
        trajectory_kwargs=TRAJ_KWARGS,
        control_dt=CONTROL_DT,
        max_episode_steps=MAX_EPISODE_STEPS,
        w_track=1.0,
        w_smooth=0.0025,
        w_velocity=0.2,
        obs_noise_std=0.001,
        action_noise_std=0.0,
        action_delay=0,
        success_threshold=SUCCESS_THRESHOLD,
    )


def evaluate_checkpoint(ckpt_path: Path, n_episodes: int, seed: int) -> dict:
    env = make_env()
    vec_env = DummyVecEnv([lambda: env])
    try:
        model = SAC.load(str(ckpt_path), env=vec_env, device="cpu")
    except Exception as e:
        vec_env.close()
        return {"error": str(e)}

    all_step_errors: list[float] = []
    all_ep_mean_errors: list[float] = []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        ep_errors: list[float] = []
        for _ in range(MAX_EPISODE_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec_env.step(action)
            info = infos[0]
            if "tracking_error" in info:
                ep_errors.append(float(info["tracking_error"]))
            if dones[0]:
                break
        if ep_errors:
            all_step_errors.extend(ep_errors)
            all_ep_mean_errors.append(float(np.mean(ep_errors)))

    vec_env.close()

    if not all_step_errors:
        return {"error": "no data"}

    errs = np.array(all_step_errors)
    return {
        "mean_tracking_error_cm": float(np.mean(errs)) * 100,
        "rmse_cm": float(np.sqrt(np.mean(errs**2))) * 100,
        "step_success_3cm_pct": float(np.mean(errs < 0.03)) * 100,
        "step_success_5cm_pct": float(np.mean(errs < 0.05)) * 100,
        "ep_mean_error_cm": float(np.mean(all_ep_mean_errors)) * 100,
    }


def main() -> None:
    ckpt_dir = _ROOT / "checkpoints"
    results = []

    print(f"{'Steps':>10}  {'Mean err':>10}  {'RMSE':>8}  {'@3cm%':>8}  {'@5cm%':>8}")
    print("-" * 55)

    for steps in STEPS_TO_SCAN:
        ckpt = ckpt_dir / f"pure_sac_{steps}_steps.zip"
        if not ckpt.exists():
            continue
        r = evaluate_checkpoint(ckpt, N_EPISODES, SEED + steps)
        if "error" in r:
            print(f"{steps:>10}  ERROR: {r['error']}")
            continue
        results.append({"steps": steps, **r})
        print(f"{steps:>10}  {r['mean_tracking_error_cm']:>9.2f}cm"
              f"  {r['rmse_cm']:>7.2f}cm"
              f"  {r['step_success_3cm_pct']:>7.1f}%"
              f"  {r['step_success_5cm_pct']:>7.1f}%")

    if not results:
        print("No results.")
        return

    # Find best by step-level success @3cm, then by mean error as tiebreak
    best = max(results, key=lambda r: (r["step_success_3cm_pct"], -r["mean_tracking_error_cm"]))
    print(f"\n{'='*55}")
    print(f"BEST CHECKPOINT: pure_sac_{best['steps']}_steps.zip")
    print(f"  Mean error:    {best['mean_tracking_error_cm']:.2f} cm")
    print(f"  RMSE:          {best['rmse_cm']:.2f} cm")
    print(f"  Success @3cm:  {best['step_success_3cm_pct']:.1f}%")
    print(f"  Success @5cm:  {best['step_success_5cm_pct']:.1f}%")

    out = _ROOT / "outputs" / "checkpoint_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    main()
