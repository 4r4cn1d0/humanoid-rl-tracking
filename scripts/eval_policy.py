#!/usr/bin/env python3
"""
Load a trained SB3 policy and evaluate trajectory tracking with quantitative metrics,
saved timeseries, plots, and optional MP4 video (long rollouts across episode resets).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Project root on path when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs.tracking_env import TrajectoryTrackingEnv


def _try_import_imageio():
    try:
        import imageio.v2 as imageio

        return imageio
    except ImportError:
        return None


def build_tracking_env(args: argparse.Namespace, *, render_mode: str | None) -> TrajectoryTrackingEnv:
    max_eps = getattr(args, "max_episode_steps", None)
    return TrajectoryTrackingEnv(
        render_mode=render_mode,
        trajectory=args.trajectory,
        control_dt=args.control_dt,
        max_episode_steps=int(max_eps) if max_eps is not None else None,
        w_track=args.w_track,
        w_smooth=args.w_smooth,
        w_velocity=args.w_velocity,
        w_orient=args.w_orient,
        track_orientation=bool(args.track_orientation),
        use_squared_error=args.squared_error,
        obs_noise_std=args.obs_noise_std,
        action_noise_std=args.action_noise_std,
        action_delay=args.action_delay,
        rng=np.random.default_rng(args.seed),
    )


def rollout_episode(
    model: PPO,
    env: TrajectoryTrackingEnv,
    *,
    max_steps: int,
    deterministic: bool,
    reset_on_done: bool = True,
) -> dict[str, np.ndarray]:
    obs, _ = env.reset()
    ts: list[float] = []
    target: list[np.ndarray] = []
    ee: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    applied: list[np.ndarray] = []
    rewards: list[float] = []
    errors: list[float] = []
    qvel_norms: list[float] = []
    orient_err: list[float] = []

    t0 = 0.0

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float64).reshape(env.action_space.shape)

        obs, reward, terminated, truncated, info = env.step(action)

        tp = np.asarray(info["target_position"], dtype=np.float64).ravel()
        ag = np.asarray(obs["achieved_goal"], dtype=np.float64).ravel()
        err = float(np.linalg.norm(ag - tp))

        ts.append(t0)
        target.append(tp.copy())
        ee.append(ag.copy())
        actions.append(action.copy())
        applied.append(np.asarray(info.get("applied_action", action), dtype=np.float64).ravel())
        rewards.append(float(reward))
        errors.append(err)
        qvel_norms.append(float(info.get("qvel_norm", 0.0)))
        orient_err.append(float(info.get("orientation_error", 0.0)))

        t0 += env.control_dt

        if terminated or truncated:
            if reset_on_done and step < max_steps - 1:
                obs, _ = env.reset()
            else:
                break

    out: dict[str, np.ndarray] = {
        "time": np.asarray(ts, dtype=np.float64),
        "target": np.stack(target, axis=0) if target else np.zeros((0, 3)),
        "ee": np.stack(ee, axis=0) if ee else np.zeros((0, 3)),
        "action": np.stack(actions, axis=0) if actions else np.zeros((0, env.action_space.shape[0])),
        "applied_action": np.stack(applied, axis=0) if applied else np.zeros((0, env.action_space.shape[0])),
        "reward": np.asarray(rewards, dtype=np.float64),
        "error": np.asarray(errors, dtype=np.float64),
        "qvel_norm": np.asarray(qvel_norms, dtype=np.float64),
        "orientation_error": np.asarray(orient_err, dtype=np.float64),
    }
    return out


def episode_metrics(series: dict[str, np.ndarray]) -> dict[str, float]:
    err = series["error"]
    if err.size == 0:
        base = {
            "rmse": 0.0,
            "mean_error": 0.0,
            "max_error": 0.0,
            "mean_abs_delta_action": 0.0,
            "mean_qvel_norm": 0.0,
            "max_qvel_norm": 0.0,
            "sum_reward": 0.0,
            "steps": 0.0,
        }
        base.update(_orient_metrics(series))
        return base

    rmse = float(np.sqrt(np.mean(err**2)))
    a = series["action"]
    if len(a) >= 2:
        da = np.linalg.norm(np.diff(a, axis=0), axis=1)
        mean_da = float(np.mean(da))
    else:
        mean_da = 0.0
    qv = series["qvel_norm"]
    out = {
        "rmse": rmse,
        "mean_error": float(np.mean(err)),
        "max_error": float(np.max(err)),
        "mean_abs_delta_action": mean_da,
        "mean_qvel_norm": float(np.mean(qv)) if qv.size else 0.0,
        "max_qvel_norm": float(np.max(qv)) if qv.size else 0.0,
        "sum_reward": float(np.sum(series["reward"])),
        "steps": float(len(err)),
    }
    out.update(_orient_metrics(series))
    return out


def _orient_metrics(series: dict[str, np.ndarray]) -> dict[str, float]:
    o = series.get("orientation_error")
    if o is None or o.size == 0 or float(np.max(np.abs(o))) < 1e-12:
        return {
            "mean_orientation_error_rad": 0.0,
            "rmse_orientation_rad": 0.0,
            "max_orientation_error_rad": 0.0,
        }
    return {
        "mean_orientation_error_rad": float(np.mean(o)),
        "rmse_orientation_rad": float(np.sqrt(np.mean(o**2))),
        "max_orientation_error_rad": float(np.max(o)),
    }


def save_plots(series: dict[str, np.ndarray], out_dir: Path, prefix: str = "ep0") -> None:
    tgt = series["target"]
    ee = series["ee"]
    if tgt.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(tgt[:, 0], tgt[:, 1], label="Target XY", color="C0", linewidth=1.5)
        ax.plot(ee[:, 0], ee[:, 1], label="EE XY", color="C1", alpha=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.set_title("Desired vs achieved (XY projection)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_path_xy.png", dpi=150)
        plt.close(fig)

    t = series["time"]
    if t.size > 0:
        oerr = series.get("orientation_error")
        nrows = 3 if oerr is not None and oerr.size and float(np.max(np.abs(oerr))) > 1e-12 else 2
        fig, axes = plt.subplots(nrows, 1, figsize=(8, 3 + 2.5 * nrows), sharex=True)
        if nrows == 2:
            axes = [axes[0], axes[1]]
        axes[0].plot(t, series["error"], color="C2")
        axes[0].set_ylabel("pos err (m)")
        axes[0].set_title("Position tracking error")
        if len(series["action"]) >= 2:
            da = np.linalg.norm(np.diff(series["action"], axis=0), axis=1)
            axes[1].plot(t[1:], da, color="C3")
        axes[1].set_ylabel("||Δa||")
        axes[1].set_title("Action smoothness")
        if nrows == 3:
            axes[2].plot(t, oerr, color="C4")
            axes[2].set_ylabel("orient err (rad)")
            axes[2].set_title("Orientation geodesic error")
            axes[2].set_xlabel("time (s)")
        else:
            axes[1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_error_smoothness.png", dpi=150)
        plt.close(fig)


def record_video(
    model: PPO,
    env: TrajectoryTrackingEnv,
    path: Path,
    *,
    n_frames: int,
    deterministic: bool,
    fps: int,
) -> None:
    imageio = _try_import_imageio()
    if imageio is None:
        raise RuntimeError("imageio not installed; pip install imageio imageio-ffmpeg")

    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    for _ in range(n_frames):
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    if not frames:
        raise RuntimeError("No frames captured; use render_mode='rgb_array'")

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO trajectory tracking policy")
    p.add_argument("--model", type=str, default="ppo_tracking_model.zip")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=str, default="outputs/eval")
    p.add_argument("--trajectory", type=str, default="circle")
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of deterministic mean",
    )
    p.add_argument("--render", action="store_true", help="Human render (slow)")
    p.add_argument("--record-mp4", type=str, default="", help="Output path for mp4 (rgb_array)")
    p.add_argument("--video-frames", type=int, default=2500, help="Number of frames for MP4 (longer video)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--control-dt", type=float, default=0.03)
    p.add_argument("--max-episode-steps", type=int, default=0, help="0 = gym default (50); use 400 for long segments")
    p.add_argument("--track-orientation", action="store_true", help="Must match training if policy used orientation")
    p.add_argument("--w-orient", type=float, default=0.22)
    p.add_argument("--obs-noise-std", type=float, default=0.0)
    p.add_argument("--action-noise-std", type=float, default=0.0)
    p.add_argument("--action-delay", type=int, default=0)
    p.add_argument("--w-track", type=float, default=1.0)
    p.add_argument("--w-smooth", type=float, default=0.1)
    p.add_argument("--w-velocity", type=float, default=0.0)
    p.add_argument("--squared-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    deterministic = not bool(args.stochastic)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    max_eps = None if args.max_episode_steps <= 0 else int(args.max_episode_steps)
    args.max_episode_steps = max_eps  # for build helper

    render_mode = "human" if args.render else ("rgb_array" if args.record_mp4 else None)
    env = build_tracking_env(args, render_mode=render_mode)

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(str(model_path), env=env)

    all_metrics: list[dict[str, float]] = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        series = rollout_episode(
            model,
            env,
            max_steps=args.max_steps,
            deterministic=deterministic,
            reset_on_done=True,
        )
        m = episode_metrics(series)
        m["episode"] = float(ep)
        all_metrics.append(m)

        np.savez_compressed(save_dir / f"episode_{ep}_timeseries.npz", **series)
        save_plots(series, save_dir, prefix=f"ep{ep}")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path.resolve()),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "trajectory": args.trajectory,
        "track_orientation": bool(args.track_orientation),
        "max_episode_steps": max_eps,
        "obs_noise_std": args.obs_noise_std,
        "action_noise_std": args.action_noise_std,
        "action_delay": args.action_delay,
        "per_episode": all_metrics,
        "mean_rmse": float(np.mean([x["rmse"] for x in all_metrics])),
        "mean_mean_error": float(np.mean([x["mean_error"] for x in all_metrics])),
    }
    (save_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

    if args.record_mp4:
        vid_env = build_tracking_env(args, render_mode="rgb_array")
        model_vid = PPO.load(str(model_path), env=vid_env)
        vid_env.reset(seed=args.seed)
        out_mp4 = Path(args.record_mp4)
        record_video(
            model_vid,
            vid_env,
            out_mp4,
            n_frames=max(1, int(args.video_frames)),
            deterministic=deterministic,
            fps=args.fps,
        )
        vid_env.close()

    env.close()


if __name__ == "__main__":
    main()
