#!/usr/bin/env python3
"""
Run the challenge-style evaluation matrix and collect one concise report.

This intentionally uses subprocesses so each condition gets a fresh simulator/env
instance and an isolated output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EVAL_SCRIPT = _ROOT / "scripts" / "eval_policy.py"


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    args: tuple[str, ...]


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        name="circle_clean",
        description="In-distribution circle tracking, no injected uncertainty",
        args=("--trajectory", "circle"),
    ),
    Condition(
        name="figure8_clean",
        description="Held-out trajectory shape with the same controller",
        args=("--trajectory", "figure8"),
    ),
    Condition(
        name="spline_clean",
        description="Held-out random waypoint path",
        args=(
            "--trajectory",
            "spline",
            "--trajectory-seed",
            "11",
            "--trajectory-kwargs",
            '{"half_extent": 0.08, "n_points": 6, "speed": 0.18}',
        ),
    ),
    Condition(
        name="circle_obs_noise",
        description="Circle tracking with Gaussian observation noise",
        args=("--trajectory", "circle", "--obs-noise-std", "0.01"),
    ),
    Condition(
        name="circle_action_noise",
        description="Circle tracking with Gaussian action noise",
        args=("--trajectory", "circle", "--action-noise-std", "0.02"),
    ),
    Condition(
        name="circle_delay_1",
        description="Circle tracking with one-step action delay",
        args=("--trajectory", "circle", "--action-delay", "1"),
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the trajectory-tracking evaluation suite")
    p.add_argument("--model", type=str, default="ppo_m4_orient.zip")
    p.add_argument("--output-root", type=str, default="outputs/challenge_eval")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control-dt", type=float, default=0.03)
    p.add_argument("--w-track", type=float, default=1.0)
    p.add_argument("--w-smooth", type=float, default=0.12)
    p.add_argument("--w-velocity", type=float, default=0.02)
    p.add_argument("--w-orient", type=float, default=0.22)
    p.add_argument(
        "--track-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --no-track-orientation for models trained without orientation keys",
    )
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--pure-sac", action="store_true", help="Use PureSACEnv + SAC loader")
    p.add_argument("--record-videos", action="store_true", help="Record an MP4 for each condition")
    p.add_argument("--video-frames", type=int, default=600, help="Frames per video")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--only",
        action="append",
        choices=[c.name for c in CONDITIONS],
        help="Run only the named condition. Can be provided multiple times.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )
    return p.parse_args()


def _common_eval_args(args: argparse.Namespace, save_dir: Path) -> list[str]:
    out = [
        sys.executable,
        str(_EVAL_SCRIPT),
        "--model",
        args.model,
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--seed",
        str(args.seed),
        "--control-dt",
        str(args.control_dt),
        "--w-track",
        str(args.w_track),
        "--w-smooth",
        str(args.w_smooth),
        "--w-velocity",
        str(args.w_velocity),
        "--w-orient",
        str(args.w_orient),
        "--save-dir",
        str(save_dir),
    ]
    if args.track_orientation:
        out.append("--track-orientation")
    if args.stochastic:
        out.append("--stochastic")
    if args.pure_sac:
        out.append("--pure-sac")
    return out


def _metric_values(metrics: dict, key: str) -> list[float]:
    return [float(ep[key]) for ep in metrics.get("per_episode", []) if key in ep]


def _mean(metrics: dict, key: str) -> float:
    values = _metric_values(metrics, key)
    return sum(values) / len(values) if values else 0.0


def _row(condition: Condition, metrics: dict) -> dict[str, str]:
    return {
        "condition": condition.name,
        "trajectory": str(metrics.get("trajectory", "")),
        "rmse_m": f"{_mean(metrics, 'rmse'):.4f}",
        "mean_error_m": f"{_mean(metrics, 'mean_error'):.4f}",
        "p95_error_m": f"{_mean(metrics, 'p95_error'):.4f}",
        "max_error_m": f"{_mean(metrics, 'max_error'):.4f}",
        "mean_delta_action": f"{_mean(metrics, 'mean_abs_delta_action'):.4f}",
        "rmse_orientation_rad": f"{_mean(metrics, 'rmse_orientation_rad'):.4f}",
        "under_5cm_pct": f"{_mean(metrics, 'pct_steps_under_5cm'):.1f}",
        "notes": condition.description,
    }


def _write_reports(out_root: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "condition",
        "trajectory",
        "rmse_m",
        "mean_error_m",
        "p95_error_m",
        "max_error_m",
        "mean_delta_action",
        "rmse_orientation_rad",
        "under_5cm_pct",
        "notes",
    ]
    csv_path = out_root / "suite_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Evaluation Suite Summary",
        "",
        "| Condition | Trajectory | RMSE (m) | p95 error (m) | Max error (m) | "
        "Mean delta action | Orient RMSE (rad) | <5 cm (%) | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['condition']} | {row['trajectory']} | {row['rmse_m']} | "
            f"{row['p95_error_m']} | {row['max_error_m']} | "
            f"{row['mean_delta_action']} | {row['rmse_orientation_rad']} | "
            f"{row['under_5cm_pct']} | {row['notes']} |"
        )
    (out_root / "suite_summary.md").write_text("\n".join(md_lines) + "\n")


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
    selected = [c for c in CONDITIONS if args.only is None or c.name in set(args.only)]

    rows: list[dict[str, str]] = []
    suite_metrics: dict[str, dict] = {}
    for condition in selected:
        save_dir = out_root / condition.name
        cmd = _common_eval_args(args, save_dir) + list(condition.args)
        if args.record_videos:
            vid_path = out_root / "videos" / f"{condition.name}.mp4"
            cmd += ["--record-mp4", str(vid_path), "--video-frames", str(args.video_frames), "--fps", str(args.fps)]
        print("\n" + " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(_ROOT), check=True)
        metrics = json.loads((save_dir / "metrics.json").read_text())
        suite_metrics[condition.name] = metrics
        rows.append(_row(condition, metrics))

    if not args.dry_run:
        (out_root / "suite_metrics.json").write_text(json.dumps(suite_metrics, indent=2))
        _write_reports(out_root, rows)
        print(f"\nWrote {out_root / 'suite_summary.md'}")


if __name__ == "__main__":
    main()
