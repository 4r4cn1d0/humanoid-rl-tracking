#!/usr/bin/env python3
"""
Generate clean summary plots for the best checkpoint evaluation.
Uses exact training parameters. Reports step-level success @3cm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = _ROOT / "outputs" / "best_model_eval"
OUT_DIR  = _ROOT / "outputs" / "best_model_eval" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_LABELS = {
    "circle_clean":       "Circle\n(clean)",
    "figure8_clean":      "Figure-8\n(clean)",
    "spline_clean":       "Spline\n(clean)",
    "circle_obs_noise":   "Circle\n+obs noise",
    "circle_action_noise":"Circle\n+act noise",
    "circle_delay_1":     "Circle\n+delay",
}

SUCCESS_THRESHOLD_CM = 3.0   # matches training


def load_suite() -> dict:
    return json.loads((EVAL_DIR / "suite_metrics.json").read_text())


def step_success_pct(per_episode: list[dict], threshold_m: float) -> float:
    """Recompute step-level success from per-episode timeseries npz files."""
    # Fall back to pct_steps_under_5cm if threshold is 5cm, else approximate
    # from mean_error (conservative). Best: load npz.
    hits, total = 0, 0
    for ep_idx, ep in enumerate(per_episode):
        cond_dir = None  # will be set per condition
        # Use pct_steps_under_5cm as proxy scaled to 3cm if available
        # We'll load the npz directly
    return 0.0  # placeholder — see below


def compute_step_success_from_npz(cond_dir: Path, n_episodes: int, threshold_m: float) -> float:
    hits, total = 0, 0
    for ep in range(n_episodes):
        f = cond_dir / f"episode_{ep}_timeseries.npz"
        if not f.exists():
            continue
        d = np.load(f)
        errs = d["error"]
        hits  += int(np.sum(errs < threshold_m))
        total += len(errs)
    return (hits / total * 100.0) if total > 0 else 0.0


def main() -> None:
    suite = load_suite()
    conditions = list(CONDITION_LABELS.keys())

    rmse_vals, mean_err_vals, success_3cm_vals, success_5cm_vals = [], [], [], []

    for cond in conditions:
        if cond not in suite:
            rmse_vals.append(0); mean_err_vals.append(0)
            success_3cm_vals.append(0); success_5cm_vals.append(0)
            continue
        m = suite[cond]
        per_ep = m.get("per_episode", [])
        n = len(per_ep)

        rmse_vals.append(np.mean([ep["rmse"] for ep in per_ep]) * 100)
        mean_err_vals.append(np.mean([ep["mean_error"] for ep in per_ep]) * 100)

        cond_dir = EVAL_DIR / cond
        success_3cm_vals.append(compute_step_success_from_npz(cond_dir, n, 0.03))
        success_5cm_vals.append(compute_step_success_from_npz(cond_dir, n, 0.05))

    labels = [CONDITION_LABELS[c] for c in conditions]
    x = np.arange(len(conditions))
    w = 0.35

    # ── Plot 1: Tracking error (RMSE + mean) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, rmse_vals,  w, label="RMSE (cm)",       color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + w/2, mean_err_vals, w, label="Mean error (cm)", color="#FF9800", alpha=0.85)
    ax.axhline(3.0, color="red",   linestyle="--", linewidth=1.2, label="3 cm threshold (training)")
    ax.axhline(5.0, color="gray",  linestyle=":",  linewidth=1.0, label="5 cm threshold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Error (cm)"); ax.set_title("Tracking Error by Condition\n(best checkpoint: 175k steps, control_dt=0.05)")
    ax.legend(fontsize=8); ax.set_ylim(0, max(max(rmse_vals), 6) * 1.2)
    for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tracking_error.png", dpi=150)
    plt.close(fig)
    print(f"Saved tracking_error.png")

    # ── Plot 2: Step-level success rate ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bars3 = ax.bar(x - w/2, success_3cm_vals, w, label="Steps <3cm (%)", color="#4CAF50", alpha=0.85)
    bars4 = ax.bar(x + w/2, success_5cm_vals, w, label="Steps <5cm (%)", color="#9C27B0", alpha=0.85)
    ax.axhline(90, color="red", linestyle="--", linewidth=1.2, label="90% target")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of timesteps"); ax.set_title("Step-Level Success Rate by Condition\n(best checkpoint: 175k steps)")
    ax.legend(fontsize=8); ax.set_ylim(0, 110)
    for bar in bars3: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7)
    for bar in bars4: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "success_rate.png", dpi=150)
    plt.close(fig)
    print(f"Saved success_rate.png")

    # ── Plot 3: Checkpoint scan curve ────────────────────────────────────────
    scan_path = _ROOT / "outputs" / "checkpoint_scan.json"
    if scan_path.exists():
        scan = json.loads(scan_path.read_text())
        steps = [r["steps"] for r in scan]
        mean_errs = [r["mean_tracking_error_cm"] for r in scan]
        success_3 = [r["step_success_3cm_pct"] for r in scan]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax1.plot(steps, mean_errs, "o-", color="#2196F3", markersize=4, linewidth=1.5)
        ax1.axhline(3.0, color="red", linestyle="--", linewidth=1, label="3 cm threshold")
        ax1.set_ylabel("Mean tracking error (cm)")
        ax1.set_title("Checkpoint Scan: circle @ speed=0.6, control_dt=0.05")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        ax2.plot(steps, success_3, "o-", color="#4CAF50", markersize=4, linewidth=1.5)
        ax2.axhline(90, color="red", linestyle="--", linewidth=1, label="90% target")
        ax2.set_ylabel("Step success @3cm (%)")
        ax2.set_xlabel("Training steps")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 105)

        # Mark best
        best_idx = int(np.argmax(success_3))
        ax1.axvline(steps[best_idx], color="orange", linestyle=":", linewidth=1.5, label=f"Best: {steps[best_idx]//1000}k")
        ax2.axvline(steps[best_idx], color="orange", linestyle=":", linewidth=1.5, label=f"Best: {steps[best_idx]//1000}k")
        ax1.legend(fontsize=8); ax2.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(OUT_DIR / "checkpoint_scan.png", dpi=150)
        plt.close(fig)
        print(f"Saved checkpoint_scan.png")

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n{'Condition':<22} {'RMSE':>8} {'Mean err':>10} {'@3cm%':>8} {'@5cm%':>8}")
    print("-" * 60)
    for i, cond in enumerate(conditions):
        print(f"{cond:<22} {rmse_vals[i]:>7.2f}cm {mean_err_vals[i]:>9.2f}cm "
              f"{success_3cm_vals[i]:>7.1f}% {success_5cm_vals[i]:>7.1f}%")
    print(f"\nPlots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
