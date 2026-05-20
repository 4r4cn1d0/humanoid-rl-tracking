#!/usr/bin/env python3
"""Plot metrics from eval_policy.py output (.npz timeseries)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=str, help="Path to episode_*_timeseries.npz")
    p.add_argument("--out", type=str, default="", help="Output directory (default: next to npz)")
    args = p.parse_args()
    path = Path(args.npz)
    data = np.load(path)
    out = Path(args.out) if args.out else path.parent
    out.mkdir(parents=True, exist_ok=True)
    prefix = path.stem

    tgt = data["target"]
    ee = data["ee"]
    if tgt.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(tgt[:, 0], tgt[:, 1], label="Target XY", color="C0")
        ax.plot(ee[:, 0], ee[:, 1], label="EE XY", color="C1", alpha=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{prefix}_path_xy.png", dpi=150)
        plt.close(fig)

    t = data["time"]
    err = data["error"]
    act = data["action"]
    if t.size > 0 and err.size > 0:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(t, err)
        axes[0].set_ylabel("error")
        if len(act) >= 2:
            da = np.linalg.norm(np.diff(act, axis=0), axis=1)
            axes[1].plot(t[1:], da)
        axes[1].set_ylabel("||Δa||")
        axes[1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(out / f"{prefix}_error_smoothness.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
