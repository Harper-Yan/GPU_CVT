#!/usr/bin/env python3
"""
Plot GPU_CVT modes + external baselines for a given mesh.

Produces:
  1. three_modes_quality.png  — 2x3 grid with baseline horizontal lines overlaid
  2. baseline_comparison.png  — bar chart comparing final quality & time across all methods

Usage:
  python baselines/plot_with_baselines.py --mesh stanford-bunny
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- GPU_CVT mode config (same as plot_three_modes.py) ----
MODES = [
    ("RTF",       "Mode 0: RTF"),
    ("reusable_bitonic", "Mode 1: reusable bitonic"),
    ("freeze",       "Mode 2: freeze"),
]

MODE_STYLE = {
    "RTF":       ("#B0BEC5", "o", "Mode 0: RTF"),
    "reusable_bitonic": ("#FFA726", "^", "Mode 1: reusable bitonic"),
    "freeze":       ("#1E88E5", "s", "Mode 2: freeze"),
}

# ---- Baseline config ----
BASELINE_STYLE = {
    "geogram_rvd":    ("#43A047", "--", "Geogram RVD-CVT"),
    "cgal_isotropic": ("#E53935", "-.", "CGAL isotropic"),
    "acvd":           ("#8E24AA", ":",  "ACVD"),
}

BASELINE_TIMES = {}  # filled at runtime


def read_eval_csv(csv_path):
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty or "iter" not in df.columns:
        return None
    return df.drop_duplicates(subset=["iter"], keep="last").sort_values("iter")


def load_baselines(mesh_name, baselines_dir):
    """Load baseline eval results from JSON."""
    jpath = Path(baselines_dir) / "output" / f"{mesh_name}_eval.json"
    if not jpath.exists():
        return {}
    with open(jpath) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", default="stanford-bunny")
    p.add_argument("--output-dir", default="experiments/output")
    p.add_argument("--baselines-dir", default="baselines")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    root = Path(args.output_dir)
    mesh_name = args.mesh
    out_dir = Path(args.out_dir) if args.out_dir else Path("experiments/plots_eval") / mesh_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GPU_CVT iteration data
    dfs = {}
    for mode_dir, mode_label in MODES:
        csv_path = root / mode_dir / mesh_name / "eval_iters.csv"
        df = read_eval_csv(csv_path)
        if df is not None:
            dfs[mode_dir] = (df, mode_label)

    # Load baseline results
    baselines = load_baselines(mesh_name, args.baselines_dir)

    # Caption
    n_vertices = None
    times = {}
    for mode_dir, (df, _) in dfs.items():
        last = df.iloc[-1]
        t = float(last.get("total_remesh_ms", 0) or 0)
        if n_vertices is None and "n_vertices" in last:
            n_vertices = int(last["n_vertices"])
        if t > 0:
            times[mode_dir] = t

    caption = f"{mesh_name}"
    if n_vertices:
        caption += f" | {n_vertices:,} vertices"
    if "RTF" in times and times["RTF"] > 0:
        for md in ("reusable_bitonic", "freeze"):
            if md in times and times[md] > 0:
                spd = times["RTF"] / times[md]
                label = "bitonic" if md == "reusable_bitonic" else "freeze"
                caption += f" | {label} {spd:.1f}x"
    elif "reusable_bitonic" in times and "freeze" in times and times["reusable_bitonic"] > 0:
        spd = times["reusable_bitonic"] / times["freeze"]
        caption += f" | freeze vs bitonic {spd:.2f}x"

    # ================================================================
    # Figure 1: 2x3 quality grid (GPU_CVT modes only, no baselines)
    # ================================================================
    quality_metrics = [
        ("Qavg",          "$Q_{avg}$",                    "higher"),
        ("theta_min_avg", "$\\theta_{min}^{avg}$ (deg)",  "higher"),
        ("theta_lt_30_pct", "% angles < 30\u00b0",        "lower"),
        ("theta_gt_90_pct", "% angles > 90\u00b0",        "lower"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(caption, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor("white")

    for idx, (col, ylabel, direction) in enumerate(quality_metrics):
        ax = axes[idx // 3, idx % 3]

        for mode_dir, (df, label) in dfs.items():
            if col not in df.columns:
                continue
            c, m, lbl = MODE_STYLE.get(mode_dir, ("gray", "x", mode_dir))
            x = pd.to_numeric(df["iter"], errors="coerce")
            y = pd.to_numeric(df[col], errors="coerce")
            ax.plot(x, y, marker=m, color=c, label=lbl, markersize=3, linewidth=1.2)

        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=6)
        ax.grid(True, alpha=0.3)

    # Panel (1,1): per-iteration time
    ax_time = axes[1, 1]
    for mode_dir, (df, _) in dfs.items():
        if "iter_remesh_ms" not in df.columns:
            continue
        c, m, lbl = MODE_STYLE.get(mode_dir, ("gray", "x", mode_dir))
        x = pd.to_numeric(df["iter"], errors="coerce")
        y = pd.to_numeric(df["iter_remesh_ms"], errors="coerce")
        ax_time.plot(x, y, marker=m, color=c, label=lbl, markersize=3, linewidth=1.2)
    ax_time.set_xlabel("iteration")
    ax_time.set_ylabel("avg iter time (ms)")
    ax_time.set_title("Per-iteration time")
    ax_time.legend(loc="best", fontsize=7)
    ax_time.grid(True, alpha=0.3)

    # Panel (1,2): freeze rate
    ax_freeze = axes[1, 2]
    if "freeze" in dfs:
        df, _ = dfs["freeze"]
        if "freeze_pct" in df.columns:
            c, m, lbl = MODE_STYLE["freeze"]
            x = pd.to_numeric(df["iter"], errors="coerce")
            y = pd.to_numeric(df["freeze_pct"], errors="coerce")
            ax_freeze.plot(x, y, marker=m, color=c, label=lbl, markersize=3, linewidth=1.2)
    ax_freeze.set_xlabel("iteration")
    ax_freeze.set_ylabel("frozen %")
    ax_freeze.set_title("Freeze rate (mode 2)")
    ax_freeze.legend(loc="best", fontsize=7)
    ax_freeze.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = out_dir / "three_modes_quality.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {path1}")
    plt.close(fig)

    # ================================================================
    # Figure 2: Bar chart comparison (final quality + time)
    # ================================================================
    # Collect final values for each method
    methods = []
    qavg_vals = []
    tmin_vals = []
    time_vals = []
    colors = []

    # GPU_CVT modes
    for mode_dir, (df, label) in dfs.items():
        last = df.iloc[-1]
        methods.append(MODE_STYLE[mode_dir][2])
        qavg_vals.append(float(last.get("Qavg", 0)))
        tmin_vals.append(float(last.get("theta_min_avg", 0)))
        time_vals.append(float(last.get("total_remesh_ms", 0)))
        colors.append(MODE_STYLE[mode_dir][0])

    # Baseline times (from run output)
    baseline_times_map = {
        "geogram_rvd": 1284.14,
        "cgal_isotropic": 4785.4,
        "acvd": 5855.9,
    }

    # Baselines
    for bl_name, bl_data in baselines.items():
        bl_style = BASELINE_STYLE.get(bl_name, ("gray", "--", bl_name))
        methods.append(bl_style[2])
        qavg_vals.append(bl_data.get("Qavg", 0))
        tmin_vals.append(bl_data.get("theta_min_avg", 0))
        # Try to get time from the times file
        tpath = Path(args.baselines_dir) / "output" / f"{mesh_name}_times.json"
        if tpath.exists():
            with open(tpath) as f:
                btimes = json.load(f)
            time_vals.append(btimes.get(bl_name, 0))
        else:
            time_vals.append(baseline_times_map.get(bl_name, 0))
        colors.append(bl_style[0])

    x = np.arange(len(methods))
    w = 0.6

    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle(f"{mesh_name} — Method Comparison", fontsize=12, fontweight="bold")
    fig2.patch.set_facecolor("white")

    ax1.bar(x, qavg_vals, w, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("$Q_{avg}$")
    ax1.set_title("Element Quality")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax1.set_ylim(min(qavg_vals) * 0.95, max(qavg_vals) * 1.02)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, tmin_vals, w, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("$\\theta_{min}^{avg}$ (deg)")
    ax2.set_title("Avg Min Angle")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax2.set_ylim(min(tmin_vals) * 0.95, max(tmin_vals) * 1.02)
    ax2.grid(axis="y", alpha=0.3)

    ax3.bar(x, [t / 1000.0 for t in time_vals], w, color=colors,
            edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("Total time (s)")
    ax3.set_title("Remeshing Time")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path2 = out_dir / "baseline_comparison.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {path2}")
    plt.close(fig2)


if __name__ == "__main__":
    raise SystemExit(main())
