#!/usr/bin/env python3
"""
Compare modes 0, 1, 2: mesh quality (all 3); freezing rate and execution time (mode 1 vs 2).

Reads from experiments/output/<mode>/<mesh_name>/eval_iters.csv where
  mode in: gpucvt (0), freeze (1), freeze_tiered (2).

Usage:
  python scripts/plot_three_modes.py [--mesh NAME] [--output-dir DIR] [--no-show]
  python scripts/plot_three_modes.py --mesh stanford-bunny
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_BASE = Path("experiments/output")
PLOTS_BASE = Path("experiments/plots_eval")
MODES = [
    ("gpucvt", "baseline (0)"),
    ("freeze", "freeze 5-tier (1)"),
    ("freeze_tiered", "freeze 6-tier (2)"),
]


def read_eval_csv(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] {csv_path}: {e}")
        return None
    if df.empty or "iter" not in df.columns:
        return None
    df = df.drop_duplicates(subset=["iter"], keep="last").sort_values("iter")
    return df


def main():
    p = argparse.ArgumentParser(description="Plot 3-mode comparison: quality, freeze rate, time")
    p.add_argument("--mesh", type=str, default="stanford-bunny", help="Mesh name (stem of .obj)")
    p.add_argument("--output-dir", type=str, default="experiments/output", help="Root output dir (e.g. experiments/output)")
    p.add_argument("--out-dir", type=str, default=None, help="Root for PNGs (default: experiments/plots_eval/<mesh>/)")
    p.add_argument("--no-show", action="store_true", help="Do not display, only save")
    args = p.parse_args()

    root = Path(args.output_dir)
    if not root.exists():
        print(f"[ERROR] Output dir not found: {root}")
        return 1

    mesh_name = args.mesh
    dfs = {}
    for mode_dir, mode_label in MODES:
        csv_path = root / mode_dir / mesh_name / "eval_iters.csv"
        df = read_eval_csv(csv_path)
        if df is not None:
            dfs[mode_dir] = (df, mode_label)
        else:
            print(f"[WARN] No data for {mode_dir}/{mesh_name}")

    if not dfs:
        print("[ERROR] No eval CSVs found for any mode.")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else (PLOTS_BASE / mesh_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Caption: vertex count and speedup over baseline
    n_vertices = None
    baseline_time = None
    speedup1 = speedup2 = None
    if "gpucvt" in dfs:
        df0 = dfs["gpucvt"][0]
        last = df0.drop_duplicates("iter", keep="last").iloc[-1]
        n_vertices = int(last["n_vertices"]) if "n_vertices" in last else None
        baseline_time = float(last["total_remesh_ms"]) if "total_remesh_ms" in last else None
    if baseline_time and baseline_time > 0:
        for mode_dir, (df, _) in dfs.items():
            last = df.drop_duplicates("iter", keep="last").iloc[-1]
            t = float(last.get("total_remesh_ms", 0) or 0)
            if t > 0:
                if mode_dir == "freeze":
                    speedup1 = baseline_time / t
                elif mode_dir == "freeze_tiered":
                    speedup2 = baseline_time / t
    caption_extra = ""
    if n_vertices is not None:
        caption_extra = f" | n_vertices={n_vertices:,}"
    if speedup1 is not None:
        caption_extra += f" | mode1 speedup={speedup1:.2f}x"
    if speedup2 is not None:
        caption_extra += f" | mode2 speedup={speedup2:.2f}x"

    # Colors and markers for the 3 modes
    style = {
        "gpucvt": ("C0", "o", "baseline (0)"),
        "freeze": ("C1", "s", "freeze 5-tier (1)"),
        "freeze_tiered": ("C2", "^", "freeze 6-tier (2)"),
    }

    # ---- 1) Mesh quality: all 3 modes ----
    quality_metrics = [
        ("Qavg", "Qavg", "higher better"),
        ("theta_min_avg", "theta_min (deg)", "higher better"),
        ("theta_lt_30_pct", "% angles < 30°", "lower better"),
        ("theta_gt_90_pct", "% angles > 90°", "lower better"),
        ("dH", "Hausdorff dH", "lower better"),
    ]

    fig1, axes1 = plt.subplots(2, 3, figsize=(14, 9))
    fig1.suptitle(f"{mesh_name}: mesh quality vs iteration{caption_extra}", fontsize=11)
    for idx, (col, ylabel, _) in enumerate(quality_metrics):
        ax = axes1[idx // 3, idx % 3]
        for mode_dir, (df, label) in dfs.items():
            if col not in df.columns or "iter" not in df.columns:
                continue
            c, m, lbl = style.get(mode_dir, ("gray", "x", mode_dir))
            x = pd.to_numeric(df["iter"], errors="coerce")
            y = pd.to_numeric(df[col], errors="coerce")
            ax.plot(x, y, marker=m, color=c, label=lbl, markersize=3)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

    # Last subplot: freeze % for mode 1 and 2 only (placeholder or actual)
    ax_freeze = axes1[1, 2]
    for mode_dir in ("freeze", "freeze_tiered"):
        if mode_dir not in dfs:
            continue
        df, _ = dfs[mode_dir]
        if "freeze_pct" not in df.columns or "iter" not in df.columns:
            continue
        c, m, lbl = style[mode_dir]
        x = pd.to_numeric(df["iter"], errors="coerce")
        y = pd.to_numeric(df["freeze_pct"], errors="coerce")
        ax_freeze.plot(x, y, marker=m, color=c, label=lbl, markersize=3)
    ax_freeze.set_xlabel("iteration")
    ax_freeze.set_ylabel("frozen %")
    ax_freeze.set_title("Freeze rate (modes 1 & 2)")
    ax_freeze.legend(loc="best", fontsize=7)
    ax_freeze.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = out_dir / "three_modes_quality.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"[SAVE] {path1}")
    if args.no_show:
        plt.close(fig1)
    else:
        plt.show()
        plt.close(fig1)

    # ---- 2) Freezing rate: mode 1 vs 2 ----
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    fig2.suptitle(f"{mesh_name}: freezing rate vs iteration{caption_extra}", fontsize=11)
    for mode_dir in ("freeze", "freeze_tiered"):
        if mode_dir not in dfs:
            continue
        df, label = dfs[mode_dir]
        if "freeze_pct" not in df.columns:
            continue
        c, m, lbl = style[mode_dir]
        x = pd.to_numeric(df["iter"], errors="coerce")
        y = pd.to_numeric(df["freeze_pct"], errors="coerce")
        ax2.plot(x, y, marker=m, color=c, label=lbl, markersize=4)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("frozen % (n_frozen / n_vertices)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = out_dir / "freeze_rate_mode1_vs_2.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"[SAVE] {path2}")
    if args.no_show:
        plt.close(fig2)
    else:
        plt.show()
        plt.close(fig2)

    # ---- 3) Execution time: mode 1 vs 2 (cumulative total_remesh_ms) ----
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    fig3.suptitle(f"{mesh_name}: execution time vs iteration{caption_extra}", fontsize=11)
    for mode_dir in ("freeze", "freeze_tiered"):
        if mode_dir not in dfs:
            continue
        df, label = dfs[mode_dir]
        if "total_remesh_ms" not in df.columns:
            continue
        c, m, lbl = style[mode_dir]
        x = pd.to_numeric(df["iter"], errors="coerce")
        y = pd.to_numeric(df["total_remesh_ms"], errors="coerce")
        ax3.plot(x, y, marker=m, color=c, label=lbl, markersize=4)
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("cumulative remesh time (ms)")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    path3 = out_dir / "time_mode1_vs_2.png"
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"[SAVE] {path3}")
    if args.no_show:
        plt.close(fig3)
    else:
        plt.show()
        plt.close(fig3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
