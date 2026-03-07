#!/usr/bin/env python3
"""
Read eval CSV under results/, compare mesh quality vs iteration and frozen rate vs iteration.
Saves and displays plots as PNG.

Usage:
  python plot_eval_iters.py [--results-dir experiments/results] [--mesh NAME] [--out-dir experiments/plots_eval] [--no-show]

Expected layout:
  experiments/results/<mesh_name>/baseline/eval_iters.csv
  experiments/results/<mesh_name>/freeze/eval_iters.csv
  (optional) experiments/results/<mesh_name>/baseline/iter_000.xyz for n_sites
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_n_sites(results_dir: Path, mesh_name: str) -> int | None:
    """Get total site count from iter_000.xyz line count."""
    for sub in ("baseline", "freeze"):
        xyz = results_dir / mesh_name / sub / "iter_000.xyz"
        if xyz.exists():
            with open(xyz) as f:
                return sum(1 for _ in f)
    return None


def get_mesh_vertices_faces(results_dir: Path, mesh_name: str) -> tuple[int, int] | None:
    """Get vertex and face count from a generated mesh.obj."""
    for sub in ("baseline", "freeze"):
        for stem in ("iter_099_mesh", "iter_000_mesh"):
            obj_path = results_dir / mesh_name / sub / f"{stem}.obj"
            if obj_path.exists():
                nv, nf = 0, 0
                with open(obj_path) as f:
                    for line in f:
                        if line.startswith("v "):
                            nv += 1
                        elif line.startswith("f "):
                            nf += 1
                if nv > 0 or nf > 0:
                    return (nv, nf)
    return None


def read_eval_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}")
        return None
    if df.empty:
        return None
    # Keep last row per (mode, iter) in case of duplicates from multiple runs
    if "iter" in df.columns and "mode" in df.columns:
        df = df.drop_duplicates(subset=["mode", "iter"], keep="last").sort_values("iter")
    return df


def main():
    p = argparse.ArgumentParser(description="Plot eval CSV: mesh quality and freeze rate vs iteration")
    p.add_argument("--results-dir", type=str, default="experiments/results", help="Root of results (e.g. experiments/results)")
    p.add_argument("--mesh", type=str, default=None, help="Mesh name to plot (default: all found)")
    p.add_argument("--out-dir", type=str, default="experiments/plots_eval", help="Output directory for PNGs")
    p.add_argument("--no-show", action="store_true", help="Do not display (only save)")
    p.add_argument("--time", action="store_true", help="Use real time (ms) on x-axis instead of iteration")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results dir not found: {results_dir}")
        return 1

    # Find mesh directories
    mesh_dirs = [d.name for d in results_dir.iterdir() if d.is_dir()]
    if args.mesh:
        if args.mesh not in mesh_dirs:
            print(f"[ERROR] Mesh '{args.mesh}' not found in {results_dir}")
            return 1
        mesh_dirs = [args.mesh]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mesh_name in mesh_dirs:
        baseline_csv = results_dir / mesh_name / "baseline" / "eval_iters.csv"
        freeze_csv = results_dir / mesh_name / "freeze" / "eval_iters.csv"

        df_baseline = read_eval_csv(baseline_csv) if baseline_csv.exists() else None
        df_freeze = read_eval_csv(freeze_csv) if freeze_csv.exists() else None

        if df_baseline is None and df_freeze is None:
            print(f"[SKIP] {mesh_name}: no eval CSV found")
            continue

        n_sites = get_n_sites(results_dir, mesh_name)
        mesh_stats = get_mesh_vertices_faces(results_dir, mesh_name)
        stats_str = ""
        if mesh_stats is not None:
            nv, nf = mesh_stats
            stats_str = f" | V={nv:,} F={nf:,}"

        # Total remesh ms and speedup for title
        time_str = ""
        if "total_remesh_ms" in (df_baseline.columns if df_baseline is not None else []):
            baseline_ms = pd.to_numeric(df_baseline["total_remesh_ms"], errors="coerce").dropna()
            baseline_ms = float(baseline_ms.iloc[-1]) if len(baseline_ms) else None
        else:
            baseline_ms = None
        if df_freeze is not None and "total_remesh_ms" in df_freeze.columns:
            freeze_ms = pd.to_numeric(df_freeze["total_remesh_ms"], errors="coerce").dropna()
            freeze_ms = float(freeze_ms.iloc[-1]) if len(freeze_ms) else None
        else:
            freeze_ms = None
        if baseline_ms is not None and freeze_ms is not None and freeze_ms > 0:
            speedup = baseline_ms / freeze_ms
            time_str = f" | baseline: {baseline_ms:.0f} ms, freeze: {freeze_ms:.0f} ms, speedup: {speedup:.2f}x"
        elif baseline_ms is not None:
            time_str = f" | baseline: {baseline_ms:.0f} ms"
        if freeze_ms is not None and (baseline_ms is None or time_str == ""):
            time_str = f" | freeze: {freeze_ms:.0f} ms"

        use_time = args.time
        xlabel = "time (ms)" if use_time else "iteration"
        xcol = "total_remesh_ms" if use_time else "iter"

        # When using time: reset freeze phase to start at 0 (freeze time = time since baseline ended)
        freeze_time_offset = 0.0
        if use_time and df_freeze is not None and "total_remesh_ms" in df_freeze.columns:
            t0 = pd.to_numeric(df_freeze["total_remesh_ms"], errors="coerce").iloc[0]
            if not np.isnan(t0):
                freeze_time_offset = float(t0)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(f"{mesh_name}: mesh quality & freeze rate vs {xlabel}{stats_str}{time_str}", fontsize=12)

        metrics = [
            ("Qavg", "Qavg", "higher better"),
            ("theta_min_avg", "theta_min (deg)", "higher better"),
            ("theta_lt_30_pct", "% angles < 30°", "lower better"),
            ("theta_gt_90_pct", "% angles > 90°", "lower better"),
            ("dH", "Hausdorff dH", "lower better"),
        ]

        for idx, (col, ylabel, _) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            if df_baseline is not None and col in df_baseline.columns and xcol in df_baseline.columns:
                xvals = pd.to_numeric(df_baseline[xcol], errors="coerce")
                vals = pd.to_numeric(df_baseline[col], errors="coerce")
                ax.plot(xvals, vals, "o-", label="baseline", color="C0", markersize=4)
            if df_freeze is not None and col in df_freeze.columns and xcol in df_freeze.columns:
                xvals = pd.to_numeric(df_freeze[xcol], errors="coerce")
                if use_time:
                    xvals = xvals - freeze_time_offset
                vals = pd.to_numeric(df_freeze[col], errors="coerce")
                ax.plot(xvals, vals, "s-", label="freeze", color="C1", markersize=4)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Freeze rate subplot: frozen % = (frozen vertices / total vertices) * 100
        ax_freeze = axes[1, 2]
        if df_freeze is not None and ("freeze_pct" in df_freeze.columns or "freeze_cell_num" in df_freeze.columns) and xcol in df_freeze.columns:
            xvals = pd.to_numeric(df_freeze[xcol], errors="coerce")
            if use_time:
                xvals = xvals - freeze_time_offset
            n_total = None
            if "n_vertices" in df_freeze.columns:
                n_total = pd.to_numeric(df_freeze["n_vertices"], errors="coerce").iloc[0] if len(df_freeze) else None
            if n_total is None or (isinstance(n_total, float) and np.isnan(n_total)):
                n_total = n_sites
            if "freeze_pct" in df_freeze.columns:
                freeze_pct = pd.to_numeric(df_freeze["freeze_pct"], errors="coerce")
            elif "freeze_cell_num" in df_freeze.columns and n_total and n_total > 0:
                n_frozen = pd.to_numeric(df_freeze["freeze_cell_num"], errors="coerce")
                freeze_pct = 100.0 * n_frozen / float(n_total)
            else:
                freeze_pct = None
            if freeze_pct is not None:
                ax_freeze.plot(xvals, freeze_pct, "s-", color="C2", markersize=4)
                ax_freeze.set_ylabel("frozen % (n_frozen / n_vertices)")
            else:
                n_frozen = pd.to_numeric(df_freeze["freeze_cell_num"], errors="coerce")
                ax_freeze.plot(xvals, n_frozen, "s-", color="C2", markersize=4)
                ax_freeze.set_ylabel("frozen count")
            ax_freeze.set_xlabel(xlabel)
            ax_freeze.set_title(f"freeze rate vs {xlabel}")
        else:
            ax_freeze.text(0.5, 0.5, "No freeze data", ha="center", va="center", transform=ax_freeze.transAxes)
        ax_freeze.grid(True, alpha=0.3)

        plt.tight_layout()
        suffix = "_time" if use_time else "_iters"
        out_path = out_dir / f"{mesh_name}_eval{suffix}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {out_path}")
        if not args.no_show:
            plt.show()
        else:
            plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
