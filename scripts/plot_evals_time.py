#!/usr/bin/env python3
"""
Usage:
  ./plot_eval_time.py <objs_dir> <runs_root> <out_dir> [--dt_ms 2.0]

Example:
  ./plot_eval_time.py ../objs . ./plots_time --dt_ms 2

Expected layout:
  <objs_dir>/<name>.obj
  <runs_root>/freeze/<name>/eval_iters.csv
  <runs_root>/gpucvt/<name>/eval_iters.csv
  <runs_root>/secured_ccu/<name>/eval_iters.csv

Output:
  <out_dir>/<objname>/<metric>_time.png
"""

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODES = ["freeze", "gpucvt", "secured_ccu"]


def sanitize(s):
    return "".join("_" if c in '<>:"/\\|?* ' else c for c in str(s))


def detect_time_col(df: pd.DataFrame) -> str | None:
    # Prefer total_remesh_ms
    candidates = [
        "total_remesh_ms",
        "total_remeshing_ms",
        "total_ms",
        "t_ms",
        "time_ms",
        "elapsed_ms",
    ]
    cols_lower = {str(c).lower(): c for c in df.columns}
    for key in candidates:
        if key in cols_lower:
            return cols_lower[key]
    return None


def read_eval_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}")
        return None
    if df.empty:
        return None

    tcol = detect_time_col(df)
    if tcol is None:
        print(f"[WARN] {csv_path}: no total_remesh_ms-like column found")
        return None

    df = df.copy()
    df["_t_ms_"] = pd.to_numeric(df[tcol], errors="coerce")
    df = df.dropna(subset=["_t_ms_"]).sort_values("_t_ms_")

    # Some CSV writers can duplicate time stamps; keep the last for each time
    df = df.groupby("_t_ms_", as_index=False).last().sort_values("_t_ms_")

    return df if not df.empty else None


def numeric_metrics(df: pd.DataFrame):
    # numeric columns excluding the time helper
    out = []
    for c in df.columns:
        if c == "_t_ms_":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def resample_to_uniform_time(df: pd.DataFrame, metric: str, dt_ms: float) -> tuple[np.ndarray, np.ndarray] | None:
    t = pd.to_numeric(df["_t_ms_"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=np.float64)

    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    if t.size < 2:
        return None

    # Ensure strictly increasing for interp
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # If time doesn't increase (shouldn't happen), bail
    if t[-1] <= t[0]:
        return None

    # Uniform sampling grid
    t0 = 0.0
    t1 = float(t[-1])
    grid = np.arange(t0, t1 + 1e-9, dt_ms, dtype=np.float64)

    # Interpolate metric onto grid
    yg = np.interp(grid, t, y)
    return grid, yg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("objs_dir")
    ap.add_argument("runs_root")
    ap.add_argument("out_dir")
    ap.add_argument("--dt_ms", type=float, default=200.0, help="sampling interval on total_remesh_ms (ms)")
    args = ap.parse_args()

    objs_dir = Path(args.objs_dir)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    dt_ms = float(args.dt_ms)

    out_dir.mkdir(parents=True, exist_ok=True)

    obj_files = sorted(objs_dir.glob("*.obj"))
    if not obj_files:
        print(f"No OBJ files in {objs_dir}")
        sys.exit(1)

    for obj in obj_files:
        name = obj.stem
        print(f"\n=== {name} ===")

        per_mode = {}
        for mode in MODES:
            csv_path = runs_root / mode / name / "eval_iters.csv"
            if not csv_path.exists():
                print(f"[MISS] {csv_path}")
                continue
            df = read_eval_csv(csv_path)
            if df is not None:
                per_mode[mode] = df
                print(f"[OK]   {mode}  {len(df)} rows")

        if not per_mode:
            print("  -> no data, skipping")
            continue

        # Collect all numeric metrics seen across modes
        metrics = set()
        for df in per_mode.values():
            metrics |= set(numeric_metrics(df))

        # Don’t plot helper columns or pure time columns
        for bad in ["iter", "iteration", "_iter_", "_t_ms_"]:
            metrics = {m for m in metrics if str(m).lower() != bad}

        obj_out = out_dir / name
        obj_out.mkdir(parents=True, exist_ok=True)

        for metric in sorted(metrics):
            plt.figure()
            has = False

            for mode in MODES:
                if mode not in per_mode:
                    continue
                df = per_mode[mode]
                if metric not in df.columns:
                    continue

                res = resample_to_uniform_time(df, metric, dt_ms)
                if res is None:
                    continue
                grid, yg = res

                plt.plot(grid, yg, label=mode)
                has = True

            if not has:
                plt.close()
                continue

            plt.title(f"{name} — {metric} (resampled every {dt_ms:g} ms)")
            plt.xlabel("total_remesh_ms (ms)")
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
            plt.legend()

            out_path = obj_out / f"{sanitize(metric)}_time.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

        print(f"  -> written to {obj_out}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
