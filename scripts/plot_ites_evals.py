#!/usr/bin/env python3
"""
Usage:
  ./plot_eval_iters.py <objs_dir> <runs_root> <out_dir>

Example:
  ./plot_eval_iters.py ../objs . ./plots

Expected layout:
  <objs_dir>/<name>.obj
  <runs_root>/freeze/<name>/eval_iters.csv
  <runs_root>/gpu_cvt/<name>/eval_iters.csv
  <runs_root>/gpucvt/<name>/eval_iters.csv

Output:
  <out_dir>/<objname>/<metric>.png
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

MODES = ["freeze","gpucvt","secured_ccu"]


def read_eval_csv(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}")
        return None

    if df.empty:
        return None

    # detect iteration column
    iter_col = None
    for c in df.columns:
        if str(c).lower() in ("iter", "iteration", "iters", "i"):
            iter_col = c
            break

    if iter_col is None:
        iter_col = df.columns[0]   # assume first column is iteration

    df = df.copy()
    df["_iter_"] = pd.to_numeric(df[iter_col], errors="coerce")
    df = df.dropna(subset=["_iter_"]).sort_values("_iter_")

    return df if not df.empty else None


def numeric_metrics(df):
    return [
        c for c in df.columns
        if c != "_iter_" and pd.api.types.is_numeric_dtype(df[c])
    ]


def sanitize(s):
    return "".join("_" if c in '<>:"/\\|?* ' else c for c in str(s))


def main():
    if len(sys.argv) != 4:
        print("Usage: ./plot_eval_iters.py <objs_dir> <runs_root> <out_dir>")
        sys.exit(1)

    objs_dir = Path(sys.argv[1])
    runs_root = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])

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
            csv = runs_root / mode / name / "eval_iters.csv"
            if not csv.exists():
                print(f"[MISS] {csv}")
                continue
            df = read_eval_csv(csv)
            if df is not None:
                per_mode[mode] = df
                print(f"[OK]   {mode}  {len(df)} rows")

        if not per_mode:
            print("  -> no data, skipping")
            continue

        metrics = set()
        for df in per_mode.values():
            metrics |= set(numeric_metrics(df))

        obj_out = out_dir / name
        obj_out.mkdir(parents=True, exist_ok=True)

        for metric in sorted(metrics):
            plt.figure()
            has = False

            for mode in MODES:
                if mode not in per_mode:
                    continue
                df = per_mode[mode]
                if metric not in df:
                    continue

                x = df["_iter_"]
                y = pd.to_numeric(df[metric], errors="coerce")
                m = ~(x.isna() | y.isna())

                if m.sum() == 0:
                    continue

                plt.plot(x[m], y[m], label=mode)
                has = True

            if not has:
                plt.close()
                continue

            plt.title(f"{name} — {metric}")
            plt.xlabel("iteration")
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
            plt.legend()

            out = obj_out / f"{sanitize(metric)}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.close()

        print(f"  -> written to {obj_out}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
