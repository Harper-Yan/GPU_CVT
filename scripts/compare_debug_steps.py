#!/usr/bin/env python3
"""
Debug script: compare first 10 points at each CVT step between testfreeze (Python)
and gpu_cvt (CUDA). Identifies which step produces mismatched results.
"""
import numpy as np
import argparse
import os
import subprocess
import sys

# Steps to compare (must match output from both pipelines)
STEPS = [
    "0_sites",           # Initial sites S (mesh vertices)
    "1_normals",         # Normals at sites (from nearest mesh vertex)
    "2_tangent_U",       # Tangent frame U
    "2_tangent_V",       # Tangent frame V
    "3_knn_sites",       # KNN indices among sites (first neighbor for pt 0)
    "4_centroids",       # Centroids before projection
    "5_after_projection", # After projection to mesh
]

TOL = 1e-6  # Relative tolerance for float comparison


def load_xyz(path):
    """Load .xyz file (one x y z per line)."""
    if not os.path.isfile(path):
        return None
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(data, dtype=np.float64) if data else None


def load_obj_vertices(path, max_pts=100):
    """Load first max_pts vertices from .obj file (v x y z lines)."""
    if not os.path.isfile(path):
        return None
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    data.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    if len(data) >= max_pts:
                        break
    return np.array(data, dtype=np.float64) if data else None


def compare_arrays(name, a, b, tol=TOL):
    """Compare two arrays, return (match, max_diff, report string)."""
    if a is None and b is None:
        return True, 0.0, f"{name}: both missing (skip)"
    if a is None:
        return False, float("inf"), f"{name}: Python missing, GPU has {len(b)} pts"
    if b is None:
        return False, float("inf"), f"{name}: GPU missing, Python has {len(a)} pts"
    n = min(len(a), len(b), 100)
    if n == 0:
        return True, 0.0, f"{name}: empty"
    a = np.asarray(a, dtype=np.float64)[:n]
    b = np.asarray(b, dtype=np.float64)[:n]
    diff = np.abs(a - b)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)
    rel_diff = diff / scale
    max_rel = float(np.max(rel_diff))
    max_abs = float(np.max(diff))
    match = max_rel <= tol
    return match, max_rel, f"{name}: match={match} max_rel_diff={max_rel:.2e} max_abs={max_abs:.2e} (n={n})"


def main():
    p = argparse.ArgumentParser(description="Compare CVT step outputs: testfreeze vs gpu_cvt")
    p.add_argument("mesh", type=str, help="Input mesh .obj path")
    p.add_argument("--freeze_dir", type=str, default="evolve_compare_eval",
                   help="Output dir from testfreeze (contains freeze/ subdir)")
    p.add_argument("--gpu_dir", type=str, default="results",
                   help="Output dir from gpu_cvt (e.g. results/stanford-bunny/baseline)")
    p.add_argument("--tolerance", type=float, default=1e-5)
    args = p.parse_args()

    mesh = os.path.abspath(args.mesh)
    objname = os.path.splitext(os.path.basename(mesh))[0]
    freeze_base = os.path.join(args.freeze_dir, objname, "freeze")
    gpu_base = os.path.join(args.gpu_dir, objname, "baseline")

    print("=" * 60)
    print("CVT step-by-step comparison (first 100 points)")
    print("=" * 60)
    print(f"Mesh: {mesh}")
    print(f"Python (testfreeze) dir: {freeze_base}")
    print(f"GPU (gpu_cvt) dir:      {gpu_base}")
    print()

    # Both pipelines write debug_*_*.xyz to their respective dirs

    # Compare before_projection (centroids) and report max relative diff
    pf_before = os.path.join(freeze_base, "before_projection_full.xyz")
    pg_before = os.path.join(gpu_base, "before_projection_full.xyz")
    if os.path.isfile(pf_before) and os.path.isfile(pg_before):
        a = load_xyz(pf_before)
        b = load_xyz(pg_before)
        if a is not None and b is not None:
            n = min(len(a), len(b))
            a = np.asarray(a, dtype=np.float64)[:n]
            b = np.asarray(b, dtype=np.float64)[:n]
            valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            if np.any(valid):
                diff = np.abs(a[valid] - b[valid])
                scale = np.maximum(np.maximum(np.abs(a[valid]), np.abs(b[valid])), 1e-30)
                rel_diff = diff / scale
                max_rel = float(np.max(rel_diff))
                max_abs = float(np.max(diff))
                print(f"\n[before_projection] max_rel_diff={max_rel:.6e} max_abs_diff={max_abs:.6e} (n_valid={np.sum(valid)})")
            else:
                print("\n[before_projection] No valid points to compare")
        else:
            print("\n[before_projection] Could not load files")
    else:
        print(f"\n[before_projection] Missing: python={os.path.isfile(pf_before)} gpu={os.path.isfile(pg_before)}")

    results = []
    first_mismatch = None

    for step in STEPS:
        # Try .xyz first (both can use it), then .obj for GPU
        pf = os.path.join(freeze_base, f"debug_{step}.xyz")
        pg_xyz = os.path.join(gpu_base, f"debug_{step}.xyz")
        pg_obj = os.path.join(gpu_base, f"debug_{step}.obj")

        a = load_xyz(pf)
        b = load_xyz(pg_xyz) if os.path.isfile(pg_xyz) else load_obj_vertices(pg_obj, 10)

        match, max_diff, report = compare_arrays(step, a, b, args.tolerance)
        results.append((step, match, report))
        print(report)
        if not match and first_mismatch is None:
            first_mismatch = step

    print()
    print("=" * 60)
    if first_mismatch is not None:
        print(f"FIRST MISMATCH at step: {first_mismatch}")
        print("This is the step where GPU and Python results diverge.")
    else:
        all_ok = all(r[1] for r in results)
        if all_ok:
            print("All steps match within tolerance.")
        else:
            print("Some steps could not be compared (missing files).")
    print("=" * 60)

    return 0 if first_mismatch is None else 1


if __name__ == "__main__":
    sys.exit(main())
