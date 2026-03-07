#!/usr/bin/env python3
"""
Experiment 3: Centroid Direction Reversal
Tracks displacement vectors across iterations and measures how often sites
reverse direction (oscillation). cos_reversal near -1 = bouncing.
"""
import numpy as np
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import trimesh
from scipy.spatial import cKDTree
from testfreeze import (
    nr, tangent, normal_variation_score,
    cell_poly2d, poly_area_centroid_2d, update_to_mesh,
    NV_K, TIER_THRESHOLDS, TIERS,
)
import matplotlib.pyplot as plt

ITERS = 50
K_NEIGH = 32
K_PROJ = 10


def lloyd_nofreeze_track_displacement(S0, P0, F0, N0, treeP0, vf, iters):
    """Run Lloyd without freezing, record displacement vectors per iteration."""
    n = len(S0)
    S = S0.copy()
    disp_vectors = []  # list of (n, 3) arrays

    for it in range(iters):
        _, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64), k=1, workers=-1)
        N = N0[idxN]
        U, V = tangent(N)

        treeS = cKDTree(np.ascontiguousarray(S, dtype=np.float64))
        _, knn = treeS.query(S, k=K_NEIGH + 1, workers=-1)
        knn = knn[:, 1:]

        bbox = S.max(axis=0) - S.min(axis=0)
        R = float(np.max(bbox))
        cent3d = np.full_like(S, np.nan, dtype=np.float64)
        for i in range(n):
            poly = cell_poly2d(i, S, U, V, knn[i], R)
            _, c2 = poly_area_centroid_2d(poly)
            if np.isfinite(c2).all():
                cent3d[i] = S[i] + c2[0] * U[i] + c2[1] * V[i]

        mask = np.isfinite(cent3d).all(axis=1)
        Snew = S.copy()
        if np.any(mask):
            _, idxC = treeP0.query(np.ascontiguousarray(cent3d[mask], dtype=np.float64),
                                   k=K_PROJ, workers=-1)
            for t_idx, i in enumerate(np.flatnonzero(mask)):
                nn = idxC[t_idx]
                faces = set()
                for v in nn:
                    for fi in vf[int(v)]:
                        faces.add(fi)
                faces = np.fromiter(faces, dtype=np.int64)
                tris = F0[faces] if faces.size else np.empty((0, 3), dtype=np.int64)
                Snew[i] = update_to_mesh(cent3d[i], tris, P0)

        disp_vectors.append(Snew - S)
        S = Snew
        print(f"  iter {it+1}/{iters}")

    return disp_vectors


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mesh", nargs="?", default=os.path.join(_ROOT, "meshes", "teapot.obj"))
    args = p.parse_args()
    in_path = args.mesh
    objname = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = os.path.join(_ROOT, "results", objname, "exp3_direction_reversal")
    os.makedirs(out_dir, exist_ok=True)

    m = trimesh.load(in_path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    P0 = np.asarray(m.vertices, dtype=np.float64)
    F0 = np.asarray(m.faces, dtype=np.int64)
    N0 = np.asarray(m.vertex_normals, dtype=np.float64)
    treeP0 = cKDTree(np.ascontiguousarray(P0, dtype=np.float64))

    vf = [[] for _ in range(len(P0))]
    for fi, (a, b, c) in enumerate(F0):
        vf[int(a)].append(fi)
        vf[int(b)].append(fi)
        vf[int(c)].append(fi)

    S0 = P0.copy()

    # Curvature and tier assignment (5-tier)
    nv_score = normal_variation_score(P0, nr(N0), NV_K)
    n_tiers = len(TIERS)
    tier_id = np.digitize(nv_score, TIER_THRESHOLDS).astype(np.int64)
    tier_names = [t[0] for t in TIERS]
    tier_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#F44336"]

    print(f"Running {ITERS} NOFREEZE iterations tracking displacement vectors...")
    disp_vectors = lloyd_nofreeze_track_displacement(S0, P0, F0, N0, treeP0, vf, ITERS)

    n = len(S0)

    # --- Compute cos_reversal between consecutive iterations ---
    # cos_reversal[t][i] = dot(d_t, d_{t+1}) / (|d_t| * |d_{t+1}|)
    cos_reversal_all = []  # (iters-1, n)
    for t in range(len(disp_vectors) - 1):
        d0 = disp_vectors[t]
        d1 = disp_vectors[t + 1]
        norm0 = np.linalg.norm(d0, axis=1)
        norm1 = np.linalg.norm(d1, axis=1)
        denom = norm0 * norm1
        valid = denom > 1e-20
        cos_val = np.zeros(n, dtype=np.float64)
        cos_val[valid] = np.sum(d0[valid] * d1[valid], axis=1) / denom[valid]
        cos_val = np.clip(cos_val, -1.0, 1.0)
        cos_val[~valid] = np.nan
        cos_reversal_all.append(cos_val)

    cos_reversal_all = np.stack(cos_reversal_all, axis=0)  # (iters-1, n)

    # Per-site mean cosine (ignoring NaN)
    with np.errstate(all="ignore"):
        mean_cos = np.nanmean(cos_reversal_all, axis=0)

    # Per-site reversal fraction: how often cos < 0
    reversal_frac = np.nanmean(cos_reversal_all < 0, axis=0)

    # Per-tier per-iteration mean cosine
    tier_cos_per_iter = np.zeros((n_tiers, cos_reversal_all.shape[0]), dtype=np.float64)
    for t in range(n_tiers):
        mask = tier_id == t
        tier_cos_per_iter[t] = np.nanmean(cos_reversal_all[:, mask], axis=1)

    # --- Print stats ---
    for t in range(n_tiers):
        mask = tier_id == t
        mc = mean_cos[mask]
        rf = reversal_frac[mask]
        mc = mc[np.isfinite(mc)]
        rf = rf[np.isfinite(rf)]
        print(f"  {tier_names[t]:8s}: n={mask.sum():5d}  "
              f"mean_cos={mc.mean():.4f}  median_cos={np.median(mc):.4f}  |  "
              f"reversal_frac mean={rf.mean():.4f}  median={np.median(rf):.4f}")

    # --- CSV output ---
    import csv
    csv_path = os.path.join(out_dir, "exp3_direction_reversal.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n", "mean_cos", "median_cos", "std_cos",
                     "reversal_frac_mean", "reversal_frac_median", "reversal_frac_std"])
        for t in range(n_tiers):
            mask = tier_id == t
            mc = mean_cos[mask]; mc = mc[np.isfinite(mc)]
            rf = reversal_frac[mask]; rf = rf[np.isfinite(rf)]
            w.writerow([tier_names[t], int(mask.sum()),
                        f"{mc.mean():.4f}", f"{np.median(mc):.4f}", f"{mc.std():.4f}",
                        f"{rf.mean():.4f}", f"{np.median(rf):.4f}", f"{rf.std():.4f}"])
    print(f"Saved: {csv_path}")

    # --- Figure ---
    x = np.arange(n_tiers)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment 3: Centroid Direction Reversal vs Curvature (5-Tier)",
                 fontsize=14, fontweight="bold")

    # (0,0) Bar: mean cosine per tier
    ax = axes[0, 0]
    cos_data = [mean_cos[(tier_id == t) & np.isfinite(mean_cos)] for t in range(n_tiers)]
    cos_means = [d.mean() for d in cos_data]
    cos_stds = [d.std() for d in cos_data]
    ax.bar(x, cos_means, yerr=cos_stds, capsize=5, color=tier_colors, alpha=0.8,
           edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Mean cos(reversal)")
    ax.set_title("Mean Displacement Cosine by Tier\n(+1=monotonic, -1=oscillating)")
    for i, (m, s) in enumerate(zip(cos_means, cos_stds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (0,1) Bar: mean reversal fraction per tier
    ax = axes[0, 1]
    rf_data = [reversal_frac[(tier_id == t) & np.isfinite(reversal_frac)] for t in range(n_tiers)]
    rf_means = [d.mean() for d in rf_data]
    rf_stds = [d.std() for d in rf_data]
    ax.bar(x, rf_means, yerr=rf_stds, capsize=5, color=tier_colors, alpha=0.8,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Reversal Fraction")
    ax.set_title("Mean Reversal Fraction by Tier")
    for i, (m, s) in enumerate(zip(rf_means, rf_stds)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (1,0) Per-tier mean cosine over iterations
    ax = axes[1, 0]
    iters_x = np.arange(1, cos_reversal_all.shape[0] + 1)
    for t in range(n_tiers):
        ax.plot(iters_x, tier_cos_per_iter[t], color=tier_colors[t],
                label=tier_names[t], linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean cos(reversal)")
    ax.set_title("Displacement Cosine Over Iterations by Tier")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Histogram: reversal fraction by tier
    ax = axes[1, 1]
    for t in range(n_tiers):
        mask = tier_id == t
        rf = reversal_frac[mask]
        rf = rf[np.isfinite(rf)]
        ax.hist(rf, bins=30, alpha=0.5, color=tier_colors[t],
                label=tier_names[t], density=True)
    ax.set_xlabel("Fraction of iterations with direction reversal")
    ax.set_ylabel("Density")
    ax.set_title("Reversal Fraction Distribution by Tier")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "exp3_direction_reversal.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
