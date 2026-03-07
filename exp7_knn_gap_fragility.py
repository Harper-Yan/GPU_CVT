#!/usr/bin/env python3
"""
Experiment 7: KNN Gap Fragility vs Curvature  (B2b-A)

For each site at each iteration, measures the gap ratio:
    gap_ratio = d_{K+1} / d_K
where d_K is distance to the K_NEIGH-th (last included) neighbor and
d_{K+1} is distance to the first excluded neighbor.

gap_ratio near 1.0 → KNN boundary is fragile: any tiny movement
can swap a neighbor in/out of the K-set, changing Voronoi topology
even when the site itself barely moves.

Hypothesis (B2a): curvature compresses inter-site distances, lowering
gap_ratio and explaining collective neighborhood instability without
needing a Voronoi-propagation argument.
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
FRAGILE_THR = 1.05   # gap_ratio below this → "fragile" KNN boundary


def lloyd_nofreeze_track_knn_gap(S0, P0, F0, N0, treeP0, vf, iters):
    """Run Lloyd without freezing; record d_{K+1}/d_K per site per iteration."""
    n = len(S0)
    S = S0.copy()
    gap_history = []   # list of (n,) float arrays

    for it in range(iters):
        _, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64), k=1, workers=-1)
        N = N0[idxN]
        U, V = tangent(N)

        treeS = cKDTree(np.ascontiguousarray(S, dtype=np.float64))
        # K_NEIGH+2 = self(1) + K_NEIGH neighbors + 1 extra outside the set
        dist_all, knn_all = treeS.query(S, k=K_NEIGH + 2, workers=-1)

        d_K  = dist_all[:, K_NEIGH]       # K_NEIGH-th neighbor (last inside)
        d_K1 = dist_all[:, K_NEIGH + 1]   # (K_NEIGH+1)-th neighbor (first outside)
        gap_ratio = np.where(d_K > 1e-12, d_K1 / d_K, 1.0)
        gap_history.append(gap_ratio.copy())

        knn = knn_all[:, 1:K_NEIGH + 1]   # (n, K_NEIGH) actual neighbor indices

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

        S = Snew
        print(f"  iter {it+1}/{iters}")

    return gap_history


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mesh", nargs="?", default=os.path.join(_ROOT, "meshes", "teapot.obj"))
    args = p.parse_args()
    in_path = args.mesh
    objname = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = os.path.join(_ROOT, "results", objname, "exp7_knn_gap_fragility")
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

    nv_score = normal_variation_score(P0, nr(N0), NV_K)
    n_tiers = len(TIERS)
    tier_id = np.digitize(nv_score, TIER_THRESHOLDS).astype(np.int64)
    tier_names = [t[0] for t in TIERS]
    tier_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#F44336"]

    print(f"Running {ITERS} NOFREEZE iterations tracking KNN gap ratio...")
    gap_history = lloyd_nofreeze_track_knn_gap(S0, P0, F0, N0, treeP0, vf, ITERS)

    n = len(S0)
    gap_stack = np.stack(gap_history, axis=0)  # (iters, n)

    # Per-site summary: mean and min gap_ratio across iterations
    mean_gap = gap_stack.mean(axis=0)   # (n,)
    min_gap  = gap_stack.min(axis=0)    # (n,) worst-case fragility per site

    # Fragile fraction: fraction of (site, iter) pairs where gap_ratio < FRAGILE_THR
    fragile_mask = gap_stack < FRAGILE_THR   # (iters, n)

    # --- Print stats ---
    print(f"\nKNN gap ratio (d_{{K+1}}/d_K) stats — K={K_NEIGH}, fragile threshold={FRAGILE_THR}:")
    for t in range(n_tiers):
        tmask = tier_id == t
        mg = mean_gap[tmask]
        mi = min_gap[tmask]
        # fraction of all (iter, site) observations in this tier that are fragile
        frac_fragile = fragile_mask[:, tmask].mean()
        print(f"  {tier_names[t]:8s}: n={tmask.sum():5d}  "
              f"mean_gap={mg.mean():.4f}  median_gap={np.median(mg):.4f}  "
              f"min_gap_mean={mi.mean():.4f}  "
              f"frac_fragile(<{FRAGILE_THR})={frac_fragile:.4f}")

    # --- CSV ---
    import csv
    csv_path = os.path.join(out_dir, "exp7_knn_gap_fragility.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n",
                    "mean_gap_mean", "mean_gap_median", "mean_gap_std",
                    "min_gap_mean",  "min_gap_median",  "min_gap_std",
                    f"frac_fragile_lt{FRAGILE_THR}"])
        for t in range(n_tiers):
            tmask = tier_id == t
            mg = mean_gap[tmask]
            mi = min_gap[tmask]
            frac_fragile = fragile_mask[:, tmask].mean()
            w.writerow([tier_names[t], int(tmask.sum()),
                        f"{mg.mean():.6f}", f"{np.median(mg):.6f}", f"{mg.std():.6f}",
                        f"{mi.mean():.6f}", f"{np.median(mi):.6f}", f"{mi.std():.6f}",
                        f"{frac_fragile:.6f}"])
    print(f"Saved: {csv_path}")

    # --- Per-tier trajectory: mean gap_ratio over iterations ---
    tier_gap_per_iter = np.zeros((n_tiers, ITERS), dtype=np.float64)
    for t in range(n_tiers):
        tmask = tier_id == t
        tier_gap_per_iter[t] = gap_stack[:, tmask].mean(axis=1)

    # --- Figure ---
    data_mean = [mean_gap[tier_id == t] for t in range(n_tiers)]
    data_min  = [min_gap[tier_id == t]  for t in range(n_tiers)]
    x = np.arange(n_tiers)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Experiment 7: KNN Gap Fragility vs Curvature (K={K_NEIGH}, 5-Tier)",
        fontsize=14, fontweight="bold"
    )

    # (0,0) Bar: mean gap_ratio per tier
    ax = axes[0, 0]
    means = [d.mean() for d in data_mean]
    stds  = [d.std()  for d in data_mean]
    ax.bar(x, means, yerr=stds, capsize=5, color=tier_colors, alpha=0.8,
           edgecolor="black", linewidth=0.5)
    ax.axhline(FRAGILE_THR, color="red", linestyle="--", linewidth=1,
               label=f"fragile threshold ({FRAGILE_THR})")
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Mean gap ratio d_{K+1}/d_K")
    ax.set_title("Mean KNN Gap Ratio by Tier")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (0,1) Box: mean gap_ratio distribution
    ax = axes[0, 1]
    bp = ax.boxplot(data_mean, labels=tier_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], tier_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(FRAGILE_THR, color="red", linestyle="--", linewidth=1,
               label=f"fragile threshold ({FRAGILE_THR})")
    ax.set_ylabel("Mean gap ratio d_{K+1}/d_K")
    ax.set_title("KNN Gap Ratio Distribution by Tier")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Line: mean gap_ratio trajectory over iterations per tier
    ax = axes[1, 0]
    iters_x = np.arange(1, ITERS + 1)
    for t in range(n_tiers):
        ax.plot(iters_x, tier_gap_per_iter[t], color=tier_colors[t],
                label=tier_names[t], linewidth=1.5)
    ax.axhline(FRAGILE_THR, color="red", linestyle="--", linewidth=1,
               label=f"fragile threshold ({FRAGILE_THR})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean gap ratio d_{K+1}/d_K")
    ax.set_title("KNN Gap Ratio Trajectory Over Iterations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Bar: fragile fraction per tier
    ax = axes[1, 1]
    frac_fragile = [fragile_mask[:, tier_id == t].mean() for t in range(n_tiers)]
    ax.bar(x, frac_fragile, color=tier_colors, alpha=0.8,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel(f"Fraction of (site, iter) with gap < {FRAGILE_THR}")
    ax.set_title(f"Fragile KNN Fraction by Tier (threshold={FRAGILE_THR})")
    for i, v in enumerate(frac_fragile):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "exp7_knn_gap_fragility.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
