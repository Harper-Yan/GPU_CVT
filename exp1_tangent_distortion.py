#!/usr/bin/env python3
"""
Experiment 1: Tangent Plane Distortion
Measures how much the 2D tangent-plane projection distorts neighbor distances
on curved vs flat regions. Plots distortion vs normal variation score.
"""
import numpy as np
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import trimesh
from scipy.spatial import cKDTree
from testfreeze import (
    nr, tangent, normal_variation_score, knn_indices,
    NV_K, TIER_THRESHOLDS, TIERS, KNN_K,
)
import matplotlib.pyplot as plt

def compute_tangent_distortion(S, P0, N0, treeP0, k_neigh=KNN_K):
    """For each site, compute mean distortion = |d_3d - d_2d| / d_3d across neighbors."""
    _, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64), k=1, workers=-1)
    N = N0[idxN]
    U, V = tangent(N)

    treeS = cKDTree(np.ascontiguousarray(S, dtype=np.float64))
    _, knn = treeS.query(S, k=k_neigh + 1, workers=-1)
    knn = knn[:, 1:]

    n_sites = len(S)
    mean_distortion = np.zeros(n_sites, dtype=np.float64)

    for i in range(n_sites):
        si = S[i]
        ui = U[i]
        vi = V[i]
        dists_3d = []
        dists_2d = []
        for j in knn[i]:
            d = S[j] - si
            d_3d = np.linalg.norm(d)
            a = float(d @ ui)
            b = float(d @ vi)
            d_2d = np.sqrt(a * a + b * b)
            if d_3d > 1e-15:
                dists_3d.append(d_3d)
                dists_2d.append(d_2d)
        dists_3d = np.array(dists_3d)
        dists_2d = np.array(dists_2d)
        if len(dists_3d) > 0:
            mean_distortion[i] = np.mean(np.abs(dists_3d - dists_2d) / dists_3d)

    return mean_distortion


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mesh", nargs="?", default=os.path.join(_ROOT, "meshes", "teapot.obj"))
    args = p.parse_args()
    in_path = args.mesh
    objname = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = os.path.join(_ROOT, "results", objname, "exp1_tangent_distortion")
    os.makedirs(out_dir, exist_ok=True)

    m = trimesh.load(in_path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    P0 = np.asarray(m.vertices, dtype=np.float64)
    F0 = np.asarray(m.faces, dtype=np.int64)
    N0 = np.asarray(m.vertex_normals, dtype=np.float64)
    treeP0 = cKDTree(np.ascontiguousarray(P0, dtype=np.float64))

    S = P0.copy()

    # Normal variation score per vertex
    nv_score = normal_variation_score(P0, nr(N0), NV_K)

    # Tangent plane distortion per site
    print("Computing tangent plane distortion...")
    distortion = compute_tangent_distortion(S, P0, N0, treeP0)

    # Tier assignment (5-tier)
    n_tiers = len(TIERS)
    tier_id = np.digitize(nv_score, TIER_THRESHOLDS).astype(np.int64)
    tier_names = [t[0] for t in TIERS]
    tier_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#F44336"]

    # --- CSV output ---
    import csv
    data_by_tier = [distortion[tier_id == t] for t in range(n_tiers)]
    means = [d.mean() for d in data_by_tier]
    stds = [d.std() for d in data_by_tier]
    csv_path = os.path.join(out_dir, "exp1_tangent_distortion.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n", "mean_distortion", "median_distortion", "std_distortion", "max_distortion"])
        for t in range(n_tiers):
            d = data_by_tier[t]
            w.writerow([tier_names[t], len(d),
                        f"{d.mean():.6f}", f"{np.median(d):.6f}", f"{d.std():.6f}", f"{d.max():.6f}"])
    print(f"Saved: {csv_path}")

    # --- Figure ---

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experiment 1: Tangent Plane Distortion vs Curvature (5-Tier)",
                 fontsize=14, fontweight="bold")

    # (0) Bar chart of mean distortion per tier
    ax = axes[0]
    x = np.arange(n_tiers)
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=tier_colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Mean Neighbor Distance Distortion")
    ax.set_title("Mean Distortion by Tier")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (1) Box plot by tier
    ax = axes[1]
    bp = ax.boxplot(data_by_tier, labels=tier_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], tier_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Mean Neighbor Distance Distortion")
    ax.set_title("Distortion Distribution by Tier")
    ax.grid(True, alpha=0.3)

    # Stats
    for t in range(n_tiers):
        d = data_by_tier[t]
        print(f"  {tier_names[t]:8s}: n={len(d):5d}  "
              f"mean={d.mean():.6f}  median={np.median(d):.6f}  max={d.max():.6f}")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "exp1_tangent_distortion.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
