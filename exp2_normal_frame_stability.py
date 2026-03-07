#!/usr/bin/env python3
"""
Experiment 2: Normal Frame Stability
Tracks how often the nearest-vertex index (and thus the tangent frame normal)
flips between iterations for each site. Plots flip count vs curvature tier.
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
    cell_poly2d, poly_area_centroid_2d, update_to_mesh,
    NV_K, TIER_THRESHOLDS, TIERS, KNN_K,
)
import matplotlib.pyplot as plt

ITERS = 50
K_NEIGH = 32
K_PROJ = 10


def lloyd_nofreeze_track_normals(S0, P0, F0, N0, treeP0, vf, iters):
    """Run Lloyd without freezing, record nearest-vertex index and normal per iteration."""
    n = len(S0)
    S = S0.copy()

    # idxN_history[t] = nearest vertex index for each site at iteration t
    idxN_history = []
    normal_history = []

    for it in range(iters):
        _, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64), k=1, workers=-1)
        N = N0[idxN]
        idxN_history.append(idxN.copy())
        normal_history.append(N.copy())

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

        S = Snew
        print(f"  iter {it+1}/{iters}")

    return idxN_history, normal_history


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mesh", nargs="?", default=os.path.join(_ROOT, "meshes", "teapot.obj"))
    args = p.parse_args()
    in_path = args.mesh
    objname = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = os.path.join(_ROOT, "results", objname, "exp2_normal_stability")
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

    # Curvature score and tier assignment (5-tier)
    nv_score = normal_variation_score(P0, nr(N0), NV_K)
    n_tiers = len(TIERS)
    tier_id = np.digitize(nv_score, TIER_THRESHOLDS).astype(np.int64)
    tier_names = [t[0] for t in TIERS]
    tier_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#F44336"]

    print(f"Running {ITERS} NOFREEZE iterations tracking normals...")
    idxN_history, normal_history = lloyd_nofreeze_track_normals(
        S0, P0, F0, N0, treeP0, vf, ITERS
    )

    n = len(S0)

    # --- Compute frame_flip: how many times nearest vertex changed ---
    frame_flip = np.zeros(n, dtype=np.int64)
    for t in range(1, len(idxN_history)):
        frame_flip += (idxN_history[t] != idxN_history[t - 1]).astype(np.int64)

    # --- Compute mean normal angle change (degrees) ---
    angle_changes = []
    for t in range(1, len(normal_history)):
        N_prev = normal_history[t - 1]
        N_curr = normal_history[t]
        cos_ang = np.sum(N_prev * N_curr, axis=1)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        angle_changes.append(np.degrees(np.arccos(cos_ang)))
    angle_changes = np.stack(angle_changes, axis=0)  # (iters-1, n)
    mean_angle_change = np.mean(angle_changes, axis=0)  # per site

    # --- Print stats ---
    for t in range(n_tiers):
        mask = tier_id == t
        ff = frame_flip[mask]
        ac = mean_angle_change[mask]
        print(f"  {tier_names[t]:8s}: n={mask.sum():5d}  "
              f"frame_flip mean={ff.mean():.2f} median={np.median(ff):.1f} max={ff.max()}  |  "
              f"angle_change mean={ac.mean():.3f} deg  median={np.median(ac):.3f} deg  max={ac.max():.3f} deg")

    # --- CSV output ---
    import csv
    data_flip = [frame_flip[tier_id == t] for t in range(n_tiers)]
    data_angle = [mean_angle_change[tier_id == t] for t in range(n_tiers)]
    csv_path = os.path.join(out_dir, "exp2_normal_frame_stability.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n", "frame_flip_mean", "frame_flip_median", "frame_flip_std", "frame_flip_max",
                     "angle_change_mean", "angle_change_median", "angle_change_std", "angle_change_max"])
        for t in range(n_tiers):
            ff = data_flip[t]
            ac = data_angle[t]
            w.writerow([tier_names[t], len(ff),
                        f"{ff.mean():.4f}", f"{np.median(ff):.1f}", f"{ff.std():.4f}", int(ff.max()),
                        f"{ac.mean():.4f}", f"{np.median(ac):.4f}", f"{ac.std():.4f}", f"{ac.max():.4f}"])
    print(f"Saved: {csv_path}")

    # --- Figure ---
    x = np.arange(n_tiers)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment 2: Normal Frame Stability vs Curvature (5-Tier)",
                 fontsize=14, fontweight="bold")

    # (0,0) Bar: mean frame_flip per tier
    ax = axes[0, 0]
    flip_means = [d.mean() for d in data_flip]
    flip_stds = [d.std() for d in data_flip]
    ax.bar(x, flip_means, yerr=flip_stds, capsize=5, color=tier_colors, alpha=0.8,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel(f"Frame Flip Count (over {ITERS} iters)")
    ax.set_title("Mean Nearest-Vertex Flips by Tier")
    for i, (m, s) in enumerate(zip(flip_means, flip_stds)):
        ax.text(i, m + s + 0.3, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (0,1) Box plot: frame_flip by tier
    ax = axes[0, 1]
    bp = ax.boxplot(data_flip, labels=tier_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], tier_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel(f"Frame Flip Count (over {ITERS} iters)")
    ax.set_title("Frame Flip Distribution by Tier")
    ax.grid(True, alpha=0.3)

    # (1,0) Bar: mean angle change per tier
    ax = axes[1, 0]
    ang_means = [d.mean() for d in data_angle]
    ang_stds = [d.std() for d in data_angle]
    ax.bar(x, ang_means, yerr=ang_stds, capsize=5, color=tier_colors, alpha=0.8,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Mean Normal Angle Change (deg)")
    ax.set_title("Mean Normal Angle Change by Tier")
    for i, (m, s) in enumerate(zip(ang_means, ang_stds)):
        ax.text(i, m + s + 0.3, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (1,1) Box plot: angle change by tier
    ax = axes[1, 1]
    bp = ax.boxplot(data_angle, labels=tier_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], tier_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Mean Normal Angle Change (deg)")
    ax.set_title("Normal Angle Change Distribution by Tier")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "exp2_normal_frame_stability.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
