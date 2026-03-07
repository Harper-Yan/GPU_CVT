#!/usr/bin/env python3
"""
Experiment 4: Reprojection Triangle Stability
Tracks which mesh triangle each site's centroid projects onto per iteration.
Measures how often the projected triangle flips, grouped by curvature tier.
"""
import numpy as np
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import trimesh
from scipy.spatial import cKDTree
from testfreeze import (
    nr, tangent, normal_variation_score, closest_point_tri,
    cell_poly2d, poly_area_centroid_2d,
    NV_K, TIER_THRESHOLDS, TIERS,
)
import matplotlib.pyplot as plt

ITERS = 50
K_NEIGH = 32
K_PROJ = 10


def update_to_mesh_track_tri(C, tris_idx, F_global_idx, P):
    """Like update_to_mesh but also returns the global face index of the winning triangle."""
    if tris_idx.shape[0] == 0 or not np.isfinite(C).all():
        return C, -1
    best = None
    best_d2 = np.inf
    best_fi = -1
    for local_t, (a, b, c) in enumerate(tris_idx):
        A = P[a].astype(np.float64)
        B = P[b].astype(np.float64)
        D = P[c].astype(np.float64)
        q = closest_point_tri(C, A, B, D)
        d = q - C
        d2 = float(d @ d)
        if d2 < best_d2:
            best_d2 = d2
            best = q
            best_fi = int(F_global_idx[local_t])
    return (best if best is not None else C), best_fi


def lloyd_nofreeze_track_triangles(S0, P0, F0, N0, treeP0, vf, iters):
    """Run Lloyd without freezing, record projected triangle index and face candidate count per site per iteration."""
    n = len(S0)
    S = S0.copy()
    tri_history = []         # list of (n,) int arrays
    face_count_history = []  # list of (n,) int arrays: number of candidate faces per site

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
        proj_tri = np.full(n, -1, dtype=np.int64)
        face_count = np.zeros(n, dtype=np.int64)

        if np.any(mask):
            _, idxC = treeP0.query(np.ascontiguousarray(cent3d[mask], dtype=np.float64),
                                   k=K_PROJ, workers=-1)
            for t_idx, i in enumerate(np.flatnonzero(mask)):
                nn = idxC[t_idx]
                faces_set = set()
                for v in nn:
                    for fi in vf[int(v)]:
                        faces_set.add(fi)
                face_indices = np.fromiter(faces_set, dtype=np.int64)
                face_count[i] = len(face_indices)
                tris = F0[face_indices] if face_indices.size else np.empty((0, 3), dtype=np.int64)
                Snew[i], proj_tri[i] = update_to_mesh_track_tri(
                    cent3d[i], tris, face_indices, P0
                )

        tri_history.append(proj_tri.copy())
        face_count_history.append(face_count.copy())
        S = Snew
        print(f"  iter {it+1}/{iters}")

    return tri_history, face_count_history


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mesh", nargs="?", default=os.path.join(_ROOT, "meshes", "teapot.obj"))
    args = p.parse_args()
    in_path = args.mesh
    objname = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = os.path.join(_ROOT, "results", objname, "exp4_reprojection_stability")
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

    print(f"Running {ITERS} NOFREEZE iterations tracking projection triangles...")
    tri_history, face_count_history = lloyd_nofreeze_track_triangles(S0, P0, F0, N0, treeP0, vf, ITERS)

    n = len(S0)

    # --- Compute mean face candidate count per site ---
    fc_stack = np.stack(face_count_history, axis=0)  # (iters, n)
    # Only count iterations where the site had a valid centroid (face_count > 0)
    fc_valid = np.where(fc_stack > 0, fc_stack, np.nan)
    mean_face_count = np.nanmean(fc_valid, axis=0)

    # --- Compute tri_flip: how many times projected triangle changed ---
    tri_flip = np.zeros(n, dtype=np.int64)
    valid_pairs = 0
    for t in range(1, len(tri_history)):
        prev = tri_history[t - 1]
        curr = tri_history[t]
        # Only count when both are valid (>= 0)
        both_valid = (prev >= 0) & (curr >= 0)
        tri_flip += (both_valid & (prev != curr)).astype(np.int64)

    # --- Compute unique triangle count per site ---
    tri_stack = np.stack(tri_history, axis=0)  # (iters, n)
    unique_tri_count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        vals = tri_stack[:, i]
        vals = vals[vals >= 0]
        unique_tri_count[i] = len(np.unique(vals))

    # --- Print stats ---
    print(f"\nFace candidate count per tier (A1 — constrains centroid search space):")
    for t in range(n_tiers):
        mask = tier_id == t
        fc = mean_face_count[mask]
        fc = fc[np.isfinite(fc)]
        print(f"  {tier_names[t]:8s}: n={mask.sum():5d}  "
              f"mean_face_count={fc.mean():.1f}  median={np.median(fc):.1f}  std={fc.std():.1f}")

    print(f"\nTriangle flip and unique triangles per tier:")
    for t in range(n_tiers):
        mask = tier_id == t
        tf = tri_flip[mask]
        utc = unique_tri_count[mask]
        print(f"  {tier_names[t]:8s}: n={mask.sum():5d}  "
              f"tri_flip mean={tf.mean():.2f} median={np.median(tf):.1f} max={tf.max()}  |  "
              f"unique_tris mean={utc.mean():.2f} median={np.median(utc):.1f} max={utc.max()}")

    # --- CSV output ---
    import csv
    csv_path = os.path.join(out_dir, "exp4_reprojection_stability.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n", "tri_flip_mean", "tri_flip_median", "tri_flip_std", "tri_flip_max",
                     "unique_tris_mean", "unique_tris_median", "unique_tris_std", "unique_tris_max",
                     "mean_face_candidates"])
        for t in range(n_tiers):
            mask = tier_id == t
            tf = tri_flip[mask]
            utc = unique_tri_count[mask]
            fc = mean_face_count[mask]
            fc_vals = fc[np.isfinite(fc)]
            w.writerow([tier_names[t], int(mask.sum()),
                        f"{tf.mean():.4f}", f"{np.median(tf):.1f}", f"{tf.std():.4f}", int(tf.max()),
                        f"{utc.mean():.4f}", f"{np.median(utc):.1f}", f"{utc.std():.4f}", int(utc.max()),
                        f"{fc_vals.mean():.2f}" if len(fc_vals) > 0 else "nan"])
    print(f"Saved: {csv_path}")

    # --- Figure ---
    data_flip = [tri_flip[tier_id == t] for t in range(n_tiers)]
    data_utc = [unique_tri_count[tier_id == t] for t in range(n_tiers)]
    x = np.arange(n_tiers)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment 4: Reprojection Triangle Stability vs Curvature (5-Tier)",
                 fontsize=14, fontweight="bold")

    # (0,0) Bar: mean tri_flip per tier
    ax = axes[0, 0]
    flip_means = [d.mean() for d in data_flip]
    flip_stds = [d.std() for d in data_flip]
    ax.bar(x, flip_means, yerr=flip_stds, capsize=5, color=tier_colors, alpha=0.8,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel(f"Triangle Flip Count (over {ITERS} iters)")
    ax.set_title("Mean Triangle Flips by Tier")
    for i, (m, s) in enumerate(zip(flip_means, flip_stds)):
        ax.text(i, m + s + 0.3, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (0,1) Box plot: tri_flip by tier
    ax = axes[0, 1]
    bp = ax.boxplot(data_flip, labels=tier_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], tier_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel(f"Triangle Flip Count (over {ITERS} iters)")
    ax.set_title("Triangle Flip Distribution by Tier")
    ax.grid(True, alpha=0.3)

    # (1,0) Bar: mean unique triangles per tier
    ax = axes[1, 0]
    utc_means = [d.mean() for d in data_utc]
    utc_stds = [d.std() for d in data_utc]
    ax.bar(x, utc_means, yerr=utc_stds, capsize=5, color=tier_colors, alpha=0.8,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Unique Triangles Visited")
    ax.set_title("Mean Unique Triangles by Tier")
    for i, (m, s) in enumerate(zip(utc_means, utc_stds)):
        ax.text(i, m + s + 0.2, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (1,1) Box plot: unique triangles by tier
    ax = axes[1, 1]
    bp = ax.boxplot(data_utc, labels=tier_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], tier_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Unique Triangles Visited")
    ax.set_title("Unique Triangles Distribution by Tier")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "exp4_reprojection_stability.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
