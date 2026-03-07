#!/usr/bin/env python3
"""
Experiment 9: Neighbor-Motion-Induced KNN Instability

Hypothesis H9: The Jaccard instability at high-curvature sites during their own
low-displacement moments is caused by the displacements of neighboring sites, not
the focal site itself.

Motivation (why Exp 7 cannot account for Exp 6a):
  Exp 7 showed that KNN boundary fragility (gap ratio d_{K+1}/d_K) is constant
  across all tiers (~1.014-1.017). Exp 7's proposed mechanism — "large displacements
  at curved cross the fragile boundary even during brief low-disp moments" — cannot
  explain Exp 6a's within-regime finding: within the low-focal-disp conditioning
  (disp_i < thr), the focal site IS moving slowly and the gap ratio is the same
  everywhere, so Exp 7 predicts no tier-dependent Jaccard instability. Yet Exp 6a
  shows 37% Jaccard failure at sharp vs 0.04% at flat, even within that conditioning.

H9 mechanism:
  At curved/sharp, sites oscillate persistently (Exp 3) with independent phases.
  When focal site i is in a momentary low-displacement phase, its K=32 neighbors
  (also at high curvature) are still actively oscillating. Those neighbor movements
  cross the Voronoi-like bisector boundary from outside i's position, changing i's
  KNN even without i itself moving. At flat, convergence is phase-correlated: when
  the focal site slows, its neighbors slow with it, so the KNN is stable.

Three predictions:
  P1. Within the low-focal-disp regime: mean displacement of i's K=32 neighbors
      increases monotonically with curvature tier.
  P2. Spearman r(mean_neighbor_disp, Jaccard_i | disp_i < thr) is negative
      (more neighbor motion → lower Jaccard), and grows in magnitude with tier.
  P3. Spearman r(focal_disp, mean_neighbor_disp | disp_i < thr) approaches 0
      at curved/sharp (phase decorrelation: focal can be still while neighbors move).
"""

import numpy as np
import os
import sys
import csv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import trimesh
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from testfreeze import (
    nr, tangent, normal_variation_score, closest_point_tri,
    cell_poly2d, poly_area_centroid_2d, jaccard_sorted_int,
    NV_K, TIER_THRESHOLDS, TIERS,
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ITERS     = 60
K_NEIGH   = 32
K_PROJ    = 10
SCATTER_N = 2000   # max scatter points per tier per panel


# ---------------------------------------------------------------------------
# Geometry helpers (same as other exps)
# ---------------------------------------------------------------------------

def update_to_mesh(C, tris, P):
    if tris.shape[0] == 0 or not np.isfinite(C).all():
        return C
    best = None
    best_d2 = np.inf
    for a, b, c in tris:
        A = P[a].astype(np.float64)
        B = P[b].astype(np.float64)
        D = P[c].astype(np.float64)
        q = closest_point_tri(C, A, B, D)
        d2 = float((q - C) @ (q - C))
        if d2 < best_d2:
            best_d2 = d2
            best = q
    return best if best is not None else C


def lloyd_nofreeze_track_all(S0, P0, F0, N0, treeP0, vf, iters):
    """Lloyd without freezing; returns disp_history (n, iters) and knn_history."""
    n = len(S0)
    S = S0.copy()
    disp_history = np.zeros((n, iters), dtype=np.float64)
    knn_history  = []   # list of (n, K_NEIGH) sorted int arrays

    for it in range(iters):
        _, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64), k=1, workers=-1)
        N = N0[idxN]
        U, V = tangent(N)

        treeS = cKDTree(np.ascontiguousarray(S, dtype=np.float64))
        _, knn = treeS.query(S, k=K_NEIGH + 1, workers=-1)
        knn_sorted = np.sort(knn[:, 1:], axis=1)
        knn_history.append(knn_sorted.copy())

        bbox = S.max(axis=0) - S.min(axis=0)
        R = float(np.max(bbox))
        cent3d = np.full_like(S, np.nan, dtype=np.float64)
        for i in range(n):
            poly = cell_poly2d(i, S, U, V, knn_sorted[i], R)
            _, c2 = poly_area_centroid_2d(poly)
            if np.isfinite(c2).all():
                cent3d[i] = S[i] + c2[0] * U[i] + c2[1] * V[i]

        mask = np.isfinite(cent3d).all(axis=1)
        Snew = S.copy()
        if np.any(mask):
            _, idxC = treeP0.query(
                np.ascontiguousarray(cent3d[mask], dtype=np.float64), k=K_PROJ, workers=-1
            )
            for t_idx, i in enumerate(np.flatnonzero(mask)):
                nn = idxC[t_idx]
                faces_set = set()
                for v in nn:
                    for fi in vf[int(v)]:
                        faces_set.add(fi)
                face_idx = np.fromiter(faces_set, dtype=np.int64)
                tris = F0[face_idx] if face_idx.size else np.empty((0, 3), dtype=np.int64)
                Snew[i] = update_to_mesh(cent3d[i], tris, P0)

        disp_history[:, it] = np.linalg.norm(Snew - S, axis=1)
        S = Snew
        print(f"  iter {it+1}/{iters}")

    return disp_history, knn_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mesh", nargs="?", default=os.path.join(_ROOT, "meshes", "teapot.obj"))
    args = p.parse_args()
    in_path = args.mesh
    objname = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = os.path.join(_ROOT, "results", objname, "exp9_neighbor_motion_induction")
    os.makedirs(out_dir, exist_ok=True)

    m = trimesh.load(in_path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    P0     = np.asarray(m.vertices,       dtype=np.float64)
    F0     = np.asarray(m.faces,          dtype=np.int64)
    N0     = np.asarray(m.vertex_normals, dtype=np.float64)
    treeP0 = cKDTree(np.ascontiguousarray(P0, dtype=np.float64))

    vf = [[] for _ in range(len(P0))]
    for fi, (a, b, c) in enumerate(F0):
        vf[int(a)].append(fi)
        vf[int(b)].append(fi)
        vf[int(c)].append(fi)

    S0 = P0.copy()
    n  = len(S0)

    nv_score          = normal_variation_score(P0, nr(N0), NV_K)
    n_tiers           = len(TIERS)
    tier_id           = np.digitize(nv_score, TIER_THRESHOLDS).astype(np.int64)
    tier_names        = [t[0] for t in TIERS]
    tier_colors       = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#F44336"]
    disp_thr_per_tier = np.array([t[1] for t in TIERS], dtype=np.float64)

    print(f"Running {ITERS} iterations (no freeze)...")
    disp_history, knn_history = lloyd_nofreeze_track_all(
        S0, P0, F0, N0, treeP0, vf, ITERS
    )

    # =========================================================================
    # Exp 9: For each (site i, iter t) where disp_i,t < disp_thr[tier_id[i]]:
    #   1. focal_disp     = disp_i,t
    #   2. mean_nbr_disp  = mean displacement of i's K=32 neighbors at iter t
    #   3. jaccard_it     = Jaccard(knn_{t-1}[i], knn_t[i])
    #
    # Per tier compute:
    #   A. mean / std of mean_nbr_disp  [P1: should increase with tier]
    #   B. Spearman r(mean_nbr_disp, jaccard_it)  [P2: should be negative, stronger at curved]
    #   C. Spearman r(focal_disp, mean_nbr_disp)  [P3: should approach 0 at curved]
    # =========================================================================
    print("\n--- Exp 9: Neighbor-motion conditioning (within low-focal-disp regime) ---")

    # Accumulators per tier
    rec_focal  = [[] for _ in range(n_tiers)]   # focal_disp
    rec_nbr    = [[] for _ in range(n_tiers)]   # mean_nbr_disp
    rec_jacc   = [[] for _ in range(n_tiers)]   # jaccard_it

    for it in range(1, ITERS):
        prev_knn = knn_history[it - 1]   # (n, K_NEIGH) sorted
        curr_knn = knn_history[it]       # (n, K_NEIGH) sorted
        disp     = disp_history[:, it]   # (n,)

        thr = disp_thr_per_tier[tier_id]   # (n,) threshold per site

        # low-focal-disp mask
        low_idx = np.flatnonzero(disp < thr)

        for i in low_idx:
            t = int(tier_id[i])

            # Jaccard of focal site
            j_val = jaccard_sorted_int(prev_knn[i], curr_knn[i])

            # Mean displacement of focal site's K=32 neighbors
            nbrs = curr_knn[i]   # sorted neighbor indices
            nbr_disp = disp[nbrs].mean()

            rec_focal[t].append(float(disp[i]))
            rec_nbr[t].append(float(nbr_disp))
            rec_jacc[t].append(float(j_val))

    # ----- Per-tier statistics -----
    stats = []
    hdr = (f"  {'Tier':>10s}  {'n':>7s}  "
           f"{'mean_nbr_d':>11s}  {'std_nbr_d':>10s}  "
           f"{'r(nbr→J)':>10s}  p_nbr_J  "
           f"{'r(foc→nbr)':>11s}  p_foc_nbr")
    print(hdr)
    for t in range(n_tiers):
        fd = np.array(rec_focal[t])
        nd = np.array(rec_nbr[t])
        jd = np.array(rec_jacc[t])
        nn = len(fd)

        if nn < 5:
            stats.append(dict(n=nn, mean_nbr=float("nan"), std_nbr=float("nan"),
                              r_nbr_j=float("nan"), p_nbr_j=float("nan"),
                              r_foc_nbr=float("nan"), p_foc_nbr=float("nan")))
            print(f"  {tier_names[t]:>10s}  {nn:>7d}  (insufficient data)")
            continue

        # P2: r(mean_neighbor_disp, jaccard)
        r_nj, p_nj = spearmanr(nd, jd)
        # P3: r(focal_disp, mean_neighbor_disp)
        r_fn, p_fn = spearmanr(fd, nd)

        stats.append(dict(
            n=nn,
            mean_nbr=float(nd.mean()),
            std_nbr=float(nd.std()),
            r_nbr_j=float(r_nj), p_nbr_j=float(p_nj),
            r_foc_nbr=float(r_fn), p_foc_nbr=float(p_fn),
        ))
        print(f"  {tier_names[t]:>10s}  {nn:>7d}  "
              f"{nd.mean():>11.5f}  {nd.std():>10.5f}  "
              f"{r_nj:>+10.4f}  {p_nj:.2e}  "
              f"{r_fn:>+11.4f}  {p_fn:.2e}")

    # ----- CSV -----
    csv_path = os.path.join(out_dir, "exp9_neighbor_motion_induction.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n_obs",
                    "mean_nbr_disp", "std_nbr_disp",
                    "spearman_r_nbr_to_jaccard", "p_nbr_j",
                    "spearman_r_focal_to_nbr", "p_foc_nbr"])
        for t in range(n_tiers):
            s = stats[t]
            w.writerow([tier_names[t], s["n"],
                        f"{s['mean_nbr']:.6f}", f"{s['std_nbr']:.6f}",
                        f"{s['r_nbr_j']:+.4f}", f"{s['p_nbr_j']:.2e}",
                        f"{s['r_foc_nbr']:+.4f}", f"{s['p_foc_nbr']:.2e}"])
    print(f"\nSaved: {csv_path}")

    # ----- Figures -----
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"Experiment 9: Neighbor-Motion-Induced KNN Instability — {objname}\n"
        "H9: Jaccard fails at high-curvature sites during their own low-disp moments\n"
        "because neighbors are still actively oscillating, reshaping KNN from outside",
        fontsize=12, fontweight="bold"
    )
    gs = GridSpec(3, 5, figure=fig, hspace=0.58, wspace=0.38)

    # --- Row 0: scatter (mean_nbr_disp vs Jaccard) within low-focal-disp regime ---
    for t in range(n_tiers):
        ax = fig.add_subplot(gs[0, t])
        nd = np.array(rec_nbr[t])
        jd = np.array(rec_jacc[t])
        if len(nd) > SCATTER_N:
            idx = np.random.default_rng(t).choice(len(nd), SCATTER_N, replace=False)
            nd, jd = nd[idx], jd[idx]
        nd = np.maximum(nd, 1e-10)
        ax.scatter(nd, jd, s=2, alpha=0.15, color=tier_colors[t], rasterized=True)
        s = stats[t]
        r_label = f"r={s['r_nbr_j']:+.3f}" if np.isfinite(s['r_nbr_j']) else "r=n/a"
        ax.set_title(f"{tier_names[t]}\n{r_label}  n={s['n']}", fontsize=9)
        ax.set_xlabel("mean_nbr_disp (log)", fontsize=7)
        ax.set_ylabel("Jaccard(focal)", fontsize=7)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.08)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    # --- Row 1: scatter (focal_disp vs mean_nbr_disp) within low-focal-disp regime ---
    for t in range(n_tiers):
        ax = fig.add_subplot(gs[1, t])
        fd = np.array(rec_focal[t])
        nd = np.array(rec_nbr[t])
        if len(fd) > SCATTER_N:
            idx = np.random.default_rng(t + 10).choice(len(fd), SCATTER_N, replace=False)
            fd, nd = fd[idx], nd[idx]
        fd = np.maximum(fd, 1e-10)
        nd = np.maximum(nd, 1e-10)
        ax.scatter(fd, nd, s=2, alpha=0.15, color=tier_colors[t], rasterized=True)
        s = stats[t]
        r_label = f"r={s['r_foc_nbr']:+.3f}" if np.isfinite(s['r_foc_nbr']) else "r=n/a"
        ax.set_title(f"{tier_names[t]}\n{r_label}", fontsize=9)
        ax.set_xlabel("focal_disp (log)", fontsize=7)
        ax.set_ylabel("mean_nbr_disp (log)", fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    # --- Row 2, cols 0-1: P1 — Mean neighbor disp per tier (within low-focal-disp) ---
    ax = fig.add_subplot(gs[2, 0:2])
    x = np.arange(n_tiers)
    means = [stats[t]["mean_nbr"] for t in range(n_tiers)]
    stds  = [stats[t]["std_nbr"]  for t in range(n_tiers)]
    ax.bar(x, means, yerr=stds, capsize=5, color=tier_colors, alpha=0.85,
           edgecolor="black", linewidth=0.5)
    for i, (m, s_) in enumerate(zip(means, stds)):
        if np.isfinite(m):
            ax.text(i, m + s_ + 1e-6, f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Mean neighbor displacement\n(conditioned on disp_focal < thr)")
    ax.set_title(
        "P1 (H9): Do neighbors move more at high curvature\n"
        "even when focal site has low disp?\n"
        "[Prediction: monotone increase → confirms neighbor-motion induction]"
    )
    ax.grid(True, alpha=0.3, axis="y")

    # --- Row 2, cols 2-3: P2 + P3 — Spearman r summary ---
    ax = fig.add_subplot(gs[2, 2:4])
    r_nbr_j  = [stats[t]["r_nbr_j"]  for t in range(n_tiers)]
    r_foc_nb = [stats[t]["r_foc_nbr"] for t in range(n_tiers)]
    width = 0.35
    ax.bar(x - width/2, r_nbr_j, width, color=tier_colors, alpha=0.85,
           edgecolor="black", linewidth=0.5,
           label="r(nbr_disp→Jaccard) [P2]")
    ax.bar(x + width/2, r_foc_nb, width, color=tier_colors, alpha=0.4,
           edgecolor="black", linewidth=0.5, hatch="//",
           label="r(focal_disp→nbr_disp) [P3]")
    ax.axhline(0, color="black", linewidth=0.7)
    for i, (rn, rf) in enumerate(zip(r_nbr_j, r_foc_nb)):
        if np.isfinite(rn):
            ax.text(i - width/2, rn - 0.01, f"{rn:+.3f}", ha="center", va="top", fontsize=7)
        if np.isfinite(rf):
            ax.text(i + width/2, rf + 0.01, f"{rf:+.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Spearman r")
    ax.set_title(
        "P2: r(nbr_disp, Jaccard) should be < 0 and grow in magnitude with tier\n"
        "P3: r(focal_disp, nbr_disp) should → 0 at curved/sharp\n"
        "[Both confirm H9 phase-decorrelation mechanism]"
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Row 2, col 4: fraction of (focal, iter) where nbr_disp > focal_disp ---
    ax = fig.add_subplot(gs[2, 4])
    frac_nbr_gt = []
    for t in range(n_tiers):
        fd = np.array(rec_focal[t])
        nd = np.array(rec_nbr[t])
        if len(fd) > 0:
            frac_nbr_gt.append(float(np.mean(nd > fd)))
        else:
            frac_nbr_gt.append(float("nan"))
    ax.bar(x, frac_nbr_gt, color=tier_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(frac_nbr_gt):
        if np.isfinite(v):
            ax.text(i, v + 0.01, f"{v:.2%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Frac(mean_nbr_disp > focal_disp)")
    ax.set_title("Fraction of low-focal-disp\nevents where neighbors\nmove more than focal site")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    out_png = os.path.join(out_dir, "exp9_neighbor_motion_induction.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close()

    # =========================================================================
    # Bonus: verify H9 against exp 6a baseline by checking whether adding
    # mean_nbr_disp as a regressor improves prediction of Jaccard within low-disp.
    # Simple: compare r(focal_disp, Jaccard) vs r(mean_nbr_disp, Jaccard) per tier.
    # =========================================================================
    print("\n--- H9 check: r(focal_disp → Jaccard) vs r(nbr_disp → Jaccard) within low-disp ---")
    print(f"  {'Tier':>10s}  {'r(focal→J)':>11s}  {'r(nbr→J)':>10s}  "
          f"  Δr (nbr adds signal beyond focal?)")
    for t in range(n_tiers):
        fd = np.array(rec_focal[t])
        nd = np.array(rec_nbr[t])
        jd = np.array(rec_jacc[t])
        if len(fd) < 5:
            print(f"  {tier_names[t]:>10s}  (n={len(fd)}, skip)")
            continue
        r_fj, _ = spearmanr(fd, jd)
        r_nj, _ = spearmanr(nd, jd)
        delta    = abs(r_nj) - abs(r_fj)
        print(f"  {tier_names[t]:>10s}  {r_fj:>+11.4f}  {r_nj:>+10.4f}  "
              f"  Δ={delta:+.4f} ({'nbr stronger' if delta > 0 else 'focal stronger'})")


if __name__ == "__main__":
    main()
