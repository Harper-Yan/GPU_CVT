#!/usr/bin/env python3
"""
Experiment 11: Oscillation Predictor — Curvature Anisotropy vs NV Score

Two questions:
  Q1. Does the Type A (position-trapped) / Type B (oscillating) split exist in
      other tiers beyond sharp, or is it sharp-specific?

  Q2. Is there a better predictor of site oscillation than the NV score?
      The NV score measures the *magnitude* of local normal variation (isotropically).
      It cannot distinguish:
        - Radially symmetric normal divergence at a peak/dome   → Type A (stable)
        - Transverse normal flip across a ridge/crease          → Type B (oscillating)
      Both can produce identical NV scores while having opposite dynamics.

Proposed predictor — Normal Covariance Linearity (L):
  For each site, compute the 3×3 covariance matrix of its K neighbor normals.
  Eigenvalues λ₁ ≥ λ₂ ≥ λ₃.
    L = (λ₁ - λ₂) / λ₁   ← one dominant direction  → ridge → Type B
    P = (λ₂ - λ₃) / λ₁   ← spread in a plane
    S = λ₃ / λ₁           ← isotropic → dome/tip   → Type A

Predictions:
  P_Q1: Within moderate and curved tiers, K-means on (frac_low_disp, mean_disp)
        should reveal a bimodal split — but narrower than at sharp.

  P_Q2a: Within the sharp tier, Type A sites (from Exp 10) have lower L (more
          isotropic normal spread) than Type B (more linear, ridge-like).

  P_Q2b: Across all sites, Spearman r(L, mean_disp) > r(NV, mean_disp) in absolute
          value — L predicts oscillation magnitude better than NV alone.

  P_Q2c: A combined predictor (NV × L) — capturing both magnitude and anisotropy —
          should further improve prediction.
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from testfreeze import (
    nr, tangent, normal_variation_score, closest_point_tri,
    cell_poly2d, poly_area_centroid_2d,
    NV_K, TIER_THRESHOLDS, TIERS,
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ITERS   = 60
K_NEIGH = 32
K_PROJ  = 10
ANISO_K = 24    # neighbors used for normal covariance (same as NV_K)


# ---------------------------------------------------------------------------
# Normal covariance anisotropy features
# ---------------------------------------------------------------------------

def normal_covariance_features(meshV: np.ndarray, meshN: np.ndarray,
                                k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each vertex, compute eigenvalues of the k-neighbor normal covariance.
    Returns L (linearity), P (planarity), S (sphericity) arrays of shape (n,).
    """
    tree = cKDTree(np.ascontiguousarray(meshV, dtype=np.float64))
    _, idx = tree.query(meshV, k=k + 1, workers=-1)
    idx = idx[:, 1:]                          # (n, k) — exclude self

    N = meshN.astype(np.float64)              # (n, 3) unit normals
    n = len(meshV)
    L = np.zeros(n, dtype=np.float64)
    P = np.zeros(n, dtype=np.float64)
    S = np.zeros(n, dtype=np.float64)

    for i in range(n):
        nbr_normals = N[idx[i]]               # (k, 3)
        mean_n = nbr_normals.mean(axis=0)
        centered = nbr_normals - mean_n       # (k, 3)
        cov = (centered.T @ centered) / k    # (3, 3)
        eigvals = np.linalg.eigvalsh(cov)    # ascending order
        eigvals = np.sort(eigvals)[::-1]     # descending: λ₁ ≥ λ₂ ≥ λ₃
        eigvals = np.maximum(eigvals, 0.0)
        denom = eigvals[0] if eigvals[0] > 1e-15 else 1e-15
        L[i] = (eigvals[0] - eigvals[1]) / denom
        P[i] = (eigvals[1] - eigvals[2]) / denom
        S[i] = eigvals[2] / denom

    return L, P, S


# ---------------------------------------------------------------------------
# Lloyd geometry helpers
# ---------------------------------------------------------------------------

def update_to_mesh(C, tris, P):
    if tris.shape[0] == 0 or not np.isfinite(C).all():
        return C
    best, best_d2 = None, np.inf
    for a, b, c in tris:
        q = closest_point_tri(C, P[a].astype(np.float64),
                              P[b].astype(np.float64), P[c].astype(np.float64))
        d2 = float((q - C) @ (q - C))
        if d2 < best_d2:
            best_d2, best = d2, q
    return best if best is not None else C


def lloyd_nofreeze_track_positions(S0, P0, F0, N0, treeP0, vf, iters):
    n = len(S0)
    S = S0.copy()
    S_hist = [S.copy()]

    for it in range(iters):
        _, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64), k=1, workers=-1)
        N = N0[idxN]
        U, V = tangent(N)

        treeS = cKDTree(np.ascontiguousarray(S, dtype=np.float64))
        _, knn = treeS.query(S, k=K_NEIGH + 1, workers=-1)
        knn_sorted = np.sort(knn[:, 1:], axis=1)

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
                np.ascontiguousarray(cent3d[mask], dtype=np.float64), k=K_PROJ, workers=-1)
            for t_idx, i in enumerate(np.flatnonzero(mask)):
                nn = idxC[t_idx]
                faces_set = set()
                for v in nn:
                    for fi in vf[int(v)]:
                        faces_set.add(fi)
                face_idx = np.fromiter(faces_set, dtype=np.int64)
                tris_sub = F0[face_idx] if face_idx.size else np.empty((0, 3), dtype=np.int64)
                Snew[i] = update_to_mesh(cent3d[i], tris_sub, P0)

        S = Snew
        S_hist.append(S.copy())
        print(f"  iter {it+1}/{iters}")

    return np.stack(S_hist, axis=0)   # (iters+1, n, 3)


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
    out_dir = os.path.join(_ROOT, "results", objname, "exp11_oscillation_predictor")
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

    # ── Tier assignment ────────────────────────────────────────────────────
    nv_score     = normal_variation_score(P0, nr(N0), NV_K)   # existing metric
    tier_id      = np.digitize(nv_score, TIER_THRESHOLDS).astype(np.int64)
    tier_names   = [t[0] for t in TIERS]
    tier_colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#F44336"]
    disp_thr_arr = np.array([t[1] for t in TIERS], dtype=np.float64)

    # ── Anisotropy features (new metric) ──────────────────────────────────
    print("Computing normal covariance anisotropy features...")
    L, Pl, Sp = normal_covariance_features(P0, nr(N0), ANISO_K)
    # L = linearity (ridge-like), Pl = planarity, Sp = sphericity (dome-like)

    # ── Lloyd 60 iterations ───────────────────────────────────────────────
    print(f"\nRunning {ITERS} NOFREEZE iterations...")
    S_hist = lloyd_nofreeze_track_positions(S0, P0, F0, N0, treeP0, vf, ITERS)

    d_vecs   = np.diff(S_hist, axis=0)           # (ITERS, n, 3)
    d_mags   = np.linalg.norm(d_vecs, axis=2)    # (ITERS, n)
    mean_disp = d_mags.mean(axis=0)              # (n,) — primary oscillation metric
    frac_low  = np.array([
        (d_mags[:, i] < disp_thr_arr[tier_id[i]]).mean() for i in range(n)
    ])

    # =========================================================================
    # Q1: Bimodal split across all tiers
    # K-means(k=2) per tier on (frac_low_disp, mean_disp)
    # =========================================================================
    print("\n--- Q1: Bimodal split by tier (K-means k=2) ---")
    tier_km_labels  = np.full(n, -1, dtype=np.int64)   # -1 = tier too small
    tier_km_typeA_n = np.zeros(len(TIERS), dtype=int)
    tier_km_typeB_n = np.zeros(len(TIERS), dtype=int)
    tier_km_typeA_fl= [0.0]*len(TIERS)
    tier_km_typeB_fl= [0.0]*len(TIERS)
    tier_km_typeA_md= [0.0]*len(TIERS)
    tier_km_typeB_md= [0.0]*len(TIERS)
    tier_km_sep     = [0.0]*len(TIERS)   # separation of cluster centroids in frac_low_disp

    print(f"  {'Tier':>10s}  {'n':>5s}  {'nA':>5s}  {'nB':>5s}  "
          f"{'A_frac_low':>11s}  {'B_frac_low':>11s}  "
          f"{'A_mean_d':>9s}  {'B_mean_d':>9s}  {'sep_frac_low':>13s}")
    for t in range(len(TIERS)):
        idx_t = np.flatnonzero(tier_id == t)
        if len(idx_t) < 4:
            print(f"  {tier_names[t]:>10s}  n={len(idx_t)} — skip")
            continue

        X = np.column_stack([frac_low[idx_t], mean_disp[idx_t]])
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=2, random_state=42, n_init=20)
        lbl = km.fit_predict(Xs)

        # Label A = higher frac_low cluster
        if frac_low[idx_t[lbl == 0]].mean() < frac_low[idx_t[lbl == 1]].mean():
            lbl = 1 - lbl   # flip so 0=A

        nA = int((lbl == 0).sum())
        nB = int((lbl == 1).sum())
        flA = float(frac_low[idx_t[lbl == 0]].mean())
        flB = float(frac_low[idx_t[lbl == 1]].mean())
        mdA = float(mean_disp[idx_t[lbl == 0]].mean())
        mdB = float(mean_disp[idx_t[lbl == 1]].mean())
        sep = flA - flB

        tier_km_typeA_n[t] = nA; tier_km_typeB_n[t] = nB
        tier_km_typeA_fl[t] = flA; tier_km_typeB_fl[t] = flB
        tier_km_typeA_md[t] = mdA; tier_km_typeB_md[t] = mdB
        tier_km_sep[t] = sep

        # store labels globally
        for pos, gidx in enumerate(idx_t):
            tier_km_labels[gidx] = int(lbl[pos])

        print(f"  {tier_names[t]:>10s}  {len(idx_t):>5d}  {nA:>5d}  {nB:>5d}  "
              f"{flA:>11.4f}  {flB:>11.4f}  "
              f"{mdA:>9.5f}  {mdB:>9.5f}  {sep:>13.4f}")

    # =========================================================================
    # Q2a: Within sharp tier — do Type A and B differ in anisotropy (L)?
    # =========================================================================
    sharp_idx = np.flatnonzero(tier_id == 4)
    if len(sharp_idx) >= 4:
        sharp_lbl = tier_km_labels[sharp_idx]
        maskA = sharp_lbl == 0
        maskB = sharp_lbl == 1

        print("\n--- Q2a: Normal covariance linearity within sharp tier ---")
        print(f"  {'':30s}  {'Type A (singularity)':>22s}  {'Type B (ridge)':>18s}")
        for label, arr in [
            ("Linearity L (ridge → high)",   L[sharp_idx]),
            ("Planarity P",                  Pl[sharp_idx]),
            ("Sphericity S (dome → high)",   Sp[sharp_idx]),
            ("NV score",                     nv_score[sharp_idx]),
            ("mean_disp",                    mean_disp[sharp_idx]),
            ("frac_low_disp",                frac_low[sharp_idx]),
        ]:
            def ms(a, m):
                v = a[m]; return f"{v.mean():.4f} ± {v.std():.4f}"
            print(f"  {label:30s}  {ms(arr, maskA):>22s}  {ms(arr, maskB):>18s}")

    # =========================================================================
    # Q2b: Global — r(L, mean_disp) vs r(NV, mean_disp)
    # Spearman rank correlations as predictors of oscillation
    # =========================================================================
    print("\n--- Q2b: Spearman r predicting mean_disp — L vs NV vs combined ---")
    print(f"  {'Tier':>10s}  {'n':>5s}  {'r(NV→disp)':>12s}  "
          f"{'r(L→disp)':>11s}  {'r(NV×L→disp)':>14s}  best")
    NV_x_L = nv_score * L   # combined predictor

    for t in range(len(TIERS)):
        idx_t = np.flatnonzero(tier_id == t)
        if len(idx_t) < 5:
            continue
        md_t  = mean_disp[idx_t]
        nv_t  = nv_score[idx_t]
        l_t   = L[idx_t]
        nl_t  = NV_x_L[idx_t]

        r_nv, _ = spearmanr(nv_t, md_t)
        r_l,  _ = spearmanr(l_t,  md_t)
        r_nl, _ = spearmanr(nl_t, md_t)

        best = max([("NV", abs(r_nv)), ("L", abs(r_l)), ("NV×L", abs(r_nl))],
                   key=lambda x: x[1])[0]
        print(f"  {tier_names[t]:>10s}  {len(idx_t):>5d}  "
              f"{r_nv:>+12.4f}  {r_l:>+11.4f}  {r_nl:>+14.4f}  {best}")

    print("\n  [Across all sites]")
    r_nv_all, _ = spearmanr(nv_score, mean_disp)
    r_l_all,  _ = spearmanr(L,        mean_disp)
    r_nl_all, _ = spearmanr(NV_x_L,   mean_disp)
    print(f"  {'all':>10s}  {n:>5d}  "
          f"{r_nv_all:>+12.4f}  {r_l_all:>+11.4f}  {r_nl_all:>+14.4f}")

    # =========================================================================
    # CSV output
    # =========================================================================
    csv_path = os.path.join(out_dir, "exp11_per_site.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_idx", "tier", "nv_score",
                    "L_linearity", "P_planarity", "S_sphericity",
                    "mean_disp", "frac_low_disp", "km_label"])
        for i in range(n):
            w.writerow([i, tier_names[int(tier_id[i])],
                        f"{nv_score[i]:.6f}",
                        f"{L[i]:.6f}", f"{Pl[i]:.6f}", f"{Sp[i]:.6f}",
                        f"{mean_disp[i]:.6f}", f"{frac_low[i]:.6f}",
                        int(tier_km_labels[i])])
    print(f"\nSaved: {csv_path}")

    # =========================================================================
    # Figures
    # =========================================================================
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"Experiment 11: Oscillation Predictor — {objname}\n"
        "Q1: Does the Type A/B split exist in other tiers?   "
        "Q2: Does normal covariance linearity (L) predict oscillation better than NV?",
        fontsize=11, fontweight="bold")
    gs = GridSpec(3, 5, figure=fig, hspace=0.50, wspace=0.38)

    # ── Row 0: Q1 — frac_low_disp histograms per tier ────────────────────
    for t in range(len(TIERS)):
        ax = fig.add_subplot(gs[0, t])
        idx_t  = np.flatnonzero(tier_id == t)
        lbl_t  = tier_km_labels[idx_t]
        maskA  = lbl_t == 0
        maskB  = lbl_t == 1
        fl     = frac_low[idx_t]
        sep    = tier_km_sep[t]

        if len(idx_t) >= 4:
            ax.hist(fl[maskA], bins=15, color="#2196F3", alpha=0.65,
                    label=f"A n={maskA.sum()}", density=True)
            ax.hist(fl[maskB], bins=15, color="#F44336", alpha=0.65,
                    label=f"B n={maskB.sum()}", density=True)
        else:
            ax.hist(fl, bins=10, color=tier_colors[t], alpha=0.7, density=True)

        ax.set_title(f"{tier_names[t]}\n(sep={sep:.3f})", fontsize=9)
        ax.set_xlabel("frac_low_disp", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    # ── Row 1: Q1 — scatter (mean_disp vs frac_low) per tier ─────────────
    for t in range(len(TIERS)):
        ax = fig.add_subplot(gs[1, t])
        idx_t = np.flatnonzero(tier_id == t)
        lbl_t = tier_km_labels[idx_t]
        for lbl_val, col in [(0, "#2196F3"), (1, "#F44336"), (-1, "#999")]:
            m = lbl_t == lbl_val
            if m.any():
                ax.scatter(frac_low[idx_t[m]], mean_disp[idx_t[m]],
                           s=6, alpha=0.5, color=col, rasterized=True)
        ax.set_xlabel("frac_low_disp", fontsize=7)
        ax.set_ylabel("mean_disp", fontsize=7)
        ax.set_title(f"{tier_names[t]} scatter", fontsize=9)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    # ── Row 2: Q2 panels ──────────────────────────────────────────────────
    # (2,0) Q2a: L per sharp type (bar)
    ax = fig.add_subplot(gs[2, 0])
    if len(sharp_idx) >= 4:
        sharp_lbl = tier_km_labels[sharp_idx]
        ma, mb = sharp_lbl == 0, sharp_lbl == 1
        labs  = ["Type A\n(singularity)", "Type B\n(ridge)"]
        means = [L[sharp_idx[ma]].mean(), L[sharp_idx[mb]].mean()]
        stds  = [L[sharp_idx[ma]].std(),  L[sharp_idx[mb]].std()]
        bars  = ax.bar([0, 1], means, yerr=stds, capsize=5,
                       color=["#2196F3", "#F44336"], alpha=0.85,
                       edgecolor="black", linewidth=0.5)
        for i, (m_, s_) in enumerate(zip(means, stds)):
            ax.text(i, m_ + s_ + 0.005, f"{m_:.3f}", ha="center", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labs, fontsize=7)
        ax.set_ylabel("Normal Covariance Linearity L")
        ax.set_title("Q2a: L in sharp tier\n(Type B should have higher L)")
        ax.grid(True, alpha=0.3, axis="y")

    # (2,1) Q2b: r comparison bar across tiers
    ax = fig.add_subplot(gs[2, 1:3])
    x = np.arange(len(TIERS))
    w = 0.28
    rs_nv, rs_l, rs_nl = [], [], []
    for t in range(len(TIERS)):
        idx_t = np.flatnonzero(tier_id == t)
        if len(idx_t) < 5:
            rs_nv.append(0); rs_l.append(0); rs_nl.append(0)
            continue
        md_t = mean_disp[idx_t]
        r1, _ = spearmanr(nv_score[idx_t], md_t)
        r2, _ = spearmanr(L[idx_t],        md_t)
        r3, _ = spearmanr(NV_x_L[idx_t],  md_t)
        rs_nv.append(r1); rs_l.append(r2); rs_nl.append(r3)

    ax.bar(x - w, rs_nv, w, label="r(NV→disp)", color="#FF9800", alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.bar(x,     rs_l,  w, label="r(L→disp)",  color="#9C27B0", alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w, rs_nl, w, label="r(NV×L→disp)", color="#009688", alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(tier_names)
    ax.set_ylabel("Spearman r")
    ax.set_title("Q2b: Predictive power of NV, L, and NV×L for mean_disp\n"
                 "(higher |r| = better predictor)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # (2,3) scatter L vs mean_disp across all sites, colored by tier
    ax = fig.add_subplot(gs[2, 3])
    for t in range(len(TIERS)):
        idx_t = np.flatnonzero(tier_id == t)
        if len(idx_t) == 0:
            continue
        rng = np.random.default_rng(t)
        sub = rng.choice(idx_t, min(500, len(idx_t)), replace=False)
        ax.scatter(L[sub], mean_disp[sub], s=4, alpha=0.3,
                   color=tier_colors[t], label=tier_names[t], rasterized=True)
    ax.set_xlabel("Linearity L")
    ax.set_ylabel("mean_disp")
    ax.set_title("L vs mean_disp (all tiers)")
    ax.legend(fontsize=6, markerscale=2)
    ax.grid(True, alpha=0.3)

    # (2,4) scatter NV vs mean_disp (baseline comparison)
    ax = fig.add_subplot(gs[2, 4])
    for t in range(len(TIERS)):
        idx_t = np.flatnonzero(tier_id == t)
        if len(idx_t) == 0:
            continue
        rng = np.random.default_rng(t + 10)
        sub = rng.choice(idx_t, min(500, len(idx_t)), replace=False)
        ax.scatter(nv_score[sub], mean_disp[sub], s=4, alpha=0.3,
                   color=tier_colors[t], label=tier_names[t], rasterized=True)
    ax.set_xlabel("NV score")
    ax.set_ylabel("mean_disp")
    ax.set_title("NV score vs mean_disp (baseline)")
    ax.legend(fontsize=6, markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(out_dir, "exp11_oscillation_predictor.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
