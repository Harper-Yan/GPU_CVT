#!/usr/bin/env python3
"""
Experiment 10: Sharp Tier Site Classification and Mesh Visualization

Distinguishes the two sub-populations within the sharp tier (first detected
in Exp 4, §3b; confirmed as a Exp 9 confound in §6d):

  Type A — Singularity / Position-trapped
    Sites at geometric singularities (peaks, corners). Displacement consistently
    low; centroid never escapes a small region; KNN stable; behaves like flat.
    These dominate the Exp 9 low-disp conditioning and cause P2/P3 to collapse.

  Type B — Ridge / Oscillating
    Sites on sharp ridges. Displacement oscillates persistently above threshold;
    almost never satisfy disp < thr; responsible for the 37-60% Jaccard failure
    rates in Exp 6a and the 64-100% windowed-Jaccard blocks in Exp 6c.

Classification features (per sharp-tier site, over ITERS Lloyd iterations):
    frac_low_disp      – fraction of iterations where ||disp|| < disp_thr_sharp
    mean_disp          – time-averaged displacement magnitude
    dir_reversal_frac  – fraction of consecutive iteration pairs where
                         cosine(d_t, d_{t-1}) < 0

Classification method: K-means (k=2) on (frac_low_disp, mean_disp).
Type A = cluster with higher frac_low_disp / lower mean_disp.

Outputs:
    exp10_classification.png       – scatter + histogram classification figure
    exp10_validation.png           – Exp-9-style P2/P3 breakdown by type
    exp10_mesh_colored.ply         – mesh with colored vertices:
                                     grey  = non-sharp tier
                                     blue  = Type A (singularity)
                                     red   = Type B (ridge)
    exp10_sites_typeA.ply          – point cloud of Type A site positions
    exp10_sites_typeB.ply          – point cloud of Type B site positions
    exp10_classification.csv       – per-site features + assigned type
    exp10_stats.csv                – summary statistics for both types
"""

import numpy as np
import os
import sys
import csv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import trimesh
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from testfreeze import (
    nr, tangent, normal_variation_score, closest_point_tri,
    cell_poly2d, poly_area_centroid_2d, jaccard_sorted_int,
    NV_K, TIER_THRESHOLDS, TIERS,
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ITERS   = 60
K_NEIGH = 32
K_PROJ  = 10

COLOR_A     = np.array([0.18, 0.45, 0.90])   # blue  – singularity / Type A
COLOR_B     = np.array([0.90, 0.18, 0.18])   # red   – ridge / Type B
COLOR_OTHER = np.array([0.75, 0.75, 0.75])   # grey  – non-sharp


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def update_to_mesh(C, tris, P):
    if tris.shape[0] == 0 or not np.isfinite(C).all():
        return C
    best, best_d2 = None, np.inf
    for a, b, c in tris:
        q = closest_point_tri(C,
                              P[a].astype(np.float64),
                              P[b].astype(np.float64),
                              P[c].astype(np.float64))
        d2 = float((q - C) @ (q - C))
        if d2 < best_d2:
            best_d2, best = d2, q
    return best if best is not None else C


def lloyd_nofreeze_track_positions(S0, P0, F0, N0, treeP0, vf, iters):
    """Lloyd without freezing. Returns S_history (iters+1, n, 3) and knn_history."""
    n = len(S0)
    S = S0.copy()
    S_hist  = [S.copy()]
    knn_hist = []

    for it in range(iters):
        _, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64), k=1, workers=-1)
        N = N0[idxN]
        U, V = tangent(N)

        treeS = cKDTree(np.ascontiguousarray(S, dtype=np.float64))
        _, knn = treeS.query(S, k=K_NEIGH + 1, workers=-1)
        knn_sorted = np.sort(knn[:, 1:], axis=1)
        knn_hist.append(knn_sorted.copy())

        bbox = S.max(axis=0) - S.min(axis=0)
        R    = float(np.max(bbox))
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

    return np.stack(S_hist, axis=0), knn_hist   # (iters+1, n, 3), list


# ---------------------------------------------------------------------------
# Open3D helpers
# ---------------------------------------------------------------------------

def save_colored_mesh_ply(path, P0, F0, colors):
    """Save mesh with per-vertex colors as PLY."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(P0.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(F0.astype(np.int64))
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0))
    o3d.io.write_triangle_mesh(path, mesh, write_vertex_colors=True)
    print(f"Saved: {path}")


def save_site_pcd_ply(path, positions, color):
    """Save point cloud of sites as PLY, all one color."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(positions, dtype=np.float64))
    n = len(positions)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.clip(color, 0.0, 1.0), (n, 1)))
    o3d.io.write_point_cloud(path, pcd)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mesh", nargs="?",
                   default=os.path.join(_ROOT, "meshes", "teapot.obj"))
    p.add_argument("--view", action="store_true",
                   help="Open interactive Open3D viewer after saving PLY")
    args = p.parse_args()
    in_path = args.mesh
    objname = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = os.path.join(_ROOT, "results", objname, "exp10_sharp_classification")
    os.makedirs(out_dir, exist_ok=True)

    # ── load mesh ──────────────────────────────────────────────────────────
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

    # ── tier assignment ────────────────────────────────────────────────────
    nv_score   = normal_variation_score(P0, nr(N0), NV_K)
    tier_id    = np.digitize(nv_score, TIER_THRESHOLDS).astype(np.int64)
    tier_names = [t[0] for t in TIERS]
    disp_thr_arr = np.array([t[1] for t in TIERS], dtype=np.float64)
    disp_thr_sharp = disp_thr_arr[4]   # 1e-3

    sharp_idx = np.flatnonzero(tier_id == 4)
    n_sharp   = len(sharp_idx)
    print(f"[{objname}] Total sites: {n}  Sharp-tier sites: {n_sharp}")
    if n_sharp < 2:
        print("Too few sharp sites — skipping.")
        return

    # ── run Lloyd ──────────────────────────────────────────────────────────
    print(f"\nRunning {ITERS} NOFREEZE iterations...")
    S_hist, knn_hist = lloyd_nofreeze_track_positions(
        S0, P0, F0, N0, treeP0, vf, ITERS)
    # S_hist: (ITERS+1, n, 3),  knn_hist: list of ITERS (n, K_NEIGH)

    # ── per-site features for sharp-tier sites ─────────────────────────────
    # displacement vectors: d[t] = S_hist[t+1] - S_hist[t],  shape (ITERS, n, 3)
    d_vecs = np.diff(S_hist, axis=0)          # (ITERS, n, 3)
    d_mags = np.linalg.norm(d_vecs, axis=2)   # (ITERS, n)

    # frac_low_disp: fraction of iterations where d_mag < disp_thr_sharp
    frac_low = (d_mags[:, sharp_idx] < disp_thr_sharp).mean(axis=0)   # (n_sharp,)
    mean_disp_s = d_mags[:, sharp_idx].mean(axis=0)                     # (n_sharp,)
    std_disp_s  = d_mags[:, sharp_idx].std(axis=0)                      # (n_sharp,)

    # direction reversal fraction: cos(d_t, d_{t-1}) < 0
    d_n = d_vecs / (np.linalg.norm(d_vecs, axis=2, keepdims=True) + 1e-30)   # unit vecs
    cosines = np.sum(d_n[:-1, sharp_idx, :] * d_n[1:, sharp_idx, :], axis=2) # (ITERS-1, n_sharp)
    dir_rev = (cosines < 0).mean(axis=0)   # (n_sharp,)

    # Jaccard-when-low-disp per sharp site (for validation)
    jacc_when_low = [[] for _ in range(n_sharp)]
    nbr_disp_when_low = [[] for _ in range(n_sharp)]

    for it in range(1, ITERS):
        prev_knn = knn_hist[it - 1]
        curr_knn = knn_hist[it]
        dM       = d_mags[it]           # (n,) magnitudes at this iter

        for s_pos, s_gidx in enumerate(sharp_idx):
            if dM[s_gidx] < disp_thr_sharp:
                j = jaccard_sorted_int(prev_knn[s_gidx], curr_knn[s_gidx])
                nbrs = curr_knn[s_gidx]
                jacc_when_low[s_pos].append(j)
                nbr_disp_when_low[s_pos].append(dM[nbrs].mean())

    mean_jacc_low  = np.array([np.mean(x) if x else np.nan for x in jacc_when_low])
    std_jacc_low   = np.array([np.std(x)  if x else np.nan for x in jacc_when_low])
    n_low_events   = np.array([len(x) for x in jacc_when_low])
    mean_nbr_when_low = np.array([np.mean(x) if x else np.nan for x in nbr_disp_when_low])

    # ── K-means classification (k=2) ──────────────────────────────────────
    X = np.column_stack([frac_low, mean_disp_s])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=2, random_state=42, n_init=20)
    labels_raw = km.fit_predict(Xs)

    # Assign Type A = higher frac_low_disp cluster
    c0_frac = frac_low[labels_raw == 0].mean()
    c1_frac = frac_low[labels_raw == 1].mean()
    if c1_frac > c0_frac:
        labels_raw = 1 - labels_raw   # flip: 0=A, 1=B

    typeA_mask = labels_raw == 0   # (n_sharp,)
    typeB_mask = labels_raw == 1
    nA = int(typeA_mask.sum())
    nB = int(typeB_mask.sum())
    print(f"\n  Type A (singularity/position-trapped): n={nA}")
    print(f"  Type B (ridge/oscillating):            n={nB}")

    # ── print stats ───────────────────────────────────────────────────────
    def fmt(arr, mask):
        v = arr[mask]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return "  —"
        return f"{v.mean():.5f} ± {v.std():.5f}"

    hdr = f"  {'':22s}  {'Type A (sing.)':>20s}  {'Type B (ridge)':>20s}"
    print(hdr)
    for label, arr in [
        ("frac_low_disp",      frac_low),
        ("mean_disp",          mean_disp_s),
        ("std_disp",           std_disp_s),
        ("dir_reversal_frac",  dir_rev),
        ("mean_jacc|low_disp", mean_jacc_low),
        ("std_jacc|low_disp",  std_jacc_low),
        ("n_low_events",       n_low_events.astype(float)),
        ("mean_nbr_d|low",     mean_nbr_when_low),
    ]:
        print(f"  {label:22s}  {fmt(arr, typeA_mask):>20s}  {fmt(arr, typeB_mask):>20s}")

    # ── validation: r(nbr_disp, Jaccard | low) per type ───────────────────
    print("\n  Validation (Exp 9 P2 breakdown by type):")
    for tname, tmask in [("Type A (singularity)", typeA_mask),
                          ("Type B (ridge)",       typeB_mask)]:
        nbr_all, jacc_all = [], []
        for s_pos in np.flatnonzero(tmask):
            nbr_all.extend(nbr_disp_when_low[s_pos])
            jacc_all.extend(jacc_when_low[s_pos])
        if len(nbr_all) >= 5:
            r, p = spearmanr(nbr_all, jacc_all)
            print(f"    {tname}: n_events={len(nbr_all):6d}  "
                  f"r(nbr_d→J)={r:+.4f}  p={p:.2e}")
        else:
            print(f"    {tname}: n_events={len(nbr_all)} — too few")

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "exp10_classification.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_global_idx", "type",
                    "frac_low_disp", "mean_disp", "std_disp", "dir_reversal_frac",
                    "mean_jacc_when_low", "std_jacc_when_low",
                    "n_low_events", "mean_nbr_disp_when_low"])
        for s_pos, gidx in enumerate(sharp_idx):
            w.writerow([
                int(gidx),
                "A" if typeA_mask[s_pos] else "B",
                f"{frac_low[s_pos]:.6f}",
                f"{mean_disp_s[s_pos]:.6f}",
                f"{std_disp_s[s_pos]:.6f}",
                f"{dir_rev[s_pos]:.6f}",
                f"{mean_jacc_low[s_pos]:.6f}"  if np.isfinite(mean_jacc_low[s_pos])  else "",
                f"{std_jacc_low[s_pos]:.6f}"   if np.isfinite(std_jacc_low[s_pos])   else "",
                int(n_low_events[s_pos]),
                f"{mean_nbr_when_low[s_pos]:.6f}" if np.isfinite(mean_nbr_when_low[s_pos]) else "",
            ])
    print(f"\nSaved: {csv_path}")

    stats_path = os.path.join(out_dir, "exp10_stats.csv")
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "n",
                    "frac_low_mean", "frac_low_std",
                    "mean_disp_mean", "mean_disp_std",
                    "dir_rev_mean",  "dir_rev_std",
                    "mean_jacc_low_mean", "mean_jacc_low_std"])
        for tname, tmask in [("A", typeA_mask), ("B", typeB_mask)]:
            def ms(arr):
                v = arr[tmask]; v = v[np.isfinite(v)]
                return (f"{v.mean():.6f}", f"{v.std():.6f}") if len(v) else ("", "")
            fl = ms(frac_low);  md = ms(mean_disp_s)
            dr = ms(dir_rev);   mj = ms(mean_jacc_low)
            w.writerow([tname, int(tmask.sum()),
                        fl[0], fl[1], md[0], md[1], dr[0], dr[1], mj[0], mj[1]])
    print(f"Saved: {stats_path}")

    # ── Figure 1: classification scatter + histograms ─────────────────────
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Experiment 10: Sharp Tier Sub-population Classification — {objname}\n"
        "Type A (blue) = singularity/position-trapped   "
        "Type B (red) = ridge/oscillating",
        fontsize=12, fontweight="bold")
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    cA = COLOR_A; cB = COLOR_B

    # (0,0) scatter: frac_low vs mean_disp
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(frac_low[typeA_mask], mean_disp_s[typeA_mask],
               s=20, alpha=0.6, color=cA, label=f"Type A (n={nA})")
    ax.scatter(frac_low[typeB_mask], mean_disp_s[typeB_mask],
               s=20, alpha=0.6, color=cB, label=f"Type B (n={nB})")
    ax.set_xlabel("frac_low_disp (fraction of iters < thr)")
    ax.set_ylabel("mean_disp")
    ax.set_title("Classification Feature Space")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) histogram: frac_low_disp for all sharp sites (should show bimodality)
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(frac_low[typeA_mask], bins=20, color=cA, alpha=0.6,
            label="Type A", density=True)
    ax.hist(frac_low[typeB_mask], bins=20, color=cB, alpha=0.6,
            label="Type B", density=True)
    ax.set_xlabel("frac_low_disp")
    ax.set_ylabel("Density")
    ax.set_title("Bimodal Distribution: frac_low_disp")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,2) scatter: mean_disp vs dir_reversal_frac
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(dir_rev[typeA_mask], mean_disp_s[typeA_mask],
               s=20, alpha=0.6, color=cA)
    ax.scatter(dir_rev[typeB_mask], mean_disp_s[typeB_mask],
               s=20, alpha=0.6, color=cB)
    ax.set_xlabel("dir_reversal_frac")
    ax.set_ylabel("mean_disp")
    ax.set_title("Direction Reversal vs Mean Displacement")
    ax.grid(True, alpha=0.3)

    # (0,3) displacement time-series: 5 random sites from each type
    ax = fig.add_subplot(gs[0, 3])
    iters_x = np.arange(1, ITERS + 1)
    rng = np.random.default_rng(0)
    for s_pos in rng.choice(np.flatnonzero(typeA_mask),
                            min(5, nA), replace=False):
        ax.plot(iters_x, d_mags[:, sharp_idx[s_pos]],
                color=cA, alpha=0.5, linewidth=0.8)
    for s_pos in rng.choice(np.flatnonzero(typeB_mask),
                            min(5, nB), replace=False):
        ax.plot(iters_x, d_mags[:, sharp_idx[s_pos]],
                color=cB, alpha=0.5, linewidth=0.8)
    ax.axhline(disp_thr_sharp, color="black", linestyle="--",
               linewidth=1, label=f"disp_thr={disp_thr_sharp}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Displacement")
    ax.set_title("Sample Displacement Trajectories (5 each)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,0) Jaccard when focal-low, per type
    ax = fig.add_subplot(gs[1, 0])
    jA = mean_jacc_low[typeA_mask]; jA = jA[np.isfinite(jA)]
    jB = mean_jacc_low[typeB_mask]; jB = jB[np.isfinite(jB)]
    ax.hist(jA, bins=20, color=cA, alpha=0.6, label="Type A", density=True)
    ax.hist(jB, bins=20, color=cB, alpha=0.6, label="Type B", density=True)
    ax.set_xlabel("Mean Jaccard | disp < thr")
    ax.set_ylabel("Density")
    ax.set_title("Jaccard Stability During Low-Disp Moments\n"
                 "(validates: Type A stable, Type B unstable)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) n_low_events per type
    ax = fig.add_subplot(gs[1, 1])
    nLA = n_low_events[typeA_mask]; nLB = n_low_events[typeB_mask]
    ax.hist(nLA, bins=20, color=cA, alpha=0.6, label="Type A", density=True)
    ax.hist(nLB, bins=20, color=cB, alpha=0.6, label="Type B", density=True)
    ax.set_xlabel("n low-disp events (out of 60 iters)")
    ax.set_ylabel("Density")
    ax.set_title("Low-Disp Event Count Per Site\n"
                 "(Type B: rarely enters low-disp regime)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) mean_nbr_disp when focal low, per type
    ax = fig.add_subplot(gs[1, 2])
    nA_vals = mean_nbr_when_low[typeA_mask]; nA_vals = nA_vals[np.isfinite(nA_vals)]
    nB_vals = mean_nbr_when_low[typeB_mask]; nB_vals = nB_vals[np.isfinite(nB_vals)]
    ax.hist(nA_vals, bins=20, color=cA, alpha=0.6, label="Type A", density=True)
    ax.hist(nB_vals, bins=20, color=cB, alpha=0.6, label="Type B", density=True)
    ax.set_xlabel("Mean neighbor disp | focal disp < thr")
    ax.set_ylabel("Density")
    ax.set_title("Neighbor Activity During Type's Low-Disp Moments\n"
                 "(Exp 9 P1 breakdown by type)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,3) bar: summary stats side-by-side
    ax = fig.add_subplot(gs[1, 3])
    metrics   = ["frac_low\ndisp", "mean\ndisp(×100)", "dir_rev\nfrac", "mean_jacc\n|low"]
    arraysA   = [frac_low, mean_disp_s * 100, dir_rev, mean_jacc_low]
    vals_A    = [np.nanmean(a[typeA_mask]) for a in arraysA]
    vals_B    = [np.nanmean(a[typeB_mask]) for a in arraysA]
    x         = np.arange(len(metrics))
    w         = 0.35
    ax.bar(x - w/2, vals_A, w, color=cA, alpha=0.85, edgecolor="black",
           linewidth=0.5, label="Type A")
    ax.bar(x + w/2, vals_B, w, color=cB, alpha=0.85, edgecolor="black",
           linewidth=0.5, label="Type B")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_title("Summary: Type A vs Type B Means")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig1_path = os.path.join(out_dir, "exp10_classification.png")
    plt.savefig(fig1_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {fig1_path}")
    plt.close()

    # ── Mesh coloring and PLY export ──────────────────────────────────────
    # 1. Build per-vertex color array for original mesh
    vert_colors = np.tile(COLOR_OTHER, (n, 1))   # all grey initially
    for s_pos, gidx in enumerate(sharp_idx):
        vert_colors[gidx] = COLOR_A if typeA_mask[s_pos] else COLOR_B

    mesh_ply = os.path.join(out_dir, "exp10_mesh_colored.ply")
    save_colored_mesh_ply(mesh_ply, P0, F0, vert_colors)

    # 2. Point clouds of final site positions for each type
    S_final = S_hist[-1]   # (n, 3) final positions
    sA_pos = S_final[sharp_idx[typeA_mask]]
    sB_pos = S_final[sharp_idx[typeB_mask]]

    save_site_pcd_ply(os.path.join(out_dir, "exp10_sites_typeA.ply"), sA_pos, COLOR_A)
    save_site_pcd_ply(os.path.join(out_dir, "exp10_sites_typeB.ply"), sB_pos, COLOR_B)

    # 3. Combined point cloud with both types + all other sites (tiny, semi-transparent)
    all_colors = np.tile(COLOR_OTHER * 0.4, (n, 1))   # dim grey for non-sharp
    for s_pos, gidx in enumerate(sharp_idx):
        all_colors[gidx] = COLOR_A if typeA_mask[s_pos] else COLOR_B

    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(S_final)
    pcd_all.colors = o3d.utility.Vector3dVector(np.clip(all_colors, 0.0, 1.0))
    pcd_all_path = os.path.join(out_dir, "exp10_sites_all_classified.ply")
    o3d.io.write_point_cloud(pcd_all_path, pcd_all)
    print(f"Saved: {pcd_all_path}")

    # ── optional interactive viewer ────────────────────────────────────────
    if args.view:
        print("\nOpening interactive viewer (close window to exit)...")
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_ply)
        mesh_o3d.compute_vertex_normals()
        pcd_A = o3d.io.read_point_cloud(
            os.path.join(out_dir, "exp10_sites_typeA.ply"))
        pcd_B = o3d.io.read_point_cloud(
            os.path.join(out_dir, "exp10_sites_typeB.ply"))
        # increase point size via VisualizerWithEditing is complex; use draw_geometries
        o3d.visualization.draw_geometries(
            [mesh_o3d, pcd_A, pcd_B],
            window_name=f"Exp10 Sharp Classification — {objname}",
            mesh_show_back_face=True,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
