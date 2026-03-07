#!/usr/bin/env python3
"""
Exp 12 — Effective Voronoi Neighbor Count vs Curvature

For each site i, only a subset of its K=32 KNN neighbors actually clip the
2D Voronoi polygon in the tangent plane — the rest produce half-planes that
do not intersect the current polygon and are geometrically redundant.

This experiment measures, per iteration and per curvature tier:
  n_eff       — how many of the K neighbours produced an effective clip
  frac_eff    — n_eff / K (fraction of budget actually used)
  cell_area   — area of the 2D polygon (proxy for Voronoi cell size)
  |centroid|  — 2D distance of centroid from origin (displacement proxy)

Hypothesis: at high curvature the tangent-plane projection distorts
inter-site distances so that fewer neighbours contribute to the boundary
(the distorted cell is "under-constrained" from certain directions),
while the centroid is pulled further off-centre.

Outputs  results/<mesh>/exp12_effective_neighbors/
    exp12_effective_neighbors.png  — 4-panel figure
    exp12_summary.csv              — per-tier summary table
    exp12_raw.npy                  — raw per-site per-iter arrays

Usage:
    python experiments/exp12_effective_neighbors.py teapot
    python experiments/exp12_effective_neighbors.py spot
"""

import sys
import os
import argparse
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import trimesh

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import testfreeze as tf

ITERS = 30          # Lloyd iterations (NOFREEZE)
K     = tf.KNN_K    # 32

# ── Modified cell_poly2d that counts effective clips ─────────────────────────

def cell_poly2d_count(i, S, U, V, neigh, R):
    """
    Like cell_poly2d but also returns:
      n_eff      — number of neighbours whose halfplane had any vertex outside it (effective clip)
      area       — 2D polygon area
      cx, cy     — 2D centroid coordinates (distance from origin = centroid offset)
    """
    si = S[i]
    poly = np.array([[-R, -R], [R, -R], [R, R], [-R, R]], dtype=np.float64)
    ui   = U[i]
    vi   = V[i]
    n_eff = 0

    for j in neigh:
        d = S[j] - si
        a = float(d @ ui)
        b = float(d @ vi)
        c = 0.5 * float(d @ d)
        # Effective clip: any vertex lies outside the halfplane (a*x + b*y > c)
        # This is correct regardless of net vertex-count change after clipping
        # (e.g. removes 1 vertex + adds 2 intersections → net +1, but IS effective)
        if np.any(a * poly[:, 0] + b * poly[:, 1] > c + 1e-12):
            n_eff += 1
        poly = tf.clip_poly_halfspace(poly, a, b, c)
        if poly.shape[0] == 0:
            break

    area, c2 = tf.poly_area_centroid_2d(poly)
    return n_eff, abs(area), c2   # c2 = [cx, cy]


# ── NOFREEZE Lloyd with per-site clip tracking ────────────────────────────────

def run_with_clip_tracking(S0, P0, F0, N0, vf, treeP0,
                           iters=ITERS, k_neigh=K, k_proj=10):
    """
    Run `iters` Lloyd iterations (no freeze) and record per-site per-iter:
      n_eff, area, centroid_dist, disp
    """
    n = len(S0)
    n_eff_hist    = np.zeros((n, iters), dtype=np.float32)
    area_hist     = np.zeros((n, iters), dtype=np.float32)
    cent_dist_hist= np.zeros((n, iters), dtype=np.float32)
    disp_hist     = np.zeros((n, iters), dtype=np.float32)

    S = S0.copy()

    for it in range(iters):
        _, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64),
                               k=1, workers=-1)
        N = N0[idxN]
        U, V = tf.tangent(N)

        treeS = cKDTree(np.ascontiguousarray(S, dtype=np.float64))
        _, idxA = treeS.query(S, k=k_neigh + 1, workers=-1)
        idxA = np.sort(idxA[:, 1:], axis=1)

        bbox = S.max(axis=0) - S.min(axis=0)
        R    = float(np.max(bbox))

        cent3d = np.full_like(S, np.nan)
        for i in range(n):
            neff, area, c2 = cell_poly2d_count(i, S, U, V, idxA[i], R)
            n_eff_hist[i, it]     = neff
            area_hist[i, it]      = area
            cent_dist_hist[i, it] = float(np.linalg.norm(c2)) if np.isfinite(c2).all() else np.nan
            if np.isfinite(c2).all():
                cent3d[i] = S[i] + c2[0] * U[i] + c2[1] * V[i]

        Snew = S.copy()
        mask = np.isfinite(cent3d).all(axis=1)
        if np.any(mask):
            _, idxC = treeP0.query(
                np.ascontiguousarray(cent3d[mask], dtype=np.float64),
                k=k_proj, workers=-1
            )
            for t, i in enumerate(np.flatnonzero(mask)):
                nn    = idxC[t]
                faces = set()
                for v in nn:
                    for fi in vf[int(v)]:
                        faces.add(fi)
                faces = np.fromiter(faces, dtype=np.int64)
                tris  = F0[faces] if faces.size else np.empty((0, 3), dtype=np.int64)
                Snew[i] = tf.update_to_mesh(cent3d[i], tris, P0)

        dd = np.linalg.norm(Snew - S, axis=1).astype(np.float32)
        disp_hist[:, it] = dd
        S = Snew

        print(f"  iter {it+1:02d}/{iters}: "
              f"mean_neff={n_eff_hist[:,it].mean():.2f}  "
              f"mean_disp={dd.mean():.4e}")

    return n_eff_hist, area_hist, cent_dist_hist, disp_hist


# ── Analysis and plots ────────────────────────────────────────────────────────

def tier_summary(arr, tier_id, skip_first=5):
    """
    Per-tier mean/std of a per-site per-iter array,
    averaged over iterations [skip_first:] to avoid transient.
    """
    vals = arr[:, skip_first:].mean(axis=1)   # per-site mean over stable iters
    rows = []
    for tid, name in enumerate(tf.TIERS):
        tname = name[0]
        mask  = (tier_id == tid)
        if not np.any(mask):
            rows.append((tname, float("nan"), float("nan"), 0))
        else:
            rows.append((tname,
                         float(vals[mask].mean()),
                         float(vals[mask].std()),
                         int(mask.sum())))
    return rows, vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mesh", nargs="?", default="teapot")
    p.add_argument("--iters", type=int, default=ITERS)
    p.add_argument("--skip",  type=int, default=5,
                   help="skip first N iters when computing tier means")
    args = p.parse_args()

    objname  = args.mesh
    obj_path = os.path.join(_ROOT, "meshes", f"{objname}.obj")
    if not os.path.exists(obj_path):
        obj_path = os.path.join(_ROOT, f"{objname}.obj")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Mesh not found: {objname}")

    out_dir = os.path.join(_ROOT, "results", objname, "exp12_effective_neighbors")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load mesh ──────────────────────────────────────────────────────────
    print(f"[Exp 12] Loading {objname} ...")
    m = trimesh.load(obj_path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    P0 = np.asarray(m.vertices,       dtype=np.float64)
    F0 = np.asarray(m.faces,          dtype=np.int64)
    N0 = np.asarray(m.vertex_normals, dtype=np.float64)
    treeP0 = cKDTree(np.ascontiguousarray(P0, dtype=np.float64))

    vf = [[] for _ in range(len(P0))]
    for fi, (a, b, c) in enumerate(F0):
        vf[int(a)].append(fi)
        vf[int(b)].append(fi)
        vf[int(c)].append(fi)

    S0 = P0.copy()
    print(f"  V={P0.shape[0]}  F={F0.shape[0]}")

    # ── Tier assignment ────────────────────────────────────────────────────
    mesh_nv = tf.normal_variation_score(P0, tf.nr(N0), tf.NV_K)
    nv      = tf.map_score_to_sites(S0, P0, mesh_nv)
    tier_id = np.digitize(nv, tf.TIER_THRESHOLDS).astype(np.int64)
    tier_names = [t[0] for t in tf.TIERS]
    for tid, name in enumerate(tier_names):
        print(f"  {name:<10}: n={int((tier_id==tid).sum())}")

    # ── Run ────────────────────────────────────────────────────────────────
    print(f"\n[Exp 12] Running {args.iters} NOFREEZE iterations ...")
    n_eff_hist, area_hist, cent_hist, disp_hist = run_with_clip_tracking(
        S0, P0, F0, N0, vf, treeP0, iters=args.iters
    )

    # ── Per-tier summaries ────────────────────────────────────────────────
    rows_neff, neff_mean = tier_summary(n_eff_hist,  tier_id, args.skip)
    rows_area, area_mean = tier_summary(area_hist,   tier_id, args.skip)
    rows_cent, cent_mean = tier_summary(cent_hist,   tier_id, args.skip)
    rows_disp, disp_mean = tier_summary(disp_hist,   tier_id, args.skip)

    print(f"\n{'Tier':<12} {'n':>5}  {'mean_neff':>10}  {'frac_eff':>9}  "
          f"{'mean_area':>10}  {'mean_cent_dist':>14}  {'mean_disp':>10}")
    print("-" * 75)
    for r_n, r_a, r_c, r_d in zip(rows_neff, rows_area, rows_cent, rows_disp):
        name  = r_n[0]
        neff  = r_n[1]
        frac  = neff / K if not np.isnan(neff) else float("nan")
        area  = r_a[1]
        cent  = r_c[1]
        disp  = r_d[1]
        n     = r_n[3]
        print(f"  {name:<10} {n:>5}  {neff:>10.3f}  {frac:>9.4f}  "
              f"{area:>10.4f}  {cent:>14.4f}  {disp:>10.5f}")

    # ── Spearman r(NV → n_eff), r(NV → disp) ──────────────────────────────
    print(f"\nGlobal Spearman correlations (site-level, mean over iters {args.skip}+):")
    for label, vals in [("n_eff",     neff_mean),
                        ("frac_eff",  neff_mean / K),
                        ("area",      area_mean),
                        ("cent_dist", cent_mean),
                        ("disp",      disp_mean)]:
        ok = np.isfinite(vals) & np.isfinite(nv)
        r, p = spearmanr(nv[ok], vals[ok])
        print(f"  r(NV → {label:<12}) = {r:+.4f}  p={p:.3e}")

    r_ne_d, _ = spearmanr(neff_mean, disp_mean)
    print(f"  r(n_eff → disp)      = {r_ne_d:+.4f}  "
          f"(negative = fewer effective clips → larger displacement)")

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "exp12_summary.csv")
    header   = ["tier", "n", "mean_neff", "std_neff", "frac_eff",
                "mean_area", "mean_cent_dist", "mean_disp"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r_n, r_a, r_c, r_d in zip(rows_neff, rows_area, rows_cent, rows_disp):
            neff = r_n[1]
            w.writerow(dict(
                tier=r_n[0], n=r_n[3],
                mean_neff=f"{neff:.4f}",
                std_neff =f"{r_n[2]:.4f}",
                frac_eff =f"{neff/K:.4f}" if not np.isnan(neff) else "nan",
                mean_area=f"{r_a[1]:.4f}",
                mean_cent_dist=f"{r_c[1]:.4f}",
                mean_disp=f"{r_d[1]:.6f}",
            ))
    print(f"\n  Saved CSV: {csv_path}")

    # ── Save raw arrays ────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "exp12_neff_hist.npy"),  n_eff_hist)
    np.save(os.path.join(out_dir, "exp12_area_hist.npy"),  area_hist)
    np.save(os.path.join(out_dir, "exp12_cent_hist.npy"),  cent_hist)
    np.save(os.path.join(out_dir, "exp12_disp_hist.npy"),  disp_hist)
    np.save(os.path.join(out_dir, "exp12_tier_id.npy"),    tier_id)
    np.save(os.path.join(out_dir, "exp12_nv.npy"),         nv)

    # ── Figure ─────────────────────────────────────────────────────────────
    colours = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]
    iters_ax = np.arange(1, args.iters + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: mean n_eff over iterations per tier
    ax = axes[0, 0]
    for tid, (col, name) in enumerate(zip(colours, tier_names)):
        mask = (tier_id == tid)
        if not np.any(mask):
            continue
        mu = n_eff_hist[mask].mean(axis=0)
        sd = n_eff_hist[mask].std(axis=0)
        ax.plot(iters_ax, mu, color=col, linewidth=1.8, label=name)
        ax.fill_between(iters_ax, mu - sd, mu + sd, alpha=0.15, color=col)
    ax.axhline(6, color="black", linestyle=":", alpha=0.5, label="6 (ideal hex)")
    ax.set_xlabel("iteration"); ax.set_ylabel("mean effective clips")
    ax.set_title("Effective neighbor count (n_eff) vs iteration")
    ax.legend(fontsize=8); ax.set_ylim(0, K + 2)

    # Panel B: fraction of K budget used per tier (stable-phase box)
    ax = axes[0, 1]
    stable_neff = n_eff_hist[:, args.skip:].mean(axis=1)
    frac_eff    = stable_neff / K
    bpdata = [frac_eff[tier_id == tid] for tid in range(len(tier_names))]
    bpdata = [d for d in bpdata if len(d) > 0]
    valid_names = [tier_names[tid] for tid in range(len(tier_names))
                   if np.any(tier_id == tid)]
    bp = ax.boxplot(bpdata, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, col in zip(bp["boxes"], [colours[i] for i, n in enumerate(tier_names)
                                        if np.any(tier_id == i)]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_xticklabels(valid_names, rotation=15)
    ax.set_ylabel("frac_eff = n_eff / K")
    ax.set_title(f"Fraction of K={K} budget used (iters {args.skip}+)")
    ax.axhline(6/K, color="black", linestyle=":", alpha=0.5, label="6/K ideal")
    ax.legend(fontsize=8)

    # Panel C: centroid offset (2D distance) vs tier
    ax = axes[1, 0]
    stable_cent = cent_hist[:, args.skip:].mean(axis=1)
    cddata = [stable_cent[tier_id == tid] for tid in range(len(tier_names))]
    cddata = [d for d in cddata if len(d) > 0]
    bp2 = ax.boxplot(cddata, patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2))
    for patch, col in zip(bp2["boxes"], [colours[i] for i, n in enumerate(tier_names)
                                         if np.any(tier_id == i)]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_xticklabels(valid_names, rotation=15)
    ax.set_ylabel("2D centroid offset from site origin")
    ax.set_title("Centroid displacement in tangent plane (iters 5+)")

    # Panel D: scatter n_eff vs displacement (site-level)
    ax = axes[1, 1]
    stable_disp = disp_hist[:, args.skip:].mean(axis=1)
    for tid, (col, name) in enumerate(zip(colours, tier_names)):
        mask = (tier_id == tid)
        if not np.any(mask):
            continue
        ax.scatter(stable_neff[mask], stable_disp[mask],
                   c=col, s=6, alpha=0.4, label=name)
    ax.set_xlabel("mean n_eff (effective clips)")
    ax.set_ylabel("mean displacement")
    ax.set_title(f"n_eff vs displacement  r={r_ne_d:+.3f}")
    ax.legend(fontsize=7)
    # add per-tier mean markers
    for tid, (col, name) in enumerate(zip(colours, tier_names)):
        mask = (tier_id == tid)
        if not np.any(mask):
            continue
        ax.scatter(stable_neff[mask].mean(), stable_disp[mask].mean(),
                   c=col, s=120, marker="D", edgecolors="black", linewidth=1, zorder=5)

    plt.suptitle(
        f"{objname} — Exp 12: Effective Voronoi Neighbor Count vs Curvature\n"
        f"(K={K} neighbours used in poly clipping; n_eff = those with any vertex outside halfplane)",
        fontsize=10
    )
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "exp12_effective_neighbors.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
