#!/usr/bin/env python3
"""
testfreeze2.py — 6-tier CVT freeze policy with L-based sharp sub-classification.

Policy vs testfreeze.py:
  Sharp tier (NV >= 0.80) is split by Normal Covariance Linearity L:
    Sharp-A  (L >  L_SHARP_THR=0.80)  singularity / position-trapped — relaxed gates
    Sharp-B  (L <= L_SHARP_THR=0.80)  ridge / oscillating             — original gates

  All other tiers (flat, gentle, moderate, curved) are unchanged.

Outputs  results/<mesh>/testfreeze2/:
    tier_assignment.png          NV distribution coloured by 6-tier assignment
    freeze_progress.png          cumulative frozen sites per tier
    performance_comparison.png   bar chart (frozen fraction + false-freeze rate)
    performance_report.txt       frozen fraction + false-freeze rate vs Exp 8 baseline
    quality_comparison.png       final-position NN-dist / coverage metrics
    position_deviation.png       site-level deviation FREEZE v2 vs NOFREEZE
    quality_over_iters.png       4-panel (avg_aspect, %bad, %skinny, frozen) vs iter
    <mesh>_FREEZEv2.csv          per-iter geogram quality for FREEZE v2
    <mesh>_NOFREEZE.csv          per-iter geogram quality for NOFREEZE

Usage:
    python testfreeze2.py teapot
    python testfreeze2.py spot
"""

import sys
import os
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import trimesh
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
import testfreeze as tf

# ── 6-tier policy table ───────────────────────────────────────────────────────
L_SHARP_THR = 0.80         # L threshold separating sharp-A from sharp-B

TIERS2 = [
    # (name,       disp_thr, streak, neigh_win, jacc_thr)
    ("flat",        5e-3,  2,  3,  0.80),   # NV < 0.15
    ("gentle",      4e-3,  3,  5,  0.85),   # NV [0.15, 0.35)
    ("moderate",    3e-3,  5,  7,  0.88),   # NV [0.35, 0.55)
    ("curved",      2e-3,  6, 10,  0.93),   # NV [0.55, 0.80)
    ("sharp_B",     1e-3,  8, 12,  0.95),   # NV >= 0.80 AND L <= 0.80  (ridge)
    ("sharp_A",     1e-3,  6,  8,  0.88),   # NV >= 0.80 AND L >  0.80  (singularity)
]
TIER_NAMES2 = [t[0] for t in TIERS2]

VORPALITE_EXE = r"E:\MY PYTHON CODES\GPUCVT\geogram\build\Windows\bin\Release\vorpalite.exe"

# Exp 8 baseline numbers for comparison (teapot, LOOKAHEAD=10)
EXP8_BASELINE = {
    "flat":     {"frozen": 965,  "total": 1537, "false_rate": 71.8},
    "gentle":   {"frozen": 369,  "total":  801, "false_rate": 72.1},
    "moderate": {"frozen": 242,  "total":  504, "false_rate": 98.8},
    "curved":   {"frozen": 211,  "total":  562, "false_rate": 91.0},
    "sharp":    {"frozen":  50,  "total":  240, "false_rate": 64.0},
}

ITERS     = 60
LOOKAHEAD = 10
NF_ITERS  = ITERS + LOOKAHEAD


# ── Tier assignment ───────────────────────────────────────────────────────────

def compute_tier_id_v2(S0, P0, N0):
    """
    Assign 6-class tier IDs using NV then L for the sharp tier.

    Returns
    -------
    tier_id : (n,) int64   values 0-5
    nv      : (n,) float64  NV score per site
    L_site  : (n,) float64  L score per site (0 for non-sharp sites)
    """
    mesh_nv = tf.normal_variation_score(P0, tf.nr(N0), tf.NV_K)
    nv      = tf.map_score_to_sites(S0, P0, mesh_nv)
    tier_id = np.digitize(nv, tf.TIER_THRESHOLDS).astype(np.int64)   # 0-4

    L_site = np.zeros(len(S0), dtype=np.float64)
    sharp_mask = (tier_id == 4)
    if np.any(sharp_mask):
        n_sharp = int(np.sum(sharp_mask))
        print(f"  Computing L for {n_sharp} sharp sites ...")
        mesh_L = tf.normal_covariance_L(P0, tf.nr(N0), tf.NV_K)
        site_L = tf.map_score_to_sites(S0, P0, mesh_L)
        L_site[sharp_mask] = site_L[sharp_mask]

        sharp_A = sharp_mask & (site_L > L_SHARP_THR)
        tier_id[sharp_A] = 5        # promote to sharp_A

        nA = int(np.sum(sharp_A))
        nB = int(np.sum(sharp_mask & ~sharp_A))
        Lvals = site_L[sharp_mask]
        print(f"  Sharp split at L>{L_SHARP_THR}: "
              f"sharp_A n={nA}  sharp_B n={nB}")
        print(f"  L range for sharp sites: "
              f"[{Lvals.min():.4f}, {Lvals.max():.4f}]  "
              f"mean={Lvals.mean():.4f}")
    return tier_id, nv, L_site


# ── Freeze-state helpers ──────────────────────────────────────────────────────

def _build_freeze_state(S0, tier_id, k_neigh=tf.KNN_K):
    disp_thr_arr  = np.array([t[1] for t in TIERS2], dtype=np.float64)
    streak_arr    = np.array([t[2] for t in TIERS2], dtype=np.int64)
    neigh_win_arr = np.array([t[3] for t in TIERS2], dtype=np.int64)
    jacc_thr_arr  = np.array([t[4] for t in TIERS2], dtype=np.float64)

    treeS0 = cKDTree(np.ascontiguousarray(S0, dtype=np.float64))
    _, idx0 = treeS0.query(S0, k=k_neigh + 1, workers=-1)
    idx0 = np.sort(idx0[:, 1:], axis=1)

    return {
        "frozen":        np.zeros(len(S0), dtype=bool),
        "low_streak":    np.zeros(len(S0), dtype=np.int64),
        "knn_hist":      [idx0.copy()],
        "idxn_cached":   idx0.copy(),
        "tier_id":       tier_id,
        "disp_thr_arr":  disp_thr_arr,
        "streak_arr":    streak_arr,
        "neigh_win_arr": neigh_win_arr,
        "jacc_thr_arr":  jacc_thr_arr,
    }


# ── Lloyd runners (fast, no geogram — for false-freeze analysis) ──────────────

def run_freeze_v2(S0, P0, F0, N0, vf, treeP0, tier_id,
                  iters=ITERS, k_neigh=tf.KNN_K, k_proj=10):
    """
    Run the new 6-tier freeze policy (no geogram reconstruction).

    Returns
    -------
    disp_hist   : (n, iters) float64
    freeze_time : (n,) int64   iteration at which each site froze; -1 = never
    frozen_final: (n,) bool
    S_final     : (n, 3) float64
    """
    fs = _build_freeze_state(S0, tier_id, k_neigh)
    S  = S0.copy()
    n  = len(S)
    disp_hist   = np.zeros((n, iters), dtype=np.float64)
    freeze_time = np.full(n, -1, dtype=np.int64)

    print(f"\n[FREEZE v2] running {iters} iterations (fast, no geogram) ...")
    for it in range(1, iters + 1):
        S, fs, dd = tf.lloyd_iter_sites_only(
            S, P0, F0, N0, vf, treeP0, fs,
            k_neigh=k_neigh, k_proj=k_proj
        )
        disp_hist[:, it - 1] = dd
        newly = fs["frozen"] & (freeze_time == -1)
        freeze_time[newly] = it
        nfr = int(np.sum(fs["frozen"]))
        if it <= 10 or it % 10 == 0:
            print(f"  iter {it:03d}: frozen={nfr}/{n} ({100*nfr/n:.1f}%)")

    return disp_hist, freeze_time, fs["frozen"], S


def run_nofreeze(S0, P0, F0, N0, vf, treeP0, tier_id,
                 iters=NF_ITERS, k_neigh=tf.KNN_K, k_proj=10):
    """
    Run Lloyd with freeze gates disabled (streak_arr = 999999).
    Returns disp_hist and S_final.
    """
    fs = _build_freeze_state(S0, tier_id, k_neigh)
    fs["streak_arr"] = np.full(len(TIERS2), 999999, dtype=np.int64)

    S = S0.copy()
    n = len(S)
    disp_hist = np.zeros((n, iters), dtype=np.float64)

    print(f"\n[NOFREEZE] running {iters} iterations (fast, no geogram) ...")
    for it in range(1, iters + 1):
        S, fs, dd = tf.lloyd_iter_sites_only(
            S, P0, F0, N0, vf, treeP0, fs,
            k_neigh=k_neigh, k_proj=k_proj
        )
        disp_hist[:, it - 1] = dd
        if it <= 10 or it % 10 == 0:
            print(f"  iter {it:03d}: mean_disp={dd.mean():.4e}")

    return disp_hist, S


# ── Per-iteration geogram quality run ─────────────────────────────────────────

def run_with_geogram_quality(mode, S0, P0, F0, N0, vf, treeP0, tier_id,
                              out_dir, objname, iters,
                              vorpalite_exe=VORPALITE_EXE,
                              k_neigh=tf.KNN_K, k_proj=10,
                              geogram_radius="5%", geogram_nb_neighbors=30):
    """
    Run Lloyd for `iters` iterations with per-iteration geogram reconstruction
    and quality measurement.

    mode : "FREEZEv2" or "NOFREEZE"

    Returns
    -------
    freeze_time  : (n,) int64  (all -1 for NOFREEZE)
    frozen_final : (n,) bool
    S_final      : (n, 3) float64
    csv_path     : str  path to the written CSV
    """
    mode_tag = mode.upper()
    fs = _build_freeze_state(S0, tier_id, k_neigh)
    if mode == "NOFREEZE":
        fs["streak_arr"] = np.full(len(TIERS2), 999999, dtype=np.int64)

    S = S0.copy()
    n = len(S)
    freeze_time = np.full(n, -1, dtype=np.int64)

    geo_dir  = os.path.join(out_dir, f"geogram_{mode_tag.lower()}")
    os.makedirs(geo_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{objname}_{mode_tag}.csv")
    header   = ["objname", "mode", "iter", "sites", "frozen",
                "site_disp_mean", "site_disp_max",
                "geogram_V", "geogram_F", "avg_aspect", "pct_gt_90", "pct_lt_30"]
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print(f"\n[{mode_tag}] running {iters} iterations with geogram quality ...")
    for it in range(0, iters + 1):
        if it == 0:
            disp_mean = disp_max = 0.0
        else:
            S, fs, dd = tf.lloyd_iter_sites_only(
                S, P0, F0, N0, vf, treeP0, fs,
                k_neigh=k_neigh, k_proj=k_proj
            )
            disp_mean = float(dd.mean())
            disp_max  = float(dd.max())
            newly = fs["frozen"] & (freeze_time == -1)
            freeze_time[newly] = it

        nFrozen = int(np.sum(fs["frozen"]))
        sites_xyz = os.path.join(geo_dir, f"{mode_tag}_{it:03d}.xyz")
        obj_path  = os.path.join(geo_dir, f"{mode_tag}_{it:03d}.obj")

        tf.write_xyz(sites_xyz, S)
        tf.geogram_reconstruct_from_sites_xyz(
            vorpalite_exe, sites_xyz, obj_path,
            radius=geogram_radius, nb_neighbors=geogram_nb_neighbors
        )

        geo      = tf.load_trimesh_any(obj_path)
        Vg       = np.asarray(geo.vertices, dtype=np.float64)
        Fg       = np.asarray(geo.faces,    dtype=np.int64)
        avg_ar, pct_gt_90, pct_lt_30 = tf.triangle_quality_metrics(Vg, Fg)

        print(f"  [{mode_tag}] iter {it:03d}:  frozen={nFrozen}/{n} "
              f" avg_aspect={avg_ar:.4f}  %>90={pct_gt_90:.2f}%  %<30={pct_lt_30:.2f}%")

        tf.append_mesh_csv(csv_path, {
            "objname": objname,   "mode": mode_tag,  "iter": it,
            "sites":   n,         "frozen": nFrozen,
            "site_disp_mean": disp_mean, "site_disp_max": disp_max,
            "geogram_V":  int(Vg.shape[0]),
            "geogram_F":  int(Fg.shape[0]),
            "avg_aspect": float(avg_ar),
            "pct_gt_90":  float(pct_gt_90),
            "pct_lt_30":  float(pct_lt_30),
        }, header)

    return freeze_time, fs["frozen"], S, csv_path


# ── Quality metrics on final positions ───────────────────────────────────────

def site_quality_metrics(S, P0, tag=""):
    """
    NN-distance CV, energy proxy, coverage mean/max (no geogram required).
    """
    tree = cKDTree(np.ascontiguousarray(S, dtype=np.float64))

    d2, _ = tree.query(S, k=2, workers=-1)
    nn      = d2[:, 1]
    nn_mean = float(nn.mean())
    nn_std  = float(nn.std())
    nn_cv   = nn_std / nn_mean if nn_mean > 0 else float("nan")
    energy  = float(np.mean(nn ** 2))

    d_cov, _ = tree.query(np.ascontiguousarray(P0, dtype=np.float64), k=1, workers=-1)
    cov_mean = float(d_cov.mean())
    cov_max  = float(d_cov.max())

    label = f"[Quality {tag}]" if tag else "[Quality]"
    print(f"{label}")
    print(f"  NN dist   : mean={nn_mean:.5f}  std={nn_std:.5f}  CV={nn_cv:.4f}")
    print(f"  Energy    : {energy:.6e}")
    print(f"  Coverage  : mean={cov_mean:.5f}  max={cov_max:.5f}")

    return dict(tag=tag, nn_mean=nn_mean, nn_std=nn_std, nn_cv=nn_cv,
                energy=energy, cov_mean=cov_mean, cov_max=cov_max)


# ── Analysis ──────────────────────────────────────────────────────────────────

def false_freeze_rate(freeze_time, nf_disp_hist, tier_id, lookahead=LOOKAHEAD):
    """
    For each frozen site, check if its displacement in NOFREEZE exceeds
    its disp_thr within `lookahead` iterations after the freeze iteration.
    """
    disp_thr_arr = np.array([t[1] for t in TIERS2], dtype=np.float64)
    n = len(freeze_time)
    false_arr = np.zeros(n, dtype=bool)
    nf_iters  = nf_disp_hist.shape[1]

    for i in np.flatnonzero(freeze_time >= 0):
        ft  = int(freeze_time[i])
        thr = disp_thr_arr[int(tier_id[i])]
        col_start = ft
        col_end   = min(col_start + lookahead, nf_iters)
        if col_start < nf_iters:
            if np.any(nf_disp_hist[i, col_start:col_end] > thr):
                false_arr[i] = True

    return false_arr


def per_tier_stats(tier_id, freeze_time, false_arr):
    rows = []
    for tid, name in enumerate(TIER_NAMES2):
        mask   = (tier_id == tid)
        total  = int(np.sum(mask))
        frozen = int(np.sum(mask & (freeze_time >= 0)))
        false  = int(np.sum(mask & false_arr))
        frac   = 100.0 * frozen / total if total > 0 else 0.0
        ffr    = 100.0 * false  / frozen if frozen > 0 else float("nan")
        rows.append(dict(tier=name, total=total, frozen=frozen,
                         frac_frozen=frac, false_rate=ffr))
    return rows


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_iter_quality_comparison(csv_v2, csv_nf, csv_5tier,
                                  out_dir, objname):
    """
    4-panel figure matching the archive_5tier style:
      avg_aspect | pct_gt_90
      pct_lt_30  | frozen count
    Compares FREEZE v2 (new), NOFREEZE, original 5-tier FREEZE.
    """
    df_v2   = pd.read_csv(csv_v2)
    df_nf   = pd.read_csv(csv_nf)   if (csv_nf   and os.path.exists(csv_nf))   else None
    df_5t   = pd.read_csv(csv_5tier) if (csv_5tier and os.path.exists(csv_5tier)) else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{objname} — FREEZE v2 (L_thr={L_SHARP_THR}) vs NOFREEZE vs 5-tier FREEZE",
        fontsize=13, fontweight="bold"
    )

    def _plot(ax, col, ylabel, title):
        ax.plot(df_v2["iter"], df_v2[col], "m-^", markersize=3,
                label="FREEZE v2 (6-tier)", alpha=0.9)
        if df_nf is not None:
            ax.plot(df_nf["iter"], df_nf[col], "b-o", markersize=3,
                    label="NOFREEZE", alpha=0.8)
        if df_5t is not None:
            ax.plot(df_5t["iter"], df_5t[col], "g-s", markersize=3,
                    label="5-tier FREEZE", alpha=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], "avg_aspect",  "Avg Aspect Ratio",
          "Average Aspect Ratio (lower = better)")
    _plot(axes[0, 1], "pct_gt_90",   "% triangles > 90°",
          "Obtuse Triangles % (lower = better)")
    _plot(axes[1, 0], "pct_lt_30",   "% triangles < 30°",
          "Skinny Triangles % (lower = better)")

    # Frozen count panel
    ax = axes[1, 1]
    n_sites = int(df_v2["sites"].iloc[0])
    ax.plot(df_v2["iter"], df_v2["frozen"], "m-^", markersize=3,
            label="FREEZE v2 (6-tier)", alpha=0.9)
    if df_5t is not None:
        ax.plot(df_5t["iter"], df_5t["frozen"], "g-s", markersize=3,
                label="5-tier FREEZE", alpha=0.8)
    ax.axhline(y=n_sites, color="gray", linestyle="--", alpha=0.5,
               label=f"Total sites ({n_sites})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Frozen sites")
    ax.set_title("Freeze Progression")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "quality_over_iters.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_quality_comparison(q_freeze, q_nofreeze, out_dir, objname):
    metrics = ["nn_cv", "energy", "cov_mean", "cov_max"]
    labels  = ["NN-dist CV\n(lower=more uniform)",
               "Energy proxy\n(mean sq NN dist, lower=better)",
               "Coverage mean\n(mesh→site, lower=better)",
               "Coverage max\n(lower=better)"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, m, lbl in zip(axes, metrics, labels):
        vf = q_freeze[m]
        vn = q_nofreeze[m]
        bars = ax.bar(["FREEZE v2", "NOFREEZE"], [vf, vn],
                      color=["#9C27B0", "#2196F3"], alpha=0.85)
        ax.bar_label(bars, fmt="%.5f", fontsize=8)
        ax.set_title(lbl, fontsize=8)
        ax.set_ylim(0, max(vf, vn) * 1.25)

    plt.suptitle(
        f"{objname} — Site-distribution quality: FREEZE v2 vs NOFREEZE (60 iters)",
        fontsize=10
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "quality_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_position_deviation(S_freeze, S_nofreeze, tier_id, frozen, out_dir, objname):
    dev = np.linalg.norm(S_freeze - S_nofreeze, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colours = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]

    ax = axes[0]
    ax.hist(dev, bins=60, color="#607D8B", alpha=0.8)
    ax.axvline(dev.mean(), color="red", linestyle="--",
               label=f"mean={dev.mean():.5f}")
    ax.set_xlabel("||S_freeze − S_nofreeze||"); ax.set_ylabel("count")
    ax.set_title(f"{objname} — Position deviation (all sites)")
    ax.legend()

    ax = axes[1]
    for tid_val, (col, name) in enumerate(zip(colours, TIER_NAMES2)):
        mask = (tier_id == tid_val)
        if not np.any(mask):
            continue
        ax.hist(dev[mask], bins=30, alpha=0.6, color=col,
                label=f"{name} mean={dev[mask].mean():.4f}")
    ax.set_xlabel("||S_freeze − S_nofreeze||"); ax.set_ylabel("count")
    ax.set_title("Per-tier position deviation")
    ax.legend(fontsize=7)

    plt.suptitle(
        f"{objname} — How much do frozen positions differ from NOFREEZE final positions?",
        fontsize=9
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "position_deviation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    print(f"\n[Position deviation  FREEZE v2 vs NOFREEZE]")
    print(f"  All sites : mean={dev.mean():.5f}  max={dev.max():.5f}")
    frozen_dev   = dev[frozen]
    unfrozen_dev = dev[~frozen]
    if frozen_dev.size:
        print(f"  Frozen    : mean={frozen_dev.mean():.5f}  max={frozen_dev.max():.5f}  "
              f"(n={frozen_dev.size})")
    if unfrozen_dev.size:
        print(f"  Unfrozen  : mean={unfrozen_dev.mean():.5f}  max={unfrozen_dev.max():.5f}  "
              f"(n={unfrozen_dev.size})")
    for tid_val, name in enumerate(TIER_NAMES2):
        mask = (tier_id == tid_val)
        if np.any(mask):
            print(f"  {name:<10}: mean={dev[mask].mean():.5f}  max={dev[mask].max():.5f}")


def plot_tier_assignment(nv, tier_id, L_site, out_dir, objname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colours = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]

    ax = axes[0]
    for tid, (col, lbl) in enumerate(zip(colours, TIER_NAMES2)):
        mask = (tier_id == tid)
        ax.hist(nv[mask], bins=60, alpha=0.7, color=col, label=f"{lbl} (n={mask.sum()})")
    for thr in tf.TIER_THRESHOLDS:
        ax.axvline(thr, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("NV score"); ax.set_ylabel("count")
    ax.set_title(f"{objname} — 6-tier NV assignment"); ax.legend(fontsize=7)

    ax = axes[1]
    sharp_mask = (tier_id >= 4)
    if np.any(sharp_mask):
        L_s   = L_site[sharp_mask]
        types = tier_id[sharp_mask]
        ax.hist(L_s[types == 4], bins=30, alpha=0.7, color="#F44336",
                label=f"sharp_B (L≤{L_SHARP_THR}) n={(types==4).sum()}")
        ax.hist(L_s[types == 5], bins=30, alpha=0.7, color="#9C27B0",
                label=f"sharp_A (L>{L_SHARP_THR}) n={(types==5).sum()}")
        ax.axvline(L_SHARP_THR, color="black", linestyle="--",
                   label=f"L threshold={L_SHARP_THR}")
    ax.set_xlabel("L (Normal Covariance Linearity)")
    ax.set_ylabel("count"); ax.set_title("Sharp-tier L distribution")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "tier_assignment.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_comparison(rows, out_dir, objname, use_baseline):
    """Bar chart: frozen fraction + false-freeze rate, V2 vs Exp 8 baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    tier_display = [r["tier"] for r in rows]
    x = np.arange(len(rows)); w = 0.35

    ax = axes[0]
    frac_v2 = [r["frac_frozen"] for r in rows]
    ax.bar(x, frac_v2, w, label="V2 (new)", color="#9C27B0", alpha=0.85)
    if use_baseline:
        frac_base = []
        for r in rows:
            name = r["tier"]
            if name in EXP8_BASELINE:
                b = EXP8_BASELINE[name]
                frac_base.append(100.0 * b["frozen"] / b["total"])
            elif name in ("sharp_A", "sharp_B"):
                b = EXP8_BASELINE["sharp"]
                frac_base.append(100.0 * b["frozen"] / b["total"])
            else:
                frac_base.append(float("nan"))
        frac_base = np.array(frac_base, dtype=np.float64)
        valid = np.isfinite(frac_base)
        ax.bar(x[valid] + w, frac_base[valid], w,
               label="Exp 8 (5-tier baseline)", color="#2196F3", alpha=0.7)
    ax.set_xticks(x + w / 2); ax.set_xticklabels(tier_display, rotation=20, ha="right")
    ax.set_ylabel("Frozen fraction (%)"); ax.set_title(f"{objname} — Frozen fraction")
    ax.legend(); ax.set_ylim(0, 110)

    ax = axes[1]
    ffr_v2 = [r["false_rate"] if np.isfinite(r["false_rate"]) else 0.0 for r in rows]
    ax.bar(x, ffr_v2, w * 1.5, color="#FF5722", alpha=0.85)
    if use_baseline:
        ffr_base = []
        for r in rows:
            name = r["tier"]
            if name in EXP8_BASELINE:
                ffr_base.append(EXP8_BASELINE[name]["false_rate"])
            elif name in ("sharp_A", "sharp_B"):
                ffr_base.append(EXP8_BASELINE["sharp"]["false_rate"])
            else:
                ffr_base.append(float("nan"))
        ffr_base = np.array(ffr_base, dtype=np.float64)
        valid = np.isfinite(ffr_base)
        ax.bar(x[valid] + w, ffr_base[valid], w,
               label="Exp 8 (5-tier baseline)", color="#2196F3", alpha=0.7)
        ax.legend()
    ax.set_xticks(x + w / 4); ax.set_xticklabels(tier_display, rotation=20, ha="right")
    ax.set_ylabel("False-freeze rate (%)"); ax.set_title(f"{objname} — False-freeze rate")
    ax.set_ylim(0, 110)

    plt.suptitle(
        f"{objname}  V2 policy (L_thr={L_SHARP_THR})  —  "
        f"sharp_A: streak=6 jacc=0.88 win=8  |  "
        f"sharp_B: streak=8 jacc=0.95 win=12",
        fontsize=9
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "performance_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_freeze_progress(freeze_time, tier_id, iters, out_dir, objname):
    fig, ax = plt.subplots(figsize=(9, 4))
    colours = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]
    for tid, (col, name) in enumerate(zip(colours, TIER_NAMES2)):
        mask = (tier_id == tid)
        if not np.any(mask):
            continue
        cum = np.array([
            np.sum(mask & (freeze_time >= 0) & (freeze_time <= it))
            for it in range(1, iters + 1)
        ])
        ax.plot(range(1, iters + 1), cum, color=col, label=name, linewidth=1.8)
    ax.set_xlabel("iteration"); ax.set_ylabel("cumulative frozen sites")
    ax.set_title(f"{objname} — Freeze progress per tier (V2)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "freeze_progress.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(rows, out_dir, objname, use_baseline):
    lines = []
    lines.append(f"testfreeze2.py — Performance Report: {objname}")
    lines.append(f"Policy: 6-tier with L_SHARP_THR={L_SHARP_THR}, "
                 f"ITERS={ITERS}, LOOKAHEAD={LOOKAHEAD}")
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"{'Tier':<12} {'Total':>6} {'Frozen':>7} {'Frac%':>7} "
                 f"{'FFR%':>8}  {'Params (streak/win/jacc)':>28}")
    lines.append("-" * 72)

    total_total = total_frozen = 0
    for r in rows:
        tid = TIER_NAMES2.index(r["tier"])
        t   = TIERS2[tid]
        ffr = f"{r['false_rate']:.1f}" if np.isfinite(r['false_rate']) else "n/a"
        lines.append(
            f"{r['tier']:<12} {r['total']:>6} {r['frozen']:>7} "
            f"{r['frac_frozen']:>6.1f}%  {ffr:>7}%  "
            f"  streak={t[2]} win={t[3]} jacc={t[4]}"
        )
        total_total  += r["total"]
        total_frozen += r["frozen"]

    lines.append("-" * 72)
    lines.append(f"{'TOTAL':<12} {total_total:>6} {total_frozen:>7} "
                 f"{100*total_frozen/total_total:>6.1f}%")
    lines.append("=" * 72)

    if use_baseline:
        lines.append("")
        lines.append("Comparison with Exp 8 baseline (teapot, same 60-iter run):")
        lines.append(f"  Exp 8 total frozen: 1837/3644 = 50.4%")
        lines.append(f"  V2   total frozen: {total_frozen}/{total_total} "
                     f"= {100*total_frozen/total_total:.1f}%")
        delta = 100*total_frozen/total_total - 50.4
        lines.append(f"  Change: {delta:+.1f} percentage points")
        lines.append("")
        lines.append("  Sharp breakdown:")
        for r in rows:
            if r["tier"] in ("sharp_A", "sharp_B"):
                lines.append(f"    {r['tier']:<10}: {r['frozen']}/{r['total']} "
                              f"({r['frac_frozen']:.1f}%)")
        b = EXP8_BASELINE["sharp"]
        lines.append(f"    Exp 8 sharp: {b['frozen']}/{b['total']} "
                     f"({100*b['frozen']/b['total']:.1f}%)")

    report = "\n".join(lines)
    print("\n" + report)
    path = os.path.join(out_dir, "performance_report.txt")
    with open(path, "w") as f:
        f.write(report + "\n")
    print(f"\n  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="testfreeze2 — 6-tier L-split policy")
    p.add_argument("mesh",       nargs="?", default="teapot")
    p.add_argument("--iters",    type=int,  default=ITERS)
    p.add_argument("--lookahead",type=int,  default=LOOKAHEAD)
    p.add_argument("--k_neigh",  type=int,  default=tf.KNN_K)
    p.add_argument("--k_proj",   type=int,  default=10)
    p.add_argument("--vorpalite",type=str,  default=VORPALITE_EXE)
    p.add_argument("--geogram_radius", type=str, default="5%")
    p.add_argument("--geogram_nb_neighbors", type=int, default=30)
    args = p.parse_args()

    objname  = args.mesh
    obj_path = os.path.join(_ROOT, "meshes", f"{objname}.obj")
    if not os.path.exists(obj_path):
        obj_path = os.path.join(_ROOT, f"{objname}.obj")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Mesh not found: {objname}")

    out_dir = os.path.join(_ROOT, "results", objname, "testfreeze2")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load mesh ──────────────────────────────────────────────────────────
    print(f"[testfreeze2] Loading {objname} ...")
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
    print(f"  V={P0.shape[0]}  F={F0.shape[0]}  n_sites={len(S0)}")

    # ── Tier assignment ────────────────────────────────────────────────────
    print("\n[testfreeze2] Computing tier assignment ...")
    tier_id, nv, L_site = compute_tier_id_v2(S0, P0, N0)
    for tid, name in enumerate(TIER_NAMES2):
        n_t = int(np.sum(tier_id == tid))
        print(f"  {name:<10}: n={n_t}")
    plot_tier_assignment(nv, tier_id, L_site, out_dir, objname)

    # ── FREEZE v2 with geogram quality ────────────────────────────────────
    freeze_time, frozen_final, S_freeze, csv_v2 = run_with_geogram_quality(
        "FREEZEv2", S0, P0, F0, N0, vf, treeP0, tier_id,
        out_dir, objname, args.iters,
        vorpalite_exe=args.vorpalite,
        k_neigh=args.k_neigh, k_proj=args.k_proj,
        geogram_radius=args.geogram_radius,
        geogram_nb_neighbors=args.geogram_nb_neighbors,
    )

    # ── NOFREEZE with geogram quality ─────────────────────────────────────
    _, _, S_nofreeze, csv_nf_new = run_with_geogram_quality(
        "NOFREEZE", S0, P0, F0, N0, vf, treeP0, tier_id,
        out_dir, objname, args.iters,
        vorpalite_exe=args.vorpalite,
        k_neigh=args.k_neigh, k_proj=args.k_proj,
        geogram_radius=args.geogram_radius,
        geogram_nb_neighbors=args.geogram_nb_neighbors,
    )

    # ── Existing 5-tier FREEZE CSV (for cross-policy comparison) ──────────
    csv_5tier = os.path.join(_ROOT, "results", objname, "freeze",
                             f"{objname}_FREEZE.csv")
    if not os.path.exists(csv_5tier):
        csv_5tier = None
        print(f"  (5-tier FREEZE CSV not found — skipping 5-tier comparison)")

    # ── 4-panel quality-over-iterations figure ────────────────────────────
    print("\n[testfreeze2] Generating quality-over-iterations figure ...")
    plot_iter_quality_comparison(csv_v2, csv_nf_new, csv_5tier, out_dir, objname)

    # ── NOFREEZE run for false-freeze evaluation (fast, no geogram) ───────
    nf_iters_ff = args.iters + args.lookahead
    nf_disp, _ = run_nofreeze(S0, P0, F0, N0, vf, treeP0, tier_id,
                               iters=nf_iters_ff,
                               k_neigh=args.k_neigh,
                               k_proj=args.k_proj)

    # ── False-freeze analysis ──────────────────────────────────────────────
    print("\n[testfreeze2] Analysing false-freeze rates ...")
    false_arr = false_freeze_rate(freeze_time, nf_disp, tier_id,
                                  lookahead=args.lookahead)
    rows = per_tier_stats(tier_id, freeze_time, false_arr)

    use_baseline = (objname == "teapot")
    write_report(rows, out_dir, objname, use_baseline)
    plot_comparison(rows, out_dir, objname, use_baseline)
    plot_freeze_progress(freeze_time, tier_id, args.iters, out_dir, objname)

    # ── Final-position quality (NN-dist / coverage) ───────────────────────
    print("\n[testfreeze2] Final-position quality metrics ...")
    q_freeze   = site_quality_metrics(S_freeze,   P0, tag="FREEZE v2")
    q_nofreeze = site_quality_metrics(S_nofreeze, P0, tag="NOFREEZE")
    plot_quality_comparison(q_freeze, q_nofreeze, out_dir, objname)
    plot_position_deviation(S_freeze, S_nofreeze, tier_id, frozen_final,
                            out_dir, objname)

    # ── Save raw arrays ────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "freeze_time.npy"),   freeze_time)
    np.save(os.path.join(out_dir, "tier_id.npy"),       tier_id)
    np.save(os.path.join(out_dir, "nv.npy"),            nv)
    np.save(os.path.join(out_dir, "L_site.npy"),        L_site)
    np.save(os.path.join(out_dir, "S_freeze.npy"),      S_freeze)
    np.save(os.path.join(out_dir, "S_nofreeze.npy"),    S_nofreeze)
    print(f"\n  Arrays saved to {out_dir}/")
    print(f"\n[testfreeze2] Done — {objname}.")


if __name__ == "__main__":
    main()
