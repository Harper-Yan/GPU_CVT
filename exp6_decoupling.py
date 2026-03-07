#!/usr/bin/env python3
"""
Experiment 6: Displacement–Jaccard Decoupling

Tests whether displacement (streak) and Jaccard (neighbor stability) carry
independent information, and whether this independence grows with curvature.

  6a. Conditional Jaccard distribution — restricted to (site, iter) where
      disp < tier_threshold (i.e., the freeze check would pass).
      At flat: Jaccard is tightly clustered near 1 (small std).
        → displacement is sufficient: if you stop moving, neighbors have too.
      At curved: Jaccard is spread wide (large std, high frac < threshold).
        → displacement is insufficient: you can stop moving while neighbors
          are still bouncing. The two signals are decoupled.
      Metric: Spearman r(disp, Jaccard) within the low-disp regime.
        Coupled (flat): even within low-disp, lower disp → higher Jaccard (r < 0).
        Decoupled (curved): within low-disp, disp does not predict Jaccard (r ≈ 0).

  6c. Windowed Jaccard predictive test — using the actual stable_by_tier
      logic from the freeze algorithm (windowed Jaccard over neigh_win iters).
      Simulates two freeze policies:
        Policy A (streak-only): freeze when streak >= tier_streak.
        Policy B (combined):    freeze when streak >= tier_streak
                                AND stable_by_tier (windowed Jaccard >= jacc_thr).
      Per tier:
        - Fraction permanently blocked by Jaccard (A fires, B never fires).
        - Fraction delayed by Jaccard (B fires later than A).
        - Mean post-freeze displacement over LOOKAHEAD iters for A vs B events.
      If Jaccard provides independent signal: B has lower post-event disp than A,
      and the gap grows with curvature tier.
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
    stable_by_tier,
    NV_K, TIER_THRESHOLDS, TIERS,
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ITERS     = 60    # enough for windowed Jaccard to accumulate (max neigh_win=12)
K_NEIGH   = 32
K_PROJ    = 10
LOOKAHEAD = 10    # post-event iters to measure false convergence
SCATTER_N = 2000  # max scatter points per tier


# ---------------------------------------------------------------------------
# Geometry helpers (identical to other exps)
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
    knn_history  = []

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
    out_dir = os.path.join(_ROOT, "results", objname, "exp6_decoupling")
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
    streak_arr        = np.array([t[2] for t in TIERS], dtype=np.int64)
    neigh_win_arr     = np.array([t[3] for t in TIERS], dtype=np.int64)
    jacc_thr_arr      = np.array([t[4] for t in TIERS], dtype=np.float64)
    maxW              = int(np.max(neigh_win_arr))

    print(f"Running {ITERS} iterations (no freeze)...")
    disp_history, knn_history = lloyd_nofreeze_track_all(
        S0, P0, F0, N0, treeP0, vf, ITERS
    )

    # =========================================================================
    # 6a. Conditional Jaccard distribution
    #     Filter: (site, iter) where disp < tier_threshold.
    #     At this moment, the displacement check would pass — the question is:
    #     how much does Jaccard vary? Low std = coupled; high std = decoupled.
    #
    #     Also compute Spearman r(disp, Jaccard) within this conditioned sample.
    #     Coupled: lower disp within the low-disp band still predicts higher Jaccard.
    #     Decoupled: r ≈ 0 — disp tells you nothing about Jaccard in this regime.
    # =========================================================================
    print("\n--- 6a: Conditional Jaccard distribution (low-disp moments) ---")
    cond_disp = [[] for _ in range(n_tiers)]   # disp values when disp < thr
    cond_jacc = [[] for _ in range(n_tiers)]   # Jaccard at those same moments

    for it in range(1, ITERS):
        prev_knn = knn_history[it - 1]
        curr_knn = knn_history[it]
        disp     = disp_history[:, it]
        thr      = disp_thr_per_tier[tier_id]
        for i in np.flatnonzero(disp < thr):
            j_val = jaccard_sorted_int(prev_knn[i], curr_knn[i])
            t     = int(tier_id[i])
            cond_disp[t].append(float(disp[i]))
            cond_jacc[t].append(float(j_val))

    cond_stats = []
    print(f"\n  {'Tier':>10s}  {'n':>7s}  {'mean_J':>8s}  {'std_J':>8s}  {'frac<thr':>9s}  {'Spearman_r':>11s}")
    for t in range(n_tiers):
        j_arr = np.array(cond_jacc[t])
        d_arr = np.array(cond_disp[t])
        if len(j_arr) < 3:
            cond_stats.append({"n": len(j_arr), "mean": float("nan"), "std": float("nan"),
                               "frac_below": float("nan"), "r": float("nan"), "p": float("nan")})
            continue
        r, pv = spearmanr(d_arr, j_arr)
        frac  = float(np.mean(j_arr < jacc_thr_arr[t]))
        cond_stats.append({
            "n": len(j_arr), "mean": float(np.mean(j_arr)), "std": float(np.std(j_arr)),
            "frac_below": frac, "r": float(r), "p": float(pv),
        })
        print(f"  {tier_names[t]:>10s}  {len(j_arr):>7d}  "
              f"{cond_stats[t]['mean']:>8.4f}  {cond_stats[t]['std']:>8.4f}  "
              f"{frac:>9.4f}  {r:>+11.4f}")

    # =========================================================================
    # 6c. Windowed Jaccard predictive test
    #     Policy A (streak-only): freeze at first iter where
    #       streak_count[i] >= streak_arr[tier_id[i]]
    #     Policy B (combined):    freeze at first iter where
    #       streak_count[i] >= streak_arr[tier_id[i]]
    #       AND stable_by_tier(knn_window, tier_id, neigh_win_arr, jacc_thr_arr)[i]
    #
    #     stable_by_tier checks ALL consecutive Jaccard pairs in the last
    #     neigh_win snapshots, so it accumulates evidence over many iterations —
    #     this is the actual freeze condition used in the algorithm.
    # =========================================================================
    print(f"\n--- 6c: Windowed Jaccard predictive test (lookahead={LOOKAHEAD}) ---")
    streak_count       = np.zeros(n, dtype=np.int64)
    first_A_iter       = np.full(n, -1, dtype=np.int64)  # streak-only freeze
    first_B_iter       = np.full(n, -1, dtype=np.int64)  # combined freeze
    knn_window         = []  # rolling window, max neigh_win snapshots

    for it in range(ITERS):
        disp         = disp_history[:, it]
        thr          = disp_thr_per_tier[tier_id]
        low          = disp < thr
        streak_count = np.where(low, streak_count + 1, 0)

        knn_window.append(knn_history[it])
        if len(knn_window) > maxW:
            knn_window.pop(0)

        streak_reached = streak_count >= streak_arr[tier_id]

        # Policy A: first time streak is satisfied
        new_A = streak_reached & (first_A_iter < 0)
        first_A_iter[new_A] = it

        # Policy B: streak satisfied AND windowed Jaccard stable
        candidates = streak_reached & (first_B_iter < 0)
        if np.any(candidates):
            stable_nbr = stable_by_tier(knn_window, tier_id, neigh_win_arr, jacc_thr_arr)
            new_B = candidates & stable_nbr
            first_B_iter[new_B] = it

    # Post-event false convergence: max disp in next LOOKAHEAD iters
    def post_event_disp(event_iters):
        """event_iters: array of shape (n,) with iter index or -1 if no event."""
        post_max = np.zeros(n, dtype=np.float64)
        for i in range(n):
            fi = int(event_iters[i])
            if fi < 0:
                post_max[i] = np.nan
                continue
            start = fi + 1
            end   = min(fi + 1 + LOOKAHEAD, ITERS)
            if start < ITERS:
                post_max[i] = disp_history[i, start:end].max()
        return post_max

    post_A = post_event_disp(first_A_iter)
    post_B = post_event_disp(first_B_iter)

    pred_stats = []
    print(f"\n  {'Tier':>10s}  {'n_A':>6s}  {'n_B':>6s}  {'blocked':>8s}  "
          f"{'delayed':>8s}  {'mean_delay':>10s}  {'false_A':>8s}  {'false_B':>8s}")
    for t in range(n_tiers):
        thr    = float(disp_thr_per_tier[t])
        mask_t = tier_id == t

        A_fired   = (first_A_iter >= 0) & mask_t
        B_fired   = (first_B_iter >= 0) & mask_t
        blocked   = A_fired & ~B_fired          # streak fired, Jaccard never confirmed
        delayed   = A_fired & B_fired & (first_B_iter > first_A_iter)

        n_A       = int(A_fired.sum())
        n_B       = int(B_fired.sum())
        n_blocked = int(blocked.sum())
        n_delayed = int(delayed.sum())

        delay_arr = (first_B_iter[delayed] - first_A_iter[delayed]).astype(float)
        mean_delay = float(delay_arr.mean()) if n_delayed > 0 else 0.0

        fr_A = float(np.nanmean(post_A[A_fired] > thr)) if n_A > 0 else float("nan")
        fr_B = float(np.nanmean(post_B[B_fired] > thr)) if n_B > 0 else float("nan")

        pred_stats.append({
            "n_A": n_A, "n_B": n_B,
            "n_blocked": n_blocked, "frac_blocked": n_blocked / max(n_A, 1),
            "n_delayed":  n_delayed, "frac_delayed":  n_delayed / max(n_A, 1),
            "mean_delay": mean_delay,
            "fr_A": fr_A, "fr_B": fr_B,
        })
        print(f"  {tier_names[t]:>10s}  {n_A:>6d}  {n_B:>6d}  "
              f"{n_blocked:>5d}({n_blocked/max(n_A,1):.1%})  "
              f"{n_delayed:>5d}({n_delayed/max(n_A,1):.1%})  "
              f"{mean_delay:>10.2f}  "
              f"{fr_A:>8.3f}  {fr_B:>8.3f}")

    # =========================================================================
    # CSV output
    # =========================================================================
    csv_6a = os.path.join(out_dir, "exp6a_conditional_jaccard.csv")
    with open(csv_6a, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n_obs", "mean_jaccard", "std_jaccard",
                    "frac_below_jacc_thr", "spearman_r_cond", "p_value"])
        for t in range(n_tiers):
            s = cond_stats[t]
            w.writerow([tier_names[t], s["n"],
                        f"{s['mean']:.4f}", f"{s['std']:.4f}",
                        f"{s['frac_below']:.4f}", f"{s['r']:+.4f}", f"{s['p']:.2e}"])
    print(f"\nSaved: {csv_6a}")

    csv_6c = os.path.join(out_dir, "exp6c_windowed_jaccard.csv")
    with open(csv_6c, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n_streak_fired", "n_combined_fired",
                    "n_blocked", "frac_blocked",
                    "n_delayed", "frac_delayed", "mean_delay_iters",
                    "false_rate_streak_only", "false_rate_combined"])
        for t in range(n_tiers):
            s = pred_stats[t]
            w.writerow([tier_names[t], s["n_A"], s["n_B"],
                        s["n_blocked"], f"{s['frac_blocked']:.4f}",
                        s["n_delayed"], f"{s['frac_delayed']:.4f}",
                        f"{s['mean_delay']:.2f}",
                        f"{s['fr_A']:.4f}", f"{s['fr_B']:.4f}"])
    print(f"Saved: {csv_6c}")

    # =========================================================================
    # Figures
    # =========================================================================
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(
        f"Experiment 6: Displacement–Jaccard Decoupling — {objname}",
        fontsize=14, fontweight="bold"
    )
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.1, 1], hspace=0.52, wspace=0.38)

    # --- Row 0: scatter plots of (disp, Jaccard) WITHIN LOW-DISP REGIME ---
    for t in range(n_tiers):
        ax  = fig.add_subplot(gs[0, t])
        d   = np.array(cond_disp[t])
        j   = np.array(cond_jacc[t])
        if len(d) > SCATTER_N:
            idx = np.random.default_rng(0).choice(len(d), SCATTER_N, replace=False)
            d, j = d[idx], j[idx]
        d = np.maximum(d, 1e-10)
        ax.scatter(d, j, s=2, alpha=0.15, color=tier_colors[t], rasterized=True)
        r   = cond_stats[t]["r"]
        std = cond_stats[t]["std"]
        ax.set_title(f"{tier_names[t]}\nr={r:+.3f}  std={std:.3f}", fontsize=9)
        ax.set_xlabel("disp (log, below thr)", fontsize=7)
        ax.set_ylabel("Jaccard", fontsize=7)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(float(jacc_thr_arr[t]), color="red", linestyle="--", linewidth=0.8,
                   alpha=0.7, label=f"thr={jacc_thr_arr[t]:.2f}")
        ax.legend(fontsize=6, loc="lower right")
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    # --- Row 1, cols 0-1: Conditional Jaccard std and Spearman r ---
    ax  = fig.add_subplot(gs[1, 0:2])
    x   = np.arange(n_tiers)
    stds = [cond_stats[t]["std"] for t in range(n_tiers)]
    bars = ax.bar(x, stds, color=tier_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    # overlay Spearman r as line on secondary axis
    ax2  = ax.twinx()
    rs   = [cond_stats[t]["r"] for t in range(n_tiers)]
    ax2.plot(x, rs, "k--o", linewidth=1.5, markersize=5, label="Spearman r (right)")
    ax2.set_ylabel("Spearman r (within low-disp)", fontsize=8)
    ax2.axhline(0, color="gray", linewidth=0.5)
    for i, r in enumerate(rs):
        ax2.text(i, r - 0.02, f"{r:+.3f}", ha="center", va="top", fontsize=8, color="black")
    for i, s in enumerate(stds):
        ax.text(i, s + 0.002, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Std(Jaccard) | disp < threshold")
    ax.set_title("6a: Jaccard Variability Within Low-Disp Regime\n"
                 "High std = Jaccard is unpredictable when disp is low = decoupled")
    ax.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=7, loc="upper right")

    # --- Row 1, cols 2-3: Fraction blocked / delayed by windowed Jaccard ---
    ax    = fig.add_subplot(gs[1, 2:4])
    width = 0.35
    fb    = [pred_stats[t]["frac_blocked"] for t in range(n_tiers)]
    fd    = [pred_stats[t]["frac_delayed"]  for t in range(n_tiers)]
    ax.bar(x - width/2, fb, width, color=tier_colors, alpha=0.85,
           edgecolor="black", linewidth=0.5, label="blocked (Jaccard never confirms)")
    ax.bar(x + width/2, fd, width, color=tier_colors, alpha=0.45,
           edgecolor="black", linewidth=0.5, hatch="//", label="delayed (confirmed later)")
    for i, (b_, d_) in enumerate(zip(fb, fd)):
        ax.text(i - width/2, b_ + 0.008, f"{b_:.1%}", ha="center", va="bottom", fontsize=7)
        ax.text(i + width/2, d_ + 0.008, f"{d_:.1%}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Fraction of streak-fired sites")
    ax.set_title("6c: Windowed Jaccard Effect on Freeze Decisions\n"
                 "'blocked' = streak passed but Jaccard never confirmed within run")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Row 1, col 4: Post-event false convergence rate A vs B ---
    ax    = fig.add_subplot(gs[1, 4])
    fr_A  = [pred_stats[t]["fr_A"] if np.isfinite(pred_stats[t]["fr_A"]) else 0.0
             for t in range(n_tiers)]
    fr_B  = [pred_stats[t]["fr_B"] if np.isfinite(pred_stats[t]["fr_B"]) else 0.0
             for t in range(n_tiers)]
    ax.bar(x - width/2, fr_A, width, color=tier_colors, alpha=0.85,
           edgecolor="black", linewidth=0.5, label="streak only (A)")
    ax.bar(x + width/2, fr_B, width, color=tier_colors, alpha=0.4,
           edgecolor="black", linewidth=0.5, hatch="//", label="streak+Jaccard (B)")
    for i, (a_, b_) in enumerate(zip(fr_A, fr_B)):
        ax.text(i - width/2, a_ + 0.01, f"{a_:.1%}", ha="center", va="bottom", fontsize=7)
        ax.text(i + width/2, b_ + 0.01, f"{b_:.1%}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Post-freeze false rate")
    ax.set_title(f"6c: Post-event False Rate\n(moved > thr in next {LOOKAHEAD} iters)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    out_png = os.path.join(out_dir, "exp6_decoupling.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
