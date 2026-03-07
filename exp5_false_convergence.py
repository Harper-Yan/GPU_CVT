#!/usr/bin/env python3
"""
Experiment 5: False Convergence Analysis
Traces directly into the freeze CVT process to justify tier-dependent
streak and Jaccard thresholds.

Three indicators:
  5a. Streak Survival Curves — P(streak >= k+1 | streak >= k) per tier.
      Shows how fragile apparent convergence is at high curvature.
  5b. Jaccard at Low-Displacement — when a site's displacement is below
      its tier threshold (would pass disp check), what is its Jaccard?
      Shows that low displacement ≠ stable neighbors for high curvature.
  5c. False Freeze Simulation — freeze uniformly at streak=2. Count how
      many "frozen" sites would have moved above threshold in the next
      5 iterations. Per-tier premature-freeze rate.
"""
import numpy as np
import os
import sys
import csv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import trimesh
from scipy.spatial import cKDTree
from testfreeze import (
    nr, tangent, normal_variation_score, closest_point_tri,
    cell_poly2d, poly_area_centroid_2d, jaccard_sorted_int,
    NV_K, TIER_THRESHOLDS, TIERS,
)
import matplotlib.pyplot as plt

ITERS = 50
K_NEIGH = 32
K_PROJ = 10


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
    """Run Lloyd without freeze, recording displacement and KNN per iteration."""
    n = len(S0)
    S = S0.copy()
    disp_history = np.zeros((n, iters), dtype=np.float64)
    # Store KNN as list of (n, K_NEIGH) arrays
    knn_history = []

    for it in range(iters):
        _, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64), k=1, workers=-1)
        N = N0[idxN]
        U, V = tangent(N)

        treeS = cKDTree(np.ascontiguousarray(S, dtype=np.float64))
        _, knn = treeS.query(S, k=K_NEIGH + 1, workers=-1)
        knn = knn[:, 1:]
        knn_sorted = np.sort(knn, axis=1)
        knn_history.append(knn_sorted.copy())

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
                faces_set = set()
                for v in nn:
                    for fi in vf[int(v)]:
                        faces_set.add(fi)
                face_indices = np.fromiter(faces_set, dtype=np.int64)
                tris = F0[face_indices] if face_indices.size else np.empty((0, 3), dtype=np.int64)
                Snew[i] = update_to_mesh(cent3d[i], tris, P0)

        disp_history[:, it] = np.linalg.norm(Snew - S, axis=1)
        S = Snew
        print(f"  iter {it+1}/{iters}")

    return disp_history, knn_history


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mesh", nargs="?", default=os.path.join(_ROOT, "meshes", "teapot.obj"))
    args = p.parse_args()
    in_path = args.mesh
    objname = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = os.path.join(_ROOT, "results", objname, "exp5_false_convergence")
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

    # Curvature and tier assignment
    nv_score = normal_variation_score(P0, nr(N0), NV_K)
    n_tiers = len(TIERS)
    tier_id = np.digitize(nv_score, TIER_THRESHOLDS).astype(np.int64)
    tier_names = [t[0] for t in TIERS]
    tier_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#F44336"]
    disp_thr_per_tier = np.array([t[1] for t in TIERS], dtype=np.float64)

    print(f"Running {ITERS} NOFREEZE iterations tracking displacement + KNN...")
    disp_history, knn_history = lloyd_nofreeze_track_all(S0, P0, F0, N0, treeP0, vf, ITERS)

    n = len(S0)

    # =====================================================================
    # 5a. STREAK SURVIVAL CURVES
    # =====================================================================
    # Use each tier's OWN disp_thr. Track streak per site. When streak
    # breaks, record the length. Compute survival: P(streak>=k+1 | streak>=k).
    print("\n--- 5a: Streak survival analysis ---")
    MAX_STREAK = 15
    # For each tier, count how many times a streak of exactly length k was observed
    # streak_starts[tier][k] = number of times a streak reached k (before possibly breaking or continuing)
    streak_reached = np.zeros((n_tiers, MAX_STREAK + 1), dtype=np.int64)  # [tier][k] = count of streaks that reached k

    low_streak = np.zeros(n, dtype=np.int64)
    for it in range(ITERS):
        disp = disp_history[:, it]
        thr = disp_thr_per_tier[tier_id]
        low = disp < thr
        # Before updating: sites with low_streak > 0 that are about to break
        breaking = (~low) & (low_streak > 0)
        for i in np.flatnonzero(breaking):
            k = int(low_streak[i])
            t = int(tier_id[i])
            # This streak reached lengths 1..k, now breaks
            for kk in range(1, min(k, MAX_STREAK) + 1):
                streak_reached[t, kk] += 1
        # At end of all iterations, surviving streaks also count
        low_streak = np.where(low, low_streak + 1, 0)

    # Count surviving streaks at end
    for i in range(n):
        k = int(low_streak[i])
        if k > 0:
            t = int(tier_id[i])
            for kk in range(1, min(k, MAX_STREAK) + 1):
                streak_reached[t, kk] += 1

    # Survival probability: P(streak >= k+1 | streak >= k) = reached[k+1] / reached[k]
    survival = np.zeros((n_tiers, MAX_STREAK), dtype=np.float64)
    for t in range(n_tiers):
        for k in range(1, MAX_STREAK):
            if streak_reached[t, k] > 0:
                survival[t, k] = streak_reached[t, k + 1] / streak_reached[t, k]

    # Print step-by-step survival
    print(f"  {'Tier':>10s}", end="")
    for k in range(1, 11):
        print(f"  P(>={k+1}|>={k})", end="")
    print()
    for t in range(n_tiers):
        print(f"  {tier_names[t]:>10s}", end="")
        for k in range(1, 11):
            print(f"       {survival[t,k]:.3f}", end="")
        print()

    # Cumulative reliability P(streak >= k | streak >= 1) = streak_reached[t,k] / streak_reached[t,1]
    print(f"\n  Cumulative P(streak >= k | streak >= 1):")
    print(f"  {'k':>4s}", end="")
    for t in range(n_tiers):
        print(f"  {tier_names[t]:>10s}", end="")
    print()
    for k in [2, 3, 4, 5, 6, 7, 8, 10]:
        print(f"  {k:>4d}", end="")
        for t in range(n_tiers):
            base = streak_reached[t, 1]
            val = streak_reached[t, k] / base if base > 0 and k <= MAX_STREAK else float("nan")
            print(f"  {val:>10.1%}", end="")
        print()

    # =====================================================================
    # 5b. JACCARD AT LOW-DISPLACEMENT MOMENTS
    # =====================================================================
    # When a site has disp < its tier threshold, compute Jaccard with prev KNN.
    # Shows whether low displacement guarantees stable neighbors.
    print("\n--- 5b: Jaccard at low-displacement moments ---")
    # Collect Jaccard values per tier when displacement is low
    jacc_at_low_disp = [[] for _ in range(n_tiers)]

    for it in range(1, ITERS):
        disp = disp_history[:, it]
        thr = disp_thr_per_tier[tier_id]
        low_mask = disp < thr
        prev_knn = knn_history[it - 1]
        curr_knn = knn_history[it]
        for i in np.flatnonzero(low_mask):
            j = jaccard_sorted_int(prev_knn[i], curr_knn[i])
            jacc_at_low_disp[int(tier_id[i])].append(j)

    jacc_stats = []
    for t in range(n_tiers):
        arr = np.array(jacc_at_low_disp[t]) if jacc_at_low_disp[t] else np.array([1.0])
        jacc_stats.append({
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "below_90": float(np.mean(arr < 0.90)),  # fraction with Jaccard < 0.90
            "below_80": float(np.mean(arr < 0.80)),
            "n_obs": len(arr),
        })
        print(f"  {tier_names[t]:>10s}: n_obs={len(arr):6d}  "
              f"mean_J={jacc_stats[t]['mean']:.4f}  "
              f"med_J={jacc_stats[t]['median']:.4f}  "
              f"frac<0.90={jacc_stats[t]['below_90']:.3f}  "
              f"frac<0.80={jacc_stats[t]['below_80']:.3f}")

    # =====================================================================
    # 5c. FALSE FREEZE SIMULATION
    # =====================================================================
    # Simulate: if we used streak=2 for ALL tiers, how many sites would be
    # "frozen" prematurely? Track post-freeze displacement.
    print("\n--- 5c: False freeze simulation (uniform streak=2) ---")
    UNIFORM_STREAK = 2
    LOOKAHEAD = 5  # check next 5 iters after false freeze

    # Find first iteration where each site achieves streak >= UNIFORM_STREAK
    sim_streak = np.zeros(n, dtype=np.int64)
    first_freeze_iter = np.full(n, -1, dtype=np.int64)
    for it in range(ITERS):
        disp = disp_history[:, it]
        thr = disp_thr_per_tier[tier_id]
        low = disp < thr
        sim_streak = np.where(low, sim_streak + 1, 0)
        newly_frozen = (sim_streak >= UNIFORM_STREAK) & (first_freeze_iter < 0)
        first_freeze_iter[newly_frozen] = it

    # For each site that was "frozen", check max displacement in next LOOKAHEAD iters
    post_freeze_max_disp = np.zeros(n, dtype=np.float64)
    was_frozen = first_freeze_iter >= 0
    for i in np.flatnonzero(was_frozen):
        fi = int(first_freeze_iter[i])
        start = fi + 1
        end = min(fi + 1 + LOOKAHEAD, ITERS)
        if start < ITERS:
            post_freeze_max_disp[i] = disp_history[i, start:end].max()

    # "False convergence" = was frozen at streak=2, but later moved > its threshold
    false_converged = was_frozen & (post_freeze_max_disp > disp_thr_per_tier[tier_id])
    fc_stats = []
    for t in range(n_tiers):
        mask_t = tier_id == t
        n_t = int(mask_t.sum())
        n_frozen_t = int((was_frozen & mask_t).sum())
        n_false_t = int((false_converged & mask_t).sum())
        rate = n_false_t / n_frozen_t if n_frozen_t > 0 else 0.0
        # Mean post-freeze displacement for this tier (among frozen sites)
        frozen_mask = was_frozen & mask_t
        mean_post = float(post_freeze_max_disp[frozen_mask].mean()) if frozen_mask.any() else 0.0
        fc_stats.append({
            "n": n_t,
            "n_frozen": n_frozen_t,
            "n_false": n_false_t,
            "false_rate": rate,
            "mean_post_disp": mean_post,
        })
        print(f"  {tier_names[t]:>10s}: n={n_t}  frozen={n_frozen_t}  "
              f"false_converged={n_false_t}  rate={rate:.3f}  "
              f"mean_post_disp={mean_post:.6f}")

    # =====================================================================
    # CSV OUTPUT
    # =====================================================================
    # 5a CSV: streak survival
    csv_5a = os.path.join(out_dir, "exp5a_streak_survival.csv")
    with open(csv_5a, "w", newline="") as f:
        w = csv.writer(f)
        header = ["tier"] + [f"P(>={k+1}|>={k})" for k in range(1, MAX_STREAK)]
        w.writerow(header)
        for t in range(n_tiers):
            row = [tier_names[t]] + [f"{survival[t,k]:.4f}" for k in range(1, MAX_STREAK)]
            w.writerow(row)
    print(f"Saved: {csv_5a}")

    # 5b CSV: Jaccard at low displacement
    csv_5b = os.path.join(out_dir, "exp5b_jaccard_at_low_disp.csv")
    with open(csv_5b, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n_obs", "mean_jaccard", "median_jaccard", "std_jaccard",
                     "frac_below_0.90", "frac_below_0.80"])
        for t in range(n_tiers):
            s = jacc_stats[t]
            w.writerow([tier_names[t], s["n_obs"],
                        f"{s['mean']:.4f}", f"{s['median']:.4f}", f"{s['std']:.4f}",
                        f"{s['below_90']:.4f}", f"{s['below_80']:.4f}"])
    print(f"Saved: {csv_5b}")

    # 5c CSV: false freeze simulation
    csv_5c = os.path.join(out_dir, "exp5c_false_freeze.csv")
    with open(csv_5c, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n", "n_frozen_streak2", "n_false_converged",
                     "false_convergence_rate", "mean_post_freeze_disp"])
        for t in range(n_tiers):
            s = fc_stats[t]
            w.writerow([tier_names[t], s["n"], s["n_frozen"], s["n_false"],
                        f"{s['false_rate']:.4f}", f"{s['mean_post_disp']:.6f}"])
    print(f"Saved: {csv_5c}")

    # =====================================================================
    # FIGURES
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"Experiment 5: False Convergence Analysis — {objname} (5-Tier)",
                 fontsize=14, fontweight="bold")

    # --- (0,0) Streak Survival Curves ---
    ax = axes[0, 0]
    ks = np.arange(1, 11)
    for t in range(n_tiers):
        ax.plot(ks, survival[t, 1:11], marker="o", color=tier_colors[t],
                label=tier_names[t], linewidth=2, markersize=5)
    ax.set_xlabel("Streak length k")
    ax.set_ylabel("P(streak ≥ k+1 | streak ≥ k)")
    ax.set_title("5a: Streak Survival Probability")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95%")

    # --- (0,1) Jaccard at Low-Displacement: bar chart ---
    ax = axes[0, 1]
    x = np.arange(n_tiers)
    means = [jacc_stats[t]["mean"] for t in range(n_tiers)]
    stds = [jacc_stats[t]["std"] for t in range(n_tiers)]
    ax.bar(x, means, yerr=stds, capsize=5, color=tier_colors, alpha=0.8,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Jaccard Similarity")
    ax.set_title("5b: Mean Jaccard When Displacement < Threshold")
    for i, (m_, s_) in enumerate(zip(means, stds)):
        ax.text(i, m_ + s_ + 0.01, f"{m_:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (1,0) Jaccard < 0.90 fraction (shows topology instability despite low displacement) ---
    ax = axes[1, 0]
    frac90 = [jacc_stats[t]["below_90"] for t in range(n_tiers)]
    frac80 = [jacc_stats[t]["below_80"] for t in range(n_tiers)]
    width = 0.35
    ax.bar(x - width / 2, frac90, width, color=tier_colors, alpha=0.7,
           edgecolor="black", linewidth=0.5, label="Jaccard < 0.90")
    ax.bar(x + width / 2, frac80, width, color=tier_colors, alpha=0.4,
           edgecolor="black", linewidth=0.5, label="Jaccard < 0.80")
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Fraction of Low-Disp Moments")
    ax.set_title("5b: Topology Instability Despite Low Displacement")
    for i, v in enumerate(frac90):
        ax.text(i - width / 2, v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (1,1) False Convergence Rate per tier ---
    ax = axes[1, 1]
    rates = [fc_stats[t]["false_rate"] for t in range(n_tiers)]
    ax.bar(x, rates, color=tier_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("False Convergence Rate")
    ax.set_title("5c: False Freeze Rate (uniform streak=2)")
    for i, v in enumerate(rates):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "exp5_false_convergence.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
