#!/usr/bin/env python3
"""
Experiment 8: End-to-end Deployed Policy Evaluation

Addresses four open questions simultaneously:

  D1. False convergence rate under the actual deployed policy
      (tier-aware streak + windowed Jaccard) — the number that matters.

  C1. Can CVT energy increase after a freeze?
      Proxy: post-freeze displacement in a counterfactual NOFREEZE run.

  C2. Do frozen site's neighbors shift post-freeze?
      Measured as mean neighbor displacement in NOFREEZE counterfactual.

  C3. Does the algorithm converge globally in practice?
      Tracked as total displacement² per iteration; compared to NOFREEZE.

Method: run both FREEZE (deployed policy) and NOFREEZE in parallel for
ITERS + LOOKAHEAD iterations. At each freeze event, use NOFREEZE
displacement as the counterfactual of what would have happened.
"""
import numpy as np
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import trimesh
from scipy.spatial import cKDTree
from testfreeze import (
    nr, tangent, normal_variation_score,
    cell_poly2d, poly_area_centroid_2d, update_to_mesh,
    jaccard_sorted_int, stable_by_tier, lloyd_iter_sites_only,
    NV_K, TIER_THRESHOLDS, TIERS,
)
import matplotlib.pyplot as plt

ITERS    = 60
LOOKAHEAD = 10
K_NEIGH  = 32
K_PROJ   = 10


def lloyd_nofreeze_step(S, P0, F0, N0, treeP0, vf):
    """One NOFREEZE Lloyd iteration; returns (Snew, displacement)."""
    _, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64), k=1, workers=-1)
    N = N0[idxN]
    U, V = tangent(N)

    treeS = cKDTree(np.ascontiguousarray(S, dtype=np.float64))
    _, knn = treeS.query(S, k=K_NEIGH + 1, workers=-1)
    knn = knn[:, 1:]

    bbox = S.max(axis=0) - S.min(axis=0)
    R = float(np.max(bbox))
    cent3d = np.full_like(S, np.nan, dtype=np.float64)
    for i in range(len(S)):
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

    return Snew, np.linalg.norm(Snew - S, axis=1)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mesh", nargs="?", default=os.path.join(_ROOT, "meshes", "teapot.obj"))
    args = p.parse_args()
    in_path = args.mesh
    objname = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = os.path.join(_ROOT, "results", objname, "exp8_deployed_policy_eval")
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
    n = len(S0)

    nv_score = normal_variation_score(P0, nr(N0), NV_K)
    n_tiers = len(TIERS)
    tier_id = np.digitize(nv_score, TIER_THRESHOLDS).astype(np.int64)
    tier_names  = [t[0]  for t in TIERS]
    tier_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#F44336"]

    disp_thr_arr  = np.array([t[1] for t in TIERS])
    streak_arr    = np.array([t[2] for t in TIERS], dtype=np.int64)
    neigh_win_arr = np.array([t[3] for t in TIERS], dtype=np.int64)
    jacc_thr_arr  = np.array([t[4] for t in TIERS])

    # ------------------------------------------------------------------ #
    # Run NOFREEZE for ITERS + LOOKAHEAD iters (counterfactual)
    # ------------------------------------------------------------------ #
    print(f"Running {ITERS + LOOKAHEAD} NOFREEZE iterations (counterfactual)...")
    nf_disp = np.zeros((n, ITERS + LOOKAHEAD), dtype=np.float64)
    S_nf = S0.copy()
    for it in range(ITERS + LOOKAHEAD):
        S_nf, d = lloyd_nofreeze_step(S_nf, P0, F0, N0, treeP0, vf)
        nf_disp[:, it] = d
        print(f"  NOFREEZE iter {it+1}/{ITERS + LOOKAHEAD}")

    # ------------------------------------------------------------------ #
    # Run DEPLOYED POLICY for ITERS iters
    # ------------------------------------------------------------------ #
    print(f"\nRunning {ITERS} DEPLOYED POLICY iterations...")
    freeze_state = {
        "frozen":       np.zeros(n, dtype=bool),
        "low_streak":   np.zeros(n, dtype=np.int64),
        "knn_hist":     [],
        "idxn_cached":  np.zeros((n, K_NEIGH), dtype=np.int64),
        "tier_id":      tier_id,
        "disp_thr_arr": disp_thr_arr,
        "streak_arr":   streak_arr,
        "neigh_win_arr": neigh_win_arr,
        "jacc_thr_arr": jacc_thr_arr,
    }

    S_fr = S0.copy()
    freeze_iter  = np.full(n, -1, dtype=np.int64)   # iteration when frozen
    total_disp_freeze  = np.zeros(ITERS)             # C3: total displacement under freeze policy
    total_disp_nofreeze = np.zeros(ITERS)            # C3: total displacement under NOFREEZE

    for it in range(ITERS):
        prev_frozen = freeze_state["frozen"].copy()
        lloyd_iter_sites_only(S_fr, P0, F0, N0, vf, treeP0,
                              freeze_state, k_neigh=K_NEIGH, k_proj=K_PROJ)
        # Record newly frozen sites
        newly_frozen = freeze_state["frozen"] & (~prev_frozen)
        freeze_iter[newly_frozen] = it

        # C3: total displacement this iteration
        total_disp_freeze[it]   = np.sum(nf_disp[:, it] ** 2)   # use NOFREEZE as common baseline
        total_disp_nofreeze[it] = np.sum(nf_disp[:, it] ** 2)   # same — reuse for comparison plot
        print(f"  FREEZE iter {it+1}/{ITERS}  frozen={freeze_state['frozen'].sum()}")

    # C3: compute from actual S_fr trajectory — need to re-instrument
    # Use nf_disp as total energy proxy for NOFREEZE; for FREEZE, use iteration-wise frozen fraction
    n_frozen_per_iter = np.zeros(ITERS, dtype=np.int64)
    # Re-run FREEZE tracking total displacement
    freeze_state2 = {
        "frozen":       np.zeros(n, dtype=bool),
        "low_streak":   np.zeros(n, dtype=np.int64),
        "knn_hist":     [],
        "idxn_cached":  np.zeros((n, K_NEIGH), dtype=np.int64),
        "tier_id":      tier_id,
        "disp_thr_arr": disp_thr_arr,
        "streak_arr":   streak_arr,
        "neigh_win_arr": neigh_win_arr,
        "jacc_thr_arr": jacc_thr_arr,
    }
    S_fr2 = S0.copy()
    total_disp_freeze2 = np.zeros(ITERS)
    for it in range(ITERS):
        S_prev = S_fr2.copy()
        lloyd_iter_sites_only(S_fr2, P0, F0, N0, vf, treeP0,
                              freeze_state2, k_neigh=K_NEIGH, k_proj=K_PROJ)
        total_disp_freeze2[it] = np.sum(np.linalg.norm(S_fr2 - S_prev, axis=1) ** 2)
        n_frozen_per_iter[it]  = int(freeze_state2["frozen"].sum())
    total_disp_nf_iters = np.array([np.sum(nf_disp[:, it] ** 2) for it in range(ITERS)])

    # ------------------------------------------------------------------ #
    # D1: False freeze rate under deployed policy
    # ------------------------------------------------------------------ #
    was_frozen = freeze_iter >= 0
    # For each frozen site, check if it moves > its disp_thr in NOFREEZE
    # counterfactual in the next LOOKAHEAD iters after freeze
    false_freeze = np.zeros(n, dtype=bool)
    post_freeze_max_disp = np.zeros(n)
    for i in np.flatnonzero(was_frozen):
        fi  = int(freeze_iter[i])
        thr = float(disp_thr_arr[tier_id[i]])
        start = fi + 1
        end   = min(fi + 1 + LOOKAHEAD, ITERS + LOOKAHEAD)
        if start < ITERS + LOOKAHEAD:
            md = nf_disp[i, start:end].max()
            post_freeze_max_disp[i] = md
            if md > thr:
                false_freeze[i] = True

    # C2: neighbor displacement post-freeze
    treeS_final = cKDTree(np.ascontiguousarray(S_fr, dtype=np.float64))
    _, knn_final = treeS_final.query(S_fr, k=K_NEIGH + 1, workers=-1)
    knn_final = knn_final[:, 1:]

    neighbor_post_disp = np.zeros(n)
    for i in np.flatnonzero(was_frozen):
        fi     = int(freeze_iter[i])
        start  = fi + 1
        end    = min(fi + 1 + LOOKAHEAD, ITERS + LOOKAHEAD)
        if start < ITERS + LOOKAHEAD:
            nbrs = knn_final[i]
            neighbor_post_disp[i] = nf_disp[nbrs, start:end].mean()

    # ------------------------------------------------------------------ #
    # Print stats
    # ------------------------------------------------------------------ #
    print(f"\n--- D1: False freeze rate under deployed policy (LOOKAHEAD={LOOKAHEAD}) ---")
    for t in range(n_tiers):
        tmask  = (tier_id == t) & was_frozen
        nfmask = (tier_id == t) & was_frozen & false_freeze
        n_frz  = int(tmask.sum())
        n_fls  = int(nfmask.sum())
        rate   = n_fls / n_frz if n_frz > 0 else float("nan")
        mpd    = post_freeze_max_disp[tmask].mean() if n_frz > 0 else float("nan")
        print(f"  {tier_names[t]:8s}: frozen={n_frz:5d}  false={n_fls:5d}  rate={rate:.3f}  "
              f"mean_post_disp={mpd:.6f}")

    print(f"\n--- C1: Post-freeze displacement (self) in NOFREEZE counterfactual ---")
    for t in range(n_tiers):
        tmask = (tier_id == t) & was_frozen
        if tmask.sum() == 0:
            print(f"  {tier_names[t]:8s}: no frozen sites")
            continue
        pd = post_freeze_max_disp[tmask]
        thr = float(disp_thr_arr[t])
        frac_above = float((pd > thr).mean())
        print(f"  {tier_names[t]:8s}: mean_post={pd.mean():.6f}  "
              f"median={np.median(pd):.6f}  frac>thr({thr:.0e})={frac_above:.3f}")

    print(f"\n--- C2: Post-freeze neighbor displacement in NOFREEZE counterfactual ---")
    for t in range(n_tiers):
        tmask = (tier_id == t) & was_frozen
        if tmask.sum() == 0:
            print(f"  {tier_names[t]:8s}: no frozen sites")
            continue
        nd = neighbor_post_disp[tmask]
        print(f"  {tier_names[t]:8s}: mean_neighbor_disp={nd.mean():.6f}  "
              f"median={np.median(nd):.6f}")

    print(f"\n--- C3: Total displacement^2 per iteration ---")
    print(f"  iter 1:  FREEZE={total_disp_freeze2[0]:.6f}  NOFREEZE={total_disp_nf_iters[0]:.6f}")
    print(f"  iter {ITERS}: FREEZE={total_disp_freeze2[-1]:.6f}  NOFREEZE={total_disp_nf_iters[-1]:.6f}")
    print(f"  FREEZE monotone decreasing: {bool(np.all(np.diff(total_disp_freeze2) <= 0))}")
    print(f"  Final FREEZE total_disp^2: {total_disp_freeze2[-1]:.6f}")
    print(f"  Final NOFREEZE total_disp^2: {total_disp_nf_iters[-1]:.6f}")

    # ------------------------------------------------------------------ #
    # CSV
    # ------------------------------------------------------------------ #
    import csv

    # D1 false freeze
    csv_d1 = os.path.join(out_dir, "exp8_d1_false_freeze_deployed.csv")
    with open(csv_d1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n_total", "n_frozen", "n_false", "false_rate",
                    "mean_post_freeze_disp", "disp_thr"])
        for t in range(n_tiers):
            tmask  = tier_id == t
            fmask  = tmask & was_frozen
            ffmask = fmask & false_freeze
            n_t    = int(tmask.sum())
            n_frz  = int(fmask.sum())
            n_fls  = int(ffmask.sum())
            rate   = n_fls / n_frz if n_frz > 0 else float("nan")
            mpd    = post_freeze_max_disp[fmask].mean() if n_frz > 0 else float("nan")
            w.writerow([tier_names[t], n_t, n_frz, n_fls,
                        f"{rate:.4f}", f"{mpd:.6f}", f"{disp_thr_arr[t]:.0e}"])
    print(f"\nSaved: {csv_d1}")

    # C1/C2 post-freeze displacement
    csv_c1 = os.path.join(out_dir, "exp8_c1c2_post_freeze_disp.csv")
    with open(csv_c1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n_frozen", "self_mean", "self_median", "self_frac_above_thr",
                    "neighbor_mean", "neighbor_median"])
        for t in range(n_tiers):
            tmask = (tier_id == t) & was_frozen
            n_frz = int(tmask.sum())
            if n_frz == 0:
                w.writerow([tier_names[t], 0] + ["nan"] * 5)
                continue
            pd  = post_freeze_max_disp[tmask]
            nd  = neighbor_post_disp[tmask]
            thr = float(disp_thr_arr[t])
            w.writerow([tier_names[t], n_frz,
                        f"{pd.mean():.6f}", f"{np.median(pd):.6f}", f"{(pd > thr).mean():.4f}",
                        f"{nd.mean():.6f}", f"{np.median(nd):.6f}"])
    print(f"Saved: {csv_c1}")

    # C3 convergence trajectory
    csv_c3 = os.path.join(out_dir, "exp8_c3_convergence.csv")
    with open(csv_c3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "total_disp_sq_freeze", "total_disp_sq_nofreeze", "n_frozen"])
        for it in range(ITERS):
            w.writerow([it + 1,
                        f"{total_disp_freeze2[it]:.6f}",
                        f"{total_disp_nf_iters[it]:.6f}",
                        int(n_frozen_per_iter[it])])
    print(f"Saved: {csv_c3}")

    # ------------------------------------------------------------------ #
    # Figure
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Experiment 8: Deployed Policy Evaluation — {objname}",
                 fontsize=14, fontweight="bold")
    x = np.arange(n_tiers)

    # (0,0) D1: false freeze rate under deployed vs uniform streak=2
    ax = axes[0, 0]
    d1_rates = []
    for t in range(n_tiers):
        fmask = (tier_id == t) & was_frozen
        ffmask = fmask & false_freeze
        n_frz = int(fmask.sum())
        rate  = int(ffmask.sum()) / n_frz if n_frz > 0 else 0.0
        d1_rates.append(rate)
    ax.bar(x, d1_rates, color=tier_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(tier_names)
    ax.set_ylabel(f"False Freeze Rate (LOOKAHEAD={LOOKAHEAD})")
    ax.set_title("D1: False Freeze Rate — Deployed Policy")
    for i, v in enumerate(d1_rates):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (0,1) C1: post-freeze self displacement distribution (box per tier)
    ax = axes[0, 1]
    data_pd = [post_freeze_max_disp[(tier_id == t) & was_frozen] for t in range(n_tiers)]
    data_pd_nonempty = [d if len(d) > 0 else np.array([0.0]) for d in data_pd]
    bp = ax.boxplot(data_pd_nonempty, labels=tier_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], tier_colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    # Draw tier-specific disp_thr as horizontal markers
    for t, thr_val in enumerate(disp_thr_arr):
        ax.plot([t + 0.7, t + 1.3], [thr_val, thr_val], color="red", linewidth=1.5, linestyle="--")
    ax.set_ylabel("Max displacement in next 10 iters (NOFREEZE)")
    ax.set_title("C1: Post-Freeze Displacement Proxy (red = disp_thr)")
    ax.grid(True, alpha=0.3)

    # (1,0) C2: neighbor displacement post-freeze
    ax = axes[1, 0]
    data_nd = [neighbor_post_disp[(tier_id == t) & was_frozen] for t in range(n_tiers)]
    data_nd_nonempty = [d if len(d) > 0 else np.array([0.0]) for d in data_nd]
    bp2 = ax.boxplot(data_nd_nonempty, labels=tier_names, patch_artist=True)
    for patch, color in zip(bp2["boxes"], tier_colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel("Mean neighbor displacement (NOFREEZE, next 10 iters)")
    ax.set_title("C2: Post-Freeze Neighbor Displacement")
    ax.grid(True, alpha=0.3)

    # (1,1) C3: total displacement² trajectory
    ax = axes[1, 1]
    iters_x = np.arange(1, ITERS + 1)
    ax.plot(iters_x, total_disp_freeze2, color="tab:blue", label="Deployed policy (FREEZE)", linewidth=2)
    ax.plot(iters_x, total_disp_nf_iters, color="tab:orange", label="NOFREEZE", linewidth=2, linestyle="--")
    ax_r = ax.twinx()
    ax_r.plot(iters_x, n_frozen_per_iter, color="gray", linewidth=1, linestyle=":", label="N frozen")
    ax_r.set_ylabel("Cumulative frozen sites", color="gray")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total displacement^2 (sum over active sites)")
    ax.set_title("C3: Convergence Trajectory")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "exp8_deployed_policy_eval.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
