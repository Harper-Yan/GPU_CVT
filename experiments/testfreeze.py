#!/usr/bin/env python3
import numpy as np
import argparse
import os
import subprocess
import time
import csv
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree

KNN_K = 32
NV_K = 24
TIER_THRESHOLDS = [0.15, 0.35, 0.55, 0.80]   # 4 boundaries -> 5 tiers
TIERS = [
    # (name,      disp_thr, streak, neigh_win, jacc_thr, extra_iters)
    ("flat",       5e-3,  2,  3,  0.80,  0),   # NV < 0.15: no instability
    ("gentle",     4e-3,  3,  5,  0.85,  0),   # 0.15-0.35: minor instability
    ("moderate",   3e-3,  5,  7,  0.88,  0),   # 0.35-0.55: transition zone
    ("curved",     2e-3,  6, 10,  0.93,  0),   # 0.55-0.80: high instability
    ("sharp",      1e-3,  8, 12,  0.95,  0),   # >= 0.80: highest instability
]
def hardness_stats_and_hist(hard, tier_thresholds, tier_names, *, bins=80, save_png="", show=True):
    import matplotlib.pyplot as plt

    hard = np.asarray(hard, dtype=np.float64)
    hard = hard[np.isfinite(hard)]
    if hard.size == 0:
        print("[hardness] no finite hardness values")
        return

    boundaries = [0.0] + list(tier_thresholds) + [float("inf")]
    print(f"[hardness] N={hard.size}")
    print(f"[hardness] min={hard.min():.6g} mean={hard.mean():.6g} max={hard.max():.6g}")
    parts = []
    for t, name in enumerate(tier_names):
        lo, hi = boundaries[t], boundaries[t + 1]
        pct = 100.0 * float(np.mean((hard >= lo) & (hard < hi)))
        if hi == float("inf"):
            parts.append(f"{name}>={lo:.2g} => {pct:.2f}%")
        else:
            parts.append(f"{name}[{lo:.2g},{hi:.2g}) => {pct:.2f}%")
    print(f"[hardness] tiers: {' | '.join(parts)}")

    plt.figure()
    plt.hist(hard, bins=bins)
    for thr in tier_thresholds:
        plt.axvline(thr, color="red", linestyle="--", alpha=0.7)
    plt.title("Hardness distribution (normal variation score)")
    plt.xlabel("hardness")
    plt.ylabel("count")
    if save_png:
        plt.savefig(save_png, dpi=200, bbox_inches="tight")
        print(f"[hardness] saved histogram: {save_png}")
    if show:
        plt.show()
    else:
        plt.close()


def color_mesh_by_hardness_open3d(P0, F0, hard, *, clip_pct=99.0, show=True, save_ply=""):
    import open3d as o3d
    hard = np.asarray(hard, dtype=np.float64)
    hard = np.where(np.isfinite(hard), hard, 0.0)

    lo = float(np.min(hard))
    hi = float(np.percentile(hard, clip_pct))
    if hi <= lo + 1e-30:
        t = np.zeros_like(hard)
    else:
        t = np.clip((hard - lo) / (hi - lo), 0.0, 1.0)

    try:
        import matplotlib.cm as cm
        C = cm.get_cmap("turbo")(t)[:, :3]
    except Exception:
        C = np.stack([t, np.zeros_like(t), 1.0 - t], axis=1)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(P0, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(F0, dtype=np.int64))
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(C)

    if save_ply:
        o3d.io.write_triangle_mesh(save_ply, mesh, write_vertex_colors=True)
        print(f"[hardness] saved colored mesh: {save_ply}")

    if show:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

def write_site_disp_matrix_txt(path, disp_hist):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    disp_hist = np.asarray(disp_hist, dtype=np.float64)
    cols = disp_hist.shape[1]
    header = " ".join([f"it{t:03d}" for t in range(cols)])
    np.savetxt(path, disp_hist, fmt="%.17g", header=header, comments="")

def append_mesh_csv(csv_path, row, header):
    first = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if first:
            w.writeheader()
        w.writerow(row)

def write_xyz(path, S):
    S = np.asarray(S, dtype=np.float64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for x, y, z in S:
            f.write(f"{x:.17g} {y:.17g} {z:.17g}\n")

def geogram_reconstruct_from_sites_xyz(vorpalite_exe, sites_xyz, out_obj, *, radius="5%", nb_neighbors=30, pre=False, remesh=False, post=True, repair=True, quiet=True):
    cmd = [
        vorpalite_exe,
        sites_xyz,
        out_obj,
        "co3ne=true",
        f"pre={'true' if pre else 'false'}",
        f"remesh={'true' if remesh else 'false'}",
        f"post={'true' if post else 'false'}",
        f"co3ne:repair={'true' if repair else 'false'}",
        f"co3ne:radius={radius}",
        f"co3ne:nb_neighbors={int(nb_neighbors)}",
        f"log:quiet={'true' if quiet else 'false'}",
        "log:pretty=false",
    ]
    subprocess.run(cmd, check=True)

def nr(x, e=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, e)

def tangent(n):
    n = nr(n)
    h = np.zeros_like(n)
    m = np.abs(n[:, 2]) > 0.9
    h[m] = [1, 0, 0]
    h[~m] = [0, 0, 1]
    u = nr(np.cross(h, n))
    v = nr(np.cross(n, u))
    return u, v

def clip_poly_halfspace(poly, a, b, c, eps=1e-12):
    if poly.shape[0] == 0:
        return poly
    def f(p):
        return a * p[0] + b * p[1] - c
    out = []
    s = poly[-1]
    fs = f(s)
    for e in poly:
        fe = f(e)
        ins_e = fe <= eps
        ins_s = fs <= eps
        if ins_e:
            if not ins_s:
                t = fs / (fs - fe + 1e-30)
                out.append(s + t * (e - s))
            out.append(e)
        elif ins_s:
            t = fs / (fs - fe + 1e-30)
            out.append(s + t * (e - s))
        s, fs = e, fe
    return np.asarray(out, dtype=np.float64)

def cell_poly2d(i, S, U, V, neigh, R):
    si = S[i]
    poly = np.array([[-R, -R], [R, -R], [R, R], [-R, R]], dtype=np.float64)
    ui = U[i]
    vi = V[i]
    for j in neigh:
        d = S[j] - si
        a = float(d @ ui)
        b = float(d @ vi)
        c = 0.5 * float(d @ d)
        poly = clip_poly_halfspace(poly, a, b, c)
        if poly.shape[0] == 0:
            break
    return poly

def poly_area_centroid_2d(poly):
    if poly.shape[0] < 3:
        return 0.0, np.array([np.nan, np.nan], dtype=np.float64)
    x = poly[:, 0]
    y = poly[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    cr = x * y2 - x2 * y
    A = 0.5 * np.sum(cr)
    if np.abs(A) < 1e-18:
        return 0.0, np.array([np.mean(x), np.mean(y)], dtype=np.float64)
    cx = (1.0 / (6.0 * A)) * np.sum((x + x2) * cr)
    cy = (1.0 / (6.0 * A)) * np.sum((y + y2) * cr)
    return A, np.array([cx, cy], dtype=np.float64)

def closest_point_tri(C, A, B, D):
    AB = B - A
    AC = D - A
    AP = C - A
    d1 = AB @ AP
    d2 = AC @ AP
    if d1 <= 0.0 and d2 <= 0.0:
        return A
    BP = C - B
    d3 = AB @ BP
    d4 = AC @ BP
    if d3 >= 0.0 and d4 <= d3:
        return B
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3 + 1e-30)
        return A + v * AB
    CP = C - D
    d5 = AB @ CP
    d6 = AC @ CP
    if d6 >= 0.0 and d5 <= d6:
        return D
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6 + 1e-30)
        return A + w * AC
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-30)
        return B + w * (D - B)
    denom = 1.0 / (va + vb + vc + 1e-30)
    v = vb * denom
    w = vc * denom
    return A + AB * v + AC * w

def update_to_mesh(C, tris_idx, P):
    if tris_idx.shape[0] == 0 or not np.isfinite(C).all():
        return C
    best = None
    best_d2 = np.inf
    for a, b, c in tris_idx:
        A = P[a].astype(np.float64)
        B = P[b].astype(np.float64)
        D = P[c].astype(np.float64)
        q = closest_point_tri(C, A, B, D)
        d = q - C
        d2 = float(d @ d)
        if d2 < best_d2:
            best_d2 = d2
            best = q
    return best if best is not None else C

def triangle_quality_metrics(Vpos, F):
    F = np.asarray(F, dtype=np.int64)
    Vpos = np.asarray(Vpos, dtype=np.float64)
    if F.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    A = Vpos[F[:, 0]]
    B = Vpos[F[:, 1]]
    C = Vpos[F[:, 2]]
    e0 = B - A
    e1 = C - B
    e2 = A - C
    a = np.linalg.norm(e1, axis=1)
    b = np.linalg.norm(e2, axis=1)
    c = np.linalg.norm(e0, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(e0, C - A), axis=1)
    area = np.maximum(area, 1e-30)
    aspect = (a * a + b * b + c * c) / (4.0 * np.sqrt(3.0) * area)
    avg_aspect = float(np.mean(aspect))
    def safe_acos(x):
        return np.arccos(np.clip(x, -1.0, 1.0))
    cosA = (b * b + c * c - a * a) / (2.0 * b * c + 1e-30)
    cosB = (c * c + a * a - b * b) / (2.0 * c * a + 1e-30)
    cosC = (a * a + b * b - c * c) / (2.0 * a * b + 1e-30)
    angA = safe_acos(cosA) * (180.0 / np.pi)
    angB = safe_acos(cosB) * (180.0 / np.pi)
    angC = safe_acos(cosC) * (180.0 / np.pi)
    max_ang = np.maximum(angA, np.maximum(angB, angC))
    min_ang = np.minimum(angA, np.minimum(angB, angC))
    pct_gt_90 = 100.0 * float(np.mean(max_ang > 90.0))
    pct_lt_30 = 100.0 * float(np.mean(min_ang < 30.0))
    return avg_aspect, pct_gt_90, pct_lt_30

def load_trimesh_any(obj_path):
    geo = trimesh.load(obj_path, process=False)
    if isinstance(geo, trimesh.Scene):
        geo = trimesh.util.concatenate([g for g in geo.geometry.values()])
    return geo

def knn_indices(P: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(np.ascontiguousarray(P, dtype=np.float64))
    _, idx = tree.query(P, k=k + 1, workers=-1)
    return idx[:, 1:]

def normal_variation_score(meshV: np.ndarray, meshN: np.ndarray, k: int) -> np.ndarray:
    idx = knn_indices(meshV, k)
    Ni = meshN
    Nj = meshN[idx]
    cos = np.sum(Ni[:, None, :] * Nj, axis=2)
    cos = np.clip(cos, -1.0, 1.0)
    score = np.mean(1.0 - cos, axis=1)
    return score

def normal_covariance_L(meshV: np.ndarray, meshN: np.ndarray, k: int) -> np.ndarray:
    """
    Per-vertex Normal Covariance Linearity  L = (λ₁ − λ₂) / λ₁.

    Fits the 3×3 covariance matrix of the k nearest-neighbour normals at each
    vertex and returns the normalised gap between the two largest eigenvalues.

    Interpretation (empirical, on triangulated meshes):
      High L  → one dominant normal-variation axis (cone-tip / singularity):
                radial triangulation fan creates one strong eigenvector.
      Low  L  → variance spread across two axes (ridge neighbourhood):
                normals from both flanks + along-ridge sites spread covariance.
    Range: [0, 1].  Returns 0 when the largest eigenvalue is near zero.
    """
    idx = knn_indices(meshV, k)          # (V, k)
    N   = np.asarray(meshN, dtype=np.float64)
    L   = np.zeros(meshV.shape[0], dtype=np.float64)
    for i in range(meshV.shape[0]):
        Ni  = N[idx[i]]                  # (k, 3)
        cov = np.cov(Ni.T)               # (3, 3)
        lam = np.sort(np.linalg.eigvalsh(cov))[::-1]   # λ₁ ≥ λ₂ ≥ λ₃
        if lam[0] > 1e-30:
            L[i] = (lam[0] - lam[1]) / lam[0]
    return L

def map_score_to_sites(siteP: np.ndarray, meshV: np.ndarray, meshScore: np.ndarray) -> np.ndarray:
    tree = cKDTree(np.ascontiguousarray(meshV, dtype=np.float64))
    _, idx = tree.query(siteP, k=1, workers=-1)
    return meshScore[idx]

def jaccard_sorted_int(a, b):
    i = 0
    j = 0
    inter = 0
    na = a.shape[0]
    nb = b.shape[0]
    while i < na and j < nb:
        va = int(a[i])
        vb = int(b[j])
        if va == vb:
            inter += 1
            i += 1
            j += 1
        elif va < vb:
            i += 1
        else:
            j += 1
    uni = na + nb - inter
    return (inter / uni) if uni > 0 else 1.0

def stable_by_tier(knn_hist, tier_id, neigh_win_arr, jacc_thr_arr):
    L = len(knn_hist)
    n = knn_hist[-1].shape[0]
    if L < 2:
        return np.zeros(n, dtype=bool)
    out = np.ones(n, dtype=bool)
    for i in range(n):
        w = int(neigh_win_arr[int(tier_id[i])])
        thr = float(jacc_thr_arr[int(tier_id[i])])
        if w <= 1:
            out[i] = False
            continue
        useL = min(L, w)
        ok = True
        for t in range(L - useL + 1, L):
            A = knn_hist[t - 1][i]
            B = knn_hist[t][i]
            if jaccard_sorted_int(A, B) < thr:
                ok = False
                break
        out[i] = ok
    return out

def lloyd_iter_sites_only(
    S, P0, F0, N0, vf, treeP0,
    freeze_state,
    *, k_neigh=32, k_proj=10
):
    frozen = freeze_state["frozen"]
    low_streak = freeze_state["low_streak"]
    knn_hist = freeze_state["knn_hist"]
    idxn_cached = freeze_state["idxn_cached"]
    tier_id = freeze_state["tier_id"]
    disp_thr_arr = freeze_state["disp_thr_arr"]
    streak_arr = freeze_state["streak_arr"]
    neigh_win_arr = freeze_state["neigh_win_arr"]
    jacc_thr_arr = freeze_state["jacc_thr_arr"]

    _, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64), k=1, workers=-1)
    N = N0[idxN]
    U, V = tangent(N)

    treeS = cKDTree(np.ascontiguousarray(S, dtype=np.float64))
    active = ~frozen
    if np.any(active):
        _, idxA = treeS.query(np.ascontiguousarray(S[active], dtype=np.float64), k=k_neigh + 1, workers=-1)
        idxA = idxA[:, 1:]
        idxA = np.sort(idxA, axis=1)
        idxn_cached[active] = idxA
    idxn_sorted = idxn_cached

    bbox = S.max(axis=0) - S.min(axis=0)
    R = float(np.max(bbox))
    cent3d = np.full_like(S, np.nan, dtype=np.float64)
    for i in range(len(S)):
        poly = cell_poly2d(i, S, U, V, idxn_sorted[i], R)
        _, c2 = poly_area_centroid_2d(poly)
        if np.isfinite(c2).all():
            cent3d[i] = S[i] + c2[0] * U[i] + c2[1] * V[i]

    mask = np.isfinite(cent3d).all(axis=1)
    Snew = S.copy()
    if np.any(mask):
        _, idxC = treeP0.query(np.ascontiguousarray(cent3d[mask], dtype=np.float64), k=k_proj, workers=-1)
        for t, i in enumerate(np.flatnonzero(mask)):
            nn = idxC[t]
            faces = set()
            for v in nn:
                for fi in vf[int(v)]:
                    faces.add(fi)
            faces = np.fromiter(faces, dtype=np.int64)
            tris = F0[faces] if faces.size else np.empty((0, 3), dtype=np.int64)
            Snew[i] = update_to_mesh(cent3d[i], tris, P0)

    disp = np.linalg.norm(Snew - S, axis=1)
    thr = disp_thr_arr[tier_id]
    low = disp < thr
    low_streak[:] = np.where(low, low_streak + 1, 0)

    knn_hist.append(idxn_sorted.copy())
    maxW = int(np.max(neigh_win_arr))
    if len(knn_hist) > maxW:
        knn_hist.pop(0)

    stable_nbr = stable_by_tier(knn_hist, tier_id, neigh_win_arr, jacc_thr_arr)
    need = streak_arr[tier_id]
    cand = (low_streak >= need) & stable_nbr & (~frozen)
    frozen[cand] = True

    freeze_state["frozen"] = frozen
    freeze_state["low_streak"] = low_streak
    freeze_state["knn_hist"] = knn_hist
    freeze_state["idxn_cached"] = idxn_cached
    return Snew, freeze_state, disp
def lloyd_iter_sites_only_block_knn_and_proj(
    S, P0, F0, N0, vf, treeP0,
    freeze_state,
    *, k_neigh=32, k_proj=10
):
    frozen = freeze_state["frozen"]
    low_streak = freeze_state["low_streak"]
    knn_hist = freeze_state["knn_hist"]
    idxn_cached = freeze_state["idxn_cached"]
    tier_id = freeze_state["tier_id"]
    disp_thr_arr = freeze_state["disp_thr_arr"]
    streak_arr = freeze_state["streak_arr"]
    neigh_win_arr = freeze_state["neigh_win_arr"]
    jacc_thr_arr = freeze_state["jacc_thr_arr"]

    # normals/tangent frame at sites (cheap-ish, keep as-is)
    _, idxN = treeP0.query(np.ascontiguousarray(S, dtype=np.float64), k=1, workers=-1)
    N = N0[idxN]
    U, V = tangent(N)

    # ---- 1) Block KNN updates for frozen sites (keep cached) ----
    treeS = cKDTree(np.ascontiguousarray(S, dtype=np.float64))
    active = ~frozen
    if np.any(active):
        _, idxA = treeS.query(
            np.ascontiguousarray(S[active], dtype=np.float64),
            k=k_neigh + 1, workers=-1
        )
        idxA = np.sort(idxA[:, 1:], axis=1)
        idxn_cached[active] = idxA
    idxn_sorted = idxn_cached  # frozen rows remain unchanged

    # ---- 2) Block centroid computation + mesh projection/update for frozen sites ----
    Snew = S.copy()

    # compute centroid candidates only for active sites
    bbox = S.max(axis=0) - S.min(axis=0)
    R = float(np.max(bbox))
    cent3d = np.full((S.shape[0], 3), np.nan, dtype=np.float64)

    if np.any(active):
        active_idx = np.flatnonzero(active)
        for i in active_idx:
            poly = cell_poly2d(i, S, U, V, idxn_sorted[i], R)
            _, c2 = poly_area_centroid_2d(poly)
            if np.isfinite(c2).all():
                cent3d[i] = S[i] + c2[0] * U[i] + c2[1] * V[i]

        # project only active+valid centroids back to mesh
        mask = active & np.isfinite(cent3d).all(axis=1)
        if np.any(mask):
            Cq = np.ascontiguousarray(cent3d[mask], dtype=np.float64)
            _, idxC = treeP0.query(Cq, k=k_proj, workers=-1)
            # idxC: (n_mask, k_proj)
            for t, i in enumerate(np.flatnonzero(mask)):
                nn = idxC[t]
                faces = set()
                for v in nn:
                    for fi in vf[int(v)]:
                        faces.add(fi)
                faces = np.fromiter(faces, dtype=np.int64)
                tris = F0[faces] if faces.size else np.empty((0, 3), dtype=np.int64)
                Snew[i] = update_to_mesh(cent3d[i], tris, P0)

    # enforce: frozen sites NEVER move
    Snew[frozen] = S[frozen]

    # displacement: frozen sites get exactly 0
    disp = np.linalg.norm(Snew - S, axis=1)
    disp[frozen] = 0.0

    # streak/freezing logic unchanged (but frozen stays frozen)
    thr = disp_thr_arr[tier_id]
    low = disp < thr
    low_streak[:] = np.where(low, low_streak + 1, 0)

    knn_hist.append(idxn_sorted.copy())
    maxW = int(np.max(neigh_win_arr))
    if len(knn_hist) > maxW:
        knn_hist.pop(0)

    stable_nbr = stable_by_tier(knn_hist, tier_id, neigh_win_arr, jacc_thr_arr)
    need = streak_arr[tier_id]
    cand = (low_streak >= need) & stable_nbr & (~frozen)
    frozen[cand] = True

    freeze_state["frozen"] = frozen
    freeze_state["low_streak"] = low_streak
    freeze_state["knn_hist"] = knn_hist
    freeze_state["idxn_cached"] = idxn_cached
    return Snew, freeze_state, disp

class LiveViewer:
    def __init__(self, title="CVT", point_size=3.0, line_width=1.0):
        self.base_title = title
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=title, width=1280, height=720)
        opt = self.vis.get_render_option()
        opt.point_size = float(point_size)
        opt.line_width = float(line_width)
        opt.mesh_show_back_face = True
        self.mesh = o3d.geometry.TriangleMesh()
        self.wire = o3d.geometry.LineSet()
        self.pc = o3d.geometry.PointCloud()
        self._added = False

    def update(self, mesh_obj_path, sites, frozen_mask, it=None, png_path=""):
        # Update window title with iteration
        if it is not None:
            try:
                self.vis.get_render_option()  # touch vis so window exists
                self.vis.get_view_control()   # same
                self.vis.get_window_name()    # may exist in some builds
            except Exception:
                pass
            # Open3D supports update_window_title on recent versions; fall back if missing.
            if hasattr(self.vis, "update_window_title"):
                self.vis.update_window_title(f"{self.base_title} | iter {int(it):03d}")
            else:
                # no-op; still saves png below
                pass

        mesh = o3d.io.read_triangle_mesh(mesh_obj_path)
        if mesh.is_empty():
            return
        mesh.compute_vertex_normals()

        wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        wire.paint_uniform_color([0, 0, 0])

        pc = o3d.geometry.PointCloud()
        S = np.asarray(sites, dtype=np.float64)
        pc.points = o3d.utility.Vector3dVector(S)

        frozen_mask = np.asarray(frozen_mask, dtype=bool)
        colors = np.zeros((len(S), 3), dtype=np.float64)
        colors[~frozen_mask] = [0.2, 0.8, 0.2]
        colors[frozen_mask]  = [0.9, 0.1, 0.1]
        pc.colors = o3d.utility.Vector3dVector(colors)

        if not self._added:
            self.mesh = mesh
            self.wire = wire
            self.pc = pc
            self.vis.add_geometry(self.mesh)
            self.vis.add_geometry(self.wire)
            self.vis.add_geometry(self.pc)
            self._added = True
        else:
            self.mesh.vertices = mesh.vertices
            self.mesh.triangles = mesh.triangles
            self.mesh.vertex_normals = mesh.vertex_normals

            self.wire.points = wire.points
            self.wire.lines = wire.lines
            self.wire.colors = wire.colors

            self.pc.points = pc.points
            self.pc.colors = pc.colors

            self.vis.update_geometry(self.mesh)
            self.vis.update_geometry(self.wire)
            self.vis.update_geometry(self.pc)

        self.vis.poll_events()
        self.vis.update_renderer()

        # Save screenshot AFTER renderer update
        if png_path:
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            self.vis.capture_screen_image(png_path, do_render=True)

    def destroy(self):
        self.vis.destroy_window()


def run_freeze_with_visualization_eval_csv(
    *,
    objname,
    S0,
    P0, F0, N0, treeP0, vf,
    iters,
    k_neigh, k_proj,
    vorpalite_exe,
    out_dir,
    vis_delay_ms,
    viewer,
    csv_path,
    geogram_radius,
    geogram_nb_neighbors
):
    os.makedirs(out_dir, exist_ok=True)
    header = [
        "objname", "mode", "iter",
        "sites", "frozen",
        "site_disp_mean", "site_disp_max",
        "geogram_V", "geogram_F",
        "avg_aspect", "pct_gt_90", "pct_lt_30",
    ]
    if os.path.exists(csv_path):
        os.remove(csv_path)

    mesh_nv = normal_variation_score(P0, nr(N0), NV_K)
    hard = map_score_to_sites(S0, P0, mesh_nv)

    tier_names = [t[0] for t in TIERS]
    hardness_stats_and_hist(
        hard, TIER_THRESHOLDS, tier_names,
        bins=100,
        save_png=os.path.join(out_dir, "hardness_hist.png"),
        show=False
    )

    tier_id = np.digitize(hard, TIER_THRESHOLDS).astype(np.int64)

    disp_thr_arr  = np.array([t[1] for t in TIERS], dtype=np.float64)
    base_streak   = np.array([t[2] for t in TIERS], dtype=np.int64)
    neigh_win_arr = np.array([t[3] for t in TIERS], dtype=np.int64)
    jacc_thr_arr  = np.array([t[4] for t in TIERS], dtype=np.float64)
    extra_iters   = np.array([t[5] for t in TIERS], dtype=np.int64)
    streak_arr = base_streak + extra_iters

    treeS0 = cKDTree(np.ascontiguousarray(S0, dtype=np.float64))
    _, idx0 = treeS0.query(np.ascontiguousarray(S0, dtype=np.float64), k=k_neigh + 1, workers=-1)
    idx0 = np.sort(idx0[:, 1:], axis=1)

    freeze_state = {
        "frozen": np.zeros(len(S0), dtype=bool),
        "low_streak": np.zeros(len(S0), dtype=np.int64),
        "knn_hist": [idx0.copy()],
        "idxn_cached": idx0.copy(),
        "tier_id": tier_id,
        "disp_thr_arr": disp_thr_arr,
        "streak_arr": streak_arr,
        "neigh_win_arr": neigh_win_arr,
        "jacc_thr_arr": jacc_thr_arr,
    }

    S = S0.copy()
    disp_hist = np.zeros((len(S), iters + 1), dtype=np.float64)

    for it in range(0, iters + 1):
        if it == 0:
            disp_mean = 0.0
            disp_max = 0.0
        else:
            S_before = S.copy()
            S, freeze_state, dd = lloyd_iter_sites_only(
            #S, freeze_state, dd = lloyd_iter_sites_only_block_knn_and_proj(
                S, P0, F0, N0, vf, treeP0,
                freeze_state,
                k_neigh=k_neigh,
                k_proj=k_proj
            )
            disp_mean = float(dd.mean())
            disp_max = float(dd.max())
            disp_hist[:, it] = dd
            # Save first 100 vertices before and after projection
            write_xyz(os.path.join(out_dir, "before_projection_100.xyz"), S_before[:100])
            write_xyz(os.path.join(out_dir, "after_projection_100.xyz"), S[:100])
            print(f"[FREEZE] saved first 100 vertices: before_projection_100.xyz, after_projection_100.xyz")

        disp_txt = os.path.join(out_dir, "FREEZE_site_disp_matrix.txt")
        write_site_disp_matrix_txt(disp_txt, disp_hist)

        sites_xyz = os.path.join(out_dir, f"FREEZE_sites_{it:03d}.xyz")
        obj_path = os.path.join(out_dir, f"FREEZE_geogram_{it:03d}.obj")
        write_xyz(sites_xyz, S)

        geogram_reconstruct_from_sites_xyz(
            vorpalite_exe, sites_xyz, obj_path,
            radius=geogram_radius,
            nb_neighbors=geogram_nb_neighbors
        )

        geo = load_trimesh_any(obj_path)
        Vg = np.asarray(geo.vertices, dtype=np.float64)
        Fg = np.asarray(geo.faces, dtype=np.int64)
        avg_ar, pct_gt_90, pct_lt_30 = triangle_quality_metrics(Vg, Fg)

        nFrozen = int(np.sum(freeze_state["frozen"]))
        frozen_mask = freeze_state["frozen"]

        print(
            f"[FREEZE] iter {it:03d}: "
            f"site_disp mean={disp_mean:.3e} max={disp_max:.3e} | "
            f"sites={S.shape[0]} frozen={nFrozen} | "
            f"Geogram V={Vg.shape[0]} F={Fg.shape[0]} "
            f"avg_aspect={avg_ar:.6g} %>90={pct_gt_90:.3f}% %<30={pct_lt_30:.3f}%"
        )

        append_mesh_csv(
            csv_path,
            row={
                "objname": objname,
                "mode": "FREEZE",
                "iter": it,
                "sites": int(S.shape[0]),
                "frozen": int(nFrozen),
                "site_disp_mean": float(disp_mean),
                "site_disp_max": float(disp_max),
                "geogram_V": int(Vg.shape[0]),
                "geogram_F": int(Fg.shape[0]),
                "avg_aspect": float(avg_ar),
                "pct_gt_90": float(pct_gt_90),
                "pct_lt_30": float(pct_lt_30),
            },
            header=header
        )

        png_path = os.path.join(out_dir, "frames", f"FREEZE_iter_{it:03d}.png")
        viewer.update(obj_path, S, frozen_mask, it=it, png_path=png_path)

        time.sleep(max(0.0, vis_delay_ms / 1000.0))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("in_path", type=str)
    p.add_argument("--vorpalite", type=str, default=r"E:\MY PYTHON CODES\GPUCVT\geogram\build\Windows\bin\Release\vorpalite.exe")
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--k_neigh", type=int, default=32)
    p.add_argument("--k_proj", type=int, default=10)
    p.add_argument("--vis_delay_ms", type=float, default=150.0)
    p.add_argument("--out_root", type=str, default="results")
    p.add_argument("--geogram_radius", type=str, default="5%")
    p.add_argument("--geogram_nb_neighbors", type=int, default=30)
    args = p.parse_args()

    in_path = os.path.abspath(args.in_path)
    objname = os.path.splitext(os.path.basename(in_path))[0]
    if not os.path.isfile(args.vorpalite):
        raise FileNotFoundError(f"vorpalite not found: {args.vorpalite}")

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

    print(f"[input] {objname}: V0={P0.shape[0]} F0={F0.shape[0]}")
    S0 = P0.copy()

    out_base = os.path.join(args.out_root, objname)
    out_fr = os.path.join(out_base, "freeze")
    os.makedirs(out_fr, exist_ok=True)

    viewer_fr = LiveViewer(title=f"{objname} - FREEZE", point_size=3.0)
    try:
        csv_fr = os.path.join(out_fr, f"{objname}_FREEZE.csv")
        run_freeze_with_visualization_eval_csv(
            objname=objname,
            S0=S0,
            P0=P0, F0=F0, N0=N0, treeP0=treeP0, vf=vf,
            iters=args.iters,
            k_neigh=args.k_neigh,
            k_proj=args.k_proj,
            vorpalite_exe=args.vorpalite,
            out_dir=out_fr,
            vis_delay_ms=args.vis_delay_ms,
            viewer=viewer_fr,
            csv_path=csv_fr,
            geogram_radius=args.geogram_radius,
            geogram_nb_neighbors=args.geogram_nb_neighbors
        )
        print(f"[FREEZE] wrote CSV: {csv_fr}")
    finally:
        viewer_fr.destroy()

if __name__ == "__main__":
    main()
