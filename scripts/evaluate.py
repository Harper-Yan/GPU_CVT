#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, os, re
from typing import Dict, Any, List, Tuple

import numpy as np


def _parse_face_token(tok: str) -> int:
    if "/" in tok:
        tok = tok.split("/")[0]
    return int(tok)


def load_obj_tri_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idx = [_parse_face_token(p) for p in parts[1:]]
                nV = len(verts)
                idx0 = []
                for i in idx:
                    if i < 0:
                        i = nV + i + 1
                    idx0.append(i - 1)
                for k in range(1, len(idx0) - 1):
                    faces.append([idx0[0], idx0[k], idx0[k + 1]])
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"OBJ '{path}': invalid vertex array shape {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3 or F.shape[0] == 0:
        raise ValueError(f"OBJ '{path}': no triangle faces found")
    return V, F


def per_triangle_min_max_angle_deg(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    ab, ac = b - a, c - a
    ba, bc = a - b, c - b
    ca, cb = a - c, b - c

    def angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        du = np.linalg.norm(u, axis=1)
        dv = np.linalg.norm(v, axis=1)
        denom = np.maximum(du * dv, 1e-30)
        cosang = np.sum(u * v, axis=1) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    A = angle(ab, ac)
    B = angle(ba, bc)
    C = angle(ca, cb)
    ang = np.stack([A, B, C], axis=1)
    return np.min(ang, axis=1), np.max(ang, axis=1), ang


def triangle_quality_q(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    l0 = np.linalg.norm(b - a, axis=1)
    l1 = np.linalg.norm(c - b, axis=1)
    l2 = np.linalg.norm(a - c, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    S = 0.5 * (l0 + l1 + l2)
    E = np.maximum(np.maximum(l0, l1), l2)
    denom = np.maximum(S * E, 1e-30)
    return (6.0 / np.sqrt(3.0)) * (area / denom)


# -------------------------
# BVH accelerated distance
# -------------------------

def _point_aabb_d2(p: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    # squared distance from point to AABB
    d = 0.0
    for k in range(3):
        v = p[k]
        if v < bmin[k]:
            t = bmin[k] - v
            d += t * t
        elif v > bmax[k]:
            t = v - bmax[k]
            d += t * t
    return d


def _point_triangle_d2(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # Squared distance point->triangle (Christer Ericson, RTCD)
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.dot(ap, ap))  # barycentric (1,0,0)

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.dot(bp, bp))  # barycentric (0,1,0)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        proj = a + v * ab
        d = p - proj
        return float(np.dot(d, d))  # on AB

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.dot(cp, cp))  # barycentric (0,0,1)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        proj = a + w * ac
        d = p - proj
        return float(np.dot(d, d))  # on AC

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        proj = b + w * (c - b)
        d = p - proj
        return float(np.dot(d, d))  # on BC

    # inside face region
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    proj = a + ab * v + ac * w
    d = p - proj
    return float(np.dot(d, d))


class TriBVH:
    """
    Very small AABB BVH over triangles.
    Query: closest squared distance from point to triangle mesh.
    """

    def __init__(self, V: np.ndarray, F: np.ndarray, leaf_size: int = 8):
        self.V = V
        self.F = F
        self.leaf_size = int(max(1, leaf_size))

        a = V[F[:, 0]]
        b = V[F[:, 1]]
        c = V[F[:, 2]]
        tri_min = np.minimum(np.minimum(a, b), c)
        tri_max = np.maximum(np.maximum(a, b), c)
        tri_ctr = (a + b + c) / 3.0

        self.tri_min = tri_min
        self.tri_max = tri_max
        self.tri_ctr = tri_ctr

        # node arrays
        self.node_bmin: List[np.ndarray] = []
        self.node_bmax: List[np.ndarray] = []
        self.node_left: List[int] = []
        self.node_right: List[int] = []
        self.node_tris: List[np.ndarray | None] = []

        all_idx = np.arange(F.shape[0], dtype=np.int64)
        self.root = self._build(all_idx)

    def _new_node(self, bmin: np.ndarray, bmax: np.ndarray,
                  left: int = -1, right: int = -1,
                  tris: np.ndarray | None = None) -> int:
        self.node_bmin.append(bmin)
        self.node_bmax.append(bmax)
        self.node_left.append(left)
        self.node_right.append(right)
        self.node_tris.append(tris)
        return len(self.node_bmin) - 1

    def _build(self, tri_idx: np.ndarray) -> int:
        bmin = np.min(self.tri_min[tri_idx], axis=0)
        bmax = np.max(self.tri_max[tri_idx], axis=0)

        if tri_idx.size <= self.leaf_size:
            return self._new_node(bmin, bmax, tris=tri_idx)

        # split by longest axis of centroids
        ctr = self.tri_ctr[tri_idx]
        extent = np.max(ctr, axis=0) - np.min(ctr, axis=0)
        axis = int(np.argmax(extent))
        order = np.argsort(ctr[:, axis], kind="mergesort")
        tri_idx = tri_idx[order]
        mid = tri_idx.size // 2

        left = self._build(tri_idx[:mid])
        right = self._build(tri_idx[mid:])
        return self._new_node(bmin, bmax, left=left, right=right, tris=None)

    def closest_d2(self, p: np.ndarray) -> float:
        best = float("inf")
        stack = [self.root]
        V = self.V
        F = self.F

        while stack:
            ni = stack.pop()
            bmin = self.node_bmin[ni]
            bmax = self.node_bmax[ni]
            if _point_aabb_d2(p, bmin, bmax) >= best:
                continue

            tris = self.node_tris[ni]
            if tris is not None:
                for tid in tris:
                    f = F[int(tid)]
                    a = V[f[0]]
                    b = V[f[1]]
                    c = V[f[2]]
                    d2 = _point_triangle_d2(p, a, b, c)
                    if d2 < best:
                        best = d2
                continue

            # internal node
            l = self.node_left[ni]
            r = self.node_right[ni]
            # push nearer first (simple heuristic)
            dl = _point_aabb_d2(p, self.node_bmin[l], self.node_bmax[l])
            dr = _point_aabb_d2(p, self.node_bmin[r], self.node_bmax[r])
            if dl < dr:
                if dr < best: stack.append(r)
                if dl < best: stack.append(l)
            else:
                if dl < best: stack.append(l)
                if dr < best: stack.append(r)

        return best


def deterministic_probe_points(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Deterministic (no randomness) set of points to probe distance:
      - all vertices
      - all triangle centroids
      - all triangle edge midpoints
    This is much stronger than vertex-only, and still deterministic.
    """
    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]
    cent = (a + b + c) / 3.0
    m01 = (a + b) / 2.0
    m12 = (b + c) / 2.0
    m20 = (c + a) / 2.0
    return np.vstack([V, cent, m01, m12, m20])


def hausdorff_bvh(V1: np.ndarray, F1: np.ndarray, V2: np.ndarray, F2: np.ndarray, leaf_size: int = 8) -> float:
    bvh1 = TriBVH(V1, F1, leaf_size=leaf_size)
    bvh2 = TriBVH(V2, F2, leaf_size=leaf_size)

    P1 = deterministic_probe_points(V1, F1)
    P2 = deterministic_probe_points(V2, F2)

    max12 = 0.0
    for p in P1:
        d2 = bvh2.closest_d2(p)
        if d2 > max12:
            max12 = d2

    max21 = 0.0
    for p in P2:
        d2 = bvh1.closest_d2(p)
        if d2 > max21:
            max21 = d2

    return float(np.sqrt(max(max12, max21)))


def evaluate_mesh(name: str, V: np.ndarray, F: np.ndarray, Vref: np.ndarray, Fref: np.ndarray,
                  bvh_leaf: int) -> Dict[str, Any]:
    per_min, _, ang = per_triangle_min_max_angle_deg(V, F)
    q = triangle_quality_q(V, F)
    row: Dict[str, Any] = {
        "Qmin": float(np.min(q)),
        "Qavg": float(np.mean(q)),
        "theta_min": float(np.min(per_min)),
        "theta_min_avg": float(np.mean(per_min)),
        "theta_lt_30_pct": float(np.mean(ang < 30.0) * 100.0),
        "theta_gt_90_pct": float(np.mean(ang > 90.0) * 100.0),
        "dH": 0.0,
    }
    try:
        row["dH"] = hausdorff_bvh(V, F, Vref, Fref, leaf_size=bvh_leaf)
    except Exception as e:
        row["dH"] = float("nan")
        row["note"] = str(e)
    return row


def list_obj_files(dir_path: str) -> list[str]:
    d = os.path.abspath(dir_path)
    files = [os.path.join(d, fn) for fn in os.listdir(d) if fn.lower().endswith(".obj")]
    files.sort()
    return files


def parse_iter_from_name(path: str) -> int:
    m = re.search(r"(?:^|/|\\)iter_(\d+)\.obj$", path, flags=re.IGNORECASE)
    return int(m.group(1)) if m else -1


def default_runs_csv_path() -> str:
    return os.path.abspath(os.path.join(os.getcwd(), "..", "build", "runs.csv"))


def upsert_run_csv(path: str, key_cols: list[str], new_row: dict):
    rows = []
    fields = []

    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            fields = list(r.fieldnames)
            for row in r:
                rows.append(dict(row))

    for k in new_row.keys():
        if k not in fields:
            fields.append(k)

    updated = False
    for r in rows:
        if all(str(r.get(k, "")) == str(new_row.get(k, "")) for k in key_cols):
            for k, v in new_row.items():
                r[k] = v
            updated = True
            break

    if not updated:
        rows.append(new_row)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_iters_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("dir")
    ap.add_argument("--bvh_leaf", type=int, default=8, help="BVH leaf size (#tris per leaf)")
    ap.add_argument("--runs_csv", default=None)
    ap.add_argument("--iters_csv", default=None)
    args = ap.parse_args()

    ref_path = os.path.abspath(args.ref)
    out_dir = os.path.abspath(args.dir)

    obj_paths = list_obj_files(out_dir)
    if not obj_paths:
        raise SystemExit(f"No .obj in {out_dir}")

    meshname = os.path.basename(os.path.normpath(out_dir))
    mode = os.path.basename(os.path.dirname(os.path.normpath(out_dir)))

    Vref, Fref = load_obj_tri_mesh(ref_path)

    cand = [(parse_iter_from_name(p), p) for p in obj_paths]
    cand = [(it, p) for (it, p) in cand if it >= 0]
    if not cand:
        raise SystemExit("No iter_XXX.obj found")
    cand.sort(key=lambda x: x[0])

    rows_iters: List[Dict[str, Any]] = []
    for it, p in cand:
        V, F = load_obj_tri_mesh(p)
        r = evaluate_mesh(os.path.basename(p), V, F, Vref, Fref, bvh_leaf=args.bvh_leaf)
        r["meshname"] = meshname
        r["mode"] = mode
        r["iter"] = int(it)
        rows_iters.append(r)

    iters_csv = args.iters_csv or os.path.join(out_dir, "eval_iters.csv")
    write_iters_csv(iters_csv, rows_iters)

    last_row = dict(rows_iters[-1])
    runs_csv = args.runs_csv or default_runs_csv_path()
    upsert_run_csv(runs_csv, ["meshname", "mode"], last_row)

    print(f"wrote {iters_csv}")
    print(f"appended {runs_csv}")


if __name__ == "__main__":
    main()
