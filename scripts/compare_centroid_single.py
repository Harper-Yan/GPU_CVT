#!/usr/bin/env python3
"""
Reproduce centroid computation for a single point to isolate GPU vs Python discrepancy.
Uses GPU's exact inputs (S, U, V, KNN order) and runs Python's cell_poly2d + poly_area_centroid_2d.
If Python with GPU KNN matches GPU centroid -> issue is KNN order (Python uses index order).
If not -> bug in GPU kernel logic.
"""
import numpy as np
import argparse
import os
import sys

# Import from testfreeze
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from testfreeze import cell_poly2d, poly_area_centroid_2d


def load_xyz(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(data, dtype=np.float64) if data else None


def load_obj_vertices(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    data.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(data, dtype=np.float64) if data else None


def load_gpu_knn_for_point(knn_full_path, point_i):
    """Parse debug_3_knn_full.txt and return neighbor indices for given point."""
    knn = None
    with open(knn_full_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                idx = int(parts[0])
                if idx == point_i:
                    knn = np.array([int(x) for x in parts[1:]], dtype=np.int64)
                    break
    return knn


def main():
    p = argparse.ArgumentParser(description="Compare centroid for single point: Python vs GPU")
    p.add_argument("mesh", type=str, help="Input mesh .obj path")
    p.add_argument("--gpu_dir", type=str, default="results/stanford-bunny/baseline",
                   help="GPU output dir (debug_* files)")
    p.add_argument("--point", type=int, default=31257, help="Point index to compare")
    args = p.parse_args()

    mesh = os.path.abspath(args.mesh)
    gpu_base = os.path.abspath(args.gpu_dir)

    # Load S: use GPU's debug_S_full.xyz (matches compacted mesh used by GPU)
    s_path = os.path.join(gpu_base, "debug_S_full.xyz")
    if not os.path.isfile(s_path):
        print(f"Missing {s_path} - run GPU with debug first to generate it")
        return 1
    S = load_xyz(s_path)
    if S is None:
        print("Failed to load S (sites)")
        return 1
    nV = len(S)

    if args.point >= nV:
        print(f"Point {args.point} out of range (nV={nV})")
        return 1

    # Load U, V from GPU
    u_path = os.path.join(gpu_base, "debug_U_full.xyz")
    v_path = os.path.join(gpu_base, "debug_V_full.xyz")
    U = load_xyz(u_path)
    V = load_xyz(v_path)
    if U is None or V is None or len(U) < nV or len(V) < nV:
        print("Failed to load U,V from GPU")
        return 1
    U = U[:nV]
    V = V[:nV]

    # Load GPU centroid
    gpu_centroids = load_xyz(os.path.join(gpu_base, "before_projection_full.xyz"))
    if gpu_centroids is None or len(gpu_centroids) < nV:
        print("Failed to load GPU centroids")
        return 1
    gpu_cent = gpu_centroids[args.point]

    # Load GPU KNN for this point (distance order)
    gpu_knn = load_gpu_knn_for_point(os.path.join(gpu_base, "debug_3_knn_full.txt"), args.point)
    if gpu_knn is None:
        print(f"Point {args.point} not in debug_3_knn_full.txt - run GPU with debug first")
        return 1

    # R = max bbox dimension (same as both pipelines)
    bbox = S.max(axis=0) - S.min(axis=0)
    R = float(np.max(bbox))

    # Python centroid with GPU's KNN order (distance order)
    poly_gpu_order = cell_poly2d(args.point, S, U, V, gpu_knn, R)
    _, c2_gpu_order = poly_area_centroid_2d(poly_gpu_order)
    py_cent_gpu_order = S[args.point] + c2_gpu_order[0] * U[args.point] + c2_gpu_order[1] * V[args.point]

    # Python centroid with index-sorted KNN (Python's default)
    idxn_sorted = np.sort(gpu_knn)
    poly_idx_order = cell_poly2d(args.point, S, U, V, idxn_sorted, R)
    _, c2_idx_order = poly_area_centroid_2d(poly_idx_order)
    py_cent_idx_order = S[args.point] + c2_idx_order[0] * U[args.point] + c2_idx_order[1] * V[args.point]

    # Also try with float32 (GPU uses float) to rule out precision
    S32 = S.astype(np.float32)
    U32 = np.asarray(U, dtype=np.float32)
    V32 = np.asarray(V, dtype=np.float32)
    R32 = np.float32(R)
    poly_f32 = cell_poly2d(args.point, S32, U32, V32, gpu_knn, R32)
    _, c2_f32 = poly_area_centroid_2d(poly_f32)
    py_cent_f32 = S32[args.point] + c2_f32[0] * U32[args.point] + c2_f32[1] * V32[args.point]

    print("=" * 60)
    print(f"Centroid comparison for point {args.point}")
    print("=" * 60)
    print(f"Sites S from mesh: {mesh}")
    print(f"U,V from: {gpu_base}/debug_U_full.xyz, debug_V_full.xyz")
    print(f"R = {R:.6g}")
    print()
    print("GPU centroid:        ", gpu_cent)
    print("Python (GPU KNN ord):", py_cent_gpu_order)
    print("Python (index order):", py_cent_idx_order)
    print("Python (float32):    ", py_cent_f32)
    print()
    diff_gpu_order = np.abs(gpu_cent - py_cent_gpu_order)
    diff_idx_order = np.abs(gpu_cent - py_cent_idx_order)
    print("|GPU - Python(GPU order)|:", diff_gpu_order, "max=", np.max(diff_gpu_order))
    print("|GPU - Python(idx order)|:", diff_idx_order, "max=", np.max(diff_idx_order))
    print()

    if np.max(diff_gpu_order) < 1e-5:
        print("Python and GPU centroids MATCH (within tolerance).")
    elif np.max(diff_idx_order) < 1e-5:
        print("Python with index order MATCHES GPU -> no discrepancy for this point.")
    else:
        print("Neither Python order matches GPU -> bug in GPU kernel (clipping or centroid formula).")
        print("Polygon from GPU order: n_verts =", len(poly_gpu_order))
        if len(poly_gpu_order) <= 20:
            print("  verts:", poly_gpu_order)

    return 0


if __name__ == "__main__":
    sys.exit(main())
