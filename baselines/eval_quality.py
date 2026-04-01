#!/usr/bin/env python3
"""
Shared quality-metric evaluation for baseline comparisons.

Computes the SAME metrics as evaluate.cuh in the GPU_CVT project:
  Qmin, Qavg           -- triangle aspect-ratio quality  (6/sqrt(3)) * area / (S * E)
  theta_min, theta_min_avg  -- minimum angle per triangle (degrees)
  theta_lt_30_pct      -- % of angles < 30 deg
  theta_gt_90_pct      -- % of angles > 90 deg
  dH                   -- one-sided Hausdorff distance (remeshed -> reference)

Usage as module:
    from eval_quality import eval_mesh, eval_hausdorff

Usage as CLI:
    python eval_quality.py <remeshed.obj> [--ref <reference.obj>]
"""

import math
import numpy as np
import trimesh
import argparse
import csv
import sys
import os


# ----------------------------------------------------------------
# Triangle quality  (matches evaluate.cuh  eval_quality_angles_cpu)
# ----------------------------------------------------------------

def eval_quality_angles(vertices: np.ndarray, faces: np.ndarray):
    """
    Compute quality and angle metrics identical to evaluate.cuh.

    Returns dict with: Qmin, Qavg, theta_min, theta_min_avg,
                       theta_lt_30_pct, theta_gt_90_pct, n_vertices, n_faces
    """
    V = vertices.astype(np.float64)
    F = faces

    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]

    ab = b - a;  ac = c - a
    ba = a - b;  bc = c - b
    ca = a - c;  cb = b - c

    def angle_deg(u, v):
        du = np.linalg.norm(u, axis=1)
        dv = np.linalg.norm(v, axis=1)
        denom = np.maximum(du * dv, 1e-30)
        cs = np.sum(u * v, axis=1) / denom
        cs = np.clip(cs, -1.0, 1.0)
        return np.degrees(np.arccos(cs))

    A = angle_deg(ab, ac)
    B = angle_deg(ba, bc)
    C = angle_deg(ca, cb)

    tmin = np.minimum(A, np.minimum(B, C))
    theta_min = float(np.min(tmin))
    theta_min_avg = float(np.mean(tmin))

    all_angles = np.concatenate([A, B, C])
    cnt_lt30 = float(np.sum(all_angles < 30.0))
    cnt_gt90 = float(np.sum(all_angles > 90.0))
    cnt_ang = float(len(all_angles))
    theta_lt_30_pct = (cnt_lt30 * 100.0 / cnt_ang) if cnt_ang > 0 else 0.0
    theta_gt_90_pct = (cnt_gt90 * 100.0 / cnt_ang) if cnt_ang > 0 else 0.0

    # Aspect ratio: q = (6/sqrt(3)) * area / (S * E)
    l0 = np.linalg.norm(b - a, axis=1)
    l1 = np.linalg.norm(c - b, axis=1)
    l2 = np.linalg.norm(a - c, axis=1)

    cross = np.cross(ab, ac)
    area = 0.5 * np.linalg.norm(cross, axis=1)

    S = 0.5 * (l0 + l1 + l2)
    E = np.maximum(l0, np.maximum(l1, l2))
    denom = np.maximum(S * E, 1e-30)
    q = (6.0 / math.sqrt(3.0)) * (area / denom)

    Qmin = float(np.min(q))
    Qavg = float(np.mean(q))

    return {
        "Qmin": Qmin,
        "Qavg": Qavg,
        "theta_min": theta_min,
        "theta_min_avg": theta_min_avg,
        "theta_lt_30_pct": theta_lt_30_pct,
        "theta_gt_90_pct": theta_gt_90_pct,
        "n_vertices": len(V),
        "n_faces": len(F),
    }


# ----------------------------------------------------------------
# One-sided Hausdorff  (remeshed -> reference)
#   matches evaluate.cuh: probe = vertices + face centroids + edge midpoints
# ----------------------------------------------------------------

def eval_hausdorff(remeshed_verts, remeshed_faces, ref_mesh: trimesh.Trimesh):
    """
    One-sided Hausdorff: max distance from probe points on remeshed surface
    to the closest point on the reference mesh.

    Probe points = vertices + face centroids + edge midpoints (same as evaluate.cuh).
    """
    V = remeshed_verts.astype(np.float64)
    F = remeshed_faces

    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]

    centroids = (a + b + c) / 3.0
    mid_ab = (a + b) / 2.0
    mid_bc = (b + c) / 2.0
    mid_ca = (c + a) / 2.0

    probes = np.vstack([V, centroids, mid_ab, mid_bc, mid_ca])

    closest, distances, _ = ref_mesh.nearest.on_surface(probes)
    dH = float(np.max(distances))
    return dH


# ----------------------------------------------------------------
# Convenience: load mesh, evaluate everything
# ----------------------------------------------------------------

def eval_mesh(remeshed_path: str, ref_path: str = None):
    """Load remeshed OBJ, compute quality metrics and optionally Hausdorff."""
    mesh = trimesh.load(remeshed_path, process=False, force="mesh")
    V = np.array(mesh.vertices)
    F = np.array(mesh.faces)

    result = eval_quality_angles(V, F)

    if ref_path is not None:
        ref = trimesh.load(ref_path, process=False, force="mesh")
        result["dH"] = eval_hausdorff(V, F, ref)
    else:
        result["dH"] = -1.0

    return result


def write_result_csv(path, mesh_name, method, time_ms, metrics):
    """Append one row to a baseline results CSV."""
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow([
                "mesh", "method", "n_vertices", "n_faces",
                "time_ms", "Qmin", "Qavg",
                "theta_min", "theta_min_avg",
                "theta_lt_30_pct", "theta_gt_90_pct", "dH",
            ])
        w.writerow([
            mesh_name, method, metrics["n_vertices"], metrics["n_faces"],
            f"{time_ms:.1f}",
            f"{metrics['Qmin']:.6f}", f"{metrics['Qavg']:.6f}",
            f"{metrics['theta_min']:.4f}", f"{metrics['theta_min_avg']:.4f}",
            f"{metrics['theta_lt_30_pct']:.5f}", f"{metrics['theta_gt_90_pct']:.5f}",
            f"{metrics['dH']:.6f}",
        ])


# ----------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate mesh quality (GPU_CVT-compatible metrics)")
    p.add_argument("mesh", help="Path to remeshed OBJ")
    p.add_argument("--ref", default=None, help="Path to reference mesh for Hausdorff")
    args = p.parse_args()

    m = eval_mesh(args.mesh, args.ref)
    print(f"n_vertices:      {m['n_vertices']}")
    print(f"n_faces:         {m['n_faces']}")
    print(f"Qmin:            {m['Qmin']:.6f}")
    print(f"Qavg:            {m['Qavg']:.6f}")
    print(f"theta_min:       {m['theta_min']:.4f}")
    print(f"theta_min_avg:   {m['theta_min_avg']:.4f}")
    print(f"theta_lt_30_pct: {m['theta_lt_30_pct']:.5f}")
    print(f"theta_gt_90_pct: {m['theta_gt_90_pct']:.5f}")
    if m["dH"] >= 0:
        print(f"dH:              {m['dH']:.6f}")
