#!/usr/bin/env python3
"""
CGAL Isotropic Remeshing baseline via PyMeshLab.

Uses the split/collapse/flip/smooth pipeline (Botsch & Kobbelt 2004).
This is NOT CVT -- it's the standard greedy local-operations approach.

Usage:
  python run_cgal_remesh.py <input.obj> <output.obj> [--iterations N]

The target edge length is set so the output vertex count approximately
matches the input vertex count (fair comparison with GPU_CVT which
keeps the same number of sites as input vertices).
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pymeshlab


def estimate_target_edge_length(ms: pymeshlab.MeshSet) -> float:
    """
    Compute the average edge length of the input mesh.
    Using this as target preserves roughly the same vertex count.
    """
    m = ms.current_mesh()
    V = m.vertex_matrix()
    F = m.face_matrix()

    # Collect all edge lengths
    e0 = np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1)
    e1 = np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1)
    e2 = np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1)

    avg_len = float(np.mean(np.concatenate([e0, e1, e2])))
    return avg_len


def main():
    p = argparse.ArgumentParser(description="CGAL isotropic remeshing baseline")
    p.add_argument("input", help="Input mesh OBJ path")
    p.add_argument("output", help="Output mesh OBJ path")
    p.add_argument("--iterations", type=int, default=10,
                   help="Number of smoothing iterations (default: 10)")
    args = p.parse_args()

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(args.input)
    m = ms.current_mesh()
    print(f"Input: {m.vertex_number()} vertices, {m.face_number()} faces",
          file=sys.stderr)

    target_len = estimate_target_edge_length(ms)
    print(f"Target edge length: {target_len:.6f}", file=sys.stderr)

    # --- timed section ---
    t0 = time.perf_counter()

    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(target_len),
        iterations=args.iterations,
        adaptive=False,
    )

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    m_out = ms.current_mesh()
    print(f"Output: {m_out.vertex_number()} vertices, {m_out.face_number()} faces",
          file=sys.stderr)
    print(f"Remesh time: {elapsed_ms:.1f} ms", file=sys.stderr)

    # Machine-readable
    print(f"TIME_MS={elapsed_ms:.1f}")
    print(f"NV_OUT={m_out.vertex_number()}")
    print(f"NF_OUT={m_out.face_number()}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    ms.save_current_mesh(args.output)
    print(f"Wrote: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
