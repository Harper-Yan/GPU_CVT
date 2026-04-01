#!/usr/bin/env python3
"""
ACVD baseline via pyacvd (Valette & Chassery 2004).

Approximated Centroidal Voronoi Diagram -- discrete clustering approach.
Trades quality for speed/scalability.

Usage:
  python run_acvd.py <input.obj> <output.obj> [--clusters N]

If --clusters is omitted, uses the input vertex count for fair comparison.
"""

import argparse
import os
import sys
import time

import numpy as np
import pyvista as pv
import pyacvd
import trimesh


def main():
    p = argparse.ArgumentParser(description="ACVD remeshing baseline")
    p.add_argument("input", help="Input mesh OBJ path")
    p.add_argument("output", help="Output mesh OBJ path")
    p.add_argument("--clusters", type=int, default=0,
                   help="Number of clusters (0 = match input vertex count)")
    args = p.parse_args()

    # Load with pyvista
    mesh_pv = pv.read(args.input)
    n_points = mesh_pv.n_points
    n_cells = mesh_pv.n_cells
    print(f"Input: {n_points} vertices, {n_cells} faces", file=sys.stderr)

    n_clusters = args.clusters if args.clusters > 0 else n_points

    # --- timed section ---
    t0 = time.perf_counter()

    clus = pyacvd.Clustering(mesh_pv)
    clus.subdivide(3)  # subdivide for better clustering
    clus.cluster(n_clusters)
    remeshed = clus.create_mesh()

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    n_out_pts = remeshed.n_points
    n_out_cells = remeshed.n_cells
    print(f"Output: {n_out_pts} vertices, {n_out_cells} faces", file=sys.stderr)
    print(f"Remesh time: {elapsed_ms:.1f} ms", file=sys.stderr)

    # Machine-readable
    print(f"TIME_MS={elapsed_ms:.1f}")
    print(f"NV_OUT={n_out_pts}")
    print(f"NF_OUT={n_out_cells}")

    # Save as OBJ via trimesh (pyvista OBJ export can be unreliable)
    # Convert pyvista PolyData to trimesh
    faces_pv = remeshed.faces.reshape(-1, 4)  # (n, 4) with leading 3
    faces_tri = faces_pv[:, 1:4]
    verts = np.array(remeshed.points)

    tm = trimesh.Trimesh(vertices=verts, faces=faces_tri, process=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tm.export(args.output)
    print(f"Wrote: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
