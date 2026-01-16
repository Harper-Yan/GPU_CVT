#!/usr/bin/env python3
"""
OBJ triangle viewer
Supports:
  v x y z
  f i j k    (1-based triangle indices)

Shows triangles clearly (faces + edges).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_obj(path):
    vertices = []
    faces = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append((float(x), float(y), float(z)))

            elif line.startswith("f "):
                parts = line.split()[1:4]
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(idx)

    return np.array(vertices), np.array(faces)


def equal_axes(ax, V):
    min_v = V.min(axis=0)
    max_v = V.max(axis=0)
    center = (min_v + max_v) * 0.5
    radius = np.max(max_v - min_v) * 0.5

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main():
    if len(sys.argv) < 2:
        print("Usage: python view_obj_triangles.py model.obj")
        return

    V, F = load_obj(sys.argv[1])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    triangles = V[F]

    mesh = Poly3DCollection(
        triangles,
        facecolor=(0.6, 0.7, 1.0, 0.6),
        edgecolor="black",
        linewidths=0.8
    )

    ax.add_collection3d(mesh)

    equal_axes(ax, V)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(f"Triangles: {len(F)}")

    plt.show()


if __name__ == "__main__":
    main()
