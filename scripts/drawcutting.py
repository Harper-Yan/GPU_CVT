import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_debug_file(filename):
    """
    Parses the debug text and extracts:
      - seed position
      - plane normal n
      - plane d
      - list of vertices (index, coords)
      - list of removed vertices
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    plane_n = None
    plane_d = None
    seed = None
    vertices = []
    removed = []

    seed_re = re.compile(r"seed:=\(([-0-9\.]+)\s+([-0-9\.]+)\s+([-0-9\.]+)\)")
    plane_re = re.compile(r"Plane:\s*n=\(([-0-9\. ]+)\),\s*d=([-0-9\.]+)")
    vert_re = re.compile(r"V\[(\d+)\].*->\s*\(([-0-9\.]+)\s+([-0-9\.]+)\s+([-0-9\.]+)\)")
    remove_re = re.compile(r"REMOVE vertex.*pos=\(([-0-9\.]+)\s+([-0-9\.]+)\s+([-0-9\.]+)\)")

    for line in lines:
        m = seed_re.search(line)
        if m:
            seed = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])

        m = plane_re.search(line)
        if m:
            nx, ny, nz = map(float, m.group(1).split())
            plane_n = np.array([nx, ny, nz])
            plane_d = float(m.group(2))

        m = vert_re.search(line)
        if m:
            idx = int(m.group(1))
            x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
            vertices.append((idx, np.array([x, y, z])))

        m = remove_re.search(line)
        if m:
            x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
            removed.append(np.array([x, y, z]))

    return seed, plane_n, plane_d, vertices, removed


def print_signed_distances(seed, plane_n, plane_d, vertices, removed):
    """Print signed distances for seed, vertices, and removed vertices."""
    sd = lambda v: float(np.dot(plane_n, v) + plane_d)

    print("\n=== SIGNED DISTANCES ===")
    print(f"Seed = {seed}, s = {sd(seed):.6f}")

    print("\nOriginal vertices:")
    for idx, v in vertices:
        print(f"  V[{idx}] = {v}, s = {sd(v):.6f}")

    if removed:
        print("\nRemoved vertices:")
        for v in removed:
            print(f"  Removed = {v}, s = {sd(v):.6f}")
    else:
        print("\n(No removed vertices recorded)")

    print("========================\n")


def make_clipping_plane(n, d, size=1500.0):
    if abs(n[0]) < 0.9:
        v = np.array([1,0,0])
    else:
        v = np.array([0,1,0])

    u = np.cross(n, v)
    u /= np.linalg.norm(u)
    v2 = np.cross(n, u)
    v2 /= np.linalg.norm(v2)

    p0 = -d * n / np.dot(n, n)

    s = size
    return [
        p0 + u*s + v2*s,
        p0 + u*s - v2*s,
        p0 - u*s - v2*s,
        p0 - u*s + v2*s,
    ]


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_clip.py debug.txt")
        return

    seed, plane_n, plane_d, vertices, removed = parse_debug_file(sys.argv[1])

    # -------------------------------
    # NEW: print signed distances!
    # -------------------------------
    print_signed_distances(seed, plane_n, plane_d, vertices, removed)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    if vertices:
        pts = np.array([v[1] for v in vertices])
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], color="blue", s=40, label="Original vertices")

    if removed:
        rem = np.vstack(removed)
        ax.scatter(rem[:,0], rem[:,1], rem[:,2], color="red", s=60, label="Removed vertices")

    if seed is not None:
        ax.scatter([seed[0]], [seed[1]], [seed[2]], color="yellow", edgecolor="black",
                   marker="*", s=300, label="Seed")

    if plane_n is not None:
        plane_poly = Poly3DCollection([make_clipping_plane(plane_n, plane_d)],
                                      alpha=0.25, facecolor='green')
        ax.add_collection3d(plane_poly)

    all_pts = []
    if vertices: all_pts.extend([v[1] for v in vertices])
    if removed:  all_pts.extend(removed)
    if seed is not None: all_pts.append(seed)

    if all_pts:
        pts = np.vstack(all_pts)
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        center = (mn + mx) / 2
        ext = (mx - mn).max() * 0.6
        ax.set_xlim(center[0]-ext, center[0]+ext)
        ax.set_ylim(center[1]-ext, center[1]+ext)
        ax.set_zlim(center[2]-ext, center[2]+ext)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Voronoi Cell Clipping Visualization (with Signed Distances)")

    plt.show()


if __name__ == "__main__":
    main()
