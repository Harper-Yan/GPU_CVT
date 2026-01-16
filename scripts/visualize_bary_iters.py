import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_points(filename):
    """Load float3 barycenters from text file."""
    pts = []
    with open(filename, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) != 3:
                continue
            pts.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return np.array(pts)


def random_sample(pts, max_samples=10000):
    n = pts.shape[0]
    if n <= max_samples:
        return pts
    idx = np.random.choice(n, max_samples, replace=False)
    return pts[idx]


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_bary_compare.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print("Error: not a directory:", directory)
        sys.exit(1)

    # Pattern: barycenters_modeX_iterYYYY.txt
    pattern = re.compile(r"barycenters_mode([01])_iter(\d+)\.txt$")

    # Collect files by iteration
    mode0 = {}
    mode1 = {}

    for fname in os.listdir(directory):
        m = pattern.match(fname)
        if m:
            mode = int(m.group(1))
            iter_id = int(m.group(2))
            path = os.path.join(directory, fname)
            if mode == 0:
                mode0[iter_id] = path
            else:
                mode1[iter_id] = path

    # Find intersection of iterations that have both modes
    common_iters = sorted(set(mode0.keys()) & set(mode1.keys()))

    if not common_iters:
        print("No matching iterations with both modes found.")
        sys.exit(0)

    for iter_id in common_iters:
        f0 = mode0[iter_id]
        f1 = mode1[iter_id]

        pts0 = load_points(f0)
        pts1 = load_points(f1)

        if pts0.size == 0 or pts1.size == 0:
            print(f"Warning: empty file at iter {iter_id}")
            continue

        pts0s = random_sample(pts0)
        pts1s = random_sample(pts1)

        fig = plt.figure(figsize=(14, 6))

        # --- MODE 0 ---
        ax0 = fig.add_subplot(121, projection="3d")
        ax0.scatter(pts0s[:,0], pts0s[:,1], pts0s[:,2], s=1)
        ax0.set_title(f"Mode 0 — Iter {iter_id}\nShowing {pts0s.shape[0]}/{pts0.shape[0]}")

        # --- MODE 1 ---
        ax1 = fig.add_subplot(122, projection="3d")
        ax1.scatter(pts1s[:,0], pts1s[:,1], pts1s[:,2], s=1)
        ax1.set_title(f"Mode 1 — Iter {iter_id}\nShowing {pts1s.shape[0]}/{pts1.shape[0]}")

        # --- equal axis scaling across both subplots ---
        combined = np.vstack([pts0s, pts1s])
        max_range = (combined.max(axis=0) - combined.min(axis=0)).max()
        mid = combined.mean(axis=0)

        for ax in (ax0, ax1):
            ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
            ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
            ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        plt.suptitle(f"Comparison of Mode 0 vs Mode 1 at Iter {iter_id}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
