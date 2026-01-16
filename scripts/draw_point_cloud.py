import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_points(filename):
    """Loads Nx3 float point cloud."""
    pts = []
    with open(filename, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 3:
                continue
            pts.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return np.array(pts)


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_point_cloud.py <point_file.txt> [max_samples]")
        sys.exit(1)

    filename = sys.argv[1]
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

    pts = load_points(filename)
    if pts.shape[0] == 0:
        print("Empty or invalid point file.")
        return

    # Optional sampling for huge clouds
    if pts.shape[0] > max_samples:
        idx = np.random.choice(pts.shape[0], max_samples, replace=False)
        pts = pts[idx]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)

    # equal axis scaling
    max_r = (pts.max(axis=0) - pts.min(axis=0)).max()
    mid = pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_r/2, mid[0] + max_r/2)
    ax.set_ylim(mid[1] - max_r/2, mid[1] + max_r/2)
    ax.set_zlim(mid[2] - max_r/2, mid[2] + max_r/2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(f"{filename} — showing {pts.shape[0]} points")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
