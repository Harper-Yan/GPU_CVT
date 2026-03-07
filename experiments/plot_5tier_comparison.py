#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

base_old = os.path.join(_ROOT, "results", "teapot")
base_new = os.path.join(_ROOT, "results", "archive_5tier", "teapot")

nofreeze  = pd.read_csv(os.path.join(base_old, "nofreeze", "teapot_NOFREEZE.csv"))
freeze3   = pd.read_csv(os.path.join(base_old, "freeze",   "teapot_FREEZE.csv"))
freeze5   = pd.read_csv(os.path.join(base_new, "freeze",   "teapot_FREEZE.csv"))

max_iter = min(nofreeze["iter"].max(), freeze5["iter"].max())
nofreeze = nofreeze[nofreeze["iter"] <= max_iter]
freeze3  = freeze3[freeze3["iter"] <= max_iter]
freeze5  = freeze5[freeze5["iter"] <= max_iter]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("5-Tier Data-Driven Freeze vs 3-Tier Original vs NOFREEZE",
             fontsize=14, fontweight="bold")

# --- avg_aspect ---
ax = axes[0, 0]
ax.plot(nofreeze["iter"], nofreeze["avg_aspect"], "b-o", markersize=3, label="NOFREEZE", alpha=0.8)
ax.plot(freeze3["iter"],  freeze3["avg_aspect"],  "r-s", markersize=3, label="3-tier FREEZE", alpha=0.8)
ax.plot(freeze5["iter"],  freeze5["avg_aspect"],  "g-^", markersize=3, label="5-tier FREEZE", alpha=0.8)
ax.set_xlabel("Iteration")
ax.set_ylabel("Avg Aspect Ratio")
ax.set_title("Average Aspect Ratio (lower = better)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- pct_gt_90 ---
ax = axes[0, 1]
ax.plot(nofreeze["iter"], nofreeze["pct_gt_90"], "b-o", markersize=3, label="NOFREEZE", alpha=0.8)
ax.plot(freeze3["iter"],  freeze3["pct_gt_90"],  "r-s", markersize=3, label="3-tier FREEZE", alpha=0.8)
ax.plot(freeze5["iter"],  freeze5["pct_gt_90"],  "g-^", markersize=3, label="5-tier FREEZE", alpha=0.8)
ax.set_xlabel("Iteration")
ax.set_ylabel("% triangles > 90 deg")
ax.set_title("Obtuse Triangles % (lower = better)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- pct_lt_30 ---
ax = axes[1, 0]
ax.plot(nofreeze["iter"], nofreeze["pct_lt_30"], "b-o", markersize=3, label="NOFREEZE", alpha=0.8)
ax.plot(freeze3["iter"],  freeze3["pct_lt_30"],  "r-s", markersize=3, label="3-tier FREEZE", alpha=0.8)
ax.plot(freeze5["iter"],  freeze5["pct_lt_30"],  "g-^", markersize=3, label="5-tier FREEZE", alpha=0.8)
ax.set_xlabel("Iteration")
ax.set_ylabel("% triangles < 30 deg")
ax.set_title("Skinny Triangles % (lower = better)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- frozen count ---
ax = axes[1, 1]
ax.plot(freeze3["iter"], freeze3["frozen"], "r-s", markersize=3, label="3-tier FREEZE", alpha=0.8)
ax.plot(freeze5["iter"], freeze5["frozen"], "g-^", markersize=3, label="5-tier FREEZE", alpha=0.8)
ax.axhline(y=3644, color="gray", linestyle="--", alpha=0.5, label="Total sites (3644)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Frozen sites")
ax.set_title("Freeze Progression")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(_ROOT, "results", "archive_5tier", "teapot", "freeze", "5tier_vs_3tier_vs_nofreeze.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
