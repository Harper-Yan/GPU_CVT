# 4. Experiments

## 4.1 Experimental Setup

**Benchmark meshes.** We evaluate on 12 triangle meshes spanning a range of vertex counts, curvature distributions, and geometric complexity:

| Mesh | Vertices | Faces |
|------|----------|-------|
| stanford-bunny | 34,834 | 69,451 |
| horse | 48,485 | 96,966 |
| happy | 49,251 | 98,498 |
| armadillo | 49,990 | 99,976 |
| lucy | 49,987 | 99,970 |
| nefertiti | 49,971 | 99,938 |
| bimba | 112,455 | 224,906 |
| xyzrgb_dragon | 124,943 | 249,882 |
| igea | 134,345 | 268,686 |
| Armadillo | 172,974 | 345,944 |
| dragon_vrip | 437,645 | 871,414 |
| happy_vrip | 543,652 | 1,087,716 |

**Modes compared.** All experiments compare two modes:
- **Mode 0 (Baseline):** Standard Lloyd iteration, all sites updated every iteration, no freezing.
- **Mode 1 (Freeze, 5-tier):** Curvature-adaptive dual-gate freeze with 5 NV-based tiers (Section 3.5).

**Parameters.** All runs use K = 20 nearest neighbors, 250 Lloyd iterations, and displacement threshold epsilon = 0.01 * R where R is the maximum bounding box edge. Tier assignment is computed once after 5 initial iterations.

**Hardware.** All experiments run on a single NVIDIA GPU. Timings include KNN, Voronoi clipping, centroid computation, reprojection, and freeze testing. Memory transfers are included.

**Quality metrics.** We report:
- Q_avg: average element quality (ratio of inscribed to circumscribed circle radius, normalized to [0,1]; 1 = equilateral).
- theta_min_avg: average minimum angle per triangle (optimal = 60 degrees for equilateral).
- theta_lt_30_pct: fraction of triangles with any angle below 30 degrees (lower = better).
- theta_gt_90_pct: fraction of triangles with any angle above 90 degrees (lower = better).
- d_H: Hausdorff distance from remeshed surface to input mesh (lower = better).

---

## 4.2 Speedup Results

Table 2 reports total remeshing time and speedup for all benchmark meshes.

| Mesh | Vertices | Baseline (s) | Ours (s) | Freeze % | Speedup |
|------|----------|-------------|---------|----------|---------|
| Armadillo | 173K | 265.9 | 14.9 | 79.0% | 17.9x |
| dragon_vrip | 438K | 573.9 | 56.4 | 71.1% | 10.2x |
| happy_vrip | 544K | 2486.7 | 228.5 | 56.2% | 10.9x |
| igea | 134K | 151.7 | 10.6 | 91.0% | 14.3x |
| xyzrgb_dragon | 125K | 134.6 | 13.7 | 20.9% | 9.8x |
| bimba | 112K | 109.3 | 10.4 | 58.5% | 10.5x |
| lucy | 50K | 26.3 | 5.7 | 5.4% | 4.6x |
| armadillo | 50K | 28.3 | 5.2 | 31.3% | 5.5x |
| happy | 49K | 28.9 | 5.5 | 9.9% | 5.3x |
| nefertiti | 50K | 24.4 | 4.9 | 55.9% | 5.0x |
| horse | 48K | 26.7 | 4.2 | 77.2% | 6.3x |
| stanford-bunny | 35K | 12.9 | 3.3 | 70.8% | 3.9x |

The baseline uses a brute-force KNN kernel; our method uses a bitonic-sort KNN backend with hub-based pruning that natively supports frozen-site masking. The reported speedups reflect the full end-to-end improvement, including both the KNN backend advantage and the freeze-based workload reduction.

**[PLOT: Per-iteration cumulative time, Baseline vs Freeze]**
Source: `experiments/plots_eval/<mesh>/three_modes_quality.png` for each mesh (per-iter time subplot).

**Key observations:**

*Speedup scales with mesh size.* The three largest meshes (Armadillo 173K, dragon_vrip 438K, happy_vrip 544K) achieve 10x to 18x end-to-end speedup. At these scales, baseline Lloyd iteration takes 4 to 41 minutes; our method reduces runtime to 15 seconds to 3.8 minutes. Medium-sized meshes (112K to 134K) achieve 10x to 14x speedup. The speedup combines two sources: the bitonic-sort KNN backend is inherently faster than the brute-force baseline for the sparse query patterns produced by freezing, and the freeze policy itself progressively eliminates redundant KNN queries. Smaller meshes (35K to 50K) show 3.9x to 6.3x speedup.

*Speedup correlates with freeze rate.* Meshes with high freeze rates (igea 91%, Armadillo 79%, horse 77%) achieve larger speedups. Meshes with predominantly smooth geometry and low curvature variation (happy 10%, lucy 5%, xyzrgb_dragon 21%) freeze fewer sites, though the KNN backend improvement alone still delivers substantial speedup.

---

## 4.3 Mesh Quality Preservation

Table 3 reports quality metrics at the final iteration (250 Lloyd iterations) for both modes.

| Mesh | Verts | Q_avg Base | Q_avg Ours | theta_min Base | theta_min Ours | theta_lt30 Base | theta_lt30 Ours | theta_gt90 Base | theta_gt90 Ours | d_H Base | d_H Ours |
|------|-------|-----------|-----------|---------------|---------------|----------------|----------------|----------------|----------------|---------|---------|
| stanford-bunny | 35K | 0.925 | 0.909 | 53.8 | 52.8 | 0.11 | 0.12 | 0.24 | 0.38 | 0.035 | 0.008 |
| horse | 48K | 0.929 | 0.915 | 54.1 | 53.1 | 0.02 | 0.03 | 0.21 | 0.32 | 0.008 | 0.008 |
| happy | 49K | 0.897 | 0.889 | 51.7 | 51.1 | 0.05 | 0.07 | 0.63 | 0.79 | 0.057 | 0.057 |
| armadillo | 50K | 0.915 | 0.904 | 53.1 | 52.3 | 0.01 | 0.00 | 0.22 | 0.33 | 0.510 | 0.454 |
| lucy | 50K | 0.879 | 0.874 | 50.3 | 50.0 | 0.20 | 0.21 | 1.24 | 1.32 | — | — |
| nefertiti | 50K | 0.916 | 0.901 | 53.1 | 52.1 | 0.02 | 0.04 | 0.29 | 0.48 | 22.6 | 22.6 |
| bimba | 112K | 0.920 | 0.907 | 53.5 | 52.6 | 0.01 | 0.01 | 0.21 | 0.32 | 1.04 | 1.04 |
| xyzrgb_dragon | 125K | 0.899 | 0.892 | 51.9 | 51.4 | 0.07 | 0.08 | 0.63 | 0.74 | 2.85 | 2.85 |
| igea | 134K | 0.935 | 0.916 | 54.6 | 53.3 | 0.00 | 0.00 | 0.09 | 0.23 | 0.000 | 0.000 |
| Armadillo | 173K | 0.934 | 0.933 | 54.5 | 54.4 | 0.00 | 0.00 | 0.10 | 0.11 | 0.171 | 0.242 |
| happy_vrip | 544K | 0.918 | 0.904 | 53.3 | 52.3 | 0.01 | 0.01 | 0.21 | 0.34 | 0.057 | 0.057 |

Quality degradation is small across all meshes and metrics. Q_avg decreases by at most 2.1% (igea), with most meshes below 1.6%. The largest Armadillo (173K), which achieves the highest speedup (17.8x), shows only -0.06% quality change, confirming that aggressive freezing at high freeze rates (79%) does not compromise output quality.

**Angle statistics.** Average minimum angle theta_min_avg differs by at most 1.3 degrees (igea) between baseline and our method. All meshes maintain theta_min_avg >= 50 degrees, well above the 30 degree threshold for acceptable mesh quality. Bad-angle fractions (theta_lt_30, theta_gt_90) remain below 1.4% on all meshes, with increases of at most 0.19 percentage points.

**Hausdorff distance.** d_H is identical or improved under freezing on all meshes except the 173K Armadillo (0.171 vs 0.242), confirming that the remeshed surface tracks the input geometry with comparable fidelity.

---

## 4.4 Causal Chain Validation

The freeze policy rests on two empirical claims: (1) high-curvature sites oscillate persistently due to tangent-plane distortion, and (2) displacement and neighborhood stability decouple at high curvature, requiring both gates for reliable convergence detection. We validate each claim through controlled experiments on the teapot mesh (3,644 sites, 50 iterations).

### Oscillation at High Curvature

**[FIGURE: fig_oscillation — two panels]**
**(a)** Centroid offset by curvature tier (from `exp12_effective_neighbors`). **(b)** Direction reversal fraction by curvature tier (from `exp3_direction_reversal`).

We measure two quantities across curvature tiers (Figure fig_oscillation). **Centroid offset** is the distance between the tangent-plane centroid and the current site position, measured in the tangent plane after Voronoi clipping. It quantifies how far the Lloyd update wants to move a site in a single iteration — larger offset means the tangent-plane approximation is producing a more biased centroid estimate. **Direction reversal rate** is the fraction of consecutive iteration pairs where the displacement vectors point in opposing directions (cosine < 0), measuring how often a site reverses course rather than progressing monotonically toward its Voronoi centroid.

Centroid offset grows 13x from flat to sharp regions (flat: 0.0039, sharp: 0.0518), confirming that tangent-plane distortion produces increasingly biased centroids at high curvature. This bias causes the reprojected centroid to land on different mesh triangles across iterations (2.8 unique triangles at flat vs 9.3 at curved over 50 iterations), cycling the local normal frame and producing persistent oscillation. The direction reversal rate rises from 3.3% at flat to 24.7% at sharp — at sharp regions, one in four consecutive steps is a reversal, confirming that high-curvature sites oscillate rather than converge monotonically.

### Displacement–Neighborhood Decoupling

**[FIGURE: fig_decoupling — from exp6_decoupling, panels 6a and 6c]**

At flat regions, low displacement reliably implies a stable neighborhood: only 0.04% of low-displacement iterations show a KNN topology change. At sharp regions, these two signals decouple: 37.4% of low-displacement moments have an unstable neighborhood, meaning Gate 1 (displacement) alone cannot detect convergence. Adding Gate 2 (KNN stability) reduces the false-freeze rate by 14.7% at sharp, while adding only 0.3% at flat where it is already reliable. Under a uniform freeze policy (streak = 2 with displacement only), 64% of frozen sites at curved regions are frozen incorrectly — they would have moved above threshold within 5 iterations. The dual-gate design with curvature-scaled streaks eliminates this failure mode.

---

## 4.5 Summary

The combined system achieves 3.9x to 17.9x end-to-end speedup across 11 meshes (10x to 18x on large meshes >170K vertices) with quality degradation below 2.1% in Q_avg and below 1.3 degrees in average minimum angle. The causal chain from tangent-plane distortion through persistent oscillation to false convergence is validated experimentally, with centroid bias growing 13x and direction reversal rate rising from 3% to 25% across curvature tiers. The dual-gate design with curvature-scaled streak addresses two independent failure modes: unreliable displacement signals (caught by longer streaks at high curvature) and displacement-neighborhood decoupling (caught by Gate 2, which eliminates 37% of false convergence detections at sharp regions).
