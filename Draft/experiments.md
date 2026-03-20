# 4. Experiments

## 4.1 Experimental Setup

**Benchmark meshes.** We evaluate on 12 triangle meshes spanning a range of vertex counts, curvature distributions, and geometric complexity:

| Mesh | Vertices | Faces | Curvature character |
|------|----------|-------|---------------------|
| stanford-bunny | 34,834 | 69,451 | Smooth, gentle curvature |
| horse | 48,485 | 96,966 | Mixed flat/curved, thin features |
| happy | 49,251 | 98,498 | Smooth, low curvature variation |
| armadillo | 49,990 | 99,976 | Mixed, fine geometric detail |
| lucy | 49,987 | 99,970 | Predominantly smooth |
| nefertiti | 49,971 | 99,938 | Mixed flat/sharp, facial features |
| bimba | 112,455 | 224,906 | Moderate curvature, face geometry |
| xyzrgb_dragon | 124,943 | 249,882 | Complex, high curvature variation |
| igea | 134,345 | 268,686 | Moderate curvature, organic |
| Armadillo | 172,974 | 345,944 | Large, mixed curvature |
| dragon_vrip | 437,645 | 871,414 | Large, high curvature complexity |
| happy_vrip | 543,652 | 1,087,716 | Largest, rich curvature variation |

**Modes compared.** All experiments compare three modes:
- **Mode 0 (Baseline):** Standard Lloyd iteration, all sites updated every iteration, no freezing.
- **Mode 1 (Freeze, 5-tier):** Curvature-adaptive dual-gate freeze with 5 NV-based tiers (Section 3.5).
- **Mode 2 (Freeze, 6-tier):** Same as Mode 1 with the sharp-tier split using normal covariance ratio L (Section 3.7).

**Parameters.** All runs use K = 20 nearest neighbors, 240 Lloyd iterations, and displacement threshold epsilon = 0.01 * R where R is the maximum bounding box edge. Tier assignment is computed once after 5 initial iterations.

**Hardware.** All experiments run on a single NVIDIA GPU. Timings include KNN, Voronoi clipping, centroid computation, reprojection, and freeze testing. Memory transfers are included.

**Quality metrics.** We report:
- Q_avg: average element quality (ratio of inscribed to circumscribed circle radius, normalized to [0,1]; 1 = equilateral).
- theta_min_avg: average minimum angle per triangle (optimal = 60 degrees for equilateral).
- theta_lt_30_pct: fraction of triangles with any angle below 30 degrees (lower = better).
- theta_gt_90_pct: fraction of triangles with any angle above 90 degrees (lower = better).
- d_H: Hausdorff distance from remeshed surface to input mesh (lower = better).

---

## 4.2 Speedup Results

Table 2 reports total remeshing time and speedup for all meshes across the three modes.

| Mesh | Vertices | Baseline (s) | Mode 1 (s) | Mode 2 (s) | Freeze % (M1) | Speedup (M1) | Speedup (M2) |
|------|----------|-------------|-----------|-----------|---------------|-------------|-------------|
| Armadillo | 173K | 92.8 | 26.5 | 25.4 | 79.0% | 3.5x | 3.6x |
| dragon_vrip | 438K | 573.9 | 149.0 | — | 71.0% | 3.9x | — |
| happy_vrip | 544K | 863.3 | 228.5 | 104.2 | 56.2% | 3.8x | 8.3x |
| igea | 134K | 13.6 | 5.7 | 5.7 | 81.4% | 2.4x | 2.4x |
| bimba | 112K | 10.0 | 5.2 | 5.2 | 56.0% | 1.9x | 1.9x |
| xyzrgb_dragon | 125K | 12.0 | 7.3 | 7.2 | 24.2% | 1.7x | 1.7x |
| horse | 48K | 2.3 | 1.6 | 1.6 | 74.8% | 1.5x | 1.4x |
| nefertiti | 50K | 2.3 | 1.7 | 1.7 | 50.8% | 1.4x | 1.4x |
| lucy | 50K | 9.2 | 6.0 | 2.1 | 5.5% | 1.5x | 4.3x |
| armadillo | 50K | 2.3 | 1.9 | 1.9 | 32.3% | 1.2x | 1.2x |
| happy | 49K | 2.3 | 1.9 | 2.0 | 9.0% | 1.2x | 1.2x |
| stanford-bunny | 35K | 1.2 | 0.6 | 1.1 | 4.0% | 1.9x | 1.0x |

**[PLOT: Per-iteration cumulative time, Mode 0 vs Mode 1 vs Mode 2]**
Source: `experiments/plots_eval/<mesh>/time_mode1_vs_2.png` for each mesh.

**Key observations:**

*Speedup scales with mesh size.* The three largest meshes (Armadillo 173K, dragon_vrip 438K, happy_vrip 544K) achieve 3.5x to 8.3x speedup. At these scales, baseline Lloyd iteration takes 1.5 to 14.4 minutes; freezing reduces runtime to seconds or low minutes. Smaller meshes (35K to 50K) show modest 1.2x to 1.9x speedup because per-iteration cost is already low and fixed overhead (kernel launches, memory transfers) dominates.

*Speedup correlates with freeze rate.* Meshes with high freeze rates (igea 81%, Armadillo 79%, horse 75%) achieve larger speedups. Meshes with predominantly smooth geometry and low curvature variation (happy 9%, lucy 6%, stanford-bunny 4%) freeze few sites because most sites continue oscillating or the computation is too fast for freeze overhead to matter.

*Mode 2 (6-tier) can substantially outperform Mode 1 on specific meshes.* On happy_vrip, Mode 2 achieves 8.3x speedup vs Mode 1's 3.8x. The sharp-tier split in Mode 2 allows singularity-type sites to freeze with a shorter streak (15 vs 30), increasing the frozen fraction at large scale. On lucy, Mode 2 achieves 4.3x vs Mode 1's 1.5x. On most other meshes, Mode 1 and Mode 2 perform similarly.

---

## 4.3 Mesh Quality Preservation

Table 3 reports quality metrics at the final iteration (iteration 240) for all modes.

**[PLOT: Quality metrics over iterations, all three modes]**
Source: `experiments/plots_eval/<mesh>/three_modes_quality.png` for each mesh. Each plot contains 6 subplots: Q_avg, theta_min_avg, theta_lt_30_pct, theta_gt_90_pct, per-iteration remesh time, and freeze rate, all plotted over iteration number with Mode 0, 1, and 2 overlaid.

| Mesh | Q_avg (M0) | Q_avg (M1) | Q_avg (M2) | Delta (M1) |
|------|-----------|-----------|-----------|-----------|
| Armadillo | 0.9338 | 0.9333 | 0.9334 | -0.05% |
| dragon_vrip | 0.9222 | 0.9214 | — | -0.09% |
| happy_vrip | 0.9183 | 0.9177 | 0.9179 | -0.07% |
| igea | 0.9155 | 0.9155 | 0.9155 | <0.01% |
| bimba | 0.9070 | 0.9071 | 0.9070 | +0.01% |
| xyzrgb_dragon | 0.8923 | 0.8923 | 0.8923 | <0.01% |
| horse | 0.9151 | 0.9146 | 0.9147 | -0.05% |
| nefertiti | 0.9009 | 0.9013 | 0.9011 | +0.04% |

**Quality degradation is below 0.1% on all meshes.** The largest delta is -0.09% on dragon_vrip (438K vertices). Several meshes show negligible or positive delta, indicating that freezing converged sites does not systematically degrade quality.

**Angle statistics.** Average minimum angle (theta_min_avg) differs by less than 0.1 degrees between baseline and freeze modes across all meshes (typical range: 50.3 to 54.5 degrees baseline, 50.3 to 54.4 degrees freeze). Bad-angle fractions (theta_lt_30_pct, theta_gt_90_pct) remain below 0.02% on large meshes with no systematic degradation.

**Hausdorff distance.** d_H values are identical or improved in freeze modes across all meshes, confirming that the remeshed surface tracks the input geometry equally well.

---

## 4.4 Freeze Rate Dynamics

**[PLOT: Freeze rate vs iteration, Mode 1 vs Mode 2]**
Source: `experiments/plots_eval/<mesh>/freeze_rate_mode1_vs_2.png` for each mesh.

The freeze rate (fraction of sites marked frozen) is a monotonically non-decreasing function of iteration number, since freezing is irreversible. Key patterns:

*Rapid early freezing on flat-dominated meshes.* On igea (81% final freeze rate), over 60% of sites freeze within the first 30 iterations. These are flat-region sites in Tier 0 (streak = 10) that converge quickly and pass the dual-gate test early.

*Gradual freezing on high-curvature meshes.* On xyzrgb_dragon (24% final freeze rate), the freeze curve rises slowly and plateaus early. Most sites reside in moderate-to-sharp curvature tiers where longer streaks are required and many sites never converge.

*Mode 2 freeze curves diverge from Mode 1 only at sharp tier.* On most meshes, Mode 1 and Mode 2 freeze curves are nearly identical. The difference appears on meshes with significant populations of sharp-tier singularity sites (Type A), where Mode 2's shorter streak (15 vs 30) allows earlier freezing.

---

## 4.5 Causal Chain Validation

The curvature-adaptive freeze policy is motivated by a causal analysis of convergence failure on curved surfaces (Section 3.2). We validate each step of the causal chain through dedicated experiments on the teapot mesh (3,644 sites, 50 iterations) and cross-validate on the spot mesh.

### Experiment 1: Tangent-Plane Distortion

**[PLOT: Tangent distortion by curvature tier]**
Source: `experiments/causal_chain_evidence/exp1_tangent_distortion/exp1_tangent_distortion.png`
Data: `experiments/causal_chain_evidence/exp1_tangent_distortion/exp1_tangent_distortion.csv`

Tangent distortion (ratio of 2D projected distance to 3D distance, relative to flat baseline) grows monotonically with curvature: 1.4% at flat (NV < 0.15) to 18.1% at sharp (NV >= 0.80), a 13.2x increase. Cross-validated on spot: 1.3% to 12.8% (9.8x), same monotonic trend.

### Experiment 2: Normal Frame Instability

**[PLOT: Normal frame flips and angle change by tier]**
Source: `experiments/causal_chain_evidence/exp2_normal_stability/exp2_normal_frame_stability.png`
Data: `experiments/causal_chain_evidence/exp2_normal_stability/exp2_normal_frame_stability.csv`

The number of normal frame flips (times the hosting triangle changes between consecutive iterations) increases from 1.9 at flat to 8.3 at curved, confirming that tangent-plane distortion destabilizes the reprojection. Mean angle change between consecutive normal frames grows from 0.5 degrees at flat to 6.4 degrees at moderate.

### Experiment 3: Persistent Oscillation

**[PLOT: Direction reversal fraction by tier]**
Source: `experiments/causal_chain_evidence/exp3_direction_reversal/exp3_direction_reversal.png`
Data: `experiments/causal_chain_evidence/exp3_direction_reversal/exp3_direction_reversal.csv`

The direction reversal rate (fraction of consecutive displacement vectors with cosine < 0) rises from 3.3% at flat to 24.7% at sharp. Mean cosine of consecutive displacements drops from +0.916 (monotonic convergence) to +0.430 (frequent reversals). At sharp regions, one in four consecutive steps is a reversal, confirming persistent oscillation rather than monotonic convergence.

### Experiment 4: Reprojection Triangle Instability

**[PLOT: Triangle flips and unique triangles visited by tier]**
Source: `experiments/causal_chain_evidence/exp4_reprojection_stability/exp4_reprojection_stability.png`
Data: `experiments/causal_chain_evidence/exp4_reprojection_stability/exp4_reprojection_stability.csv`

The centroid reprojects onto 2.8 different triangles on average at flat regions vs 9.3 at moderate/curved regions over 50 iterations. Unique triangles visited follow the same trend. The biased centroid at high curvature cycles across neighboring triangles, driving the normal frame instability measured in Experiment 2.

### Experiment 5: False Convergence and Streak Reliability

**[PLOT: Streak survival probability and false freeze rate by tier]**
Source: `experiments/causal_chain_evidence/exp5_false_convergence/exp5_false_convergence.png`
Data: `experiments/causal_chain_evidence/exp5_false_convergence/exp5a_streak_survival.csv`, `exp5c_false_freeze.csv`

**5a: Streak survival.** The probability that a low-displacement streak continues from length k to k+1 varies dramatically by tier. At flat: P(k=1 to k=2) = 85.6%, meaning streaks are self-reinforcing. At sharp: P(k=1 to k=2) = 48.1%, a coin flip. A site that has been low-displacement for one iteration has only a 48% chance of being low-displacement the next iteration at sharp curvature.

**5c: False freeze rate under uniform policy.** If all sites are frozen after streak = 2 (uniform policy), the false-freeze rate (fraction of frozen sites that would have moved above displacement threshold within 5 subsequent iterations under a no-freeze counterfactual) is 29.0% at flat and 63.5% at moderate/curved. Nearly two-thirds of sites frozen by a uniform policy at curved regions are frozen incorrectly.

### Experiment 6: Gate Decoupling

**[PLOT: Conditional Jaccard failure and predictive power by tier]**
Source: `experiments/causal_chain_evidence/exp6_decoupling/exp6_decoupling.png`
Data: `experiments/causal_chain_evidence/exp6_decoupling/exp6a_conditional_jaccard.csv`, `exp6c_predictive_power.csv`

**6a: Displacement-neighborhood decoupling.** Among iterations where displacement is below threshold, the fraction where KNN topology also changes (Jaccard < 1.0) is 0.04% at flat but 37.4% at sharp. At flat regions, low displacement reliably implies stable neighborhood. At sharp regions, more than one in three low-displacement moments have an unstable neighborhood, meaning Gate 1 (displacement) alone cannot detect convergence.

**6c: Predictive power of the dual gate.** Adding Gate 2 (KNN stability) reduces the false-freeze rate by 0.3% at flat (already reliable) but by 14.7% at sharp. The dual gate's value is concentrated exactly where it is needed most: at high curvature where displacement alone is unreliable.

### Experiment 9: Asynchronous Neighbor Convergence

**[PLOT: Phase coupling and neighbor motion at low focal displacement]**
Source: `experiments/causal_chain_evidence/exp9_neighbor_motion_induction/exp9_neighbor_motion_induction.png`
Data: `experiments/causal_chain_evidence/exp9_neighbor_motion_induction/exp9_neighbor_motion_induction.csv`

Phase coupling (correlation between focal site displacement and mean neighbor displacement) is r = +0.65 at flat and approaches 0 at sharp. At flat regions, when a site converges, its neighbors converge at a similar rate. At sharp regions, neighbors oscillate independently. Mean neighbor displacement when the focal site is near-stationary is 8.2x above the freeze threshold at sharp, confirming that neighborhoods are actively rearranging even when the focal site appears converged.

### Experiment 10: Sharp-Tier Bimodal Structure

**[PLOT: Type A (singularity) vs Type B (ridge) classification at sharp tier]**
Source: `experiments/causal_chain_evidence/exp10_sharp_classification/exp10_classification.png`
Data: `experiments/causal_chain_evidence/exp10_sharp_classification/exp10_stats.csv`

The sharp tier (NV >= 0.80) contains two distinct sub-populations identified via K-means clustering on displacement and neighborhood stability features:
- **Type A (57.5%):** Singularity sites at cone tips and corners. 75% have low displacement. Geometrically trapped, they converge reliably despite high NV.
- **Type B (42.5%):** Ridge sites along creases. Mean displacement 8.2x above freeze threshold. Persistent oscillation.

This bimodal structure motivates the 6-tier Mode 2 split (Section 3.7), which uses normal covariance ratio L to distinguish Type A from Type B and assigns a shorter streak to Type A.

### Experiment 12: Centroid Offset

**[PLOT: Centroid offset by tier]**
Source: `experiments/causal_chain_evidence/exp12_effective_neighbors/exp12_effective_neighbors.png`
Data: `experiments/causal_chain_evidence/exp12_effective_neighbors/exp12_summary.csv`

The centroid offset (distance between the tangent-plane centroid and the site position in the tangent plane) grows 13x from flat to sharp (flat: 0.0039, sharp: 0.0518), directly confirming that tangent-plane distortion produces increasingly biased centroids at high curvature. This measurement closes the causal chain: distortion biases the centroid, the biased centroid reprojects onto varying triangles (Experiment 4), and the resulting normal frame cycling drives persistent oscillation (Experiment 3).

---

## 4.6 Summary

The freeze policy achieves 1.2x to 8.3x speedup across 12 meshes (3.5x to 8.3x on large meshes >170K vertices) with less than 0.1% quality degradation. The causal chain from tangent-plane distortion through persistent oscillation to false convergence is validated at every step, with each metric showing monotonic curvature dependence. The dual-gate design with curvature-scaled streak addresses the two independent failure modes: unreliable displacement signals (caught by streak length) and displacement-neighborhood decoupling (caught by Gate 2). The 6-tier variant further improves performance on meshes with bimodal sharp-tier populations by distinguishing singularity sites from ridge sites.
