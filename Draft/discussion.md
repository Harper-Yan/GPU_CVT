# 5. Discussion

## 5.1 Why Speedup Scales with Mesh Size

The freeze policy's primary mechanism is skipping KNN queries for frozen sites. KNN search dominates per-iteration GPU cost and scales with the number of active query sites. The remaining per-iteration work, Voronoi clipping, centroid computation, and reprojection, runs for all sites (frozen and unfrozen) every iteration. Speedup is therefore bounded by the fraction of iteration time spent in KNN.

On small meshes (35K to 50K vertices), KNN completes quickly even without freezing, and fixed overheads (kernel launches, memory transfers, freeze bookkeeping) consume a larger share of iteration time. Even with 75% freeze rate (horse), speedup is only 1.5x because the absolute time saved per iteration is small relative to fixed costs.

On large meshes (170K to 544K vertices), KNN dominates iteration time. At 173K vertices (Armadillo), baseline per-iteration KNN takes hundreds of milliseconds; skipping 79% of queries produces substantial absolute savings that compound over 240 iterations, yielding 3.5x speedup. At 544K vertices (happy_vrip), the effect is amplified further to 3.8x (Mode 1) and 8.3x (Mode 2).

The practical implication is that the freeze policy is most valuable precisely in the regime where acceleration is most needed: large meshes where baseline GPU CVT becomes impractical.

## 5.2 Freeze Rate vs Speedup: Why the Correlation Is Imperfect

The final freeze rate (fraction of sites frozen at the last iteration) does not perfectly predict speedup for two reasons.

**Timing of freezing matters.** Freeze rate is reported at the final iteration, but speedup depends on the time-weighted integral of frozen fraction across all iterations. A mesh where 64% of sites freeze late (e.g., stanford-bunny: freeze grows slowly from 0% to 64% between iterations 0 and 90) skips little work during most of the run and achieves only 1.0x speedup. A mesh where 79% of sites freeze early (Armadillo: most sites freeze within the first 30 iterations) accumulates savings across the full run.

**Algorithmic differences between modes.** In our implementation, freeze mode uses a bitonic-sort KNN path while baseline uses a brute-force path for some queries. On meshes where brute-force is the bottleneck (e.g., xyzrgb_dragon), freeze mode is faster even at low freeze rates because of the algorithm switch, not the freezing itself. A fairer comparison would use the same KNN backend in both modes; the reported speedups therefore include both the freeze benefit and any backend difference.

## 5.3 The Bimodal Structure of the Sharp Tier and the 6-Tier Variant

The sharp curvature tier (NV >= 0.80) contains two qualitatively different sub-populations (Experiment 10):

- **Type A (singularity, 57.5% of sharp sites):** Sites at cone tips, corners, and topological features. Despite high normal variation, these sites are geometrically trapped: their centroids cannot escape the local geometric pocket, producing low displacement and stable neighborhoods. 75% of their iterations are below the displacement threshold. They behave like flat-tier sites in terms of convergence dynamics.

- **Type B (ridge, 42.5%):** Sites along edges and creases. These oscillate persistently with mean displacement 8.2x above the freeze threshold and 25% direction-reversal rate. Only 6% of their iterations fall below the displacement threshold.

The 5-tier policy (Mode 1, Section 3.3) treats both types identically with streak = 30. This is conservative: Type A sites would safely freeze with a much shorter streak, but are forced to wait 30 consecutive passes.

**The 6-tier variant (Mode 2)** introduces a second curvature descriptor to distinguish the two sub-populations: the normal covariance ratio L. For each site, we compute the 3x3 covariance matrix of the normals of its K neighbors, extract the eigenvalues lambda_1 >= lambda_2 >= lambda_3, and define:

$$L(s_i) = \frac{\lambda_1 - \lambda_2}{\lambda_1}$$

L near 0 indicates normals spread uniformly (isotropic curvature, typical of singularities where normals fan out in all directions). L near 1 indicates normals spread along a dominant direction (anisotropic curvature, typical of ridges where normals vary across the ridge but are consistent along it).

Mode 2 splits Tier 4 based on L:
- **Tier 4 (sharp, ridge):** NV >= 0.80 and L <= 0.80. Streak = 30. These are the oscillating Type B sites that require the strictest convergence criteria.
- **Tier 5 (sharp, singularity):** NV >= 0.80 and L > 0.80. Streak = 15. These are the geometrically trapped Type A sites that can safely freeze earlier.

L is computed from the same KNN normals used for NV, adding negligible cost (a 3x3 covariance matrix and eigenvalue extraction per site, performed once during preprocessing).

**Impact on performance.** Mode 2 substantially outperforms Mode 1 on meshes where the sharp tier contains many singularity-type sites. On happy_vrip, Mode 2 achieves 8.3x speedup vs Mode 1's 3.8x: the earlier freezing of Type A sites increases the frozen fraction during the critical middle iterations where KNN cost is highest. On lucy, Mode 2 achieves 4.3x vs Mode 1's 1.5x. On meshes where the sharp tier is small or homogeneous (most other benchmarks), Mode 1 and Mode 2 perform identically, confirming that the split adds no overhead when it is not needed.

## 5.4 What Frozen Sites Track and What They Miss

A frozen site's KNN list is held fixed, but its centroid and projection continue updating because unfrozen neighbors may still be moving. This design means:

- **What frozen sites track:** Changes in Voronoi cell geometry caused by neighbor movement. As unfrozen neighbors shift, the Voronoi cell of a frozen site reshapes, and the centroid adjusts accordingly. When all neighbors eventually stabilize, the frozen site's position reflects the final neighborhood geometry.

- **What frozen sites miss:** Changes in the KNN set itself. If a distant site moves closer and should enter the frozen site's K-nearest set, or if a current neighbor moves away and should leave, the frozen site's neighbor list does not update. The dual-gate requirement (especially Gate 2, KNN stability) mitigates this: a site is only frozen when its KNN set has been stable for multiple consecutive iterations, so the frozen neighbor list is a good approximation for the remaining iterations.

Experiment 8 measures neighbor displacement after freezing: mean neighbor displacement is 0.006 to 0.009 across all tiers, with no strong curvature gradient. Neighbors of frozen sites continue evolving, but the continued centroid computation allows frozen sites to passively adapt to these changes.

## 5.5 The Role of Irreversible Freezing

Freezing is irreversible in our policy: once frozen, a site remains frozen for all remaining iterations. An alternative design would allow unfreezing (releasing a site back to active status) if its neighborhood changes significantly after freezing. We chose irreversible freezing for three reasons:

1. **Simplicity.** Unfreezing requires monitoring frozen sites for KNN changes, which partially negates the computational savings of skipping KNN queries.

2. **The dual gate already filters aggressively.** The curvature-scaled streak ensures that a site is only frozen when there is strong evidence of genuine convergence. At the sharp tier, a site must pass both gates for 30 consecutive iterations. The probability of a false positive surviving this filter is low.

3. **Empirical quality preservation.** Across all 12 benchmark meshes, irreversible freezing produces less than 0.1% quality degradation. The cost of occasional false freezes is empirically negligible.

A potential failure mode is that a frozen site's neighborhood could undergo a large-scale rearrangement late in the iteration sequence (e.g., a cascade of unfrozen sites converging to new positions). In practice, we do not observe this: the freeze rate curve plateaus early (most freezing occurs within the first 30 to 50 iterations), and by the time late iterations run, the remaining unfrozen sites produce only small displacements.

## 5.6 Relationship to CVT Energy

Lloyd iteration is not merely an iterative heuristic: it provably decreases the CVT energy functional at every step in the Euclidean setting [Du et al. 1999]. Each site moves to the centroid of its Voronoi cell, which is the unique position minimizing the integral of squared distances within that cell. The energy-descent property is what gives CVT its theoretical grounding and distinguishes it from ad hoc smoothing methods.

Our freeze policy intervenes in this energy-descent process. By holding a site's KNN list fixed, we alter the Voronoi cell geometry that determines the centroid: the frozen neighbor list may not reflect the true K nearest neighbors as other sites continue moving. The centroid computed from a stale neighbor list is not guaranteed to be the true Voronoi centroid, and the energy-descent property may not hold for iterations involving frozen sites.

We do not attempt to prove that the energy-descent property is preserved under freezing. The freeze policy is explicitly a heuristic, a spatially adaptive iteration scheduler designed to reduce computational cost, not an energy-minimization algorithm. Its correctness claim is operational, not theoretical: empirically, frozen sites produce less than 0.1% quality degradation across all benchmark meshes, and Hausdorff distances are identical or improved. The dual-gate design with curvature-scaled streak is calibrated to freeze sites only when there is strong empirical evidence that the site and its neighborhood have stabilized, making the stale-KNN approximation a good one in practice.

A formal analysis of CVT energy behavior under partial site freezing, bounding the energy increase caused by stale neighbor lists as a function of freeze timing and neighborhood stability, would strengthen the theoretical foundation. We leave this as future work and note that the empirical quality results provide a practical substitute: if the final mesh quality is indistinguishable from the unfrozen baseline, the energy trajectory, whatever its intermediate behavior, has reached an equally good final state.

## 5.7 Limitations

**False freeze rate under counterfactual analysis.** Experiment 8 measures false-freeze rates using a no-freeze counterfactual (running the same initial configuration without any freezing). Because the freeze and no-freeze trajectories diverge immediately (50% of sites freeze by iteration 8), the counterfactual displacement overestimates the true false-freeze rate. A perturbation experiment (releasing individual frozen sites while keeping others frozen) would provide a more accurate estimate but has not been performed.

**Tier thresholds are empirically calibrated, not optimized.** The NV boundaries [0.15, 0.35, 0.55, 0.80] and streak lengths [10, 15, 20, 25, 30] are chosen based on observed breakpoints in instability metrics. No sensitivity analysis or ablation study has been performed. Robustness to perturbation (e.g., shifting boundaries by +/- 0.05 or streak lengths by +/- 2) is not formally tested. The thresholds are consistent across the two validation meshes (teapot and spot), but generalization to meshes with very different curvature distributions is not guaranteed.

**Sites that never freeze.** On our benchmark meshes, 19% to 96% of sites never freeze (depending on mesh geometry). These are sites in moderate-to-sharp curvature tiers whose displacement or KNN set never stabilizes for the required streak length. Some of these sites are genuinely oscillating due to the tangent-plane distortion mechanism. Others may be converging slowly and would freeze given more iterations. A 200-iteration extended run on persistently unfrozen sites would distinguish the two cases but has not been performed.

**Scope of evaluation.** The causal chain experiments (Experiments 1 through 12) are performed on the teapot (3,644 sites) and spot meshes. The end-to-end speedup and quality evaluation covers 12 meshes. The causal analysis has not been repeated on all 12 benchmark meshes, though the monotonic curvature dependence of all metrics on both teapot and spot suggests generality.

**Assumption of uniform isotropic CVT.** Our method and analysis assume standard (unweighted) CVT, where all sites have equal weight and the target tessellation is uniform. In power diagrams (weighted Voronoi diagrams), sites carry weights that produce non-uniform cell sizes, adapting triangle density to local curvature or a prescribed sizing field. Under non-uniform tessellations, convergence dynamics may differ from the isotropic case in ways our curvature analysis does not capture. Sites near density transitions, where the target cell size changes rapidly, may experience additional oscillation as the power diagram adjusts to competing size constraints, requiring different freeze criteria. Sites in high-density regions have shorter inter-site distances and may converge faster, potentially allowing more aggressive freezing. Extending the curvature-convergence analysis and the dual-gate freeze policy to weighted CVT and power diagrams is a natural next step but would require re-characterizing the convergence landscape under non-uniform cell targets.

**Centroid and projection are not skipped.** Unlike a full "sleep" policy that would skip all computation for frozen sites, our policy only skips KNN queries. Voronoi clipping, centroid computation, and reprojection run for all sites every iteration. The speedup is therefore bounded by the fraction of iteration time spent in KNN. On implementations where KNN is a smaller fraction of per-iteration cost (e.g., restricted Voronoi diagrams with expensive clipping), the freeze policy would yield smaller speedup. The policy is most effective for tangent-plane Lloyd iteration where KNN dominates.
