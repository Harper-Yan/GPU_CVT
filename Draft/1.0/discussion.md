# 5. Discussion

## 5.1 Why Speedup Scales with Mesh Size

The freeze policy's primary mechanism is skipping KNN queries for frozen sites. KNN search dominates per-iteration GPU cost and scales with the number of active query sites. The remaining per-iteration work, Voronoi clipping, centroid computation, and reprojection, runs for all sites (frozen and unfrozen) every iteration. Speedup is therefore bounded by the fraction of iteration time spent in KNN.

On small meshes (35K to 50K vertices), KNN completes quickly even without freezing, and fixed overheads (kernel launches, memory transfers, freeze bookkeeping) consume a larger share of iteration time. Even with 75% freeze rate (horse), speedup is only 6.4x because the absolute time saved per iteration is small relative to fixed costs.

On large meshes (170K to 544K vertices), KNN dominates iteration time. At 173K vertices (Armadillo), baseline per-iteration brute-force KNN takes over 700ms; switching to the bitonic-sort backend and skipping 79% of queries reduces this to under 10ms, compounding over 250 iterations to yield 17.9x end-to-end speedup. At 544K vertices (happy\_vrip), the effect produces 10.9x speedup. The improvement comes from two complementary sources: the bitonic-sort backend is inherently faster for the sparse, skewed query patterns produced by freezing (where most sites are masked out), and the freeze policy itself eliminates the queries entirely for converged sites.

The practical implication is that the freeze policy is most valuable precisely in the regime where acceleration is most needed: large meshes where baseline GPU CVT becomes impractical.

## 5.2 Freeze Rate vs Speedup: Why the Correlation Is Imperfect

The final freeze rate (fraction of sites frozen at the last iteration) does not perfectly predict speedup for two reasons.

**Timing of freezing matters.** Freeze rate is reported at the final iteration, but speedup depends on the time-weighted integral of frozen fraction across all iterations. A mesh where 64% of sites freeze late (e.g., stanford-bunny: freeze grows slowly from 0% to 64% between iterations 0 and 90) skips little work during most of the run and achieves only 1.0x speedup. A mesh where 79% of sites freeze early (Armadillo: most sites freeze within the first 30 iterations) accumulates savings across the full run.

**The KNN backend contributes to speedup.** Our method uses a bitonic-sort KNN backend with hub-based pruning that natively supports frozen-site masking, while the baseline uses a brute-force KNN kernel. The bitonic-sort backend is inherently faster for the sparse query patterns produced by freezing and is a deliberate part of the system design: the freeze policy produces a progressively sparser set of active queries, and the backend is optimized to exploit this sparsity. The reported speedups therefore reflect the full end-to-end improvement of the combined system. On meshes where the freeze rate is low (e.g., xyzrgb\_dragon at 24\%), the backend difference still contributes modest speedup even without substantial freezing.

## 5.3 Curvature Anisotropy as a Secondary Descriptor

Normal variation (NV) captures the magnitude of local curvature but not its directional structure. K-means clustering on displacement and neighborhood stability features within the sharp tier (NV >= 0.80) reveals two sub-populations with opposite convergence behavior (Figure fig_sharp_bimodal): singularity sites (57.5%) that are geometrically trapped and converge reliably (75% of iterations below the displacement threshold), and ridge sites (42.5%) that oscillate persistently (mean displacement 8.2x above threshold). This bimodal split is unique to the sharp tier — lower tiers show unimodal convergence statistics because NV alone sufficiently predicts behavior.

The normal covariance ratio $L = (\lambda_1 - \lambda_2) / \lambda_1$, computed from the eigenvalues of the KNN normal covariance matrix, is a natural candidate to distinguish the two types: $L$ near 0 indicates isotropic curvature (singularities), while $L$ near 1 indicates anisotropic curvature (ridges). Because this distinction only affects the sharp tier — a small fraction of total sites on most meshes — we retain the simpler 5-tier policy in the current work. Incorporating $L$ as a secondary descriptor to allow shorter streaks for singularity sites is a targeted future extension that could improve freeze rates on meshes with large sharp-tier populations.

## 5.4 Relationship to CVT Energy

Lloyd iteration provably decreases the CVT energy functional at every step in the Euclidean setting [Du et al. 1999]. Each site moves to the centroid of its Voronoi cell, which is the unique position minimizing the integral of squared distances within that cell.

Our freeze policy intervenes in this energy-descent process. By holding a site's KNN list fixed, we alter the Voronoi cell geometry that determines the centroid: the frozen neighbor list may not reflect the true K nearest neighbors as other sites continue moving. The centroid computed from a stale neighbor list is not guaranteed to be the true Voronoi centroid, and the energy-descent property may not hold for iterations involving frozen sites.

We do not attempt to prove that the energy-descent property is preserved under freezing. The freeze policy is explicitly a heuristic — a spatially adaptive iteration scheduler designed to reduce computational cost, not an energy-minimization algorithm. Its correctness claim is operational, not theoretical: empirically, frozen sites produce small quality degradation across all benchmark meshes, and Hausdorff distances are identical or improved. A formal analysis of CVT energy behavior under partial site freezing would strengthen the theoretical foundation and is left as future work.

## 5.5 Limitations

**Tier thresholds are empirically calibrated, not optimized.** The NV boundaries [0.15, 0.35, 0.55, 0.80] and streak lengths [10, 15, 20, 25, 30] are chosen based on observed breakpoints in instability metrics. No sensitivity analysis or ablation study has been performed. The thresholds are consistent across the two validation meshes (teapot and spot), but generalization to meshes with very different curvature distributions is not guaranteed.

**Sites that never freeze.** On our benchmark meshes, 19% to 96% of sites never freeze (depending on mesh geometry). These are sites in moderate-to-sharp curvature tiers whose displacement or KNN set never stabilizes for the required streak length. Some are genuinely oscillating; others may be converging slowly and would freeze given more iterations.

**Scope of evaluation.** The causal chain experiments are performed on the teapot (3,644 sites) and spot meshes. The end-to-end evaluation covers 11 meshes. The causal analysis has not been repeated on all benchmark meshes, though the monotonic curvature dependence on both teapot and spot suggests generality.

**Assumption of uniform isotropic CVT.** Our method assumes standard (unweighted) CVT, where all sites have equal weight and the target tessellation is uniform. In power diagrams (weighted Voronoi), sites carry weights that produce non-uniform cell sizes. Under non-uniform tessellations, convergence dynamics may differ: sites near density transitions may experience additional oscillation, while sites in high-density regions may converge faster. Extending the freeze policy to weighted CVT is a natural next step.

**Centroid and projection are not skipped.** Unlike a full "sleep" policy that would skip all computation for frozen sites, our policy only skips KNN queries. Voronoi clipping, centroid computation, and reprojection run for all sites every iteration. The speedup is therefore bounded by the fraction of iteration time spent in KNN. On implementations where KNN is a smaller fraction of per-iteration cost (e.g., restricted Voronoi diagrams with expensive clipping), the freeze policy would yield smaller speedup.
