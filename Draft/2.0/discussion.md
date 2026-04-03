# 5. Discussion

## Topics

| Topic | Key point |
|-------|-----------|
| 5.1 Relationship to CVT energy | Freezing fixes stale KNN, breaking formal energy descent; periodic refresh bounds error; empirically quality preserved |
| 5.2 Tangent-plane quality gap | Q_avg ~0.917 vs Geogram ~0.929 is from tangent-plane approximation, not freezing; shared by all KNN-based CVT |
| 5.3 Why speedup scales with mesh size | KNN dominates cost; larger meshes have more flat sites that freeze early; fixed overheads dilute at small scale |
| 5.4 Freeze rate vs. speedup | Final freeze % doesn't fully predict speedup — when sites freeze matters; KNN backend contributes independently |
| 5.5 Limitations | Empirical thresholds; centroid not skipped; irreversible; uniform CVT only; large meshes lack full baselines |
| 5.6 Generality and future work | Drop-in for tangent-plane Lloyd; orthogonal to multi-facet clipping, power diagrams, faster solvers; CLOVER integration |

---

## 5.1 Relationship to CVT Energy

Lloyd iteration provably decreases the CVT energy functional at every step in the Euclidean setting [Du et al. 1999]: each site moves to the centroid of its Voronoi cell, the unique position minimizing the integral of squared distances within that cell. Our freeze policy intervenes in this energy-descent process. By holding a frozen site's KNN list fixed, we alter the Voronoi cell geometry from which the centroid is computed. As unfrozen neighbors continue to move, the stored neighbor list diverges from the true current KNN, and the centroid computed from stale neighbors is not guaranteed to be the true Voronoi centroid. The energy-descent property may therefore not hold for iterations involving frozen sites.

We do not claim that energy descent is preserved under freezing. The freeze policy is explicitly a heuristic — a spatially adaptive iteration scheduler designed to reduce computational cost, not a modification to the CVT energy functional. Its correctness is operational rather than theoretical: the dual-gate test with curvature-scaled streaks ensures that a site is frozen only when its position and neighborhood have stabilized, meaning the stale KNN is a close approximation of the true KNN at the time of freezing. Periodic refresh every $R = 50$ iterations bounds the accumulated divergence, and our experiments confirm that quality metrics track the unfrozen baseline within 0.1\% across all 11 meshes over 250 iterations.

A formal analysis of CVT energy behavior under partial site freezing — bounding the energy perturbation as a function of frozen fraction and refresh interval — would strengthen the theoretical foundation and is an interesting direction for future work.

## 5.2 The Tangent-Plane Quality Gap

Across all meshes, our method produces $Q_{\mathrm{avg}} \approx 0.917$ compared to Geogram's $Q_{\mathrm{avg}} \approx 0.929$, a gap of approximately 1.2\%. This gap is not caused by the freeze policy: Mode 1 (reusable KNN without freezing) achieves the same quality as Mode 2 (with freezing), confirming that the freeze mechanism does not degrade the output.

The gap is inherent to the tangent-plane approximation used by all KNN-based CVT methods, including RTF [Yao et al. 2023] and multi-facet clipping [Fei et al. 2025]. The tangent-plane construction projects 3D Voronoi geometry onto a local 2D plane, introducing distance distortion that grows with local curvature (Section 3.2, Mechanism 1). Geogram, by contrast, computes the exact restricted Voronoi diagram via Delaunay triangulation, avoiding this approximation entirely. The quality gap is the price paid for the simpler GPU-friendly tangent-plane framework — a trade-off shared by all methods in this family and independent of our freeze contribution.

Reducing this gap would require improving the per-site Voronoi approximation (e.g., via multi-facet clipping [Fei et al. 2025]) rather than modifying the freeze policy. Such improvements are orthogonal to our work and could be combined with our freeze mechanism.

## 5.3 Why Speedup Scales with Mesh Size

The freeze policy's speedup grows from 1.6–1.8$\times$ at 35–50K vertices to 3.7–4.2$\times$ at 1.2–2.8M vertices. This scaling behavior has two complementary causes.

First, KNN search accounts for a larger fraction of per-iteration cost on bigger meshes. On small meshes, fixed overheads — kernel launches, memory transfers, freeze bookkeeping, and the lightweight per-site operations (clipping, centroid, reprojection) — consume a significant share of iteration time. Even at 75\% freeze rate, the absolute time saved by skipping KNN is modest relative to these fixed costs. On large meshes, KNN dominates iteration time overwhelmingly, and eliminating 97\% of queries translates almost directly into proportional wall-clock reduction.

Second, the freeze rate itself increases with mesh size. At higher sampling density, flat regions of the surface — which constitute the majority of area on most meshes — are represented by proportionally more sites. These sites converge rapidly and pass the freeze test early. The fraction of sites near sharp features, which resist freezing, shrinks relative to the total count. The combination of higher freeze rates and larger KNN dominance produces the observed superlinear scaling of the freeze advantage.

## 5.4 Freeze Rate versus Speedup

The final freeze rate (fraction of sites frozen at the last iteration) does not perfectly predict total speedup for two reasons.

First, the *timing* of freezing matters. Speedup depends on the time-weighted integral of the frozen fraction across all iterations, not just the terminal value. A mesh where sites freeze gradually over 200 iterations accumulates less savings than one where most sites freeze within the first 30 iterations, even if both reach the same final freeze rate. This explains why Armadillo (79\% frozen, most sites freezing early) achieves 2.5$\times$ while happy\_vrip (56\% frozen at a different pace) achieves 2.4$\times$ — the relationship between final freeze rate and speedup is modulated by the freeze curve shape.

Second, the reusable KNN backend contributes speedup independently of freezing. The hub-grid bitonic structure with warm-starting is inherently faster than brute-force KNN even for the unfrozen queries, because it exploits spatial locality and seeds each query with the previous iteration's result. Mode 1 (no freeze) already achieves 5.5–7.9$\times$ over Mode 0 on small meshes, entirely from KNN structure improvements. The reported Mode 1 $\to$ Mode 2 speedups isolate the freeze contribution on top of this already-optimized baseline.

## 5.5 Limitations

**Empirically calibrated thresholds.** The NV tier boundaries $\{0.15, 0.35, 0.55, 0.80\}$ and streak lengths $\{10, 15, 20, 25, 30\}$ are chosen based on observed breakpoints in convergence instability metrics. No systematic sensitivity analysis or hyperparameter optimization has been performed. The thresholds produce consistent results across our 11-mesh benchmark, but generalization to meshes with unusual curvature distributions (e.g., predominantly sharp features with minimal flat area) is not guaranteed.

**Centroid and projection are not skipped.** Our policy only removes frozen sites from KNN queries. Voronoi clipping, centroid computation, and reprojection run for all sites at every iteration, because unfrozen neighbors may still reshape frozen sites' Voronoi cells. Speedup is therefore bounded by the fraction of iteration time spent in KNN. On implementations where clipping or projection is expensive relative to KNN, the freeze policy would yield smaller gains.

**Irreversible freezing.** Once frozen, a site remains frozen for all subsequent iterations. If a frozen site's neighborhood changes significantly due to distant cascading effects, the site cannot re-enter the active set. Periodic refresh mitigates this by updating stored KNN every $R$ iterations, but the frozen decision itself is never revisited. In practice, we observe no quality degradation from irreversibility, but a reversible variant with per-site unfreeze monitoring could be more robust for pathological cases.

**Uniform isotropic CVT only.** The current method assumes standard unweighted CVT, where all sites have equal weight and the target tessellation is uniform. In power diagrams (weighted Voronoi), sites carry weights that produce non-uniform cell sizes, and convergence dynamics near density transitions may differ. Extending the freeze policy to weighted CVT would require adapting the displacement threshold and KNN stability criteria to account for weight-induced asymmetries.

**Incomplete baselines at large scale.** Mode 0 and Geogram are not run on the three large meshes (1.2–2.8M) due to prohibitive runtime. While extrapolation from observed scaling trends strongly suggests that Mode 2 is faster, the absence of direct measurements at this scale is a limitation. The large-mesh evaluation is restricted to the Mode 1 vs. Mode 2 comparison.

## 5.6 Generality and Future Work

The freeze policy is a drop-in modification to any tangent-plane Lloyd iteration loop. It requires no changes to the per-site update rule, no additional mesh queries beyond what the Lloyd loop already computes, and no global synchronization. The curvature tiering is computed once from existing KNN normals; the per-iteration freeze test adds two comparisons and one counter increment per unfrozen site.

The freeze mechanism is orthogonal to several complementary acceleration strategies. Curvature-adaptive multi-facet clipping [Fei et al. 2025] improves per-site Voronoi accuracy; our freeze policy could be applied on top, using multi-facet clipping for active sites while skipping frozen sites entirely. Power diagram extensions [De Goes et al. 2012] generalize CVT to non-uniform tessellations; the freeze policy's dual-gate structure could be adapted to weight-aware convergence criteria. Faster-converging solvers such as L-BFGS or Anderson acceleration reduce the number of iterations required; fewer iterations with freeze-aware KNN would compound both sources of speedup.

Several concrete extensions are suggested by our experimental findings. A systematic sensitivity analysis of tier boundaries and streak lengths would quantify robustness and potentially yield tighter thresholds. For sites whose entire $K$-neighborhood is frozen, clipping and centroid computation could also be skipped, further reducing per-iteration cost in the late-convergence phase when freeze rates exceed 90\%. Full integration with CLOVER's spatio-graph KNN [Kamel et al. 2025] — whose hub structure natively supports partial rebuilds and non-uniform distributions — could improve KNN efficiency at larger scales where our current uniform-grid hub structure becomes less optimal. Finally, the bimodal convergence behavior observed within the sharp tier (Section 3.2) — where singularity sites converge reliably but ridge sites oscillate — suggests that incorporating curvature anisotropy as a secondary descriptor could improve freeze rates on geometrically complex meshes without additional false-freeze risk.
