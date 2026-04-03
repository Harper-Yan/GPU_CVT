# 4. Experiments

## Section Overview

| Section | Main point | Data source |
|---------|-----------|-------------|
| 4.1 Setup | Hardware, 11 meshes (35K–2.8M), 3 modes, Geogram 250L baseline, quality metrics | objs/, baselines/ |
| 4.2 Ablation: KNN reuse + freeze | Mode 1 gives 5–8× over Mode 0; Mode 2 adds 1.6–4.2× on top — synergistic, advantage grows with scale | eval CSVs, three_modes_quality.png |
| 4.3 Freeze behavior & quality | Freeze rate rises to 50–80% (medium) and 97%+ (large); quality tracks unfrozen baseline within negligible margin | three_modes_quality.png |
| 4.4 vs. Geogram (CPU baseline) | Mode 2 is 3–10× faster than Geogram 250L at all tested scales (35K–544K) while matching quality | baseline_comparison.png |
| 4.5 Scalability | M1→M2 speedup grows from 1.7× at 35K to 4.2× at 2.8M; at million-vertex scale, freeze reduces 16.7 min to 4.0 min | eval CSVs |

## Timing Data

| Mesh | Vertices | Mode 0 (ms) | Mode 1 (ms) | Mode 2 (ms) | M0→M1 | M1→M2 | Geogram 250L (ms) | vs Geogram (M2) |
|------|----------|-------------|-------------|-------------|-------|-------|-------------------|-----------------|
| stanford-bunny | 34,834 | 12,292 | 2,255 | 1,348 | 5.5× | 1.7× | 10,737 | 8.0× |
| horse | 48,485 | 25,403 | 3,212 | 1,748 | 7.9× | 1.8× | 18,209 | 10.4× |
| nefertiti | 49,971 | 23,300 | 3,347 | 2,151 | 7.0× | 1.6× | 18,570 | 8.6× |
| bimba | 112,455 | — | 8,004 | 4,268 | — | 1.9× | 36,091 | 8.5× |
| igea | 134,345 | — | 9,897 | 4,110 | — | 2.4× | 22,588 | 5.5× |
| Armadillo | 172,974 | — | 14,000 | 5,601 | — | 2.5× | 17,593 | 3.1× |
| dragon_vrip | 437,645 | — | 48,391 | 17,849 | — | 2.7× | 100,203 | 5.6× |
| happy_vrip | 543,652 | — | 66,162 | 27,141 | — | 2.4× | 117,299 | 4.3× |
| Lucy | 1,264,847 | — | 237,889 | 64,998 | — | 3.7× | — | — |
| Trumpet | 1,225,855 | — | 220,257 | 52,641 | — | 4.2× | — | — |
| Samothrace | 2,836,106 | — | 1,003,270 | 241,466 | — | 4.2× | — | — |

---

## 4.1 Experimental Setup

We evaluate our method on 11 triangle meshes spanning three orders of magnitude in vertex count: three small meshes (34K–50K vertices), five medium meshes (112K–544K), and three large meshes (1.2M–2.8M). The meshes are selected to cover a range of geometric complexity, from smooth surfaces (Stanford Bunny) through moderate detail (Igea, Bimba) to fine features and large flat regions (Dragon, Happy Buddha, Lucy, Samothrace).

All experiments run on [GPU model] with [RAM] and [CPU]. We implement three evaluation modes. Mode 0 is a brute-force RTF baseline that recomputes all-pairs KNN for every site at every iteration, serving as the unoptimized reference. Mode 1 employs our reusable bitonic KNN structure (Section 3.4) with hub-grid spatial indexing, warm-starting, and mesh KNN caching, but without freezing. Mode 2 adds the curvature-adaptive freeze policy (Section 3.3) on top of Mode 1. All modes use $K = 32$ neighbors, run for 250 Lloyd iterations, and use a displacement threshold of $\epsilon = 0.01 \times R$ where $R$ is the bounding box diagonal. The periodic refresh interval is $R = 50$ iterations.

Mode 0 is run only on small meshes, as its brute-force KNN makes it prohibitively expensive at larger scales. Modes 1 and 2 are run on all 11 meshes.

**Baseline selection.** We consider three CPU remeshing baselines: Geogram RVD-CVT [Lévy and Liu 2010], CGAL isotropic remeshing [Botsch and Kobbelt 2004], and ACVD [Valette et al. 2008]. Among these, Geogram is the most relevant comparison: it implements exact restricted Voronoi diagram construction via Delaunay triangulation, the CPU-side state of the art for CVT-based remeshing, and consistently produces the highest element quality ($Q_{\mathrm{avg}} \approx 0.929$). CGAL isotropic remeshing uses greedy local operations (split/collapse/flip/smooth) rather than CVT optimization, producing lower quality ($Q_{\mathrm{avg}} \approx 0.85$–$0.90$) at 2–5$\times$ slower runtime than Geogram. ACVD approximates CVT via cluster-based decimation but likewise produces lower quality ($Q_{\mathrm{avg}} \approx 0.88$) and is the slowest of the three (3–7$\times$ slower than Geogram). Since both CGAL and ACVD are strictly dominated by Geogram in both quality and speed, we use Geogram as our primary CPU baseline and report CGAL and ACVD results only for completeness on the small meshes where all methods were evaluated.

Geogram is configured with 250 Lloyd iterations and 0 Newton iterations, with $n_{\mathrm{samples}} = n_{\mathrm{vertices}}$ to match our site count and iteration count for a fair comparison. Geogram is run on the 8 small and medium meshes (up to 544K vertices). At larger scales, Mode 0 and Geogram are not run: extrapolating Geogram's observed $O(n \log n)$ scaling from 544K (117s) to 2.8M yields an estimated runtime of 700–800s, far exceeding Mode 2's 241s and not justifying the computational cost.

We report four quality metrics: average element quality $Q_{\mathrm{avg}}$ (where 1.0 denotes a perfect equilateral triangle), average minimum angle $\theta_{\min}^{\mathrm{avg}}$, and the percentages of angles below 30° and above 90°.

## 4.2 Ablation: KNN Reuse and Freeze Contributions

To isolate the contribution of each component, we compare Mode 0 (brute-force), Mode 1 (reusable KNN only), and Mode 2 (reusable KNN + freeze) across all meshes where each mode is available.

**Reusable KNN (Mode 0 $\to$ Mode 1).** On the three small meshes where Mode 0 is available, replacing brute-force KNN with our hub-grid bitonic structure yields 5.5–7.9$\times$ speedup. The reusable spatial index avoids the $O(n^2)$ all-pairs scan at every iteration, providing substantial acceleration even without any freezing. The warm-start mechanism further reduces query cost in later iterations when site displacements are small.

**Freeze policy (Mode 1 $\to$ Mode 2).** Adding the freeze policy on top of KNN reuse provides an additional speedup that grows consistently with mesh size:

- Small meshes (35–50K): 1.6–1.8$\times$, with freeze rates of 50–75\%.
- Medium meshes (112–544K): 1.9–2.7$\times$, with freeze rates of 56–81\%.
- Large meshes (1.2–2.8M): 3.7–4.2$\times$, with freeze rates exceeding 97\%.

The two components are synergistic: the freeze policy progressively removes converged sites from the KNN query set, and the reusable KNN structure's frozen-mask compaction ensures that this sparsity translates directly into proportional wall-clock reduction. Without the compaction mechanism, frozen sites would still consume warp occupancy; without the freeze policy, the compaction has nothing to skip.

Quality is preserved across all three modes. Figure [three_modes_quality] shows $Q_{\mathrm{avg}}$, $\theta_{\min}$, and angle distribution metrics tracking closely between Mode 1 and Mode 2 throughout iteration on all 11 meshes, confirming that the freeze policy does not degrade mesh quality. The cumulative time plots (Figure [time_mode1_vs_2]) show the timing curves of Mode 1 and Mode 2 diverging progressively as the freeze rate grows, with the gap most dramatic on large meshes where Mode 2 completes in a fraction of Mode 1's time.

## 4.3 Freeze Behavior and Quality Preservation

We examine the freeze policy's behavior in detail to verify that the curvature-adaptive design achieves high freeze rates without compromising output quality.

**Progressive freezing.** The freeze rate rises monotonically over the course of iteration. On medium meshes, sites in flat regions (Tier 0–1) begin freezing around iteration 30–50, once the streak requirement of 10–15 consecutive passes is met. Sites in moderate-curvature regions (Tier 2–3) follow at iteration 60–100. Sites near sharp features (Tier 4) freeze late or remain active throughout, as their longer streak requirements (25–30 iterations) and stricter KNN stability checks reflect the genuine difficulty of convergence in these regions. The final freeze rate on medium meshes ranges from 56\% (happy\_vrip, which has extensive high-curvature regions) to 81\% (Igea, which is predominantly smooth).

On large meshes, the freeze rate exceeds 97\% on all three test cases. At high sampling density, the surface is dominated by flat regions that gain proportionally more sites, all of which converge early and pass the freeze test quickly. The small fraction of sites near sharp features — typically under 3\% of the total at million-vertex scale — remains active, preserving detail fidelity.

**Quality preservation.** Across all 11 meshes, the quality metrics of Mode 2 (freeze) track those of Mode 1 (no freeze) within negligible margin throughout iteration. The $Q_{\mathrm{avg}}$ curves overlap to within 0.1\%, and the angle distribution metrics (\%$< 30°$, \%$> 90°$) show no systematic divergence. This confirms that the dual-gate test with curvature-scaled streaks successfully avoids false freezing: only sites whose position and neighborhood have genuinely stabilized are removed from KNN queries.

**Per-iteration time reduction.** As the freeze rate grows, Mode 2's per-iteration time drops correspondingly. On medium meshes, Mode 2 is initially comparable to Mode 1 but becomes 2–3$\times$ faster per iteration by iteration 50–100. On large meshes, the per-iteration speedup reaches 4–5$\times$ in the later phase when over 97\% of sites are frozen and only 3\% of the KNN workload remains active.

## 4.4 Comparison with CPU Baselines

We compare against Geogram's RVD-CVT, the state-of-the-art CPU implementation of surface CVT based on exact restricted Voronoi diagrams with Delaunay triangulation. For a fair comparison, Geogram is configured with 250 Lloyd iterations (matching our iteration count) and $n_{\mathrm{samples}} = n_{\mathrm{vertices}}$ (matching our site count).

On small meshes (35–50K vertices), Mode 2 is 8.0–10.4$\times$ faster than Geogram while achieving comparable quality ($Q_{\mathrm{avg}} \approx 0.917$ vs. Geogram's 0.929). The quality gap of approximately 1.2\% is inherent to the tangent-plane approximation used by all KNN-based CVT methods versus Geogram's exact RVD computation; it is not caused by freezing, as Mode 1 (without freeze) produces the same quality.

On medium meshes (112–544K), Mode 2 maintains a 3.1–8.5$\times$ speedup over Geogram. At the largest medium mesh (happy\_vrip, 544K), Geogram takes 117 seconds while Mode 2 completes in 27 seconds, a 4.3$\times$ advantage. The freeze policy is essential at this scale: Mode 1 alone achieves only 1.8$\times$ over Geogram, while Mode 2's freeze-driven compaction provides the additional factor needed to maintain a clear advantage.

On large meshes (1.2–2.8M vertices), Geogram is not run. Extrapolating from its observed $O(n \log n)$ per-iteration scaling — 117 seconds at 544K vertices — we estimate a runtime on the order of 700–800 seconds for 2.8M vertices (Samothrace). Mode 2 completes the same mesh in 241 seconds, approximately 3$\times$ faster than the extrapolated baseline. Mode 0 (brute-force RTF) is also not run at this scale; its $O(n^2)$ KNN cost would require hours. At million-vertex scale, only Mode 2 with freeze-aware KNN reuse produces results within a practical time budget.

## 4.5 Scalability

The central result of our evaluation is that the freeze advantage grows consistently with mesh size. The Mode 1 $\to$ Mode 2 speedup increases from 1.6–1.8$\times$ at 35–50K vertices, through 1.9–2.7$\times$ at 112–544K, to 3.7–4.2$\times$ at 1.2–2.8M. This trend is driven by a simple geometric argument: at higher sampling density, flat regions of the surface — which dominate most meshes — are represented by proportionally more sites. These flat-region sites converge rapidly and pass the freeze test early, while the fraction of sites near sharp features shrinks relative to the total count. The result is that the freeze rate grows from 50–75\% at small scale to over 97\% at million-vertex scale.

The practical consequences are significant. On the Samothrace mesh (2.84M vertices), Mode 1 without freezing takes 16.7 minutes. Mode 2 with freeze completes the same 250 iterations in 4.0 minutes — a 4.2$\times$ reduction. On Trumpet (1.23M), Mode 2 reduces 3.7 minutes to 53 seconds. Without the freeze policy, GPU-based CVT at million-vertex scale is impractical for iterative design workflows; with it, the runtime falls within a range that permits interactive exploration of remeshing parameters.

The effective per-iteration complexity of Mode 2 is $O((1 - f) \cdot n \cdot K)$, where $f$ is the frozen fraction. As $f$ grows toward 1 with increasing $n$, the active query count $(1 - f) \cdot n$ grows sublinearly in $n$ — at 97\% frozen, only 3\% of sites are queried per iteration. This freeze-driven sublinear scaling is the mechanism through which our method extends KNN-based GPU CVT to scales where prior methods are impractical, and through which it maintains a consistent advantage over CPU Delaunay methods whose per-iteration cost scales as $O(n \log n)$ with no convergence-aware reduction.
