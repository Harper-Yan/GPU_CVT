# 2. Related Work

## Isotropic Remeshing: A Taxonomy

Isotropic surface remeshing, the problem of redistributing vertices to produce well-shaped, nearly equilateral triangles, has been approached through several fundamentally different strategies. We survey the major families below and argue that CVT-based methods occupy a unique position in the quality-scalability tradeoff, motivating the need for acceleration at large mesh scales.

**Greedy local operations.** The earliest and most widely adopted remeshing pipeline combines edge splits, collapses, flips, and tangential smoothing in a greedy loop [Botsch and Kobbelt 2004; Hoppe 1996]. Dunyach et al. [2013] extended the approach with adaptive sizing fields. These methods are simple to implement and handle topology changes gracefully, but element quality depends heavily on the number of smoothing passes. Convergence is heuristic: there is no energy being minimized, so the process is typically run for a fixed number of iterations without formal quality guarantees. Reported evaluations rarely exceed 100K vertices.

**Particle-based repulsion.** Turk [1992] distributed vertices on surfaces by mutual repulsion, producing approximately uniform distributions. Witkin and Heckbert [1994] generalized the idea with adaptive particle systems. These methods are intuitive but slow to converge due to long-range interactions and provide no direct control over the resulting tessellation quality. Evaluations are limited to meshes of a few thousand vertices.

**Variational and parameterization-based methods.** Alliez et al. [2003; 2005] formulated remeshing as error-diffusion over a global parameterization, achieving high-quality results but requiring a valid parameterization that is expensive to compute for complex geometry. Cohen-Steiner et al. [2004] introduced variational shape approximation, partitioning the surface into planar proxy regions. These methods scale poorly: parameterization cost is superlinear, and evaluations are typically confined to meshes under 50K faces.

**Optimal transport and blue noise.** De Goes et al. [2012] formulated CVT as an optimal transport problem, computing capacity-constrained Voronoi tessellations for blue-noise sampling. Mehta et al. [2012] connected Lloyd iteration to stochastic optimization for blue noise on surfaces. These approaches produce excellent distributions but require expensive transport solvers; reported meshes are under 10K sites.

**Learning-based remeshing.** Recent work applies neural networks to mesh processing. Sharp et al. [2022] introduced DiffusionNet for discretization-agnostic learning on surfaces. Liu et al. [2020] proposed neural mesh generation via subdivision. Potamias et al. [2022] learned remeshing as a vertex displacement problem. While these methods handle large inputs at inference time, they target different quality objectives (feature preservation, task-specific adaptation) rather than the provable energy-minimization properties of CVT. Training data requirements and generalization to unseen geometry remain open challenges.

**Power diagrams and non-uniform tessellations.** Standard CVT produces uniform cell sizes, but many applications require adaptive resolution: smaller triangles in regions of high curvature or geometric detail, larger triangles in flat regions. Power diagrams (weighted Voronoi diagrams) generalize CVT by assigning a weight to each site, shifting the bisector between adjacent cells and producing non-uniform cell sizes [Aurenhammer 1987]. The weighted centroidal power diagram, where each site is the weighted centroid of its power cell, can be computed via a weighted Lloyd iteration [Levy and Liu 2010]. De Goes et al. [2012] connected power diagrams to optimal transport, enabling capacity-constrained tessellations with prescribed cell areas. Budninskiy et al. [2016] used power diagrams for adaptive surface sampling with guaranteed triangle quality bounds. These methods produce tessellations where cell size varies smoothly across the surface, matching a prescribed density function. The non-uniform cell sizes mean that convergence behavior varies not only with curvature but also with the local density gradient: sites in high-density regions have smaller cells and shorter inter-site distances, potentially converging faster, while sites near density transitions may oscillate as the power diagram adapts to competing size constraints.

**CVT-based remeshing** produces the highest isotropic element quality among non-learning methods by minimizing a well-defined energy functional [Du et al. 1999]. The resulting meshes have provably optimal point distributions in the Euclidean case and near-optimal distributions on surfaces [Du et al. 2003]. The quality advantage of CVT over greedy and particle methods has been demonstrated repeatedly [Yan et al. 2009; Levy and Liu 2010; Yan et al. 2014]. However, CVT's iterative nature (Lloyd iteration converges linearly [Du and Emelianenko 2006]) makes it computationally expensive, and the per-iteration cost of KNN and Voronoi computation scales with the total number of sites. As a result, **every published CVT remeshing method evaluates on meshes of at most 50K to 100K vertices**. The computational burden of Lloyd iteration is the primary factor limiting problem scale in the CVT literature, leaving a gap precisely where high-quality remeshing is most needed: large, detailed meshes with hundreds of thousands of vertices.

Our work addresses the scalability bottleneck of CVT directly. Rather than replacing CVT with a faster but lower-quality method, we accelerate CVT itself by exploiting the spatially non-uniform convergence behavior of Lloyd iteration on curved surfaces.

## Surface CVT and the Tangent-Plane Framework

Extending CVT from the plane to surfaces requires defining Voronoi diagrams with respect to surface distances. Two main approaches exist. Yan et al. [2009] compute restricted Voronoi diagrams (RVD) by intersecting 3D Voronoi cells with mesh faces, yielding exact surface CVT at the cost of geometric intersection queries. This approach was refined by Yan et al. [2014] for improved robustness and by Levy [2015] for efficient GPU implementation of RVD clipping. The alternative is the tangent-plane approximation: construct the Voronoi diagram in a local 2D coordinate frame and project the centroid back onto the surface [Levy and Liu 2010; Rong et al. 2011; Valette et al. 2008]. The tangent-plane approach is simpler, more GPU-friendly, and widely used in practice, but introduces systematic centroid bias at high-curvature regions.

Du et al. [2003] analyzed geodesic CVT where distances follow the surface metric, providing the theoretical basis for surface remeshing quality. The gap between geodesic and tangent-plane CVT is well-known qualitatively but has not been analyzed in terms of per-site convergence dynamics. Our work fills that gap: we trace how tangent-plane distortion causes curvature-dependent convergence instability and use the analysis to design an adaptive acceleration policy.

## GPU-Accelerated CVT

GPU parallelism is a natural fit for Lloyd iteration, where each site's update is independent. Rong and Tan [2006] introduced jump flooding for discrete Voronoi diagrams on GPU. Rong et al. [2011] extended the approach to surface CVT, achieving real-time performance on meshes up to 50K sites. Levy and Liu [2010] presented a GPU implementation of Lp-CVT using tangent-plane construction at similar scale. Ray et al. [2018] used GPU parallelism for restricted Voronoi diagrams in hex-dominant meshing. More recently, Basselin et al. [2021] accelerated 3D CVT computation on GPU for tetrahedral meshing.

A consistent pattern across all GPU CVT work is that **each Lloyd iteration treats all sites uniformly**, performing KNN, centroid, and projection for every site regardless of convergence state. At the scales evaluated (10K to 50K sites), per-iteration cost is milliseconds and total runtime is seconds, so selective computation offers little benefit. At 500K+ vertices, where a single Lloyd iteration costs over 10 seconds and 240 iterations take 41 minutes, the uniform approach becomes the dominant bottleneck. No prior GPU CVT method exploits spatially non-uniform convergence to reduce per-iteration work. Our freezing strategy is complementary to any GPU CVT backend: it reduces the active site count fed to the KNN kernel without changing the per-site computation.

## Acceleration of Lloyd Iteration

The linear convergence rate of Lloyd iteration [Du and Emelianenko 2006] has motivated several acceleration strategies. Liu et al. [2009] applied L-BFGS quasi-Newton optimization to CVT energy, achieving superlinear convergence at the cost of gradient and Hessian-vector evaluations. Hateley et al. [2015] proposed Anderson acceleration for CVT. Xin et al. [2016] introduced intrinsic CVT computation that avoids the tangent-plane approximation entirely using exact geodesic distances, improving convergence reliability at high curvature but at significantly higher per-iteration cost.

These methods reduce the *number* of iterations needed for convergence. Our approach is orthogonal: we reduce the *cost per iteration* by identifying and excluding converged sites. The two strategies could in principle be combined, applying L-BFGS or Anderson acceleration to the remaining active sites while frozen sites are skipped.

## Adaptive and Active-Set Methods

Excluding converged variables from iterative computation is a classical idea. Active-set methods in constrained optimization [Nocedal and Wright 2006] maintain a working set and update only active constraints. In mesh optimization, Freitag and Ollivier-Gooch [1997] selectively smoothed vertices whose quality fell below a threshold. Xu et al. [2018] applied local termination criteria to parallel Laplacian smoothing. Chen and Holst [2011] used adaptive refinement indicators to concentrate computation in regions of high error.

Our freeze policy shares the active-set spirit but addresses a challenge absent in prior mesh smoothing work: on surfaces, the convergence signal (displacement) is systematically unreliable in curved regions due to tangent-plane distortion. A single displacement threshold applied uniformly produces 64% false-freeze rates at sharp features. We introduce curvature as the governing variable for convergence reliability and add neighborhood stability as a second gate, neither of which has been proposed in prior adaptive mesh processing.

## Convergence Behavior on Surfaces

Planar CVT convergence is well understood [Du and Emelianenko 2006; Du et al. 1999]. On surfaces, convergence behavior is less studied. Levy and Liu [2010] noted empirically that convergence slows near sharp features. De Goes et al. [2012] analyzed CVT energy landscapes for blue noise but not per-site convergence dynamics. Yan et al. [2014] observed that restricted Voronoi diagrams improve convergence stability compared to the tangent-plane approach, implicitly confirming that the tangent-plane approximation is the source of instability, without further analysis.

Our causal analysis establishes that high-curvature convergence failure is not merely "slower convergence" but a qualitatively different regime: persistent oscillation with 25% direction-reversal rates, phase-diverse convergence among neighbors, and systematic decoupling of displacement and neighborhood stability signals. The analysis identifies curvature as the root cause, traces the complete mechanism through five measurable intermediate steps, and validates each step across multiple meshes. No prior work provides a comparable per-site convergence analysis for surface CVT.

## Scale in Surface Remeshing Evaluation

The meshes evaluated in surface remeshing papers have grown slowly over two decades. Table 1 summarizes the maximum mesh scale reported across representative methods in each category.

| Method | Category | Year | Max vertices reported |
|--------|----------|------|-----------------------|
| Turk [1992] | Particle | 1992 | ~5K |
| Alliez et al. [2003] | Variational | 2003 | ~30K |
| Botsch and Kobbelt [2004] | Greedy local | 2004 | ~100K |
| Yan et al. [2009] | RVD-CVT | 2009 | ~50K |
| Levy and Liu [2010] | GPU-CVT | 2010 | ~50K |
| Rong et al. [2011] | GPU-CVT | 2011 | ~50K |
| Yan et al. [2014] | RVD-CVT | 2014 | ~100K |
| Hu et al. [2020] | Tetrahedral | 2020 | ~500K (volumetric) |
| Potamias et al. [2022] | Learning | 2022 | ~100K (inference) |

For surface CVT specifically, the evaluation ceiling has been 50K to 100K vertices for over a decade. The reason is computational: at 100K sites, Lloyd iteration already takes minutes; at 500K, it takes tens of minutes per run. Our benchmark of 12 meshes up to 544K vertices (1.09M triangles) pushes surface CVT evaluation into a regime that has been computationally inaccessible to prior work, demonstrating that CVT-quality remeshing can be achieved at scales previously reserved for faster but lower-quality methods.
