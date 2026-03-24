# 2. Related Work

## Isotropic Remeshing: A Taxonomy

Isotropic surface remeshing, the problem of redistributing vertices to produce well-shaped, nearly equilateral triangles, has been approached through several fundamentally different strategies. We survey the major families below and argue that CVT-based methods occupy a unique position in the quality-scalability tradeoff, motivating the need for acceleration at large mesh scales.

**Greedy local operations.** The earliest and most widely adopted remeshing pipeline combines edge splits, collapses, flips, and tangential smoothing in a greedy loop [Botsch and Kobbelt 2004; Hoppe 1996]. Dunyach et al. [2013] extended the approach with adaptive sizing fields. These methods are simple to implement and handle topology changes gracefully, but element quality depends heavily on the number of smoothing passes. Convergence is heuristic: there is no energy being minimized, so the process is typically run for a fixed number of iterations without formal quality guarantees. Reported evaluations rarely exceed 100K vertices.

**Particle-based repulsion.** Turk [1992] distributed vertices on surfaces by mutual repulsion, producing approximately uniform distributions. Witkin and Heckbert [1994] generalized the idea with adaptive particle systems. These methods are intuitive but slow to converge due to long-range interactions and provide no direct control over the resulting tessellation quality. Evaluations are limited to meshes of a few thousand vertices.

**Variational and parameterization-based methods.** Alliez et al. [2003; 2005] formulated remeshing as error-diffusion over a global parameterization, achieving high-quality results but requiring a valid parameterization that is expensive to compute for complex geometry. Cohen-Steiner et al. [2004] introduced variational shape approximation, partitioning the surface into planar proxy regions. These methods scale poorly: parameterization cost is superlinear, and evaluations are typically confined to meshes under 50K faces.

**Optimal transport and blue noise.** De Goes et al. [2012] formulated CVT as an optimal transport problem, computing capacity-constrained Voronoi tessellations for blue-noise sampling. Mehta et al. [2012] connected Lloyd iteration to stochastic optimization for blue noise on surfaces. These approaches produce excellent distributions but require expensive transport solvers; reported meshes are under 10K sites.

**Learning-based remeshing.** Recent work applies neural networks to mesh processing. Sharp et al. [2022] introduced DiffusionNet for discretization-agnostic learning on surfaces. Liu et al. [2020] proposed neural mesh generation via subdivision. Potamias et al. [2022] learned remeshing as a vertex displacement problem. While these methods handle large inputs at inference time, they target different quality objectives (feature preservation, task-specific adaptation) rather than the provable energy-minimization properties of CVT. Training data requirements and generalization to unseen geometry remain open challenges.

**Power diagrams and non-uniform tessellations.** Standard CVT produces uniform cell sizes, but many applications require adaptive resolution: smaller triangles in regions of high curvature or geometric detail, larger triangles in flat regions. Power diagrams (weighted Voronoi diagrams) generalize CVT by assigning a weight to each site, shifting the bisector between adjacent cells and producing non-uniform cell sizes [Aurenhammer 1987]. The weighted centroidal power diagram, where each site is the weighted centroid of its power cell, can be computed via a weighted Lloyd iteration [Levy and Liu 2010]. De Goes et al. [2012] connected power diagrams to optimal transport, enabling capacity-constrained tessellations with prescribed cell areas. Budninskiy et al. [2016] used power diagrams for adaptive surface sampling with guaranteed triangle quality bounds. These methods produce tessellations where cell size varies smoothly across the surface, matching a prescribed density function. The non-uniform cell sizes mean that convergence behavior varies not only with curvature but also with the local density gradient: sites in high-density regions have smaller cells and shorter inter-site distances, potentially converging faster, while sites near density transitions may oscillate as the power diagram adapts to competing size constraints.

**CVT-based remeshing** produces the highest isotropic element quality among non-learning methods by minimizing a well-defined energy functional [Du et al. 1999]. The resulting meshes have provably optimal point distributions in the Euclidean case and near-optimal distributions on surfaces [Du et al. 2003]. The quality advantage of CVT over greedy and particle methods has been demonstrated repeatedly [Yan et al. 2009; Levy and Liu 2010; Yan et al. 2014; Yan and Wonka 2016]. However, CVT's iterative nature (Lloyd iteration converges linearly [Du and Emelianenko 2006]) makes it computationally expensive, and the per-iteration cost of KNN and Voronoi computation scales with the total number of sites. Recent GPU approximations such as RTF and PowerRTF [Yao et al. 2023] and curvature-adaptive multi-facet clipping [Fei et al. 2025] improve efficiency via specialized tangent-plane shortcuts and achieve higher practical scales than pure standard Lloyd, while a discrete approximated CVT was added to CGAL in 2025 for meshes up to several million triangles. As a result, **most published CVT remeshing methods using standard RVD or tangent-plane Lloyd evaluate on meshes of at most 50K to 100K vertices** (specialized caching or approximation approaches reach higher).

Our work addresses the scalability bottleneck of CVT directly. Rather than replacing CVT with a faster but lower-quality method or an approximate shortcut, we accelerate the standard tangent-plane Lloyd iteration itself by exploiting the spatially non-uniform convergence behavior of Lloyd iteration on curved surfaces.

## Surface CVT and the Tangent-Plane Framework

[unchanged, except add at end of tangent-plane paragraph:]
Yan and Wonka [2016] further extended the CVT energy with a penalty term to guarantee non-obtuse triangles. The recent RTF/PowerRTF family [Yao et al. 2023] and adaptive clipping CVT [Fei et al. 2025] operate in the same tangent-plane framework but replace expensive clipping/traversal with direct GPU-computable planar facets.

## GPU-Accelerated CVT

GPU parallelism is a natural fit for Lloyd iteration, where each site's update is independent. Rong and Tan [2006] introduced jump flooding for discrete Voronoi diagrams on GPU. Rong et al. [2011] extended the approach to surface CVT, achieving real-time performance on meshes up to 50K sites. Levy and Liu [2010] presented a GPU implementation of Lp-CVT using tangent-plane construction at similar scale. Ray et al. [2018] used GPU parallelism for restricted Voronoi diagrams in hex-dominant meshing. More recently, Basselin et al. [2021] accelerated 3D CVT computation on GPU for tetrahedral meshing.

**Recent GPU CVT accelerations focus on approximation shortcuts rather than uniform Lloyd.** Yao et al. [2023] introduced restricted tangent faces (RTF) and power-diagram variants (PowerRTF) computed directly on GPU without auxiliary points or full Voronoi traversal. Fei et al. [2025] proposed curvature-adaptive multiple facet clipping of 3D Voronoi cells with per-cell GPU parallelism. In 2025, CGAL added an approximated discrete CVT remesher (GSoC project) that uses entirely discrete geometry processing and scales to several million triangles with low memory overhead.

A consistent pattern across *all* GPU CVT work (including these approximations) is that **each Lloyd iteration treats all sites uniformly** (or approximates the entire diagram uniformly), performing KNN/centroid/projection or equivalent for every site regardless of convergence state. [...]

## Scale in Surface Remeshing Evaluation

| Method | Category | Year | Max vertices reported |
|--------|----------|------|-----------------------|
| Turk [1992] | Particle | 1992 | ~5K |
| Alliez et al. [2003] | Variational | 2003 | ~30K |
| Botsch and Kobbelt [2004] | Greedy local | 2004 | ~100K |
| Yan et al. [2009] | RVD-CVT | 2009 | ~50K |
| Fuhrmann et al. [2010] | Direct Resampling (CVT) | 2010 | ~500K |
| Levy and Liu [2010] | GPU-CVT | 2010 | ~50K |
| Rong et al. [2011] | GPU-CVT | 2011 | ~50K |
| Yao et al. [2023] | GPU RTF/PowerRTF | 2023 | ~30K sites |
| Yan et al. [2014] | RVD-CVT | 2014 | ~100K |
| Fei et al. [2025] | Adaptive Clipping CVT | 2025 | ~100K input verts |
| CGAL (2025) | Approximated Discrete CVT | 2025 | several million triangles |
| Hu et al. [2020] | Tetrahedral | 2020 | ~500K (volumetric) |
| Potamias et al. [2022] | Learning | 2022 | ~100K (inference) |

While recent approximations (Yao et al. 2023; Fei et al. 2025) and discrete methods (CGAL 2025) reach higher scales via specialized shortcuts, the core uniform tangent-plane Lloyd loop that our acceleration targets has historically been limited to 50K–100K vertices for over a decade. [...]
