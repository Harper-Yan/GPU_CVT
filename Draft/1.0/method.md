# 3. Method: Curvature-Adaptive Freeze Policy

## 3.1 Background: Surface CVT via Lloyd Iteration

Given a triangle mesh M with vertices V and faces F, and a set of N sites S distributed on M, surface CVT seeks a configuration where each site coincides with the centroid of its Voronoi cell on the surface. The standard algorithm is Lloyd iteration, which repeats the following steps:

1. **KNN query.** For each site s_i, find its K nearest neighboring sites. The neighbor set defines the local Voronoi topology.
2. **Tangent-plane Voronoi.** Project s_i and its K neighbors into the local 2D tangent plane at s_i. Clip the Voronoi cell of s_i against the neighbors to obtain a convex polygon.
3. **Centroid computation.** Compute the centroid of the clipped Voronoi cell in the tangent plane.
4. **Reprojection.** Project the 2D centroid back onto the mesh surface M to obtain the updated site position.

Steps 1 through 4 are repeated for all N sites in parallel on GPU. KNN search dominates per-iteration cost: it requires distance computations between all active query sites and candidate neighbors using a spatial acceleration structure (in our implementation, a bitonic sort on a uniform grid with hub-based pruning). The centroid clipping and reprojection are comparatively cheap.

Convergence is monitored by total displacement: the sum of squared distances between old and new site positions across all sites. In practice, total displacement decreases monotonically but per-site displacement varies widely depending on local surface geometry.

## 3.2 Curvature and Convergence on Surfaces

Lloyd iteration on surfaces exhibits a striking spatial pattern: sites in flat regions converge rapidly and monotonically, while sites in high-curvature regions oscillate persistently. We trace the mechanism through five measurable steps and introduce the curvature measure that governs our freeze policy.

**Tangent-plane distortion.** The tangent plane at a site s_i approximates the local surface as a plane. On curved surfaces, distances measured in the tangent plane diverge from true surface distances. We define tangent distortion as the ratio of projected 2D distance to true 3D distance for each site-neighbor pair. Across our benchmark meshes, tangent distortion grows from 1.4% at flat regions to 18.1% at sharp regions, a 13x increase.

**Centroid bias.** Because the Voronoi cell is clipped in the distorted tangent plane, the computed centroid is systematically biased. The offset between the tangent-plane centroid and the true surface centroid grows proportionally to tangent distortion: 13x larger at sharp than flat regions.

**Reprojection instability.** The biased centroid reprojects onto different mesh triangles across consecutive iterations. At sharp regions, the hosting triangle changes 9.3 times on average over 60 iterations, compared to 2.8 at flat regions. Each triangle change alters the local normal frame used for the next tangent-plane construction.

**Persistent oscillation.** The cycling normal frame causes the site to oscillate rather than converge. At sharp regions, 25% of consecutive displacement vectors point in opposing directions (direction reversal), compared to 3% at flat regions.

**Unreliable convergence signals.** A site that momentarily pauses during oscillation produces a low-displacement reading that does not indicate genuine convergence. We measure streak survival probability: given that a site has had k consecutive low-displacement iterations, the probability that the k+1-th iteration is also low-displacement. At flat regions, P(k=1 to k=2) = 86%, meaning low-displacement streaks are self-reinforcing. At sharp regions, P(k=1 to k=2) = 48%, a coin flip, meaning a single or short low-displacement streak is unreliable.

**Asynchronous neighborhood convergence.** In flat regions, when a site converges, its neighbors converge at a similar rate (phase-synchronous convergence, correlation r = +0.65 between site displacement and neighbor displacement). In high-curvature regions, neighbors oscillate with independent phases (r approaches 0). A focal site may pause while its neighbors continue moving, reshaping the Voronoi geometry. At sharp regions, 37% of low-displacement moments have unstable KNN neighborhoods, meaning the neighbor set changes despite the focal site being nearly stationary. Displacement alone cannot detect whether the local Voronoi topology has stabilized.

**Consequence.** A uniform freeze policy that freezes any site after a short low-displacement streak (e.g., streak = 2) produces 64% false-freeze rates at moderate and curved regions. False-frozen sites are locked into positions they would have moved away from within 5 iterations, degrading mesh quality.

**Curvature measure: normal variation.** We need a per-site curvature proxy that is cheap to compute and monotonically correlated with convergence instability. We use normal variation (NV), defined as:

$$NV(s_i) = 1 - \frac{1}{K} \sum_{j=1}^{K} \cos(\mathbf{n}_i, \mathbf{n}_{k_j})$$

where n_i is the surface normal at site s_i and k_1, ..., k_K are its K nearest neighbors. NV = 0 when all neighbors share the same normal (perfectly flat); NV approaches 1 when normals are orthogonal or opposing (sharp features). NV is computed from normals already available in the Lloyd loop (no additional mesh queries), correlates monotonically with every instability metric in the causal chain above, and produces consistent tier thresholds across meshes with different topology and curvature distributions.

## 3.3 Freeze Policy

We now describe the complete freeze policy: what constitutes convergence, how the convergence criteria scale with curvature, and what computation is skipped for frozen sites.

### Dual-gate convergence test

A site is a freeze candidate when two conditions hold simultaneously:

**Gate 1: Low displacement.** The squared displacement between the current and updated site position is below a threshold:

$$\|s_i^{(t+1)} - s_i^{(t)}\|^2 \leq \epsilon^2$$

where epsilon = 0.01 * R and R is the maximum bounding box edge length of the mesh.

**Gate 2: Stable KNN topology.** The first K_check nearest neighbors must be positionally identical to the previous iteration, where K_check is curvature-dependent:

$$\text{KNN}^{(t)}_{1..K_c}(s_i) = \text{KNN}^{(t-1)}_{1..K_c}(s_i)$$

The check requires both the same neighbor identity and the same rank ordering in the first K_c positions. Rank changes indicate that inter-site distances shifted, which means the Voronoi cell geometry changed even if the neighbor set is identical.

The number of neighbors checked, K_c, scales with curvature tier:

| Tier | NV range | K_check (fraction of K) | Neighbors checked (K=32) |
|------|----------|------------------------|--------------------------|
| 0 (flat) | < 0.15 | 50% | 16 |
| 1 (gentle) | [0.15, 0.35) | 60% | 19 |
| 2 (moderate) | [0.35, 0.55) | 75% | 24 |
| 3 (curved) | [0.55, 0.80) | 85% | 27 |
| 4 (sharp) | >= 0.80 | 100% | 32 |

The rationale: the Voronoi cell of a site is primarily determined by its closest ~6-8 neighbors, which contribute bisector edges to the cell boundary. More distant neighbors (rank 20-32) rarely contribute bisector edges, especially on flat regions where sites are evenly spaced. On flat regions, rank swaps among distant neighbors reflect negligible distance perturbations rather than meaningful Voronoi geometry changes. On sharp regions, all neighbors potentially contribute to the cell boundary due to tangent-plane distortion, so all K neighbors are checked.

Displacement and neighborhood topology measure structurally different aspects of convergence. At flat regions they are redundant: when a site stops moving, its neighbors stop too, and the KNN set stabilizes. At curved regions they decouple: a site may pause while neighbors continue oscillating, producing low displacement with an unstable neighborhood. Gate 2 catches the 37% of false convergence events at sharp regions where Gate 1 alone would incorrectly signal convergence. Both gates must pass simultaneously.

### Curvature-scaled streak with density adaptation

Even with dual gating, a single iteration of passing both gates is insufficient at high curvature. We require both gates to pass for a number of consecutive iterations (the "streak") that increases with curvature. The base streak lengths are:

| Tier | Base streak |
|------|------------|
| 0 (flat) | 10 |
| 1 (gentle) | 15 |
| 2 (moderate) | 20 |
| 3 (curved) | 25 |
| 4 (sharp) | 30 |

On dense meshes, the base streaks are scaled by a density factor:

$$\tau_t = \max(3, \lfloor \tau_t^{\text{base}} \cdot (N_{\text{ref}} / N)^{0.3} \rceil)$$

where N_ref = 50,000 is a reference mesh size and N is the number of sites. For meshes smaller than N_ref, streaks are unchanged. For larger meshes, streaks are shortened: a 544K-vertex mesh uses scale ≈ 0.49, reducing the flat-tier streak from 10 to 5. The exponent 0.3 is chosen so that the scaling is mild — a 10x increase in mesh size reduces streaks by approximately 50%.

The density scaling is motivated by the observation that on dense meshes, KNN stabilization takes longer in absolute iterations (more neighbors at similar distances leading to rank permutations), but each iteration produces proportionally smaller geometric changes. A shorter streak on a dense mesh provides equivalent confidence to a longer streak on a sparse mesh, because the per-iteration displacement magnitude and centroid change are correspondingly smaller.

A streak counter c_i tracks consecutive iterations where both gates pass. If both gates pass, c_i increments by 1. If either gate fails, c_i resets to 0. When c_i reaches the tier-specific threshold, site s_i is marked as frozen. Tier assignment is performed once, after the first few Lloyd iterations stabilize the KNN structure, and does not change during iteration.

### What freezing skips and what continues

Once a site s_i is frozen:

- **KNN queries are skipped.** The frozen site does not participate as a query in the KNN search. Its neighbor list from the last iteration before freezing is stored and reused. The freeze-aware bitonic-sort KNN backend (Section 3.4) excludes frozen sites at kernel level with negligible overhead, so per-iteration KNN cost scales with the number of unfrozen sites. Since KNN dominates GPU per-iteration cost, this is the primary source of speedup.

- **Centroid computation and reprojection continue.** The frozen site's Voronoi cell is still clipped and its centroid is still computed and projected, because unfrozen neighbors may still be moving. As neighbors shift position, the Voronoi cell geometry of the frozen site changes, and the centroid tracks these changes. Allowing the centroid to update ensures that when a frozen site's neighborhood eventually stabilizes, the site's position reflects the final neighborhood geometry rather than a stale snapshot.

Freezing is not "stopping all computation for a site." It is removing the site from the most expensive computation (KNN) while allowing it to passively adapt to ongoing changes in its neighborhood. The dual-gate requirement ensures that a site is only frozen when its neighborhood is already converging, so the frozen neighbor list remains a good approximation for the remaining iterations.

### Periodic KNN refresh

Although frozen sites' neighbor lists are a good approximation at the time of freezing, they can become stale over many iterations as unfrozen neighbors continue moving. Over hundreds of iterations, the accumulated drift can cause quality divergence between frozen and unfrozen configurations.

To bound this staleness, we perform a full KNN refresh for all sites (including frozen) every R iterations (R = 50 in our implementation). During a refresh iteration, the frozen mask is temporarily cleared so that the bitonic KNN backend queries all N sites, producing fresh neighbor lists for every site. The frozen mask is then restored — no sites are unfrozen, but their neighbor lists are updated to reflect the current global configuration.

The cost of this refresh is one full KNN query every R iterations, amounting to approximately 1/R = 2% overhead on per-iteration KNN cost. In exchange, the stale-KNN error is bounded to at most R iterations of drift. In our experiments, this eliminates quality divergence entirely: with periodic refresh, the freeze mode's quality curves track the unfrozen baseline within 0.05% across 1000 iterations, compared to a 2% divergence without refresh.

The freeze decision remains irreversible: once frozen, a site stays frozen for all subsequent iterations. Periodic refresh updates the frozen site's neighbor list but does not re-evaluate the freeze decision. This keeps the implementation simple while preventing stale-KNN accumulation.

## 3.4 Freeze-Aware KNN Backend

The freeze policy produces a progressively sparser set of active query sites: as iterations advance, the frozen fraction grows from 0% to 80-95%, and the KNN kernel must efficiently handle a skewed workload where most sites require no computation.

Standard brute-force KNN computes all $N^2$ pairwise distances regardless of which sites are active, making it unable to exploit this sparsity. We adopt a bitonic-sort KNN backend with hub-based spatial pruning that natively integrates a per-site frozen mask. The key design elements are:

**Uniform grid with hub structure.** Sites are bucketed into a uniform 3D grid. Each grid cell maintains a compact "hub" of representative points. During query, a site first scans hubs of nearby cells to establish a distance bound, then refines by scanning individual points only in cells within that bound. This avoids the $O(N)$ scan of brute-force.

**Frozen-site masking.** The query kernel accepts a per-site frozen flag array. At kernel launch, each thread checks its frozen flag and returns immediately if set. This early exit costs a single global memory read per frozen site and avoids all subsequent hub scanning, distance computation, and sorting work for that site. Critically, frozen sites remain in the spatial grid as *candidates* for other sites' neighbor queries — only the query is skipped, not the candidacy.

**Bitonic merge network.** Each query thread maintains a register-resident sorted list of K best neighbors, updated via a bitonic sorting network operating within a warp. The in-register design avoids shared memory bank conflicts and enables high throughput on the remaining active queries.

The combination of spatial pruning and frozen-site masking means that per-iteration KNN cost scales with the number of *unfrozen* sites rather than total sites. As the frozen fraction grows, the backend's effective workload shrinks proportionally, compounding with the inherent speed advantage of hub-based pruning over brute-force.

## 3.5 Algorithm Summary

The complete algorithm integrates into any GPU Lloyd iteration loop with minimal modification:

**Preprocessing (once):**
1. Run a few initial Lloyd iterations to stabilize KNN structure.
2. Compute NV for each site from KNN normals.
3. Assign each site to a curvature tier. Compute density-scaled streak thresholds and curvature-scaled K_check values. Initialize all streak counters to 0 and all frozen flags to false.

**Per iteration:**
1. **KNN query** for unfrozen sites only. Frozen sites reuse stored neighbor lists. Every R iterations, temporarily query all sites (including frozen) to refresh stale neighbor lists.
2. **Voronoi clipping and centroid computation** for all sites (frozen and unfrozen).
3. **Reprojection** of centroids onto mesh for all sites.
4. **Freeze test** for each unfrozen site:
   - Compute Gate 1 (displacement < threshold) and Gate 2 (first K_check neighbors positionally unchanged).
   - If both pass, increment streak counter. If either fails, reset to 0.
   - If streak counter reaches tier- and density-dependent threshold, mark site as frozen and store its current KNN list.

The per-iteration cost is $O((1 - f) \cdot N \cdot C_{\text{KNN}} + N \cdot C_{\text{light}})$, where f is the frozen fraction and $C_{\text{light}} \ll C_{\text{KNN}}$ covers centroid, projection, and freeze testing. Every R iterations, one additional $O(N \cdot C_{\text{KNN}})$ refresh is incurred, amortized to $O(N \cdot C_{\text{KNN}} / R)$ per iteration. Since KNN search accounts for over 90% of per-iteration GPU time, speedup scales approximately linearly with the frozen fraction.
