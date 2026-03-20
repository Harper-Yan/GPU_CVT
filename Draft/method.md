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

**Gate 2: Stable KNN topology.** The K-nearest-neighbor set is identical to the previous iteration:

$$\text{KNN}^{(t)}(s_i) = \text{KNN}^{(t-1)}(s_i)$$

where equality requires exact matching of all K neighbor indices. Any single index change constitutes a failure.

Displacement and neighborhood topology measure structurally different aspects of convergence. At flat regions they are redundant: when a site stops moving, its neighbors stop too, and the KNN set stabilizes. At curved regions they decouple: a site may pause while neighbors continue oscillating, producing low displacement with an unstable neighborhood. Gate 2 catches the 37% of false convergence events at sharp regions where Gate 1 alone would incorrectly signal convergence. Both gates must pass simultaneously.

### Curvature-scaled streak

Even with dual gating, a single iteration of passing both gates is insufficient at high curvature. We require both gates to pass for a number of consecutive iterations (the "streak") that increases with curvature. Sites are assigned to tiers based on NV:

| Tier | NV range | Required streak | Streak survival at k=1 to 2 |
|------|----------|----------------|------------------------------|
| 0 (flat) | < 0.15 | 10 | 86% |
| 1 (gentle) | [0.15, 0.35) | 15 | 78% |
| 2 (moderate) | [0.35, 0.55) | 20 | 50% |
| 3 (curved) | [0.55, 0.80) | 25 | 48% |
| 4 (sharp) | >= 0.80 | 30 | 25% reversal rate |

A streak counter c_i tracks consecutive iterations where both gates pass. If both gates pass, c_i increments by 1. If either gate fails, c_i resets to 0. When c_i reaches the tier-specific threshold, site s_i is marked as frozen.

The tier boundaries [0.15, 0.35, 0.55, 0.80] correspond to empirical breakpoints where instability metrics show qualitative transitions. The streak lengths are the minimum values for which cumulative streak-survival probability at each tier converges toward a plateau. Tier assignment is performed once, after the first few Lloyd iterations stabilize the KNN structure, and does not change during iteration.

### What freezing skips and what continues

Once a site s_i is frozen:

- **KNN queries are skipped.** The frozen site does not participate as a query in the KNN search. Its neighbor list from the last iteration before freezing is stored and reused. Since KNN search dominates GPU per-iteration cost, skipping it for frozen sites is the primary source of speedup.

- **Centroid computation and reprojection continue.** The frozen site's Voronoi cell is still clipped and its centroid is still computed and projected, because unfrozen neighbors may still be moving. As neighbors shift position, the Voronoi cell geometry of the frozen site changes, and the centroid tracks these changes. Allowing the centroid to update ensures that when a frozen site's neighborhood eventually stabilizes, the site's position reflects the final neighborhood geometry rather than a stale snapshot.

Freezing is not "stopping all computation for a site." It is removing the site from the most expensive computation (KNN) while allowing it to passively adapt to ongoing changes in its neighborhood. The dual-gate requirement ensures that a site is only frozen when its neighborhood is already converging, so the frozen neighbor list remains a good approximation for the remaining iterations.

The freeze decision is irreversible: once frozen, a site remains frozen for all subsequent iterations. The curvature-scaled streak provides high confidence of genuine convergence; unfreezing would add complexity and overhead without measurable quality benefit in our experiments.

## 3.4 Algorithm Summary

The complete algorithm integrates into any GPU Lloyd iteration loop with minimal modification:

**Preprocessing (once):**
1. Run a few initial Lloyd iterations to stabilize KNN structure.
2. Compute NV for each site from KNN normals.
3. Assign each site to a curvature tier. Initialize all streak counters to 0 and all frozen flags to false.

**Per iteration:**
1. **KNN query** for unfrozen sites only. Frozen sites reuse stored neighbor lists.
2. **Voronoi clipping and centroid computation** for all sites (frozen and unfrozen).
3. **Reprojection** of centroids onto mesh for all sites.
4. **Freeze test** for each unfrozen site:
   - Compute Gate 1 (displacement < threshold) and Gate 2 (KNN unchanged).
   - If both pass, increment streak counter. If either fails, reset to 0.
   - If streak counter reaches tier threshold, mark site as frozen and store its current KNN list.
