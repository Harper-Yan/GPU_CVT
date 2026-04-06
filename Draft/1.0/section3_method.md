# 3. Method: Curvature-Adaptive Freeze Policy

We propose a curvature-adaptive freeze policy that eliminates redundant computation in surface CVT by selectively removing converged sites from KNN queries. Our key observation is that convergence behavior in Lloyd iteration is strongly governed by surface curvature: sites on flat regions stabilize early and reliably, while sites on curved regions exhibit persistent oscillation and unstable neighborhood topology.  

Based on this observation, we design a **dual-gate convergence criterion** combined with a **curvature-scaled confidence test** to safely identify converged sites. Frozen sites are excluded from KNN queries—the dominant computational cost—while continuing lightweight updates to preserve correctness. This yields substantial speedup with negligible impact on mesh quality.

---

## 3.1 Background: Surface CVT via Lloyd Iteration

Given a triangle mesh \( M = (V, F) \) and a set of sites \( S = \{s_i\}_{i=1}^N \) on the surface, surface CVT seeks a configuration where each site coincides with the centroid of its Voronoi cell. This is typically solved via Lloyd iteration:

1. **KNN query.** For each site \( s_i \), find its \( K \)-nearest neighbors.
2. **Tangent-plane Voronoi.** Project the site and its neighbors to a local tangent plane and construct a clipped Voronoi cell.
3. **Centroid computation.** Compute the centroid of the cell in the tangent plane.
4. **Reprojection.** Project the centroid back onto the surface.

These steps are executed in parallel on the GPU for all sites. Among them, **KNN search dominates runtime**, while centroid computation and reprojection are relatively inexpensive.

In practice, although global convergence is gradual, **most sites stabilize early**, leading to significant redundant computation in later iterations.

---

## 3.2 Curvature-Dependent Convergence Behavior

We observe that convergence behavior in Lloyd iteration varies systematically with surface curvature.

- **Flat regions.** The tangent-plane approximation is accurate, centroid estimates are unbiased, and sites converge monotonically.
- **Curved regions.** Tangent-plane distortion introduces centroid bias, causing oscillation and unstable neighborhood topology.

As a result, **displacement alone is not a reliable indicator of convergence**.

---

**Figure 2 (Curvature vs. convergence behavior).**  
*Left:* Flat region with smooth convergence.  
*Right:* Curved region with oscillation and changing neighbors.

---

We define a curvature proxy using **normal variation (NV)**:

\[
NV(s_i) = 1 - \frac{1}{K} \sum_{j=1}^{K} \cos(\mathbf{n}_i, \mathbf{n}_{k_j})
\]

---

## 3.3 Curvature-Adaptive Freeze Policy

### Design Principles

1. **Structural convergence detection**  
2. **Curvature-adaptive confidence**  
3. **Selective workload elimination**

---

### Dual-Gate Convergence Test

A site is converged when:

- **Low displacement**
\[
\|s_i^{(t+1)} - s_i^{(t)}\|^2 \le \epsilon^2
\]

- **Stable neighborhood**
\[
\text{KNN}^{(t)}(s_i) = \text{KNN}^{(t-1)}(s_i)
\]

---

**Figure 3 (Failure of displacement-only criterion).**  
Shows incorrect convergence detection when neighbors change.

---

### Curvature-Scaled Streak

A site is frozen when:

\[
(\text{disp} \le \epsilon^2) \land (\text{KNN stable}) \quad \text{for } c_i \ge \tau(NV)
\]

---

### Selective Work Skipping

- Skip KNN for frozen sites  
- Continue centroid + reprojection  

---

**Figure 4 (Freezing mechanism).**  
Frozen sites are skipped in KNN while active sites continue updates.

---

## 3.4 Algorithm Summary and Complexity

**Preprocessing**
- Initialize NV and tiers  

**Per iteration**
1. KNN for unfrozen sites  
2. Centroid + projection for all  
3. Freeze decision  

---

**Figure 5 (Pipeline overview).**  
Shows partial KNN + full centroid + freeze decision.

---

Runtime:

\[
O((1 - f)N \cdot C_{KNN} + N \cdot C_{light})
\]

Speedup scales with frozen fraction.
