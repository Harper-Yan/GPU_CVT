# GPU_CVT Revision Plan

Based on review notes in `GPU_CVT_review_notes.pdf`.

---

## High-Impact Issues (Critical)

### H1. Ablation: Separate KNN backend from freeze policy

**Problem:** Current results conflate two sources of speedup — the bitonic-sort KNN backend and the freeze policy. Reviewers cannot assess the contribution of each.

**Action:** Run all benchmark meshes under 3 configurations:
1. **Brute-force KNN, no freeze** (current baseline = Mode 0)
2. **Bitonic-sort KNN, no freeze** (new experiment — isolates backend improvement)
3. **Bitonic-sort KNN + freeze** (current method = Mode 1)

**Experiment:**
- Modify `main.cu` to support a mode that uses bitonic-sort KNN but disables freezing (set all streak thresholds to infinity, or skip freeze test).
- Run all 12 meshes, 250 iterations, record per-iteration time and total time.
- Report: Table with 3 columns of total time + 2 speedup columns (backend-only speedup, freeze-only speedup, combined speedup).
- Add a stacked bar chart or grouped bar chart showing the decomposition.

**Where in paper:** New Table + Figure in Section 4.2. Update text to discuss the two sources separately.

---

### H2. Comparison against alternative acceleration strategies

**Problem:** No comparison to other ways of accelerating CVT (early stopping, approximate CVT, prior GPU baselines).

**Action:** Implement and compare against:
1. **Early stopping:** Run baseline Lloyd and stop when global displacement drops below a threshold (match the quality of our method's output). Report time-to-equivalent-quality.
2. **Uniform freeze (no curvature awareness):** Freeze all sites after a fixed streak (e.g., streak=10 for all tiers). Shows the value of curvature-adaptive tiers.
3. **Prior GPU CVT baselines:** Compare against Rong et al. 2011, Yao & Liu 2023 if code is available, or cite their reported numbers at comparable mesh sizes.

**Experiment:**
- Early stopping: binary search for the iteration count that matches our final Q_avg on each mesh. Record time.
- Uniform freeze: run with streak=10 for all tiers. Record speedup and quality degradation.
- Optional: single-gate freeze (displacement only, no Gate 2). Shows the value of the dual gate.

**Where in paper:** New subsection 4.X "Comparison with Alternative Strategies" or fold into Section 4.2.

---

### H3. Sensitivity analysis on parameters

**Problem:** Tier thresholds, streak lengths, and displacement threshold are empirically chosen with no robustness analysis.

**Action:** Sweep each parameter independently:
1. **NV thresholds:** Perturb each boundary by ±0.05 and ±0.10. Record freeze rate and quality.
2. **Streak lengths:** Perturb each tier's streak by ±5. Record freeze rate, quality, and false-freeze rate.
3. **Displacement threshold epsilon:** Test 0.005R, 0.01R (default), 0.02R, 0.05R. Record freeze rate, quality, and speedup.

**Experiment:**
- Run on 3 representative meshes (one small smooth, one medium mixed, one large complex — e.g., horse 48K, igea 134K, Armadillo 173K).
- For each parameter sweep, hold all others at default.
- Report: line plots or heatmaps showing quality and freeze rate as a function of each parameter.

**Where in paper:** New subsection 4.X "Sensitivity Analysis" or new Section 4.6. Update Limitations to reference results.

---

## Medium-Impact Issues

### M1. Runtime breakdown (KNN vs centroid vs projection)

**Problem:** Paper claims "KNN dominates" but provides no profiling data.

**Action:** Profile per-iteration time broken down by kernel:
- KNN search
- Voronoi clipping + centroid
- Reprojection
- Freeze test overhead

**Experiment:**
- Use CUDA events to time each kernel per iteration on 3 meshes (small, medium, large).
- Report: stacked bar chart of per-iteration time breakdown for baseline vs. ours, at iteration 0 (no freeze), iteration 50 (partial freeze), iteration 200 (high freeze).

**Where in paper:** Section 4.2 or Discussion 5.1. Replaces the hand-wavy "KNN dominates" claim with data.

---

### M2. Frozen fraction over iterations

**Problem:** Paper reports final freeze rate but not the dynamics of how it evolves.

**Action:** Already have this data in `eval_iters.csv` (freeze_rate column per iteration).

**Experiment:**
- Plot frozen fraction vs. iteration for all meshes (or a representative subset).
- Highlight the rapid-early-freeze (igea) vs. gradual-freeze (xyzrgb_dragon) patterns.

**Where in paper:** Section 4.4 "Freeze Rate Dynamics" — currently text-only, add figure.

---

### M3. Generalization: different K values or weighted CVT

**Problem:** All experiments use K=20. No evidence the method works under different settings.

**Action:** Run with K=10, K=15, K=20 (default), K=30 on 3 representative meshes.

**Experiment:**
- Record: speedup, freeze rate, quality for each K.
- If time permits: test on one weighted CVT example (power diagram).

**Where in paper:** Sensitivity Analysis section or Discussion.

---

### M4. Measure freeze logic overhead

**Problem:** Paper claims freeze overhead is "negligible" but provides no measurement.

**Action:** Measure the per-iteration cost of the freeze test (displacement check, KNN identity check, counter increment) as a fraction of total iteration time.

**Experiment:**
- Run with and without the freeze test enabled (but without actually freezing — i.e., always reset counter). Compare per-iteration time.
- Or: use CUDA profiler to isolate the freeze-test kernel time.

**Where in paper:** Section 3.4 or 4.2. Replace "negligible overhead" with a number.

---

## Low-Impact Improvements

### L1. Reframe as adaptive computation / skewness-aware parallelism

**Action:** Add 1-2 paragraphs in Introduction or Related Work connecting to the broader theme of adaptive/non-uniform GPU computation. The freeze policy is an instance of dynamic workload pruning where convergence state determines per-element compute intensity.

---

### L2. Expand related work on dynamic workload pruning

**Action:** Add references to:
- Adaptive mesh refinement (AMR) on GPU
- Dynamic load balancing in GPU particle simulations
- Early-exit strategies in neural network inference
- Any prior work on selective iteration in geometric optimization

---

### L3. Improve figures

**Action:**
- Pipeline figure (Fig 3): already generated, verify clarity.
- Add a "workload shrink" visualization: heatmap of active sites on mesh surface at iterations 0, 50, 100, 200.
- Ensure all figures are publication-quality (vector PDF, consistent fonts, proper axis labels).

---

## Experiment Priority Order

| Priority | Task | Effort | Blocked by |
|----------|------|--------|------------|
| 1 | H1: Ablation (3 KNN variants) | High — needs code change + full re-run | Nothing |
| 2 | M1: Runtime breakdown | Medium — CUDA profiling | Nothing |
| 3 | M4: Freeze overhead | Low — minor profiling | Nothing |
| 4 | H3: Sensitivity sweeps | Medium — parameter sweep scripts | Nothing |
| 5 | M2: Freeze fraction plot | Low — data exists | Nothing |
| 6 | H2: Alternative strategies | Medium-High — implement early stop + uniform freeze | Nothing |
| 7 | M3: K-value sweep | Medium — re-run with different K | Nothing |
| 8 | L1-L3: Writing/figures | Low | Results from above |
