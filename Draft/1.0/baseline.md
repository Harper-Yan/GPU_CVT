# Baselines for GPU_CVT Paper

## Primary comparison: 3-mode ablation (self-contained, reproducible)

| Mode | KNN Backend | Freezing | Purpose |
|------|------------|----------|---------|
| 0 | Brute-force | No | Standard tangent-plane Lloyd (equivalent to RTF 2023 baseline) |
| 1 | Bitonic-sort + hubs | No | Isolates KNN backend improvement |
| 2 | Bitonic-sort + hubs | Yes (5-tier + refresh) | Full method |

This ablation cleanly separates the two sources of speedup without requiring external code.

## External baselines considered

### Runnable (code available)

**CGAL isotropic remeshing (greedy local ops)**
- Split/collapse/flip/smooth pipeline
- Available in CGAL and PMP library
- Any scale, CPU-only
- Purpose: show CVT produces higher quality than greedy methods, justifying computational cost
- Not a direct competitor (different algorithm class)

**CGAL ACVD (Valette & Chassery 2004, CGAL 6.1)**
- Discrete clustering approximation of CVT
- CPU, scales to millions of triangles
- Code: https://github.com/valette/ACVD and CGAL 6.1
- Purpose: alternative that trades quality for scalability
- Not directly comparable (discrete clustering vs continuous CVT, no KNN, CPU vs GPU)

**Geogram RVD-CVT (Levy et al.)**
- Exact restricted Voronoi diagram on surfaces
- CPU, tested up to ~50K
- Already used in our pipeline for evaluation (vorpalite)
- Purpose: quality reference — our method should match RVD quality

### Not runnable (no public code)

**RTF / PowerRTF (Yao et al. 2023)**
- GPU tangent-plane CVT with restricted tangent faces
- Evaluated at ~30K sites
- No public repository
- Our mode 0 (brute-force tangent-plane Lloyd) is the same framework

**Fei et al. 2025 (Adaptive multi-facet clipping)**
- GPU CVT with 1-3 facet clips per cell based on curvature
- Evaluated at ~9K output sites, 100K input vertices
- No public repository
- Orthogonal approach: improves per-iteration accuracy, we reduce per-iteration workload
- Higher quality than PowerRTF (Q_avg 0.917 vs 0.869 on Duck) but slower

## Recommendation for paper

**Include in experiments:**
1. 3-mode ablation (mode 0/1/2) — primary results
2. CGAL isotropic remeshing — non-CVT quality/speed reference (if reviewer requests)

**Cite in related work only (no direct comparison):**
- RTF/PowerRTF — same framework, no code, smaller scale
- Fei et al. 2025 — complementary approach, no code
- CGAL ACVD — different algorithm class (discrete clustering)
- Geogram RVD — quality reference, already used for evaluation

**Justification for self-contained ablation:**
- No external GPU CVT code is publicly available at our scale (>100K vertices)
- Our mode 0 IS the standard baseline everyone compares against
- The ablation isolates each contribution (KNN backend vs freeze policy) transparently
- A reviewer can reproduce all results from our code alone
