# Revision Plan

## 1. Parameter Sensitivity Study ~~(new experiment, high priority)~~ DONE

Ran Mode 2 on Horse (48K) and Armadillo (172K), varying 5 parameters (20 settings total, 40 runs).

**Results:** `experiments/ablation/` — summary.csv, per-parameter plots, analysis.md
**In paper:** experiments.tex §4.5 Parameter Sensitivity, fig11 (refresh), fig12 (streak)
**In figs/:** fig11_sensitivity_refresh.png, fig12_sensitivity_streak.png

**Key finding:** Refresh R is the only correctness parameter (R=never degrades quality). All others (streak, NV tiers, Kc/K, warmup) affect only efficiency. Quality varies <0.002 across all finite-R settings. No per-mesh tuning needed.

## 2. Hardware and Mode 0 Clarification ~~(text fix, high priority)~~ DONE

- Filled hardware: NVIDIA RTX 4070 Laptop (8GB), i7-14650HX, 16GB RAM
- Replaced "Mode 0" with "RTF [Yao et al. 2023]" everywhere in .md and .tex
- Regenerated fig5 and fig10 with RTF labels
- Fixed Geogram config inconsistency (related_work said "5 Lloyd + 30 Newton", experiments said "250 Lloyd + 0 Newton" — unified to the latter)

## 3. Number Consistency Audit ~~(text fix, high priority)~~ DONE

- Updated Qavg from 0.917 to 0.926 (actual 8-mesh average)
- Updated quality gap from 1.2% to 0.3%
- Updated freeze rates: small 80–93%, medium 84–98%, large 97–98%
- Fixed specific freeze citations (Armadillo 95%, happy_vrip 93%, etc.)
- Fixed Geogram Qavg in fig10 caption (0.924 → 0.929)

---

## Remaining TODO (sorted by impact to paper: high → low)

### ~~4. Time Breakdown by Stage~~ DONE

Stacked bar chart (fig13) showing 7 pipeline stages at iter 0/50/100/200 for Horse, Armadillo, Happy Buddha. KNN shrinks from 24–32% to 42–58% of a much smaller total. Added to experiments.tex §4.3 and experiments.md.

### ~~5. False-Freeze Definition and Measurement~~ DONE

The "64% → 2%" claim has backing data in `experiments/causal_chain_evidence/exp5_false_convergence/` but lacks a clear definition in the paper text. 

**Definition:** A false freeze is a site frozen at iteration t (by a given policy) whose displacement exceeds ε within the next W iterations — it wasn't actually converged and resumed moving.

**Existing data (exp5c):**
- Naive policy (Gate 1 only, streak=2): false-freeze rate = 29% (flat), 34% (gentle), 64% (moderate), 63% (curved), 57% (sharp)
- Full policy (dual-gate + curvature-scaled streak): not directly in exp5c but implied <2% by streak survival curves in exp5a

**Note:** KNN drift over R=50 iterations (measured via diagnostic kernel) shows 92–99% of frozen sites have changed KNN at refresh — but this is expected natural drift, NOT false freezing. This is exactly what refresh corrects. The false-freeze metric measures whether the freeze *decision* was correct at the time it was made, not whether KNN drifts later.

**Action:** Add 1–2 sentences in method.tex §3.3 defining false freeze precisely. Reference exp5c data in experiments.tex or supplementary.

### ~~6. Memory Footprint Table~~ DONE

Measured peak GPU via nvidia-smi polling (6 meshes). Sublinear scaling: 58× vertices → 9.7× memory due to CUDA lazy backing. Limit ~3.5–4M on 8GB. Added to experiments.tex and experiments.md.

### ~~7. Failure Cases~~ DONE

Added "Reduced benefit on high-curvature meshes" to limitations in discussion.tex and discussion.md. Nefertiti (80% freeze, 1.6×) vs Igea (98%, 2.4×). Also updated "Empirically calibrated thresholds" limitation to reference the sensitivity study.

### 8. Scope Tightening (text-only, ~10 min) — LOW-MEDIUM IMPACT

Prevents reviewer objection about overclaiming. Currently implies applicability to adaptive/power variants without experimental evidence. One sentence fix in abstract/intro.

### 9. Reversible vs Irreversible Freeze Validation (data extraction, ~30 min) — LOW IMPACT

Nice-to-have but unlikely to be a reviewer blocker. The sensitivity study already shows quality is stable, which implicitly validates irreversible freezing. Extracting "would-have-unfrozen" counts from refresh logs would strengthen the argument but is not essential.
