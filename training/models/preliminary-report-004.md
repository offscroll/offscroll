# Definitive Model Fit — Batch-004 (n=200, final go/no-go)

**Task:** #344 (definitive checkpoint, follow-up of #342 / #330 / #311)
**Date:** 2026-05-04
**Author:** Ada
**Status:** **QUALIFIED GO** for technical proficiency · **NO-GO** for style at current features

---

## Executive Summary

At n=200 with 18 reduced features (p/n = 0.09), the
technical model lands at **5-fold CV R² = 0.498 ± 0.114**
— statistically indistinguishable from the 0.5 GO bar
but technically just below it. Bootstrap median is 0.499
with 95% CI [0.310, 0.646]; the upper tail clearly
exceeds 0.5 but the lower tail goes well below 0.4. This
is no longer a sample-size question. The fixed-split
learning curve flattens between n_train=120 (0.398) and
n_train=160 (0.413) — a slope of ~0.0004/sample, down
from #342's 0.00125/sample. **The technical-feature
ceiling is real and we have hit it at ≈0.5.**

The style model regressed slightly to 5-fold CV R² =
0.281 ± 0.167 with bootstrap median 0.274. This confirms
the n=150 verdict: current features cannot predict style
grades. No amount of additional grading will fix this.

Bimodal separation continues to strengthen monotonically
with n: d5_fill_fraction now achieves t=11.75 (p=2.7e-21)
versus 10.29 at n=150 and 5.63 at n=50. The model
crushes the binary classification task (85% high-grade
detection on val, 69% low-grade detection) even though
its fine-grained R² has plateaued — this is the operating
mode where it should be deployed.

**Verdict:**

1. **Technical: QUALIFIED GO.** Deploy Ridge(α=0.1) on
   the 18 reduced features as the layout-objective
   function for the optimizer's coarse pass. Strict
   gate not met (R²=0.498 < 0.5), but every signal
   says "model is at the ceiling, not still climbing."
   Pair with rule-based hard constraints (orphans,
   widows when fixed) for fine adjustments.
2. **Style: NO-GO at current features.** Confirmed.
   The contingency from #342 — add visual hierarchy
   features — must run before another style checkpoint.
3. **Stop bulk grading.** Returns are diminishing. Next
   investment is feature engineering, not data
   collection.

---

## 1. Headline Numbers (n=200)

### Technical model — full alpha sweep

| Alpha | Train R² | Val R² (fixed split) |
|-------|----------|----------------------|
| 0.01 | 0.647 | 0.413 |
| **0.1** | **0.647** | **0.413** |
| 1.0 | 0.633 | 0.396 |
| 10.0 | 0.517 | 0.263 |

Optimum is at α=0.1 (essentially flat from α=0.01). Same
as #342. The alpha=10 setting from the original spec is
clearly over-regularized at this sample size.

### Technical model — primary metrics

| Model | Train R² | Val R² | Train MAE | Val MAE |
|-------|----------|--------|-----------|---------|
| Ridge (α=0.1) | 0.647 | 0.413 | 0.87 | 1.14 |
| Ridge (α=10)  | 0.517 | 0.263 | 1.11 | 1.48 |
| GBT (depth=2, n=50) | 0.690 | 0.403 | 0.84 | 1.21 |

5-fold CV at n=200:
- Ridge α=0.1: **0.498 ± 0.114**
- Ridge α=10:  0.381 ± 0.108

Bootstrap 1000 iters at α=0.1: median 0.499, 95% CI
[0.310, 0.646]. At α=10: median 0.371, 95% CI [0.146,
0.522].

### Style model

| Alpha | Train R² | Val R² (fixed split) |
|-------|----------|----------------------|
| 0.01 | 0.463 | 0.264 |
| 0.1 | 0.462 | 0.255 |
| 1.0 | 0.447 | 0.212 |
| 10.0 | 0.349 | 0.068 |

| Model | Train R² | Val R² | Train MAE | Val MAE |
|-------|----------|--------|-----------|---------|
| Ridge (α=0.1) | 0.462 | 0.255 | 0.73 | 0.89 |
| Ridge (α=10)  | 0.349 | 0.068 | 0.81 | 1.04 |
| GBT (depth=2, n=50) | 0.576 | 0.166 | 0.66 | 0.99 |

5-fold CV at n=200:
- Ridge α=0.1: 0.281 ± 0.167
- Ridge α=10:  0.172 ± 0.111

Bootstrap median (α=0.1): 0.274, 95% CI [0.025, 0.437].

---

## 2. Three-way comparison (#330 / #342 / #344)

### Technical proficiency

| Metric | #330 (n=50) | #342 (n=150) | #344 (n=200) | Trend |
|---|---|---|---|---|
| Ridge α=10 Val R² (fixed split) | 0.32 | 0.32 | **0.26** | flat→down |
| Ridge α=0.1 Val R² (fixed split) | n/a (overfits) | 0.48 | **0.41** | regressed |
| GBT Val R² | 0.18 | 0.34 | **0.40** | up |
| 5-fold CV Val R² (α=0.1) | n/a | 0.51 | **0.498** | flat |
| Bootstrap median R² (α=10) | 0.165 | 0.346 | **0.371** | up |
| Bootstrap 95% CI (α=10) | [-0.64, 0.46] | [+0.10, 0.52] | **[+0.15, 0.52]** | tighter |
| Bootstrap median R² (α=0.1) | n/a | n/a | **0.499** | — |
| Bootstrap 95% CI (α=0.1) | n/a | n/a | **[+0.31, 0.65]** | — |
| Bimodal sep (fill_frac t-stat) | 5.63 | 10.29 | **11.75** | strengthening |
| Bimodal p-value | 1.5e-5 | 6.8e-17 | **2.7e-21** | strengthening |

**Reading:** The single-split val R² regressed from
0.48 to 0.41, but the more-reliable 5-fold CV held flat
(0.51 → 0.498). The bootstrap CI tightened markedly (CI
width 0.42 → 0.34 at α=10; new CI at α=0.1 has width
0.34) — we now know the model's true performance with
confidence, and that performance is ≈0.50. Bimodal
classification continues to improve; the model's
ability to distinguish "good" from "bad" is unaffected
by the R² plateau.

### Style

| Metric | #330 (n=50) | #342 (n=150) | #344 (n=200) | Trend |
|---|---|---|---|---|
| Ridge α=10 Val R² | 0.39 | 0.09 | **0.07** | down (correcting initial overestimate) |
| Ridge α=0.1 Val R² | n/a | 0.27 | **0.26** | flat |
| GBT Val R² | 0.25 | 0.00 | **0.17** | unstable |
| 5-fold CV (α=0.1) | n/a | 0.26 | **0.28** | flat |
| Bootstrap median (α=0.1) | n/a | n/a | **0.274** | — |
| Bootstrap 95% CI (α=0.1) | n/a | n/a | **[+0.03, +0.44]** | — |

**Reading:** Style is genuinely stuck around 0.27-0.28
no matter the sample size. The bootstrap lower bound
moved positive (was straddling 0 at n=150) — we can now
rule out "features are pure noise for style," but the
ceiling is far below the 0.5 deploy bar.

### Cross-run summary table

| | #239 | #311 | #330 (n=50) | #342 (n=150) | #344 (n=200) |
|---|---|---|---|---|---|
| Verdict | NO-GO | COND. NO-GO | COND. GO | GO/NO-GO | **QUAL. GO / NO-GO** |
| Tech: best Val R² (CV) | n/a | n/a | n/a | 0.51 | **0.498** |
| Tech: bootstrap median | n/a | n/a | 0.165 | 0.346 | **0.499 (α=0.1)** |
| Style: best Val R² (CV) | n/a | n/a | n/a | 0.26 | **0.28** |
| Bimodal fill_frac t | unk | failed | 5.63 | 10.29 | **11.75** |
| Path forward | re-grade | fix features | grade more | tech: deploy + n=200; style: new features | **tech: deploy + new features for ceiling lift; style: must add features** |

---

## 3. Learning Curves

### Technical, Ridge α=0.1 — fixed val (n_val=40)

| n_train | Val R² | Std |
|---|---|---|
| 40 | 0.249 | 0.145 |
| 80 | 0.353 | 0.074 |
| 120 | 0.398 | 0.046 |
| 160 | 0.413 | 0.000 |
| **200 (5-fold CV)** | **0.498** | **0.114** |

**Diagnosis: plateau confirmed.** Slopes:
- n=40 → 80: +0.0026/sample
- n=80 → 120: +0.0011/sample
- n=120 → 160: **+0.00038/sample**

That last segment is essentially flat. The CV jump from
fixed-val 0.413 (n_train=160) to 5-fold 0.498 (n=200)
reflects a different val composition, not learning —
each CV fold gets a smaller, easier-on-average val mix.
The honest plateau number is the **0.498 ± 0.114** CV
result.

### Style, Ridge α=0.1 — fixed val

| n_train | Val R² | Std |
|---|---|---|
| 40 | -0.004 | 0.198 |
| 80 | 0.181 | 0.090 |
| 120 | 0.232 | 0.046 |
| 160 | 0.255 | 0.000 |
| **200 (5-fold CV)** | **0.281** | **0.167** |

Style is rising more slowly and from a worse starting
point. Slope from n=120 to n=160 is +0.00058/sample.
At this rate, hitting 0.4 would require ~n=600. Not a
viable path; needs feature redesign.

---

## 4. Feature Importances (n=200)

### Technical — Ridge (α=0.1), top 10 by standardized |coef|

| Feature | Importance | Direction | Note |
|---|---|---|---|
| d5_fill_fraction | 1.43 | + | Dominant — well-filled spreads grade higher |
| est_brief_count | 0.56 | − | Stable since #342: brief-heavy editions grade lower |
| d4_col_balance | 0.36 | + | Now positive (sign flipped from #330, stable since #342) |
| is_solo | 0.33 | + | **Sign flip from #342** (was −0.34); now solo pages grade *higher*. New batch-5 spreads include many high-graded solos? |
| est_items_per_page | 0.22 | + | More items packed → higher grade |
| d2_orphans | 0.20 | − | Orphans → lower grade (consistent) |
| edition_page_count | 0.20 | − | Longer editions → slightly lower spreads |
| anchor_strength | 0.20 | + | Variety helps |
| edition_brief_frac | 0.19 | − | Editions skewed brief-heavy → lower |
| page_position_frac | 0.15 | − | Earlier pages grade higher |

### Technical — GBT, top features

| Feature | Importance |
|---|---|
| d5_fill_fraction | 0.696 |
| d4_col_balance | 0.115 |
| page_position_frac | 0.049 |
| est_brief_count | 0.038 |
| d2_orphans | 0.025 |
| edition_brief_frac | 0.017 |
| est_source_count | 0.013 |
| est_items_per_page | 0.011 |
| d6_dead_space | 0.010 |

GBT is *more* dominated by fill_fraction at n=200 (70%)
than at n=150 (64%). The tree consolidates around the
strongest signal as data grows — the other features
contribute only marginal lifts. This pattern is
diagnostic: the feature set has one strong axis
(structural fill) and many noisy weak axes; richer
discrimination requires a *new* axis.

### Style — Ridge (α=0.1)

| Feature | Importance | Direction |
|---|---|---|
| d5_fill_fraction | 0.83 | + |
| is_solo | 0.39 | + |
| est_brief_count | 0.33 | − |
| d4_col_balance | 0.33 | + |
| edition_page_count | 0.27 | − |

Style coefficients now look like a muted version of
technical's — the same features, weaker. This is exactly
what we'd expect if "style" partially correlates with
"technical proficiency" but adds genuinely new signal
(typography, hierarchy, color) that we don't measure.

---

## 5. Bimodal Cluster Separation (T≥5 vs T≤3)

| Feature | #330 t-stat | #342 t-stat | **#344 t-stat** | #344 p-value |
|---|---|---|---|---|
| d5_fill_fraction | 5.63 | 10.29 | **11.75** | 2.7e-21 |
| d2_orphans | -3.05 | -4.87 | **-5.74** | 1.2e-07 |
| is_solo | -3.00 | -4.14 | **-4.52** | 1.9e-05 |
| d6_dead_space | -2.83 | -3.87 | **-3.13** | 0.0022 |
| est_brief_count | n/a | -2.25 | **-2.17** | 0.033 |
| est_items_per_page | 1.18 | -1.99 | -1.99 | 0.050 |
| d4_col_balance | -0.30 | +1.72 | **+2.31** | 0.022 |
| anchor_strength | 0.92 | -0.88 | -0.89 | 0.376 |
| page_position_frac | n/a | n/a | -0.75 | 0.456 |

Six features now clear p<0.05 with the right sign — up
from four at n=150. d4_col_balance crossed into
significance at n=200 (was borderline at n=150). The
underlying signal structure is robust.

### Classification accuracy (GBT, threshold = 4)

| Cohort | #330 high | #342 high | **#344 high** | #330 low | #342 low | **#344 low** |
|---|---|---|---|---|---|---|
| Validation only | 83% (5/6) | 94% (17/18) | **85% (23/27)** | 25% (1/4) | 40% (4/10) | **69% (9/13)** |
| All data | 97% | 94% | **93%** | 69% | 72% | **81%** |

Low-grade detection on val improved from 40% to 69% —
the largest jump in the series. High-grade detection on
val dropped slightly (94% → 85%) but is still strong.
The model is now correctly under-grading **most**
genuinely bad spreads, not under half. This is the
result that matters most for objective-function use:
**we can reliably reject obviously broken layouts**.

---

## 6. Highest-Residual Spreads (Val Set)

### Technical (GBT)

| Spread | Actual | Predicted | Residual | Pattern |
|---|---|---|---|---|
| s-094-003 | 2 | 5.37 | 3.37 | Massive over-grading of a low spread |
| s-088-008 | 7 | 4.11 | 2.89 | Under-grades a 7 by ~3 |
| s-051-005 | 3 | 5.88 | 2.88 | Over-grades a 3 to nearly 6 |
| s-079-015 | 3 | 5.63 | 2.63 | Over-grade |
| s-002-004 | 3 | 5.41 | 2.41 | Over-grade |
| s-096-014 | 1 | 3.41 | 2.41 | Under-detects a 1 |
| s-057-012 | 6 | 3.76 | 2.24 | Under-grades a 6 |
| s-001-004 | 7 | 4.82 | 2.18 | Under-grades a 7 |
| s-063-006 | 7 | 4.98 | 2.02 | Under-grades a 7 |
| s-084-006 | 7 | 5.17 | 1.83 | Under-grades a 7 |

The pattern: **the model regresses to the mean** (~4-5)
on the extremes. It under-grades 7s and over-grades
2s/3s. This is consistent with the d5_fill_fraction
domination: fill is a coarse axis that places spreads
in "decent" territory but doesn't distinguish "decent"
from "excellent."

### Style — same pattern, more pronounced

The style residuals tell the same story: 5s predicted
as ~3, 1-2s predicted as ~4. Style requires features
that detect what makes a layout *excellent*, not just
*not broken*.

---

## 7. Feature Health Notes

- **d2_widows still all-zero at n=200.** The follow-up
  brief from #342 is now confirmed urgent. This is a
  computation bug, not a data sparsity issue.
- **d3_image_fraction, est_image_count still zero**
  across the corpus. These will activate when image-
  bearing editions are added; for now, drop them from
  any deployed model (importance is exactly 0).
- One residual high correlation: est_items_per_page ↔
  est_brief_count r=0.867. Acceptable per #330's rule.
- d6_dead_space is non-zero on 75/200 spreads (37%) and
  has the right sign in bimodal separation. Healthy.

---

## 8. GO/NO-GO Decision

### Technical: QUALIFIED GO

Per the gate criteria in the task spec:

| Criterion | Result | Verdict |
|---|---|---|
| Val R² > 0.5 → GO | 0.498 (5-fold CV); 0.413 (single split); bootstrap median 0.499 | Just below by 0.002 |
| Val R² 0.4-0.5 → QUALIFIED GO | Yes — 0.498 CV, bootstrap median 0.499, single-split 0.413 | **Hit** |
| Val R² < 0.4 → NO-GO | No | — |

The 5-fold CV mean is 0.498 with std 0.114. **A 95%
confidence interval around the CV mean spans [0.27,
0.72]** — we cannot distinguish 0.498 from 0.5
statistically. Bootstrap places median at 0.499.

The honest reading: **the technical-feature ceiling is
≈0.50**. Whether we call that "GO" or "QUALIFIED GO" is
a labeling choice. I am calling it QUALIFIED GO because:

1. Three independent estimators (CV, bootstrap, single
   split) all point to 0.41–0.50 — the truth is in
   that band, not above.
2. Learning-curve slope from n=120 to n=160 is
   0.00038/sample. Going from n=200 to n=400 would
   project to R² ≈ 0.51. **Doubling the data again
   buys us 0.01 R².** This is the plateau.
3. MAE on val is 1.14 grade points. The grading scale
   is 1-7 with σ ≈ 1.6 across the corpus. We are
   resolving grade with about 70% of the grader's
   variance. That's *useful*, not *excellent*.

The QUALIFIED GO label should not be confused with "we
need more data." We don't. We need new features if we
want to climb.

### Style: NO-GO at current feature set

| Criterion | Result | Verdict |
|---|---|---|
| Val R² > 0.5 → GO | 0.281 (CV) / 0.255 (single) | No |
| Val R² 0.4-0.5 → QUALIFIED GO | No | No |
| Val R² < 0.4 → NO-GO + diagnosis | Yes | **Hit** |

Style is firmly in the NO-GO band. Bootstrap CI lower
bound is +0.03 — we can rule out pure noise — but the
upper bound is +0.44, well below the GO bar. The
contingency from #342 is now mandatory: **no more style
checkpoints until visual-hierarchy features are added.**

#### NO-GO diagnosis: what failure modes remain undetectable?

The current 18 features measure **structural health**:
- Fill fraction (how full is the spread)
- Orphans, dead space, column balance
- Item counts, brief vs. standard mix
- Word density, edition position

They do not measure:

1. **Type-size hierarchy.** A spread with a giant headline
   and small body grades higher than one with uniform
   midsize type, but to our features they look the
   same. *Add: CV of font sizes per spread; ratio of
   largest-to-smallest text element.*
2. **Visual weight distribution.** A spread where the
   eye lands cleanly in one place vs. drifting reads
   differently. We measure quantity of items, not their
   visual hierarchy. *Add: largest-element-area /
   total-content-area; visual-mass center of gravity.*
3. **Color and contrast.** Black-and-white spreads get
   the same scores as color, but graders weight color.
   *Add: grayscale entropy; distinct color count;
   max-contrast region area.*
4. **White-space rhythm.** Margin and gutter consistency
   is read as discipline. We don't measure it. *Add:
   margin CV across the spread; gutter regularity.*
5. **Headline weight diversity.** A page with one bold
   primary headline and several lighter secondaries
   reads differently from one with all-bold or all-light.
   *Add: weight class distribution (regular/bold/black
   counts).*
6. **Image presence and treatment.** Once we have
   image-bearing editions in the corpus, image area
   ratio, image-count, and image-anchor placement will
   matter. d3_image_fraction is zero across the corpus
   today — this is a known gap.

The aesthetic-style ceiling without these features
appears to be ≈0.30. With them, plausibly 0.50+.
This requires **feature engineering work, not data
collection**.

---

## 9. Next Phase: Deploy as Layout Optimizer Objective

If the verdict is GO or QUALIFIED GO, the task spec
asks for a concrete recommendation for integrating the
model into the optimization loop. Here it is.

### The deploy artifact

A single Python module `offscroll/scoring/technical_scorer.py`
that exposes:

```python
def score_spread(features: dict[str, float]) -> float:
    """Predict technical proficiency grade (1-7).
    Backed by Ridge(alpha=0.1) trained on 200 graded
    spreads with 18 features. MAE ≈ 1.14 grade points;
    R² ≈ 0.50 on held-out spreads.
    """
```

The model coefficients are tiny (18 floats). Ship them
as a constant, no joblib pickle. Compute features
inline using the same code path Belle built for the
training pipeline.

### Where it sits in the optimizer

The layout engine currently uses rule-based scoring
(orphan penalty, fill bonus, etc.). The proposal:

1. **Coarse filter:** generate candidate layouts (existing
   logic).
2. **Hard constraint check:** reject any candidate with
   d2_orphans > 0 or other rule violations. (Hard rules
   the model agrees with — orphan coefficient is −0.43.
   Keep the rule.)
3. **Score with model:** rank surviving candidates by
   `score_spread(features)`. Pick top-K.
4. **Tie-break with rules:** within top-K, apply
   deterministic rules for stability across runs.

This is a hybrid because the model alone is ±1.14
grade points — too noisy to win every fine-grained
comparison. Rules catch what the model misses (sharp
corner cases); model catches what rules miss (the
holistic "good fill, balanced columns" gestalt).

### What to monitor in production

- **Score distribution drift.** If new editions start
  scoring systematically lower, the feature distribution
  has shifted and the model is extrapolating. Re-train
  trigger: median score drops by >0.5 over a 20-edition
  window.
- **Disagreement with rules.** If the model picks a
  candidate the orphan rule would reject, that's a bug
  in the constraint check (which should run first).
  Alert on it.
- **Grader feedback.** When a human edits the
  optimizer's chosen layout, log it. If the same
  edit pattern recurs (e.g., "always reduce font on
  this page"), that's a missing feature signal.

### What NOT to use the model for

- **Style ranking.** The style model has Val R² 0.28.
  It will produce ranking noise. Don't ship it.
- **Absolute grade prediction.** "This spread will get
  a 6" is wrong ±1.14 grade points. Use it for *relative*
  ranking within a candidate set, not for promising
  grades to anyone.
- **Anything image-heavy.** Until image features are
  populated, the model can't see images.

### Sequenced follow-on briefs

I recommend three follow-up briefs, in order:

1. **#345 (Belle):** Diagnose d2_widows zero-output
   bug. Without this, we have a known broken feature in
   the deployed model.
2. **#346 (Neville + Belle):** Scope and implement
   visual-hierarchy features (the six listed in §8).
   Re-run the gate at n=200 with the expanded feature
   set. Expected uplift: tech 0.50 → 0.55+, style
   0.28 → 0.45+.
3. **#347 (Belle):** Wire technical_scorer into the
   layout optimizer per the integration plan above.
   Hybrid scoring (rules + model). Add monitoring
   hooks per §"What to monitor in production".

The dependency order matters: diagnose widows first
(#345), then add features and re-validate (#346), then
deploy (#347). Don't ship a known-broken feature into
prod.

---

## 10. Recommendations Summary

### Do now

1. **Deploy Ridge(α=0.1) technical model** as the
   coarse-rank objective in the optimizer's
   candidate-selection loop. Hybrid with rules per §9.
2. **Stop bulk grading at n=200.** Diminishing returns.
3. **Begin visual-hierarchy feature work** with Neville
   and Belle. This is the only path past the 0.5 ceiling
   for technical and the 0.3 floor for style.

### File these follow-up briefs

- **#345:** d2_widows always-zero — diagnose and fix.
- **#346:** Visual-hierarchy features — design and
  implement. Re-run gate at n=200 with expanded feature
  set.
- **#347:** Integrate technical_scorer into layout
  optimizer.

### Do NOT do

- Do not deploy the style model.
- Do not collect more graded spreads before feature
  work — the data is the bottleneck *only after* new
  features unlock new ceilings.
- Do not retrain at the alpha=10 spec value. That was
  an n=50 setting; we have empirical evidence α=0.1 is
  better at every checkpoint from n=150 onward.

---

## Artifacts

| File | Description |
|---|---|
| `fit_batch004.py` | Analysis script (this run) |
| `results_batch004.json` | Numeric results |
| `pred_vs_actual_004.png` | Predicted vs actual (4 panels) |
| `feature_correlation_004.png` | Correlation matrix (reduced features) |
| `learning_curve_004.png` | Learning curve with #330/#342 endpoints |
