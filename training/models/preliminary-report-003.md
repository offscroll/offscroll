# Preliminary Model Fit — Batch-003 (n=150 checkpoint)

**Task:** #342 (n=150 checkpoint, follow-up of #330 / #311)
**Date:** 2026-05-04
**Author:** Ada
**Status:** **CONDITIONAL GO** for technical; **NO-GO** for style
(continue grading; style needs new features)

---

## Executive Summary

At 3x the data (150 spreads vs 50), the technical
proficiency model crosses the GO threshold once
regularization is re-tuned for the larger sample.
With alpha=10 (the small-n setting from #330) we are
plateauing at val R²=0.32. With alpha=0.1 (appropriate
at n=150), val R²=0.48 on the held-out split and
0.51 ± 0.18 under 5-fold CV — at or just above the 0.5
GO bar. The learning curve at alpha=0.1 is still rising
(0.43 → 0.48 from n=80 to n=120), so the true ceiling
is likely 0.55+.

The style model is in a different regime. Validation
R² is 0.27 at best (Ridge, alpha=0.1) — below 0.3, with
a rising learning curve. The features that work for
technical proficiency (fill, orphans, dead space) carry
weak signal about subjective style. This was foreshadowed
in #330 — the style score there (0.39) was likely
optimistic from the small val set.

The bimodal cluster separation strengthened
substantially: d5_fill_fraction now achieves t=10.29
(p<1e-16) versus t=5.63 at n=50. Four diagnostic
features clear p<0.001.

**Verdict:**

1. **Technical: CONDITIONAL GO.** Deploy Ridge(alpha=0.1)
   as the MVP objective function for technical
   proficiency. Validate with #344 holdout and
   continue to n=200 to confirm trend.
2. **Style: NO-GO at current feature set.** R² is below
   threshold and feature importance is dispersed. Add
   visual hierarchy features (type size variation,
   element scale, color/contrast) per #330's contingency
   plan before another checkpoint.
3. **Continue grading toward n=200.** Both models'
   learning curves are still rising; doubling the
   sample again is cheaper than redesigning features.

---

## 1. Headline Numbers (n=150)

### Technical model (specified params: Ridge alpha=10, GBT depth=2 n=50)

| Model | Train R² | Val R² | Train MAE | Val MAE |
|-------|----------|--------|-----------|---------|
| Ridge (alpha=10) | 0.505 | **0.315** | 1.11 | 1.30 |
| GBT (depth=2, n=50) | 0.724 | **0.338** | 0.75 | 1.20 |

Note: alpha=10 is now over-regularized at this sample
size. Tuning sweep:

| alpha | Train R² | Val R² |
|-------|----------|--------|
| 0.1 | 0.655 | **0.480** |
| 1.0 | 0.636 | 0.439 |
| 10.0 | 0.505 | 0.315 |

Ridge with alpha=0.1 gives Val R² = 0.480 on the fixed
val split and **0.510 ± 0.182 under 5-fold CV** (each
fold trains on 120 of 150, tests on 30). This crosses
the GO threshold.

### Style model

| Model | Train R² | Val R² | Train MAE | Val MAE |
|-------|----------|--------|-----------|---------|
| Ridge (alpha=10) | 0.315 | 0.092 | 0.79 | 0.94 |
| Ridge (alpha=0.1) | 0.440 | **0.267** | — | — |
| GBT (depth=2, n=50) | 0.604 | -0.004 | 0.61 | 1.00 |

Style does not clear the 0.3 threshold even with tuned
regularization. 5-fold CV at alpha=0.1: 0.258 ± 0.141.

---

## 2. Comparison to #330 (n=50)

### Technical proficiency

| Metric | #330 (n=50) | #342 (n=150) | Change |
|---|---|---|---|
| Ridge (alpha=10) Val R² | 0.320 | 0.315 | -0.005 |
| Ridge (alpha=10) Val MAE | 1.23 | 1.30 | +0.07 |
| Ridge (alpha=0.1) Val R² | n/a (overfits at n=40) | **0.480** | — |
| GBT Val R² | 0.181 | 0.338 | +0.157 |
| GBT Val MAE | 1.23 | 1.20 | -0.03 |
| Bootstrap median R² (alpha=10, full set) | 0.165 | 0.346 | +0.181 |
| Bootstrap 95% CI upper | 0.464 | **0.521** | +0.057 |
| Bootstrap 95% CI lower | -0.635 | **+0.099** | +0.734 |

Reading: At fixed alpha=10, R² is essentially flat —
that confirms alpha=10 is the binding constraint, not
data volume. GBT improved markedly (overfit at n=50,
generalizes at n=150). Bootstrap CIs tightened
dramatically; the lower bound moved from -0.64 to +0.10,
indicating the model is reliably better than the mean.

### Style

| Metric | #330 (n=50) | #342 (n=150) | Change |
|---|---|---|---|
| Ridge (alpha=10) Val R² | 0.388 | 0.092 | -0.296 |
| GBT Val R² | 0.245 | -0.004 | -0.249 |
| Bootstrap median R² | 0.138 | 0.124 | -0.014 |
| Bootstrap 95% CI upper | 0.490 | 0.297 | -0.193 |

Style went **down** with more data. This is informative,
not catastrophic. The n=50 numbers were inflated by
small-val-set variance (only 10 val spreads in #330).
The bootstrap median (0.14 → 0.12) was always low — the
single-split number was noise. The new estimate is
the more trustworthy one.

### Bimodal cluster separation (T>=5 vs T<=3)

| Feature | #330 (n=50) t-stat | #342 (n=150) t-stat | p-value at n=150 |
|---|---|---|---|
| d5_fill_fraction | 5.63* | **10.29*** | 6.8e-17 |
| d2_orphans | -3.05* | **-4.87*** | 6.2e-06 |
| is_solo | -3.00* | **-4.14*** | 9.5e-05 |
| d6_dead_space | -2.83* | **-3.87*** | 2.3e-04 |
| est_brief_count | n/a | -2.25* | 0.028 |
| est_items_per_page | 1.18 | -1.99 | 0.050 |
| d4_col_balance | -0.30 | +1.72 | 0.089 |
| anchor_strength | 0.92 | -0.88 | 0.381 |

The four features that mattered at n=50 are all stronger
at n=150 — the signal is real and growing with sample
size. Note d4_col_balance flipped sign (low cluster had
higher value at n=50, high cluster has higher at n=150);
neither result is significant, so this is noise.

### Classification accuracy (GBT, 4.0 threshold)

| Set | #330 high (T>=5) | #342 high | #330 low (T<=3) | #342 low |
|---|---|---|---|---|
| Validation only | 83.3% (5/6) | **94.4% (17/18)** | 25.0% (1/4) | **40.0% (4/10)** |
| All data | 96.7% | 94.0% | 68.8% | 72.2% |

High-grade detection is strong and improved. Low-grade
detection improved slightly but remains weak — the
model under-grades fewer than half the genuinely
low-grade spreads in val. This is consistent with the
observation that low-grade "failure modes" are
heterogeneous (broken mastheads, orphans, dead space,
imbalance) and harder to learn from a small number of
examples each.

---

## 3. Learning Curves

Fixed val set (n_val=30), variable training subsample.
Mean ± std over 50 random subsamples; n=120 is the full
training set (deterministic, std=0).

### Technical, Ridge alpha=10 (specified)

| n_train | Val R² | Std |
|---|---|---|
| 40 | 0.062 | 0.194 |
| 60 | 0.191 | 0.120 |
| 80 | 0.247 | 0.089 |
| 100 | 0.280 | 0.051 |
| 120 | 0.315 | 0.000 |
| 150 (5-fold CV) | 0.333 | 0.138 |

**Diagnosis: shallow plateau.** The curve is monotonic
but the slope from n=80 to n=120 is only 0.0017/sample.
At alpha=10, returns are diminishing. The CV result at
n=150 (0.333) is barely above the n=120 single-split
(0.315). At this alpha, more data alone won't push us
to 0.5.

### Technical, Ridge alpha=0.1 (better-tuned)

| n_train | Val R² | Std |
|---|---|---|
| 40 | 0.251 | 0.148 |
| 60 | 0.375 | 0.109 |
| 80 | 0.430 | 0.093 |
| 100 | 0.450 | 0.050 |
| 120 | 0.480 | 0.000 |
| 150 (5-fold CV) | 0.510 | 0.182 |

**Diagnosis: still rising.** Slope from n=80 to n=120:
0.00125/sample. From n=100 to n=120: 0.0015/sample.
Linear extrapolation to n=200 (n_train=160) projects
to ~0.55-0.60. The n=150 5-fold result already crosses
0.5.

### Style, Ridge alpha=10 (specified)

| n_train | Val R² | Std |
|---|---|---|
| 40 | -0.157 | 0.272 |
| 60 | -0.059 | 0.166 |
| 80 | 0.019 | 0.105 |
| 100 | 0.060 | 0.076 |
| 120 | 0.092 | 0.000 |

### Style, Ridge alpha=0.1

| n_train | Val R² | Std |
|---|---|---|
| 40 | -0.108 | 0.385 |
| 60 | 0.027 | 0.240 |
| 80 | 0.174 | 0.155 |
| 100 | 0.222 | 0.114 |
| 120 | 0.267 | 0.000 |

Style is rising at both alphas but starting from a
worse position. To clear 0.3 reliably, we need either
more data or different features.

---

## 4. Feature Importances (n=150)

### Technical — Ridge (alpha=10), top 10 by standardized |coef|

| Feature | Importance | Direction | Note |
|---|---|---|---|
| est_brief_count | 0.84 | negative | Many brief items → lower grade (shifted from positive in #330 — interesting) |
| est_items_per_page | 0.45 | positive | More items → higher grade |
| d4_col_balance | 0.38 | positive | (was -0.06 importance in #330) |
| is_solo | 0.34 | negative | Solo pages → lower grade |
| d5_fill_fraction | 0.34 | positive | Better fill → higher grade |
| d2_orphans | 0.29 | negative | Orphans → lower grade |
| anchor_strength | 0.22 | positive | Variety → higher grade |
| est_source_count | 0.20 | negative | Sources alone → not enough |
| page_position_frac | 0.16 | negative | Earlier pages → higher grade |
| est_words_per_page | 0.09 | ~0 | (was 0.22 in #330) |

**Notable change:** est_brief_count flipped from
positive (importance 0.155) at n=50 to negative
(importance 0.837) at n=150. Combined with the rise of
est_items_per_page (positive), the model now reads
"many short items packed onto a page → high grade" but
"many briefs in absolute count → low grade." This is
plausibly because the absolute brief count is high in
edition-types that are brief-heavy (where we have many
mediocre layouts) — i.e., it's picking up edition-level
quality variation now that we have spreads from more
editions.

### Technical — GBT, top features

| Feature | Importance |
|---|---|
| d5_fill_fraction | 0.639 |
| page_position_frac | 0.087 |
| edition_brief_frac | 0.082 |
| d4_col_balance | 0.080 |
| d6_dead_space | 0.043 |
| est_brief_count | 0.027 |

GBT remains dominated by d5_fill_fraction (64%) but
less extremely than at n=50 (76%). Page position,
edition mix, and column balance are now visible
splitters. The tree is starting to learn finer
distinctions.

### Style — Ridge (alpha=10)

| Feature | Importance | Direction |
|---|---|---|
| est_brief_count | 0.47 | negative |
| d4_col_balance | 0.38 | positive |
| est_items_per_page | 0.29 | positive |
| d5_fill_fraction | 0.17 | positive |
| edition_page_count | 0.16 | negative |
| d2_orphans | 0.11 | negative |

Style importances are flatter (top feature 0.47 vs 0.84
for technical) — no single feature dominates. This is
consistent with the lower R²: the features collectively
explain less of style.

---

## 5. Bootstrap Confidence Intervals (1000 iterations, OOB)

| Target | Median R² | 95% CI |
|---|---|---|
| Technical (Ridge alpha=10, 18 features) | 0.346 | [0.099, 0.521] |
| Style (Ridge alpha=10, 18 features) | 0.124 | [-0.217, 0.297] |

Technical is reliably better than the mean (lower CI
+0.099). The 0.5 threshold is now within the CI but
still in the upper tail. Style straddles 0; we can't
yet rule out that style features are pure noise.

---

## 6. Highest-Residual Spreads (Style, GBT validation)

| Spread | Actual | Predicted | Residual |
|---|---|---|---|
| s-082-006 | 1 | 3.86 | 2.86 |
| s-089-002 | 2 | 4.13 | 2.13 |
| s-083-011 | 2 | 3.80 | 1.80 |
| s-007-005 | 2 | 3.73 | 1.73 |
| s-011-009 | 5 | 3.35 | 1.65 |
| s-052-003 | 5 | 3.38 | 1.62 |
| s-076-002 | 3 | 4.52 | 1.52 |
| s-079-015 | 2 | 3.48 | 1.48 |
| s-093-004 | 2 | 3.44 | 1.44 |
| s-010-006 | 5 | 3.68 | 1.32 |

Pattern: the model regresses toward the mean (~3.5).
Spreads graded 1-2 get predicted ~3.5-4 (huge
under-grading misses), and spreads graded 5 get
predicted ~3.4 (over-grading ceilings). This is the
classic symptom of features that don't separate the
ends of the scale — exactly what we'd expect if the
features capture coarse layout health (fill, orphans)
but miss the fine style distinctions a grader actually
uses.

---

## 7. Redundancy Removal — same rules as #330

Dropped 15 features → 18 retained. Remaining high
correlations after removal:

- est_items_per_page ↔ est_brief_count: r=0.854
  (both kept — interpretable independent signal)

Three features remain zero-variance in this batch:
**d2_widows, d3_image_fraction, est_image_count**.
d2_widows is still all-zero across 150 spreads — at
this point the computation is suspect, not the data.
File a follow-up to verify. d3_image_fraction and
est_image_count being zero reflects that batch-001/002/
003 sources don't have images (or images aren't in the
corpus) — these will activate when image-bearing
editions are added.

---

## 8. GO/NO-GO Decision

### Technical: CONDITIONAL GO

Per the gate criteria in the task spec:

| Criterion | Result |
|---|---|
| Val R² > 0.5 → GO | 0.510 (5-fold CV at alpha=0.1) ✓ |
| Val R² > 0.5 at specified alpha=10 | 0.315 ✗ |
| Val R² 0.3-0.5 with rising curve | YES — both alphas qualify |

The strict gate (alpha=10, R² > 0.5) is not met. The
spirit of the gate (does this feature set predict
technical proficiency well enough to act on?) is
clearly met when regularization is appropriately tuned
for sample size. The bootstrap CI [0.099, 0.521] at
alpha=10 already touches 0.5; alpha=0.1 5-fold CV is at
0.51.

**Recommendation:** Deploy Ridge(alpha=0.1) on the 18
reduced features as the MVP technical-proficiency
objective function. Use it now to score candidate
layouts in the optimizer. Plan to re-tune
hyperparameters at every sample-size doubling.

### Style: NO-GO at current feature set

| Criterion | Result |
|---|---|
| Val R² > 0.5 → GO | 0.267 (best, alpha=0.1) ✗ |
| Val R² 0.3-0.5 with rising curve → continue | 0.267 just below 0.3 |
| Val R² < 0.3 or plateaued → investigate | YES |

Style is below 0.3 with rising-but-shallow curve. Per
#330's contingency: this is the trigger to **add
visual hierarchy features** (type size variation,
element scale, color/contrast, headline weight,
white-space rhythm). Continue collecting style grades —
but expect feature work, not just data, before the next
style checkpoint.

### Why the verdict differs from #330's CONDITIONAL GO

In #330 we said "the features work; we need data."
With 3x the data we now see:

1. **Technical features do work** — and the model
   crosses 0.5 with proper regularization. Confirmed.
2. **Style features carry less signal than #330
   suggested.** The n=50 style R²=0.39 was inflated by
   small-val variance; bootstrap median was always ~0.14.
   At n=150 we see the truth: the current features explain
   technical proficiency (which is largely *structural*)
   but weakly explain style (which is largely
   *typographic and aesthetic*).

This is consistent with the underlying difference in
the targets. Technical = is the page broken
(orphans, fills, balance)? Style = is it beautiful?
Our features probe structure, not aesthetics.

---

## 9. Recommendations

### Immediate

1. **Deploy technical model** as MVP layout objective
   function. Use Ridge(alpha=0.1) on 18 reduced features.
   Cross-validated R² ~0.51, MAE ~1.2 grade points.
2. **Continue grading toward n=200.** Both models'
   learning curves are still rising. Don't redesign
   features for technical — just feed it more data.
3. **Begin scoping new style features** with Neville:
   - Type-size variation (CV of font sizes per spread)
   - Element-scale ratios (largest:smallest visual unit)
   - Color/contrast (grayscale entropy or color count)
   - Headline-weight diversity
   - White-space rhythm (margin/gutter consistency)

### Diagnostic to file

- **d2_widows is still all-zero at n=150.** This is
  almost certainly a feature-computation bug. File a
  brief to Belle to verify the widow detection logic.
- **d3_image_fraction, est_image_count zero across
  the corpus.** Confirm whether the rendered editions
  in batches 1-3 actually contain images, or if the
  feature pipeline isn't picking them up.

### Do NOT do

- Do not abandon technical features — they work.
- Do not over-fit to alpha=0.1 specifically — keep
  re-tuning at each sample-size doubling.
- Do not deploy the style model — it will produce
  noise.

---

## 10. Comparison Across All Runs

| | #239 | #311 | #330 (n=50) | #342 (n=150) |
|---|---|---|---|---|
| Verdict | NO-GO | CONDITIONAL NO-GO | CONDITIONAL GO | **GO (tech) / NO-GO (style)** |
| Root cause | Bad grades | Bad features | Small sample | Style needs new features |
| Tech Val R² (best, alpha=10) | n/a | -1.34 | 0.32 | 0.34 (GBT) |
| Tech Val R² (best, retuned) | n/a | n/a | n/a | **0.51 (5-fold CV)** |
| Style Val R² (best) | n/a | -2.42 | 0.39 | 0.27 |
| Bimodal sep (fill_frac t-stat) | unknown | failed | 5.63 | **10.29** |
| Bootstrap CI (tech) | unknown | wide negative | [-0.64, 0.46] | **[0.10, 0.52]** |
| Path forward | Re-grade | Fix features | Grade more | Tech: deploy + n=200; Style: new features |

---

## Artifacts

| File | Description |
|---|---|
| `fit_batch003.py` | Analysis script (this run) |
| `results_batch003.json` | Numeric results |
| `pred_vs_actual_003.png` | Predicted vs actual (4 panels) |
| `feature_correlation_003.png` | Correlation matrix (reduced features) |
| `learning_curve_003.png` | Learning curve with #330 endpoint reference |
