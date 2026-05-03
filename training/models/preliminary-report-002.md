# Preliminary Model Fit — Batch-002

**Task:** #311  
**Date:** 2026-05-03  
**Author:** Ada  
**Status:** CONDITIONAL NO-GO — feature engineering incomplete

---

## Executive Summary

The learned objective function approach is sound in
principle, but the current feature set cannot predict
grades — not even the bimodal cluster separation that
should be trivial. The root cause is not the modeling
approach; it's that 4 of 6 diagnostic features are
missing (all NA) and the remaining features are
dominated by content-quantity proxies that don't
capture layout quality.

**Verdict:** PAUSE grading. Fix feature computation
first (Belle's domain). Then re-run this gate on the
same 50 spreads.

---

## 1. Technical Model Results

| Model | Train R² | Val R² | Train MAE | Val MAE |
|-------|----------|--------|-----------|---------|
| Ridge | 0.660 | -1.340 | 0.80 | 2.21 |
| GBT | 0.998 | 0.071 | 0.05 | 1.40 |

Both models overfit catastrophically. GBT memorizes
training data perfectly but fails on 10 hold-out
spreads. Ridge generalizes even worse (negative R²
= worse than predicting the mean).

**Why this happens:** 34 features with 40 training
samples. The p/n ratio ≈ 0.85 guarantees overfitting
for any flexible model.

### Feature Importances (GBT)

| Feature | Importance | Comment |
|---------|-----------|---------|
| page_position_frac | 0.542 | Spurious — position doesn't cause quality |
| is_front | 0.124 | Confounded — front pages are structurally different |
| d5_fill_fraction | 0.058 | See issue below |
| anchor_strength | 0.049 | Derived from item count (r=0.86) |
| est_word_count_mean | 0.045 | Content proxy |

The model latches onto page position because front
covers and early pages happen to get different grades
in this small sample. This is a textbook spurious
correlation from small n.

---

## 2. Style Model Results

| Model | Train R² | Val R² | Train MAE | Val MAE |
|-------|----------|--------|-----------|---------|
| Ridge | 0.648 | -2.423 | 0.40 | 1.21 |
| GBT | 0.998 | -0.151 | 0.03 | 0.79 |

As expected, worse than technical. Same overfitting
pattern. The style model picks up on word count CV
and template entropy — both proxies for content
variety, not compositional quality.

### Highest-Residual Spreads (style, validation)

| Spread | Actual | Predicted | Residual |
|--------|--------|-----------|----------|
| s-010-002 | 4 | 2.60 | 1.40 |
| s-036-002 | 3 | 1.67 | 1.33 |
| s-084-006 | 4 | 2.78 | 1.22 |

These spreads score higher on style than features
predict. The features cannot see what Neville sees:
compositional tension, rhythm, visual hierarchy,
whitespace used purposefully vs. accidentally.

---

## 3. Critical Issues Found

### Issue 1: Four features are ALL NA

```
d2_orphans:     50/50 NA
d2_widows:      50/50 NA
d4_col_balance: 50/50 NA
d6_dead_space:  50/50 NA
```

These are the layout quality features that should
separate "competent fill" from "structural failure."
Without them, the model only has content-quantity
features — which tell you how much text is on the
page, not whether it's laid out well.

**This is the primary blocker.** Belle's feature
computation (#310) either didn't compute these or
they weren't joinable. These need to be present
before the gate can be evaluated.

### Issue 2: d5_fill_fraction is misscaled

Values range around 1.6–2.5 (should be 0–1 for a
fraction). It's perfectly correlated with
`est_words_per_page` (r=1.000), suggesting it's
actually a word-density metric, not a spatial fill
measure. Additionally, **the direction is wrong:**
high-grade spreads have *lower* d5_fill_fraction
(2.14 vs 2.58 for low-grade). If this were true fill
fraction, high-grade spreads should fill *more* space.

### Issue 3: Severe multicollinearity

Perfectly or near-perfectly correlated pairs:
- `is_solo` ↔ `is_terminal` ↔ `n_pages_in_spread` (r=±1.0)
- `edition_brief_frac` ↔ `edition_standard_frac` (r=-1.0)
- `est_words_per_page` ↔ `d5_fill_fraction` (r=1.0)
- `est_item_count` ↔ `est_standard_count` (r=0.93)
- `d8_word_count_cv` ↔ `anchor_strength` (r=0.90)

After removing redundant pairs we'd have ~20
features, which is still too many for 40 samples.

### Issue 4: Sample size

Bootstrap 95% CI on R² (OOB): [-8.5, 0.37] for
technical, [-8.6, 0.31] for style. The estimates are
completely unstable. Learning curve shows no
convergence — R² gets worse, not better, as we add
training samples (because more samples means the
model can't memorize as effectively, and there's no
true signal in these features).

---

## 4. Bimodal Separation Check

**Minimum bar: can the model separate T≥5 from T≤3?**

Distribution: 30 spreads at T≥5, 16 at T≤3, 4 at T=4.

The GBT achieves 96.7% correct on T≥5 and 81.2% on
T≤3 — but this is on train+val combined, and the GBT
memorizes training data (R²=0.998). On the 10
validation spreads alone, R²=0.07 means it barely
separates anything.

Feature means show the clusters are NOT well
separated by available features:

| Feature | High (T≥5) | Low (T≤3) | Diff |
|---------|-----------|----------|------|
| d5_fill_fraction | 2.14 | 2.58 | -0.43 |
| est_items_per_page | 1.97 | 1.47 | +0.50 |
| est_words_per_page | 965 | 1159 | -194 |
| anchor_strength | 1.97 | 1.86 | +0.11 |

The differences are small relative to variance and
sometimes in the *wrong* direction (low-grade
spreads have more words per page — because they're
dense text walls with no structure, which the model
reads as "more content = good").

**Minimum bar NOT met.** The features cannot
distinguish the two clusters.

---

## 5. What's Missing (for Neville)

The style residual analysis points to features the
current set lacks entirely:

1. **Spread balance / visual weight distribution** —
   d4_col_balance is NA. High-residual spreads
   likely have intentional asymmetry that reads as
   "designed" rather than "broken."

2. **Purposeful whitespace vs. dead space** —
   d6_dead_space is NA. The model can't distinguish
   "minimalist elegance" from "forgot to fill the
   page."

3. **Visual hierarchy / contrast** — no feature
   captures type size variation, weight contrast, or
   element scale relationships.

4. **Compositional rhythm** — alternation between
   dense and sparse elements. Currently all features
   are per-spread aggregates; no sequence-level
   information.

5. **Image/typography interaction** — d3_image_fraction
   is 0.0 for all spreads in this batch (no images).
   Style scoring may be more relevant for
   image-bearing spreads.

---

## 6. Sample Size Assessment

**Question:** Do we need 1,010 spreads or would 500
suffice?

**Answer:** Premature question. With the current
feature set, MORE data won't help — the learning
curve is flat-to-negative. The features lack signal.

Once features are fixed (4 NAs resolved, fill
fraction corrected, redundancies removed), we should
target ~10× the number of useful features. If we get
down to 8–10 non-redundant features, 100–150 spreads
would be adequate for ridge regression. For GBT with
interactions, 200–300 would be comfortable.

**Recommendation:** Don't grade more until features
are fixed. Re-evaluate sample size needs after the
feature fix.

---

## 7. Recommendation

### Immediate actions (blocking)

1. **Belle:** Fix feature computation for d2_orphans,
   d2_widows, d4_col_balance, d6_dead_space. These
   were specified in the grading protocol but are all
   NA in the output. Investigate whether the
   computation failed silently or these require a
   different extraction approach.

2. **Belle:** Investigate d5_fill_fraction. Values
   >1.0 and perfect correlation with word count
   suggest this is miscomputed. Should be spatial
   area fraction, not a word-density proxy.

3. **Ada:** Once features are recomputed, re-run this
   exact analysis on the same 50 spreads. No new
   grading needed — the grades are clean.

### After feature fix

4. Remove redundant features (target 8–12
   non-redundant features from the ~34 we have)
5. Re-evaluate bimodal separation with corrected
   features
6. If technical R² > 0.5 on validation → GO,
   proceed with grading 100–200 more spreads
7. If still failing → rethink feature specification
   with Neville (what does the grader actually look
   at that we're not measuring?)

### What NOT to do

- Don't grade more spreads yet (waste of Neville's
  time if features are broken)
- Don't try fancier models (the issue is features,
  not model complexity)
- Don't abandon the approach (the logic is sound;
  the implementation has a data bug)

---

## Artifacts

| File | Description |
|------|-------------|
| `fit_batch002.py` | Analysis script (reproducible) |
| `results_batch002.json` | Numeric results |
| `pred_vs_actual_002.png` | Predicted vs actual scatter (4 panels) |
| `feature_correlation_002.png` | Full correlation matrix |

---

## Comparison with #239 (Previous Attempt)

| | #239 | #311 (this) |
|---|------|-------------|
| Verdict | NO-GO (rendering bug) | CONDITIONAL NO-GO (feature bug) |
| Root cause | Bad grades (unreliable due to rendering) | Bad features (4 missing, 1 miscomputed) |
| Grades reliable? | No | Yes (batch-002 is clean) |
| Features reliable? | Unknown | No (4/6 diagnostic features NA) |
| Path forward | Re-grade after render fix | Re-compute features, re-run gate |

The situation is better than #239: we know the grades
are good and the problem is isolated to feature
computation. This is a fixable engineering issue, not
a fundamental approach problem.
