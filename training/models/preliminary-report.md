# Preliminary Model Fitting Report — OffScroll Layout Quality

**Author:** Ada (Algorithm Architect, IRAS)
**Date:** 2026-04-02
**Task:** #239
**Data:** 50 graded spreads (batch-001), 40 train / 10 val

---

## Executive Summary

**The learned objective function approach is sound, but batch 1
cannot validate it.** The grades measure "did the rendering bug
destroy this spread" — a signal invisible to metadata-derived
features. The GBT technical model posts R²=0.53 on 10 validation
samples, but bootstrap analysis reveals this is noise: median
out-of-bag R² is −1.1 for technical and −0.8 for style, with
97% and 89% of bootstraps negative respectively.

**This is not a model failure. It is a data/feature mismatch.**
The features describe what *should* be on the page. The grades
measure what *actually rendered*. Until the }{ rendering bug is
fixed or rendered-output features are computed, no model can
bridge this gap.

**Recommendation: Fix the rendering bug, re-render and re-grade
~50 spreads, then re-run this analysis.** Do not invest in grading
1,496 spreads on the current rendering pipeline.

---

## 1. Technical Proficiency Model

### Results

| Model | R² (train) | R² (val) | MAE (train) | MAE (val) |
|-------|-----------|---------|------------|----------|
| Ridge | 0.491 | 0.030 | 1.00 | 1.16 |
| GBT   | 0.799 | 0.533 | 0.62 | 0.75 |

### Interpretation

The GBT model's R²=0.53 on validation technically crosses the
0.5 "proceed" threshold. **Do not trust this number.** With only
10 validation samples and a bimodal grade distribution (44% score
2, 18% score 6), a model that learns to predict the mean for most
spreads and gets lucky on 1–2 validation points can hit 0.5.

The bootstrap analysis confirms this: fitting Ridge on random 80/20
splits of the full 50 samples, the median out-of-bag R² is **−1.1**
with a 95% CI of [−8.5, 0.01]. 97.2% of bootstrap iterations
produce negative R². The signal is not real.

### Top Features (GBT Importance)

| Feature | Importance |
|---------|-----------|
| page_position_frac | 0.313 |
| est_word_count | 0.111 |
| edition_template_entropy | 0.108 |
| edition_word_count_total | 0.071 |
| est_word_count_mean | 0.059 |

`page_position_frac` dominates because front pages (position 0.0)
are solo pages that don't have the }{ bug on a facing page, so
they score higher. This is the model learning a proxy for "which
page roles avoid the bug" — not learning layout quality.

---

## 2. Style Model

### Results

| Model | R² (train) | R² (val) | MAE (train) | MAE (val) |
|-------|-----------|---------|------------|----------|
| Ridge | 0.643 | 0.260 | 0.57 | 0.57 |
| GBT   | 0.790 | −0.016 | 0.38 | 0.62 |

### Interpretation

Both models fail on validation. The GBT overfits catastrophically
(R² from 0.79 train to −0.02 val). Ridge shows the same pattern
in a milder form.

Style grades are even harder to predict than technical grades
because:
1. Style has lower variance (std 1.2 vs 1.7) — less signal
2. Style depends more on compositional relationships between
   facing pages — features the metadata cannot capture
3. The best style scores (5–6) come from front pages with
   masthead design — a qualitative feature invisible to word counts

### Top Features (GBT Importance)

| Feature | Importance |
|---------|-----------|
| est_word_count | 0.276 |
| page_position_frac | 0.238 |
| anchor_strength | 0.070 |
| edition_standard_frac | 0.064 |
| edition_template_entropy | 0.053 |

Same story: the model gropes for any proxy of "does this spread
have content" and finds word count and page position.

### Highest-Residual Spreads (Style GBT)

| Spread | Actual | Predicted | |Residual| | What Neville Sees |
|--------|--------|-----------|-----------|-------------------|
| s-085-011 | 1 | 2.96 | 1.96 | Right page empty (}{ bug), left is a text wall — model predicted average but it's broken |
| s-078-007 | 5 | 3.43 | 1.57 | **Best spread in batch** — feature article facing briefs page. Template variety creates compositional dialogue. Features can't capture this |
| s-078-002 | 1 | 2.42 | 1.42 | Raw website search statistics rendered as content — data quality issue invisible to features |
| s-086-019 | 1 | 2.22 | 1.22 | Both pages completely empty — total rendering failure |
| s-086-003 | 4 | 2.85 | 1.15 | Q&A format providing natural rhythm — conversational structure invisible to word counts |

**Pattern:** Residuals are highest where the model can't distinguish
between "intended content that rendered" and "intended content that
didn't render." The features describe the same content for both
cases.

---

## 3. Diagnostic Analysis

### Feature Correlation

Several feature groups are perfectly or near-perfectly correlated
(see `feature_correlation.png`):

- `is_solo` / `n_pages_in_spread` / `spread_type_spread` / `page_role_interior` — all encode the same binary
- `edition_brief_frac` / `edition_standard_frac` — sum to 1.0
- `est_words_per_page` / `d5_fill_fraction` — r=1.000 (fill is just words/capacity)
- `d8_word_count_cv` / `anchor_strength` — r=0.918

**Action for future models:** Drop one from each perfectly
correlated pair. For Ridge this causes rank deficiency; for tree
models it wastes splits but doesn't hurt.

### Sample Size Assessment

The learning curve analysis (Ridge, random train/val splits from
n=15 to n=40) shows **no improvement** as n increases. Mean R² on
held-out data remains deeply negative at all sample sizes:

| n_train | Mean R² (tech) | Mean R² (style) |
|---------|---------------|-----------------|
| 15 | −19.4 | −15.2 |
| 25 | −3.6 | −3.7 |
| 35 | −1.1 | −1.8 |
| 40 | −1.1 | −1.0 |

**This is not a sample size problem.** More data won't help when
the features don't capture the signal. The curve is converging
toward negative R², not toward positive — the features are
fundamentally misaligned with what the grades measure.

### Bootstrap Confidence Intervals

| Target | Median R² | 95% CI | % Negative |
|--------|----------|--------|-----------|
| Technical | −1.101 | [−8.46, 0.01] | 97.2% |
| Style | −0.818 | [−9.13, 0.23] | 88.7% |

---

## 4. Verdict

### Why This Happened

The analysis reveals a clean diagnosis:

1. **Neville's grades in batch 1 are dominated by the }{ rendering
   bug.** ~50% of spreads have empty pages. The grade distribution
   is bimodal: 1–2 (broken) vs 5–6 (rendered). Neville explicitly
   notes: "the grades are measuring 'does the page have content'
   rather than 'is the composition good.'"

2. **Belle's features are computed from edition metadata, not
   rendered output.** They describe what content *should* be on
   the spread. They are blind to whether it actually rendered.

3. **The model is asked to predict a rendering bug from content
   metadata.** This is an impossible task. The bug is a pipeline
   failure unrelated to content features.

### What This Does NOT Mean

- **It does not mean the learned objective function won't work.**
  It means batch 1 doesn't test it.
- **It does not mean the features are bad.** Belle's features are
  well-designed proxies for layout characteristics. They'll become
  useful when grades measure layout quality rather than rendering
  success.
- **It does not mean 50 spreads is too few.** The learning curve
  shows the problem is feature-grade mismatch, not sample size.

### Technical Model Verdict

**Below 0.3 on any trustworthy metric → Diagnose.**
Diagnosis: feature-grade mismatch. Features describe intended
content; grades measure rendered output. Root cause: }{ rendering
bug in ~50% of spreads.

### Style Model Verdict

**Weaker than technical, as expected.** But the weakness isn't
about missing style features — it's the same rendering-bug
mismatch, compounded by lower grade variance.

### Missing Feature Candidates

For when grades measure actual layout quality (post-bug-fix):

1. **d2_orphans / d2_widows** — already in the schema as NA.
   Neville grades these harshly. High causal link to technical
   grade.
2. **d4_col_balance** — "the single most visible technical defect"
   per Neville. Already in schema as NA.
3. **d6_dead_space** — trapped white space. Already in schema as NA.
4. **Facing-page template contrast** — Neville's highest style
   scores go to spreads with different templates on facing pages
   (e.g., s-078-007: feature article vs briefs page). A binary or
   categorical feature for "do the two facing pages use different
   primary templates" would capture this.
5. **Has-image binary** — even a 0/1 "does this spread have any
   image" would help when the image pipeline is working.
6. **Content-rendered binary** — for now, a simple "did both pages
   actually render content" would explain most of the grade
   variance. This is a debugging signal, not a layout quality
   feature.

### Sample Size Assessment

**Do not grade 1,496 spreads on the current pipeline.** The
grading effort would produce ~750 spreads graded 2 (broken) and
~750 spreads graded 5–6 (rendered but monotonous). This is
expensive noise.

After the rendering bug is fixed:
- Start with **100 re-rendered spreads** (new batch, fresh grades)
- If R² > 0.3 on technical with metadata features alone: proceed
  to 500
- If R² > 0.5 with rendered-output features (d2, d4, d6): the
  objective function works — scale to full corpus
- If R² < 0.3 even with rendered features: pause and diagnose with
  Neville what dimensions are still missing

500 spreads is likely sufficient if the signal is real. 1,496
would reduce variance but wouldn't fix a feature gap.

### Recommendation

**Pause grading. Fix the rendering pipeline. Then:**

1. **Fix the }{ rendering bug** — highest priority, blocks all
   downstream work
2. **Fix HTML/CSS attribute leakage** — raw markup in text
3. **Verify image pipeline** — only 1 image in 50 spreads
4. **Re-render 100 spreads** from diverse editions
5. **Compute rendered-output features** (d2, d4, d6 — currently
   NA) from the re-rendered pages
6. **Have Neville re-grade** the 100 re-rendered spreads
7. **Re-run this analysis** — if metadata features alone hit R² >
   0.3 or metadata + rendered features hit R² > 0.5, commit to
   scaling the grading effort

**This is not a setback — it's the system working as designed.**
The preliminary fitting exists precisely to catch this kind of
mismatch before investing in 1,496 grades. The go/no-go gate did
its job: no-go on grading, go on fixing the rendering pipeline.

---

## Artifacts

| File | Description |
|------|-------------|
| `fit_preliminary_models.py` | Full analysis code (reproducible) |
| `results_summary.json` | Numeric results for all models |
| `pred_vs_actual.png` | Scatter plots: predicted vs actual grades |
| `feature_correlation.png` | Feature correlation heatmap |
| `preliminary-report.md` | This report |

---

*Ada — Algorithm Architect & Reviewer, IRAS*
