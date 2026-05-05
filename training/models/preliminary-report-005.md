# Visual-Hierarchy Feature Re-run — Batch-005 (n=200, features-006)

**Task:** #378 (visual-hierarchy features + re-run gate; follow-up of #344 / #346)
**Date:** 2026-05-05
**Author:** Belle
**Status:** **NO-GO for ceiling lift.** Tech holds at QUALIFIED GO; style remains NO-GO.

---

## Executive Summary

The expected uplift from visual-hierarchy features did not
materialize. With 14 new features added on top of the 18-feature
structural baseline (32 raw → 29 after redundancy reduction), the
5-fold CV R² is **statistically indistinguishable from #344**:

| Model | #344 (structural) | **#378 (+ visual)** | Δ |
|---|---|---|---|
| Tech Ridge α=0.1, 5-fold CV | 0.498 ± 0.114 | **0.493 ± 0.178** | −0.005 |
| Style Ridge α=0.1, 5-fold CV | 0.281 ± 0.167 | **0.267 ± 0.191** | −0.014 |
| Tech bootstrap median (α=0.1) | 0.499 | **0.472** | −0.027 |
| Style bootstrap median (α=0.1) | 0.274 | **0.231** | −0.043 |

Both numbers regressed slightly — by less than one standard
deviation, well within sampling noise. **No lift, in either
direction.** Ada's projected uplift (tech 0.50 → 0.55+, style
0.28 → 0.45+) is not supported by the data.

The diagnosis is straightforward and useful: most of the new
visual features correlate with `d5_fill_fraction` and `is_solo`
(|r| > 0.7 for the strongest ones). They duplicate the dominant
structural axis rather than complement it. A targeted ablation
isolates **5 ``orthogonal'' visual features** (those with |r| < 0.2
to fill and < 0.4 to solo) that produce a microscopic uplift
(+0.003 tech, +0.002 style) — within noise. The remaining 9
``aligned'' visual features actively *hurt* CV when added
(struct + aligned: tech 0.463, style 0.241) by introducing
multicollinearity at this sample size.

**Verdicts:**

1. **Technical: still QUALIFIED GO.** The structural model
   from #344 is the artifact to deploy. Visual features
   should not be added to the deployed model.
2. **Style: still NO-GO.** No movement from 0.28 floor. The
   contingency from #344 (visual-hierarchy features) has been
   tested and rejected as the path forward. A different
   intervention is required.
3. **Hypothesis revision needed.** The hypothesis that ``visual
   hierarchy is the missing axis'' is not refuted, but the
   hypothesis that ``Typst-PDF text-span statistics measure
   visual hierarchy'' is. The corpus's typographic uniformity
   (one body font, one headline family, four distinct sizes
   total) leaves visual-hierarchy features confounded with
   spread density.

---

## 1. What was added

`compute_features_006.py` adds 14 visual-hierarchy features
computed from the rendered PDF text spans:

| Group | Features |
|---|---|
| Type-size variation | `h_distinct_font_sizes`, `h_size_std_chars`, `h_max_size_to_body` |
| Weight / family | `h_distinct_weights`, `h_bold_char_frac`, `h_italic_char_frac`, `h_sans_char_frac` |
| Element-scale | `h_block_area_max_to_median`, `h_block_area_cv` |
| White-space rhythm | `h_gap_cv`, `h_max_gap_to_median` |
| Headlines / pull quotes | `h_headline_count`, `h_headline_area_frac`, `h_pull_quote_count` |

**Health:** all 14 features are non-NA across 200 spreads.
`h_distinct_font_sizes` ranges 2-6, `h_distinct_weights` 2-6,
`h_headline_count` 1-8 — non-degenerate distributions. No NA
or zero-variance issues. Schema is a clean superset of
features-005.

The 80/20 split is identical to features-004/005 (seed 42 over
lex-sorted IDs) so the gate comparison is apples-to-apples.

---

## 2. Headline numbers (n=200, features-006)

### Technical model — alpha sweep

| Alpha | Train R² | Val R² (single split, n_val=40) |
|---|---|---|
| 0.01 | 0.709 | 0.453 |
| **0.1** | **0.707** | **0.453** |
| 1.0 | 0.682 | 0.428 |
| 10.0 | 0.558 | 0.339 |

Optimum still at α=0.1 (consistent with #344). Train R²
increased from 0.647 (#344) to 0.707 — the model is overfitting
*more* with the extra features, while validation is flat.

### Technical model — primary metrics

| Model | Train R² | Val R² | Train MAE | Val MAE |
|---|---|---|---|---|
| Ridge (α=0.1) | 0.707 | **0.453** | 0.76 | 1.00 |
| Ridge (α=10)  | 0.558 | 0.339 | 1.04 | 1.34 |
| GBT (depth=2, n=50) | 0.711 | 0.415 | 0.81 | 1.17 |

**5-fold CV at n=200:**
- Ridge α=0.1: **0.493 ± 0.178**
- Ridge α=10:  0.371 ± 0.196

**Bootstrap 1000 iters (α=0.1):** median 0.472, 95% CI [0.201, 0.643].

### Style model

| Alpha | Train R² | Val R² |
|---|---|---|
| 0.01 | 0.554 | 0.233 |
| **0.1** | **0.553** | **0.218** |
| 1.0 | 0.536 | 0.166 |
| 10.0 | 0.428 | 0.089 |

| Model | Train R² | Val R² | Val MAE |
|---|---|---|---|
| Ridge (α=0.1) | 0.553 | **0.218** | 0.87 |
| Ridge (α=10)  | 0.428 | 0.089 | 1.00 |
| GBT (depth=2, n=50) | 0.590 | 0.164 | 0.98 |

**5-fold CV:** Ridge α=0.1: **0.267 ± 0.191**.
**Bootstrap median (α=0.1):** 0.231, 95% CI [−0.137, 0.445].

The style bootstrap CI now includes negative values — wider than
#344's [+0.025, +0.437] — which is what we expect when adding
features that don't pay for themselves at fixed n.

---

## 3. Three-way comparison: #330 / #342 / #344 / #378

### Technical proficiency

| Metric | #330 (n=50) | #342 (n=150) | #344 (n=200) | **#378 (+visual)** |
|---|---|---|---|---|
| Ridge α=0.1 Val R² (single split) | n/a | 0.48 | 0.41 | **0.45** |
| 5-fold CV Val R² (α=0.1) | n/a | 0.51 | 0.498 | **0.493** |
| Bootstrap median R² (α=0.1) | n/a | n/a | 0.499 | **0.472** |
| Bootstrap 95% CI (α=0.1) | n/a | n/a | [0.31, 0.65] | **[0.20, 0.64]** |
| GBT Val R² | 0.18 | 0.34 | 0.40 | **0.41** |
| Bimodal sep (fill_frac t) | 5.63 | 10.29 | 11.75 | 11.75 |

**Reading:** single-split val R² nudged up (+0.04), but the
5-fold CV — the more reliable estimator — moved down (−0.005).
Both moves are within the standard deviations of the estimators.
The honest read is "flat at the structural ceiling."

### Style

| Metric | #330 (n=50) | #342 (n=150) | #344 (n=200) | **#378 (+visual)** |
|---|---|---|---|---|
| Ridge α=0.1 Val R² | n/a | 0.27 | 0.26 | **0.22** |
| 5-fold CV (α=0.1) | n/a | 0.26 | 0.281 | **0.267** |
| Bootstrap median (α=0.1) | n/a | n/a | 0.274 | **0.231** |
| Bootstrap 95% CI | n/a | n/a | [0.025, 0.437] | **[−0.137, 0.445]** |

Style is still stuck. The bootstrap lower bound went *negative*
again — visually, the cloud of bootstrap R²s expanded without
shifting upward.

---

## 4. Bimodal separation — visual features ARE separating, but on the same axis

The new visual features individually achieve strong bimodal
separation (T≥5 vs T≤3). Top 12 by |t-stat|:

| Feature | High mean | Low mean | Diff | t-stat | p |
|---|---|---|---|---|---|
| d5_fill_fraction | 0.603 | 0.270 | +0.333 | **+11.75** | 2.7e-21 |
| h_bold_char_frac | 0.025 | 0.194 | −0.169 | **−6.68** | 5.2e-9 |
| h_sans_char_frac | 0.034 | 0.381 | −0.347 | **−6.18** | 4.3e-8 |
| h_size_std_chars | 0.755 | 6.403 | −5.648 | **−5.90** | 1.3e-7 |
| h_headline_area_frac | 0.043 | 0.343 | −0.300 | **−5.83** | 1.7e-7 |
| d2_orphans | 0.078 | 0.500 | −0.422 | −5.74 | 1.2e-7 |
| is_solo | 0.078 | 0.368 | −0.289 | −4.52 | 1.9e-5 |
| h_block_area_cv | 0.911 | 1.072 | −0.161 | −3.65 | 4.1e-4 |
| h_block_area_max_to_median | 6.084 | 9.319 | −3.236 | −3.63 | 4.2e-4 |
| h_headline_count | 2.583 | 2.000 | +0.583 | **+3.48** | 6.3e-4 |
| d6_dead_space | 0.014 | 0.035 | −0.021 | −3.13 | 0.002 |
| h_distinct_font_sizes | 3.209 | 3.529 | −0.321 | −3.04 | 0.003 |

Eight visual features clear p<0.001. So the features *do*
separate the bimodal — the problem is they don't add
*orthogonal* signal.

### Why no R² lift then?

**Multicollinearity with `d5_fill_fraction` and `is_solo`:**

| Visual feature | r(d5_fill_fraction) | r(is_solo) |
|---|---|---|
| h_bold_char_frac | **−0.726** | **+0.778** |
| h_sans_char_frac | **−0.718** | **+0.795** |
| h_size_std_chars | **−0.713** | **+0.815** |
| h_headline_area_frac | **−0.732** | **+0.835** |
| h_block_area_cv | −0.505 | +0.473 |
| h_block_area_max_to_median | −0.483 | +0.515 |
| h_distinct_font_sizes | −0.415 | +0.484 |
| h_gap_cv | −0.284 | +0.164 |
| h_italic_char_frac | −0.207 | +0.158 |
| h_max_size_to_body | +0.134 | −0.159 |
| h_distinct_weights | +0.075 | −0.331 |
| h_pull_quote_count | −0.085 | −0.145 |
| h_headline_count | +0.074 | −0.109 |
| h_max_gap_to_median | −0.070 | −0.058 |

The four most-separating visual features (`h_bold_char_frac`,
`h_sans_char_frac`, `h_size_std_chars`, `h_headline_area_frac`)
are essentially restatements of "this spread has unusually little
body text relative to headline/bold material" — which is what
**low fill** and **solo pages** already encode. They re-detect
the same thing through a different sensor.

The mechanism is corpus-driven: when `d5_fill_fraction` is low,
absolute character count is low, so headline/bold characters
(which are present on every page as section labels and item
titles) make up a larger *fraction* of total characters. The
visual features end up being noisy proxies for "low fill" rather
than independent measurements of "good or bad hierarchy."

---

## 5. Ablation — orthogonal vs aligned visual features

| Feature subset | n_features | Tech CV | Style CV |
|---|---|---|---|
| Structural-only baseline | 18 | **0.498 ± 0.114** | **0.281 ± 0.167** |
| Struct + 5 orthogonal h_* | 23 | 0.501 ± 0.144 | 0.283 ± 0.170 |
| Struct + 9 aligned h_* | 27 | 0.463 ± 0.186 | 0.241 ± 0.214 |
| Struct + ALL 14 h_* | 29 | 0.493 ± 0.178 | 0.267 ± 0.191 |
| 5 orthogonal h_* alone | 5 | 0.067 ± 0.123 | 0.098 ± 0.097 |
| 9 aligned h_* alone | 9 | 0.297 ± 0.179 | 0.113 ± 0.102 |

(Orthogonal subset = `{h_max_size_to_body, h_distinct_weights,
h_max_gap_to_median, h_headline_count, h_pull_quote_count}`,
defined as |r(d5_fill_fraction)| < 0.2 and |r(is_solo)| < 0.4.)

**The orthogonal subset adds essentially nothing** (+0.003 tech,
+0.002 style) — within sampling noise. The aligned subset
*hurts* (−0.035 tech, −0.040 style) because the model can't
disentangle them from the existing fill/solo features.

In retrospect this makes the negative result less surprising: the
tightly-fill-correlated features were likely to add noise rather
than signal once the model already had `d5_fill_fraction`. The
orthogonal subset *is* genuinely new information — it just
isn't enough information to move the ceiling.

---

## 6. Why didn't this work?

Three plausible explanations, in decreasing order of how
confidently the data supports each:

### 1. The corpus is typographically homogeneous (well supported)

The corpus uses a fixed font system: SourceSerif4 (body, italics)
and SourceSans3 (bold headlines). Across the 200 graded spreads,
the union of font sizes is **{7, 8, 9, 10, 12, 14, 48}** — and
the 48pt size only appears on edition page 1 (the nameplate).
That leaves *six* sizes the layout system varies across, and most
of them are minor (caption, byline, sub-headline). **There is no
real headline-to-body type-size choice happening in this corpus** —
all headlines are 14pt SourceSans3-Bold or 14pt SourceSerif4-It
(pull quotes), all body is 10pt SourceSerif4-Regular.

Without variation in the underlying typography, "type-size
variation" features have no ground to measure against. They end
up encoding density artifacts (how much headline-to-body ratio
the page happens to have given its content), not hierarchy
choices.

This is the explanation I have most confidence in. It also
suggests an experiment: regenerate a slice of the corpus with
deliberate type-size variation (one font size up for some
headlines, larger pull quotes, etc.) and see whether the visual
features then have signal to learn from.

### 2. Visual hierarchy ≠ what graders are penalizing on style (medium support)

Looking at the lowest-style spreads in the corpus, they tend to
be either: (a) front pages with the giant nameplate and minimal
content (the fill problem), or (b) ``deal list'' terminal pages
with all-bold sans-serif content (which inflates the visual
features but is graded low for being *too* hierarchy-loud, not
too flat). The style scale isn't measuring "lack of visual
hierarchy" — it's measuring "compositional intentionality,"
which is harder to instrument.

Neville's grading criteria specifically call out things like
"reading path with a satisfying arc," "facing-page conversation,"
"productive friction" — these are perceptual gestalt judgments,
not statistics over text spans.

### 3. Pixel-level signal is needed, not text-span statistics (lower support, but worth testing)

Text-span features measure typography *as-coded*, not typography
*as-perceived*. Two spreads with identical font/size choices
can read very differently if one has tighter line lengths,
better paragraph balance, or more breathing room around the
anchor element. The current features can't see any of that.

A future experiment would rasterize each PDF page and compute
features from the image: visual mass distribution, eye-tracking
saliency proxies, white-space "breathability" via local
gradient. This is a heavier engineering lift but adds a layer
of measurement that text-span statistics structurally cannot.

---

## 7. What changed between #344 and #378

Compared to #344's deployment recommendation, three things have
moved:

1. **The "QUALIFIED GO" verdict for tech still stands**, but it
   stands at the same R² ≈ 0.50 ceiling — slightly down from
   bootstrap median 0.499 to 0.472. We now have stronger
   evidence the ceiling is at 0.50, not 0.55+, because we tried
   the obvious next-feature-set and got nothing.
2. **The "NO-GO" verdict for style stands**, and we now know
   one specific path (typography-from-PDF features) does NOT
   unlock it. This is useful negative information — it deletes
   one branch of the search tree.
3. **Ada's specific uplift estimate was wrong.** Her #344 report
   projected tech 0.55+, style 0.45+ from visual features.
   That projection assumed visual hierarchy provides orthogonal
   information. In this corpus, on these features, it does not.

This is not a critique of Ada — it's the kind of empirical
revision that data is for. The hypothesis was reasonable;
the data refuted it.

---

## 8. Highest-residual spreads (Val Set) — same pattern as #344

### Technical (GBT)

| Spread | Actual | Predicted | Residual | Pattern |
|---|---|---|---|---|
| s-094-003 | 2 | 5.40 | 3.40 | Over-grades a 2 (still — #344 had 3.37 here) |
| s-088-008 | 7 | 4.05 | 2.95 | Under-grades a 7 (similar to #344) |
| s-051-005 | 3 | 5.89 | 2.89 | Over-grade |
| s-096-014 | 1 | 3.78 | 2.78 | Under-detects a 1 |
| s-079-015 | 3 | 5.73 | 2.73 | Over-grade |
| s-002-004 | 3 | 5.55 | 2.55 | Over-grade |
| s-057-012 | 6 | 3.82 | 2.18 | Under-grades a 6 |
| s-084-006 | 7 | 4.90 | 2.10 | Under-grades a 7 |

The same regression-to-the-mean pattern from #344: the model
under-predicts 7s (~5) and over-predicts 1-3s (~4-5). Visual
features did not help with the extremes — they just pulled them
slightly tighter to the mean. The honest read: the **model still
cannot tell "excellent" from "decent"**, and adding the new
features did not give it that ability.

### Style (GBT)

The top-residual spreads on style are the same ones on
technical — confirming that style and technical share the same
unresolved residual structure.

---

## 9. Verdict and recommendations

### GO/NO-GO

| Track | #344 verdict | **#378 verdict** | Movement |
|---|---|---|---|
| Technical | QUALIFIED GO | **QUALIFIED GO (no change)** | Flat at R²≈0.50 |
| Style | NO-GO + diagnose | **NO-GO + revised diagnosis** | Tested, refuted |

### Do now

1. **Ship the #344 technical scorer** — Ridge(α=0.1) on the 18
   structural features. Do not add visual features to the
   deployed model. The ablation is conclusive: aligned
   features actively hurt CV.
2. **Stop pursuing PDF-text-span visual features for the
   style model.** This branch of the search tree is closed.
3. **Park the style scorer.** The style model is not deployable
   and we now have evidence the obvious feature lift does not
   work in this corpus.

### Recommended next steps (in priority order)

**Highest priority — corpus diversity experiment (~1 sprint, low engineering cost):**

Generate a controlled slice of new editions with deliberately
varied typography:
- Headline sizes drawn from {14, 16, 18, 22} pt (currently fixed at 14)
- A "feature" variant with 18pt SourceSans3-Black for the lead item
- Pull quotes at {12, 14, 18} pt (currently fixed at 14)
- Optional drop-cap on lead items

Render, grade ~30 of these spreads, and retrain. If the visual
features now show R² lift, the diagnosis is confirmed (#1 from
§6) and the path forward is *generate first, then measure*.

**If that fails — pixel-level visual features (~2-3 sprints, medium cost):**

Rasterize each page at 150 DPI and compute:
- Visual-mass center of gravity (per page and per spread)
- Visual-mass *direction* — what fraction of mass is in each
  third of the page (eye-entry analysis)
- Local-gradient "breathability" — variance of pixel intensity in
  100x100 patches; high in dense text zones, low in white space
- Saliency-proxy: dilated stroke density (rough analogue of eye-
  tracking saliency without a vision model)

Two engineering decisions for that work:
- Use Pillow + numpy for the pixel work — no need for PyMuPDF's
  vector path, just rasterize the PDF pages once and cache.
- Cache rasterized PNGs alongside the PDFs in
  `editions/<id>/raster/` so feature recomputation doesn't need
  to re-rasterize.

**If both fail — accept the ceiling:**

Ship the structural model with the quality-of-life improvements
from #344's §9 (hard-constraint check, then model rank,
deterministic tie-break) and treat style as a non-modeled
dimension. The optimizer ranks on technical; humans approve on
style. This is acceptable for the editorial-print use case
because a human is in the loop for layout review.

### What NOT to do

- **Do not collect more graded spreads.** #344 already
  established the ceiling is feature-driven, not data-driven.
  This run reinforces that. More grades won't help.
- **Do not try other text-span feature variations.** We have
  surveyed the typography universe — the corpus has limited
  variation to learn from. The bottleneck is corpus diversity
  or measurement modality, not feature engineering on the
  current corpus.
- **Do not deploy the style model in any form.** The style
  ceiling has now been measured at three feature-set
  generations and held flat at ≈0.27.

---

## 10. Sequenced follow-on briefs

In order:

1. **#379 (Belle + Neville):** Corpus typographic-diversity
   experiment. Generate 30 spreads with controlled variation
   in headline sizes, pull-quote sizes, and lead-item
   treatments. Re-grade. Re-fit on features-006. Re-evaluate.
   Expected outcome: either visual features now show signal
   (path forward is template diversity), or they don't (move
   to #380).
2. **#380 (Belle, conditional on #379 negative):** Pixel-level
   feature pipeline. Add raster generation step to the
   training data pipeline; implement visual-mass and
   breathability features; re-fit.
3. **#381 (Belle):** Wire technical_scorer (Ridge α=0.1, 18
   features) into the layout optimizer per #344's §9 plan.
   This is independent of #379/#380 and can run in parallel.

---

## 11. Artifacts

| File | Description |
|---|---|
| `/home/modus/offscroll/training/features/compute_features_006.py` | Feature computation (18 prior + 14 visual-hierarchy) |
| `/home/modus/offscroll/training/features/features-006.csv` | Computed feature matrix (n=200) |
| `/home/modus/offscroll/training/models/fit_batch005.py` | Analysis script (this run) |
| `/home/modus/offscroll/training/models/results_batch005.json` | Numeric results |
| `/home/modus/offscroll/training/models/pred_vs_actual_005.png` | Predicted vs actual (4 panels) |
| `/home/modus/offscroll/training/models/feature_correlation_005.png` | Correlation matrix (reduced features) |

---

## Appendix A: New feature definitions

| Feature | Definition |
|---|---|
| `h_distinct_font_sizes` | Count of distinct font sizes on the spread (rounded to 0.1pt) |
| `h_size_std_chars` | Character-weighted std of font sizes across all spans |
| `h_max_size_to_body` | Max non-nameplate (size < 36pt) font size / 10pt body baseline |
| `h_distinct_weights` | Count of distinct (family, bold, italic) combinations |
| `h_bold_char_frac` | Fraction of characters in Bold or Black weights |
| `h_italic_char_frac` | Fraction of characters in italic styles |
| `h_sans_char_frac` | Fraction of characters in sans-serif family |
| `h_block_area_max_to_median` | Largest text-block area / median text-block area |
| `h_block_area_cv` | Coefficient of variation of text-block areas |
| `h_gap_cv` | Coefficient of variation of inter-block gaps within columns |
| `h_max_gap_to_median` | Max inter-block gap / median inter-block gap |
| `h_headline_count` | Count of blocks classified as "headline" (sans-serif, ≥12pt) |
| `h_headline_area_frac` | Total headline-block area / total block area |
| `h_pull_quote_count` | Count of blocks classified as "pull quote" (italic-serif, ≥12pt) |

Pull-quote and headline classification are mutually exclusive
(pull-quote takes precedence) and computed per block based on the
union of spans inside the block, not the dominant span.

## Appendix B: Per-batch row counts

| Batch | Rows |
|---|---|
| batch-002 | 50 |
| batch-003 | 50 |
| batch-004 | 50 |
| batch-005 | 50 |
| **Total** | **200** |

Train/val: 160 / 40, identical seed-42 split as features-005.

---

*Belle — Systems Architect & Integration Lead, IRAS*

COMPLETED
