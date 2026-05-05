# technical_scorer.py — validation results

**Brief:** #379 — OffScroll: integrate technical scorer into layout optimizer
**Author:** Belle
**Date:** 2026-05-05
**Run by:** `/home/modus/offscroll/.venv/bin/python -m scoring.test_scorer`

---

## 1. Numerical regression

The trained model coefficients embedded in `technical_scorer.py`
reproduce a freshly-fit `Ridge(alpha=0.1)` on the n=200 training
matrix to within **1.2e-14** absolute error. No drift between the
constants and re-trained sklearn output.

## 2. Held-out performance (matches #344)

Running the constants-only scorer over the held-out validation
split of features-004.csv:

| Metric | scorer | #344 reported | Δ |
|--------|--------|---------------|---|
| Val R² | 0.4128 | 0.4128 | 0.0000 |
| Val MAE | 1.1364 | 1.1364 | 0.0000 |

Confirms the scorer is bit-equivalent to the EXP-004 deployed
checkpoint.

## 3. Score distribution (200 spreads)

```
min      p10      median   p90      max
1.62     2.39     5.11     6.47     6.93
```

Range: 5.31 grade points across the corpus. Median 5.11 is
slightly above the corpus mean technical grade (~4.5) — model
exhibits a mild positive bias in this distribution. Worth
tracking in production: a 20-edition rolling median that drifts
below 4.6 or above 5.6 is the trigger to check feature
distribution drift.

## 4. Model-vs-rule top-1 disagreement

Treating each multi-spread training edition as a candidate set
and ranking spreads by (a) model score, (b) pure rule score:

- 72 editions had ≥2 spreads available.
- Top-1 disagreed on **14/72 (19%)** of editions.
- On those 14 disagreements, the model's pick had a higher
  actual grade than the rule's pick **8 times**, lower **2
  times**, and tied **4 times**.
- Mean grade lift when they disagree: **+0.86 grade points** in
  favor of the model.

This is the headline finding for "is hybrid scoring better than
pure-rule." When the rules and the model agree (81% of the time),
the choice is unambiguous. When they disagree, the model is right
about 4× more often than the rules. That is the value-add we are
buying with the hybrid design.

The 19% disagreement rate is also a healthy signal: it confirms
the model is contributing rankings the rules would not have
produced. A near-zero disagreement rate would mean the model is
echoing the rules, which would mean it's not adding signal.

## 5. Hard-constraint filtering

- Spreads with `d2_orphans > 0`: 46/200 (23%).
- Spreads with `d2_widows > 0`: 0/200 — confirms the broken-
  feature state described in EXP-004 (now fixed via brief #377;
  retraining will activate the widow signal).
- Total spreads rejected by hard constraints: 46/200 (23%).

The 23% orphan-rejection rate on training-style candidates is
high but expected — the training corpus was deliberately
generated to span the quality range, so a quarter of spreads
have visible orphans. In production candidate generators
(sequential greedy + height estimator) the rejection rate
should be much lower; if it's not, the candidate generator has
regressed.

## 6. `select_best_candidate` demo

Exercised on training-060 (4 spreads) as a synthetic candidate
set:

- 4 candidates seen, 3 passed hard constraints (1 orphan
  rejection).
- Winner: `s-060-009` — actual grade 6 (high), model score 5.72,
  matched both the model top-1 and the rule top-1 (no
  disagreement on this edition).
- `ScoringMonitor.summary()` returned the expected JSON shape
  with score-min/median/max, rejections-by-rule, and
  disagreement record.

---

## What this exercises end-to-end

1. ✅ Constants ship as 18 floats + 1 intercept, no joblib pickle.
2. ✅ `score_spread(features: dict[str, float]) -> float` works
   from a plain dict — same feature schema as the training
   pipeline (compute_features_004.py).
3. ✅ Hard-constraint check rejects orphan-bearing candidates
   before the model sees them.
4. ✅ Hybrid ranker (model top-K + rule tiebreak) selects the
   best candidate or returns None if all are rejected.
5. ✅ Monitoring hooks capture score distribution, rejection
   reasons, and model-vs-rule disagreement on every decision.
6. ✅ Empirical: when rule and model disagree, the model's pick
   has a +0.86 grade-point average lift on actual graded data.

## What this does NOT exercise

- **Live candidate generator integration.** The sequential
  greedy generator + height estimator (per
  `LAYOUT-OPT-ARCHITECTURE-REPORT.md`) is not yet built. Once
  it lands, `select_best_candidate` is the entry point — wiring
  is a few lines.
- **Image-bearing editions.** `est_image_count`,
  `d3_image_fraction` had zero variance in the training corpus
  and have zero coefficients. They will need to be re-validated
  after image content enters production.
- **Style scoring.** Per #344, the style model is NO-GO at
  current features. Not deployed. This module scores technical
  proficiency only.

## Follow-ups

Per Ada's recommendations in EXP-004 §10:

- After **#377** (d2_widows fix) lands and is re-validated, the
  scorer should be re-trained on a fresh feature matrix that
  includes non-zero widow data. The `d2_widows` coefficient will
  move off zero. Update RIDGE_COEFS and re-run this validator;
  the held-out R²/MAE may shift.
- After **#378** (visual-hierarchy features) lands, the feature
  set will expand. FEATURE_ORDER and RIDGE_COEFS will need to be
  regenerated. The scorer's contract (`dict[str, float]`) is
  stable across feature-set changes; only the constants update.
