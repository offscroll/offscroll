# Batch 1 Re-Grading — BLOCKED

**Grader:** Neville (Layout and Publishing Expert)
**Date:** 2026-04-03
**Task:** #253 — OffScroll re-grade spreads on clean training set (batch 1)
**Status:** BLOCKED — the `}{` rendering bug persists in re-rendered pages

---

## Summary

Task #253 assumed the `}{` rendering bug was fixed (#243) and all
100 editions re-rendered (#244) to produce a clean training set.
**This is not the case.** The re-rendered pages (timestamped
2026-04-01) still contain the `}{` bug at essentially the same
rate as the original batch 1 grading (#237).

I cannot grade this as a "clean training set" because it is not
clean. Grading broken renders as training data for the layout
optimization model would teach the model to predict rendering
failures, not layout quality.

---

## Evidence

### Full manifest analysis (1,496 spreads, 2,837 pages)

- **893 pages (31.5%) are broken** — nearly blank, showing only
  `}` and/or `{` characters at the top with the footer rendered
  correctly
- **92 out of 100 editions** have broken pages
- Broken rates per edition range from 30-44%
- No consistent even/odd page number pattern

### Batch 1 (first 50 spreads, 95 pages)

- **26 out of 50 spreads (52%)** have at least one broken page
- 26 broken pages out of 95 total
- Broken page rate consistent with the full manifest

### Detection method

Broken `}{` pages are consistently 12-13KB PNG files at
1275x1650 resolution. Normal content pages range from 30KB to
620KB. A threshold of 15KB cleanly separates the two populations
with zero ambiguous cases. I visually confirmed multiple pages
at both ends of this gap.

### Comparison to old batch 1 (#237)

The old batch 1 grading found 24/50 spreads broken (48%). This
re-render shows 26/50 broken (52%). The rates are statistically
identical. The "fix" did not change the rendered output.

---

## Observations on the ~24 clean spreads

While I did not complete formal grading (a partial batch would
produce miscalibrated training data), I visually inspected
several clean spreads. Qualitative observations:

### Genuine improvements visible (vs. what the bug obscured)

- **Two-column text layouts** render correctly with consistent
  column widths, margins, and gutters
- **Front pages** have proper mastheads with typographic weight
- **Section headers** (FEATURES, BRIEFS, ANALYSIS) render with
  horizontal rules — structurally correct
- **Pull quotes** are formatted with italic text and attribution
- **Drop caps** appear on some front pages

### Persistent quality issues (separate from the bug)

- **No images** on interior pages — the image pipeline appears
  non-functional. D3 (image ratio) will be near zero.
- **No template variety** within pages — every content page is
  a single long article in two columns. Template-type entropy
  will be near zero.
- **Visible HTML/CSS markup in text** — `class="pt"`,
  `class="highlight"`, `id=`, `style=` attributes rendered as
  visible text. Content sanitization is failing.
- **Monotonous spreads** — even clean spreads show two identical
  text walls with no compositional dialogue between facing pages

### Expected scores once rendering is fixed

Based on the clean spreads I examined, I estimate:
- Technical median: **5-6** (adequate to good)
- Style median: **3-4** (flat to tentative)
- Score ceiling: **6-7** (limited by lack of images and template
  variety, not by compositional errors)

This would represent a dramatic improvement over the old batch 1
median of 2, confirming that the `}{` bug was the dominant
quality factor — but only once the bug is actually fixed.

---

## What needs to happen

1. **Diagnose why #243 fix did not take effect.** The re-render
   was done 2026-04-01. Either the fix was not applied before
   rendering, the fix was incomplete, or it addresses a different
   variant of the `}{` problem.

2. **Automated verification.** Before the next re-render, add a
   post-render check: any page PNG under 15KB is broken. This is
   a zero-false-positive, zero-false-negative detector for this
   specific bug. The render pipeline should flag and report
   broken pages automatically.

3. **Re-render all 100 editions** with the actual fix in place,
   verified by the automated check.

4. **Then re-run this grading task** (#253) on the verified clean
   set.

---

*Neville — Layout and Publishing Expert, IRAS*
