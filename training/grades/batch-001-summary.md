# Batch 1 Grading — 50 Spreads

**Grader:** Neville (Layout and Publishing Expert)
**Date:** 2026-04-06
**Task:** #253 — OffScroll re-grade spreads on clean training set (batch 1)
**Status:** COMPLETED with caveats — grading done, but rendering
bug persists (see below)

---

## Critical Finding: `}{` Rendering Bug Persists

The re-render from #244 did NOT fix the `}{` bug. Of the 50
spreads graded, **26 (52%) have one `}{`-bugged page.** These are
the same blank pages (12-13KB PNGs showing only `}` and `{`
characters) seen in the original batch 1 (#237).

Additionally, 4 "CLEAN" spreads (above the 20KB threshold)
exhibit partial `}{` corruption — pull quotes render correctly
but are followed by `}{` and dead space, leaving the page
~85-90% empty (spreads 22, 27, 36; spread 43 has an empty
BRIEFS section). These are **not caught by the file-size
threshold** but represent the same underlying bug.

I graded all 50 spreads rather than blocking. The content pages
where rendering succeeds show genuine layout quality, and this
data is useful for the optimization pipeline even if downstream
consumers need to filter by bug status.

---

## Score Distributions

### Technical Proficiency (Scale A)

| Score | Count | Pct  |
|-------|-------|------|
| 3     | 4     | 8%   |
| 4     | 4     | 8%   |
| 5     | 15    | 30%  |
| 6     | 20    | 40%  |
| 7     | 7     | 14%  |

- **Median: 6** (Good)
- **Mean: 5.44**
- **Range: 3-7**

### Style (Scale B)

| Score | Count | Pct  |
|-------|-------|------|
| 1     | 2     | 4%   |
| 2     | 13    | 26%  |
| 3     | 17    | 34%  |
| 4     | 11    | 22%  |
| 5     | 5     | 10%  |
| 6     | 2     | 4%   |

- **Median: 3** (Flat)
- **Mean: 3.20**
- **Range: 1-6**

---

## Comparison to Old Batch 1 (#237)

The old batch 1 scored a median of 2/10 because ~50% of pages
were blank from the `}{` bug. That data was garbage — it measured
rendering failures, not layout quality.

This batch grades the *content that actually renders*, producing
meaningful signal:

- **Technical median 6 vs old ~2**: The layout system produces
  competent pages when rendering succeeds. Column balance, fill,
  and spacing are adequate to good.
- **Style median 3 vs old ~2**: Style is low but for real reasons
  (monotonous content, no images, no template variety) rather
  than blank pages.

The improvement confirms that the `}{` bug was the dominant
quality factor in the old batch, as expected.

---

## Calibration Observations

### What drives high technical scores (6-7)

- Two-column layouts with balanced column fill
- Multiple articles per page with consistent inter-item spacing
- Front pages with clear masthead hierarchy
- Page fill above 85%
- The 7 spreads scoring technical 7 are: s-074-004 (has an
  IMAGE), s-051-031, s-020-016, s-068-012, s-078-007, s-041-001,
  s-020-019

### What drives low technical scores (3-4)

- Pull quote pages with `}{` trailing — massive dead space
  (spreads 22, 27, 36)
- Pages with inline `}{` artifacts and broken section rendering
  (spread 15, 43)
- Raw data dumps without editorial formatting (spread 10)
- Repeated identical content — podcast boilerplate (spread 41)

### What drives high style scores (5-6)

- **Images**: Only 1 spread (s-074-004) had a real image. It
  scored style 5. Images are the single biggest style lever and
  they are almost entirely absent.
- **Pull quotes**: When properly placed (s-020-019), pull quotes
  anchor the page and create rhythm. Style 6.
- **Mixed content types**: Spreads with podcast briefs, book
  reviews, and articles (s-030-006) score higher than single
  long-form articles.
- **Q&A format**: Interview-style content (s-086-003) creates
  natural rhythm from speaker alternation. Style 5.
- **Front pages**: Mastheads with bold headlines (s-041-001)
  create the "pick up and read" response. Style 6.

### What drives low style scores (1-2)

- **Single long-form article gray walls**: The most common
  pattern. One article fills the page with body text and nothing
  else. No anchor element, no visual interruption, no rhythm.
  13 spreads scored style 2.
- **Content quality**: Some pages render raw data (real estate
  listings), repeated boilerplate (podcast episodes), or
  metadata as if it were editorial content.
- **Identical facing pages**: Many clean spreads have two pages
  that look interchangeable — same density, same treatment,
  no compositional dialogue.

---

## Feature Engineering Observations

Things I responded to that may not be in the current feature set:

1. **Inline rendering artifacts.** `class=""`, `id=""`,
   `style=""`, `[`, `]` HTML/markup tags visible in body text.
   A content sanitization quality signal, not a layout signal,
   but it degrades perceived quality.

2. **Content repetition.** Multiple identical or near-identical
   items (repeated podcast descriptions). This is a curation
   failure, not a layout failure, but it produces monotonous
   pages that score low on S1 (visual rhythm).

3. **Pull quote isolation.** Pull quotes rendering correctly but
   followed by `}{` and dead space. The pull quote itself is
   well-formatted but the page is 85% empty. This is a distinct
   failure mode from the full-page `}{` blank.

4. **Single-article vs multi-article pages.** Pages with
   multiple articles consistently score higher on style because
   headline repetition creates natural rhythm. The number of
   distinct items per page is a strong style predictor.

5. **Q&A and list-format content.** Content with inherent
   structural formatting (interviews, bullet lists, recipes)
   scores higher on style than equivalent-length prose. Content
   structure type is a feature worth extracting.

6. **Front page hierarchy.** The presence and visual weight of
   the masthead relative to the content strongly predicts style
   on page 1. Mastheads are present and correctly formatted;
   the question is whether they *command*.

---

## Recommendations

1. **Fix the `}{` bug for real.** 26/50 spreads are contaminated.
   The previous attempt (#243/#244) did not work. The bug appears
   to correlate with content items that contain certain template
   delimiters — it is not random. The pull-quote variant (where
   the pull quote renders but everything after it is `}{`) may
   be a different code path from the full-page blank variant.

2. **Add images to the training set.** The single spread with
   an image (s-074-004, Bihar map) scored dramatically higher
   on style. D3 (image ratio) is near-zero across almost all
   pages. The image pipeline appears non-functional for training
   editions.

3. **Increase template variety.** Almost every page is a single
   or double long-form article. Template-type entropy is near
   zero. The feature and brief templates appear unused.

4. **Filter before training.** If using this batch for the
   optimization model, filter to the 20 fully clean spreads
   (excluding spreads with any `}{` pages or pull-quote-then-
   blank patterns). The MIXED spreads grade the content page's
   own quality but lack spread context, making them less reliable
   as spread-level training data.

---

*Neville — Layout and Publishing Expert, IRAS*
