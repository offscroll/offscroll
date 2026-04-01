# Batch 001 Grading Summary

**Grader:** Neville (Layout and Publishing Expert)
**Date:** 2026-04-01
**Spreads graded:** 50 (grading_index 1-50)
**Grading protocol:** LAYOUT-GRADING-PROTOCOL.md (blind grading, spread context)

---

## Score Distributions

### Technical Proficiency (Scale A)

| Score | Count | Percentage |
|-------|-------|------------|
| 1     | 3     | 6%         |
| 2     | 22    | 44%        |
| 3     | 7     | 14%        |
| 4     | 2     | 4%         |
| 5     | 7     | 14%        |
| 6     | 9     | 18%        |

**Mean: 3.0 | Median: 2 | Std Dev: 1.7**

### Style (Scale B)

| Score | Count | Percentage |
|-------|-------|------------|
| 1     | 4     | 8%         |
| 2     | 23    | 46%        |
| 3     | 13    | 26%        |
| 4     | 6     | 12%        |
| 5     | 2     | 4%         |
| 6     | 2     | 4%         |

**Mean: 2.6 | Median: 2 | Std Dev: 1.2**

---

## Critical Finding: The System Has Fundamental Problems

The grading protocol states: "The median score for a competent automated
layout should be 6-7. If the median is below 5, the system has
fundamental problems."

**Both medians are 2.** This is not a calibration issue on my part — the
layouts have a catastrophic, pervasive rendering bug that makes grading
the design system nearly impossible to separate from grading the
rendering pipeline.

---

## The Dominant Defect: The `}{` Empty Page Bug

**~50% of all pages in this batch are broken.** They render as nearly
empty pages containing only `}` and `{` characters (JSON/template
delimiters leaking through to the rendered output) plus the publication
footer. This is not a layout quality problem — it is a rendering pipeline
failure.

Of the 50 spreads graded:
- **24 spreads** (48%) have at least one completely empty page (}{ bug)
- **3 spreads** (6%) have BOTH pages empty — complete spread failures
- **4 spreads** have a pull quote rendered at the top of an otherwise empty
  page (the pull quote template works, but no content follows)
- Only **~18 spreads** (36%) have content on both pages

This bug must be fixed before grading produces useful training signal.
Every spread with an empty page scores technical 1-2 and style 1-2,
regardless of the quality of the content page. The grades are measuring
"does the page have content" rather than "is the composition good,"
which defeats the purpose of the learned objective function.

---

## Observations on Spreads Without the Bug

When both pages have content (roughly 18 spreads), the layouts show:

### What Works
- **Two-column text layout** is consistently executed. Column widths,
  margins, and gutters are uniform and readable.
- **Front pages** (4 solo front pages seen) have proper mastheads with
  strong typographic weight. "Morning Globe" (s-026-001) and "The Clear
  Courier" (s-041-001) are the best — score 6 on both scales.
- **Page footers** (publication name + date) are consistently placed.
- **Section headers** (FEATURES, BRIEFS, ANALYSIS) are visually distinct
  with horizontal rules.
- **Pull quotes** are formatted with italic text, attribution, and
  horizontal rules — typographically correct.
- **Drop caps** appear on some front pages (s-041-001) — a nice
  editorial detail.

### What Does Not Work
- **No images** on any interior spread. Only one page in 50 spreads had
  an embedded image (s-088-006, page 10). The image ratio feature (D3)
  is effectively zero across the batch.
- **No template variety within pages.** Almost every content page is a
  single long article in two columns. No mixing of features, standards,
  and briefs on the same page. The template-type entropy feature will
  be near zero.
- **Visible HTML/CSS markup in text.** Many pages render `class="pt"`,
  `class="highlight"`, `id=`, `style=` attributes as visible text.
  Content sanitization is failing.
- **Monotonous spreads.** Even when both pages have content, the spread
  typically shows two identical text walls. No anchor element, no
  visual hierarchy beyond headline sizes, no compositional dialogue
  between facing pages.
- **Dense text walls with no breathing room.** Interior pages are packed
  edge-to-edge with body text. No pull quotes break the rhythm on
  content pages (pull quotes only appear on otherwise-empty pages).
- **Column balance issues.** Spread s-087-011 confined all content to a
  single narrow right-side column, leaving most of both pages empty.
- **Raw data rendered as content.** Spread s-078-002 rendered raw website
  search statistics (city names with search counts) as if they were
  editorial content.

### The Best Spread in the Batch

**s-078-007** (spread 40): Left page has a feature-length article about
React in production. Right page has a brief-group section with multiple
short tech blog entries (Neon Tech Blog, Serverless Blog). This is the
only spread that shows template variety between facing pages — a
feature article facing a briefs page. Technical 6, Style 5. This is
what the system should be producing consistently.

---

## Calibration Notes

### First-Session Calibration
This is the first grading session. Per protocol, I should grade 30-40
pages to establish the anchor set. However, the overwhelming dominance
of the }{ bug means the score distribution is bimodal: scores cluster
at 1-2 (broken pages) and 5-6 (functional pages), with almost nothing
in between. The scale is not failing to discriminate — the layouts
genuinely fall into "broken" and "functional but unremarkable."

### Anchor Set Candidates
From this batch, I would nominate these as anchor candidates once the
rendering bug is fixed:
- **Bad:** s-086-019 (tech 1, style 1) — both pages empty
- **Below average:** s-087-011 (tech 3, style 2) — single-column layout failure
- **Adequate:** s-027-010 (tech 5, style 3) — both pages filled but monotonous
- **Good:** s-041-014 (tech 6, style 4) — two clean articles facing each other
- **Best available:** s-026-001 (tech 6, style 6) — strong front page

The anchor set should be rebuilt after the rendering bug is fixed, as
the quality distribution will shift dramatically.

### Scale Compression Warning
My grades cluster at 2 (broken) and 5-6 (functional). I have not
given any score above 6. This is appropriate for the current state of
the system — there are no layouts that reach "very good" (7) because
even the best spreads lack images, template variety, and compositional
sophistication. No score inflation is occurring.

---

## Recommendations Before Continuing Grading

1. **Fix the }{ rendering bug before batch 2.** This is the single
   highest-priority fix. It makes ~50% of all pages ungradeable for
   layout quality. Every grade assigned to a spread with an empty page
   is measuring "pipeline failure" not "design quality."

2. **Fix HTML/CSS attribute leakage in content text.** Multiple pages
   render raw HTML class attributes as visible text. This is a content
   sanitization issue.

3. **Investigate the single-column rendering bug** (s-087-011). Content
   rendering into one narrow column instead of two suggests a CSS or
   template selection failure.

4. **Verify image pipeline.** Only 1 of ~95 content pages had an
   embedded image. Either the training editions were generated without
   images, or the image embedding pipeline is broken.

5. **After fixes, regenerate and re-grade.** The current batch provides
   a useful baseline ("this is how bad it was") but will not produce
   meaningful training signal for the learned objective function. The
   technical model would learn to predict "does the page have content"
   rather than "is the composition good."

---

*Neville — Layout and Publishing Expert, IRAS*
*Batch 1 of ~30 | Task #237*
