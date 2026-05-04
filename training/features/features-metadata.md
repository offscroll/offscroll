# Feature Metadata — OffScroll Layout Feature Vectors

**Author:** Belle (Systems Architect, IRAS)
**Date:** 2026-05-04 (Task #343 — n=200 final set; supersedes #341)
**Source data:** `training/grades/batch-002.csv` + `batch-003.csv` +
`batch-004.csv` + `batch-005.csv`
(50 × 4 = 200 graded spreads, no duplicate IDs)
**Output:** `training/features/features-004.csv`
**Seed:** 42 | **Split:** 80% train / 20% val → 160 train, 40 val
**Previous checkpoints:**
  - `features.csv` (batch-001, corrupted by `}{` bug — superseded)
  - `features-002.csv` (batch-002 only, n=50; preserved, do not overwrite)
  - `features-003.csv` (batches 2+3+4, n=150; preserved, do not overwrite)
**Computation script:** `compute_features_004.py`
(identical feature pipeline to `compute_features_003.py`; only the
input batch list and split logic differ, plus two data-quality
guards described below)

## Data-quality guards added in Task #343

Two anomalies surfaced during the n=200 merge. Both are handled in
the loader; the source files are unchanged.

1. **Trailing `COMPLETED` row in `batch-005.csv`.** A runner artifact
   appended an extra line after the 50 graded spreads. The loader
   filters any row whose `spread_id` does not start with `s-` or whose
   `technical`/`style` cells are empty. 1 row filtered.

2. **Six spread IDs missing from `edition-map.json`.** `s-028-004`,
   `s-045-007`, `s-047-011`, `s-090-023`, `s-093-005`, `s-095-011` are
   real terminal spreads that exist in their edition's
   `metadata.json`, but the `edition-map.json` index only covers
   spreads 1..N−1 for those editions. The loader falls back to
   deriving `edition_id` from the `s-{nnn}-{idx}` ID format and
   verifies the edition's `metadata.json` exists before accepting
   it. All 6 spreads recovered. The map file should be regenerated
   to cover all spreads — surfaced as a follow-up.

---

## Context and Method

Features are computed from two sources:

1. **Edition metadata** (`metadata.json`, `edition.json`): structural
   facts about spreads — page positions, edition composition, item word
   counts, template types, image counts, source diversity. These are
   exact or estimated via linear word-count allocation.

2. **Rendered PDFs** (via PyMuPDF): spatial layout features extracted
   from the actual rendered output. Text block bounding boxes from the
   PDF give precise measurements of fill, column balance, dead space,
   and orphan/widow detection. This was added in Task #329 to fix the
   five features that were previously NA or miscomputed.

**n=200 final-set note (Task #343):** `features-004.csv` is the
combined feature matrix for batches 2 + 3 + 4 + 5 (50 spreads each,
200 total, no duplicate IDs). Computation is identical to
`features-003.csv` — same script logic in `compute_features_004.py`,
same feature set, same PDF analysis pass. The train/val split is
freshly assigned on the 200-spread set; it does not preserve the
n=150 assignment. See "n=200 Final-Set Summary Statistics" below for
the full n=200 stats.

The remainder of this document describes the batch-002 baseline; the
n=150 and n=200 statistics extend (not replace) it. The feature schema
and methods are unchanged.

**Batch 2 quality note:** This batch uses the re-rendered training set
(#309), which fixes the `}{` rendering bug that corrupted batch-001.
The grade distribution is bimodal:

- **Technical grade:** mean=4.8, median=6, std=1.7. Cluster at T:6
  (25 spreads — competent layouts) and cluster at T:2-3 (16 spreads —
  structural failures).
- **Style grade:** mean=2.9, median=3, std=0.9. Concentrated at 2-3
  (37 of 50 spreads). Style features are weak by design — the style
  model is expected to fail initially per the pipeline spec.

**Structural failure modes now detectable:**
- Empty masthead pages — `d5_fill_fraction` now correctly reports
  0.03–0.06 for these pages (was 0.79–8.09 with the word-density
  proxy). All 7 T:2 front-page spreads show fill < 0.07.
- Pull-quote-only pages — `d5_fill_fraction` reports 0.1–0.3 for
  these. Combined with `d6_dead_space`, the model can now identify
  trapped whitespace within the content flow.

---

## Spread Identification

| Column | Type | Description |
|--------|------|-------------|
| `spread_id` | string | Unique spread identifier (`s-{edition}-{spread_index}`). Edition index is the training-NNN directory number; spread index is 1-based within the edition's spread list. |
| `split` | string | `train` or `val`. Assigned randomly with seed 42. 80%/20% split. |

---

## Target Variables (Neville's Grades)

| Column | Type | Description |
|--------|------|-------------|
| `technical_grade` | int 1–10 | Scale A: technical proficiency. Measures composition errors, column balance, spacing, fill, and image placement. Batch-002 mean=4.8, median=6. Bimodal: cluster at 6 (competent) and 2-3 (broken). |
| `style_grade` | int 1–10 | Scale B: editorial style. Measures visual rhythm, surprise calibration, front-page impact, and section transitions. Batch-002 mean=2.9, median=3. |

---

## Spread Structure Features

Derived from `metadata.json`. These are exact, not estimated.

| Column | Type | Description |
|--------|------|-------------|
| `spread_type` | string | `solo` (single page) or `spread` (two facing pages). Front pages (page 1) and terminal pages are often solo. |
| `is_solo` | int 0/1 | 1 if spread_type == solo. |
| `is_front` | int 0/1 | 1 if the spread contains page 1. Front pages have different technical and style priorities (widow elimination, S3 front-page impact). |
| `is_terminal` | int 0/1 | 1 if the spread contains the last page of the edition. Terminal pages have relaxed fill requirements. |
| `page_role` | string | Categorical position: `front`, `interior`, `terminal`, `solo_terminal`. Per grading protocol: page role affects which trade-offs the compositor is expected to make. |
| `left_page` | int | Page number of the left (or only) page in the spread. 1-indexed. |
| `right_page` | int | Page number of the right page in the spread. Equals `left_page` for solo spreads. |
| `n_pages_in_spread` | int | 1 (solo) or 2 (two-page spread). |
| `edition_page_count` | int | Total pages in the edition. Determines what "interior" means. |
| `page_position_frac` | float [0,1] | (left_page − 1) / (edition_page_count − 1). 0.0 = page 1, 1.0 = last page. Continuous position signal for the model. |

---

## Edition-Level Aggregate Features

Derived from `edition.json`. These describe the full edition, not just
the spread's pages. They are exact.

| Column | Type | Description |
|--------|------|-------------|
| `edition_item_count` | int | Total items across all sections in the edition. Proxy for edition complexity. |
| `edition_word_count_total` | int | Sum of word counts across all items. Measures total content volume. |
| `edition_word_count_mean` | float | Mean word count per item. Indicates whether the edition is brief-heavy (short items, low mean) or feature-heavy (long items, high mean). |
| `edition_brief_frac` | float [0,1] | Fraction of items with `layout_hint = brief`. |
| `edition_standard_frac` | float [0,1] | Fraction of items with `layout_hint = standard`. |
| `edition_template_entropy` | float ≥ 0 | Shannon entropy (bits) of the `layout_hint` distribution across all edition items. 0 = all items same template; 1 = equal mix of two templates. Higher = more template variety. |
| `edition_image_count_total` | int | Total number of embedded images across the edition. Batch-002 note: all zero — image pipeline not yet active for training data. |
| `edition_source_count` | int | Number of distinct source names (feeds) contributing to the edition. Measures editorial diversity. |
| `edition_section_count` | int | Number of sections in the edition (Front Page, Features, Briefs, etc.). Determines how many section transitions exist. |

---

## Spread-Estimated Features

These are approximations. The item layout system places items
sequentially (section by section, page by page). Without a rendered
output, we estimate which items land on the spread's pages using a
linear cumulative word-count allocation: items are assigned to pages
proportionally to their word counts.

**Accuracy:** This estimate is best for editions with uniformly-sized
items and worst for editions with a few very long items. It is a rough
proxy, not a ground-truth measurement. Actual per-page item placement
requires either: (a) a rendering pass, or (b) integration with the
layout engine to extract page assignments.

| Column | Type | Description |
|--------|------|-------------|
| `est_item_count` | int | Estimated number of items on the spread's pages (items with any word-count overlap in the linear allocation). |
| `est_items_per_page` | float | `est_item_count / n_pages_in_spread`. Items per page proxy. More items per page → more inter-item gaps → harder spacing consistency (D8). |
| `est_word_count` | float | Estimated total word count on the spread's pages (pro-rated for partial item overlaps). |
| `est_words_per_page` | float | `est_word_count / n_pages_in_spread`. Content density proxy. |
| `est_word_count_mean` | float | `est_word_count / est_item_count`. Average item length on the spread. |
| `est_brief_count` | int | Estimated number of brief-template items on the spread. |
| `est_standard_count` | int | Estimated number of standard-template items on the spread. |
| `est_image_count` | int | Estimated number of images on the spread pages. From edition.json images arrays; batch-002: all zero. |
| `est_source_count` | int | Estimated number of distinct sources on the spread pages. |

---

## D-Feature Proxies (Metadata-Derivable)

These approximate the D-dimension scores from the grading protocol
using metadata. They are not equivalent to Neville's visual
assessments — they are computable proxies that the model can use
until rendered-output features are available.

| Column | Type | Description | Limitation |
|--------|------|-------------|------------|
| `d3_image_fraction` | float [0,∞) | Images per item on estimated spread pages (`est_image_count / est_item_count`). Proxy for D3 (image ratio). **Batch-002:** all zero — image pipeline not active. | Does not capture image size or position; those require rendered output. |
| `d5_fill_fraction` | float [0,1] | Spatial fill fraction: ratio of rendered text block area to total printable area across the spread's pages. Computed from PDF text block bounding boxes (PyMuPDF). Printable area = (page_width − 2×margin) × (page_height − 2×margin − footer). **Batch-002 range:** 0.03–0.84, mean=0.53. Correlation with `est_words_per_page`: r=−0.05. Correlation with `technical_grade`: r=+0.76. | Measures bounding-box area of text blocks, not ink pixels. Includes inter-line spacing within blocks. Does not account for images (image pipeline not yet active). |
| `d7_template_entropy` | float ≥ 0 | Shannon entropy of `layout_hint` distribution across estimated spread items. 0 = all same template; 1 = equal brief/standard mix. Proxy for template diversity. **Batch-002:** mostly 0.0 (only 6% of spreads have mixed templates in estimation). | Uses estimated item set; accuracy depends on item placement estimate. |
| `d8_word_count_cv` | float ≥ 0 | Coefficient of variation of word counts across estimated spread items. High CV = items vary widely in length → harder to achieve uniform spacing (D8). **Batch-002 range:** 0.0–1.43, mean=0.61. | Word count variation is a proxy for height variation, not a direct measure. |
| `anchor_strength` | float ≥ 1 | Max word count / mean word count across estimated spread items. 1.0 = all items same length; higher = one dominant long item. Proxy for anchor strength. **Batch-002 range:** 1.0–4.63, mean=1.91. | Word count is a proxy for rendered area. |

---

## Rendered-Output Features (Computed from PDFs)

These features are extracted from the rendered edition PDFs using
PyMuPDF. Text block bounding boxes give precise spatial measurements
that metadata alone cannot provide. Added in Task #329.

| Column | Type | Description | Method |
|--------|------|-------------|--------|
| `d2_orphans` | int ≥ 0 | Count of orphaned elements on the spread: headlines at the bottom of a column with no body text following (large empty space below), or isolated single body-text lines at the bottom of a column separated from the preceding content by a gap. **Batch-002 range:** 0–1, mean=0.16. | From PDF: detect headline blocks (`SourceSans3-Bold`) at the bottom of a column with > 100pt empty space below. Also detect single body-text lines at column bottom separated by > 20pt gap from preceding content. |
| `d2_widows` | int ≥ 0 | Count of widow lines on the spread: single body-text lines at the top of a column that are remnants of a paragraph from the previous column/page. **Batch-002:** all zero. The Typst layout engine places complete article blocks rather than breaking paragraphs across columns, so widow lines do not occur in the current training data. | From PDF: detect first block in a column that is a single body-text line (`SourceSerif4-Regular`, 10pt) with height < 20pt, followed by a gap > 15pt. |
| `d4_col_balance` | float ≥ 0 | Column height imbalance in points. Maximum difference between the bottom positions of columns on any page in the spread. 0 = perfectly balanced or single-column page. Target per grading protocol: ≤ 36pt. **Batch-002 range:** 0–501pt, mean=159pt. Correlation with `technical_grade`: r=−0.02 (weak — many spreads have single-column pages or are balanced). | From PDF: detect columns by clustering text block x-coordinates. For each page, measure the y-position of the lowest text block in each column. Balance = max − min across columns on the worst page. |
| `d6_dead_space` | float [0,1] | Trapped whitespace ratio: area of large gaps (> 36pt) between consecutive text blocks within a column, divided by total printable area. Does NOT include unfilled space below content (that is captured by `d5_fill_fraction`). **Batch-002 range:** 0–0.06, mean=0.01. Correlation with `technical_grade`: r=−0.52. | From PDF: for each column, sort blocks by y-position. Gaps between consecutive blocks exceeding 36pt (0.5in) are counted as dead space. Dead space area = gap height × column width. Ratio = total dead area / total printable area across spread pages. |

---

## Split Assignment

The 80/20 train/val split was assigned with `random.Random(42)` over
the lexicographically-sorted spread ID list (so the split is
deterministic and independent of CSV row order across the three
batch files):
- 120 spreads in training set
- 30 spreads in validation set

The split is recorded in the `split` column.

**Per-batch split breakdown (n=150):**
- batch-002: 39 train / 11 val
- batch-003: 43 train / 7 val
- batch-004: 38 train / 12 val

Note: `features-003.csv` does not preserve the 40/10 split that was
in `features-002.csv`. Spreads from batch-002 are re-shuffled into the
n=150 split — train/val membership for a given spread_id will differ
between the two files. This is intentional: with three times the data,
re-shuffling gives a single coherent split rather than carrying forward
a smaller-population assignment.

**Stratification note:** The technical grade distribution remains
roughly bimodal (peaks at 2 and 6). With 150 spreads, the random split
holds up well — both classes are represented in both partitions — but
stratification by technical grade should be considered as the graded
set continues to grow.

---

## Batch 2 Summary Statistics (Task #329 — corrected features, n=50)

```
n_spreads:               50
train / val:             40 / 10
technical_grade mean:    4.8   (median 6, std 1.7)
style_grade mean:        2.9   (median 3, std 0.9)

Rendered-output features (NEW — computed from PDFs):
  d5_fill_fraction:      min=0.03  max=0.84  mean=0.53
  d4_col_balance (pt):   min=0.00  max=501   mean=159
  d6_dead_space:         min=0.00  max=0.06  mean=0.01
  d2_orphans:            min=0     max=1     mean=0.16
  d2_widows:             min=0     max=0     mean=0.00

Metadata-derived features (unchanged):
  d7_template_entropy:   min=0.00  max=1.00  mean=0.11
  d8_word_count_cv:      min=0.00  max=1.55  mean=0.60
  anchor_strength:       min=1.00  max=4.79  mean=1.96
  est_items_per_page:    min=0.50  max=8.00  mean=1.93
  est_words_per_page:    min=484   max=3724  mean=1060
  page_position_frac:    min=0.00  max=0.84  mean=0.35

Technical grade distribution:
  2: 7  |  3: 9  |  4: 4  |  5: 2  |  6: 25  |  7: 3

Style grade distribution:
  1: 1  |  2: 17  |  3: 20  |  4: 10  |  5: 2

Key correlations with technical_grade:
  d5_fill_fraction:  r = +0.76  (higher fill → higher grade)
  d6_dead_space:     r = −0.52  (more dead space → lower grade)
  d4_col_balance:    r = −0.02  (weak — many single-col pages)
  d5 vs est_words:   r = −0.05  (was r=1.00 — independence verified)
```

---

## n=150 Checkpoint Summary Statistics (Task #341 — batches 2+3+4)

```
n_spreads:               150
train / val:             120 / 30
technical_grade mean:    4.55  (median 5, std 1.86)
style_grade mean:        3.13  (median 3, std 1.21)

Rendered-output features:
  d5_fill_fraction:      min=0.0305  max=0.8430  mean=0.4716
  d4_col_balance (pt):   min=0.00    max=637.29  mean=173.14
  d6_dead_space:         min=0.0000  max=0.2424  mean=0.0216
  d2_orphans:            min=0       max=2       mean=0.25
  d2_widows:             min=0       max=0       mean=0.00

Metadata-derived features:
  d7_template_entropy:   min=0.00    max=1.00    mean=0.21
  d8_word_count_cv:      min=0.00    max=2.91    mean=0.68
  anchor_strength:       min=1.00    max=16.90   mean=2.33
  est_items_per_page:    min=0.50    max=19.00   mean=2.68
  est_words_per_page:    min=363     max=3724    mean=994
  page_position_frac:    min=0.00    max=0.97    mean=0.39

Technical grade distribution:
  1: 3  |  2: 31  |  3: 20  |  4: 12  |  5: 15  |  6: 49  |  7: 20

Style grade distribution:
  1: 5  |  2: 51  |  3: 44  |  4: 26  |  5: 18  |  6: 6

Per-batch row counts (with split):
  batch-002: 50 spreads (39 train / 11 val)
  batch-003: 50 spreads (43 train /  7 val)
  batch-004: 50 spreads (38 train / 12 val)

Correlations with technical_grade (n=150):
  d5_fill_fraction:    r = +0.6964   (strong, was +0.76 at n=50)
  d2_orphans:          r = -0.4483   (strengthened from sparse signal at n=50)
  d6_dead_space:       r = -0.4221   (was -0.52 at n=50)
  est_items_per_page:  r = -0.2629
  d4_col_balance:      r = +0.1822   (sign flip from n=50 noise)
  anchor_strength:     r = -0.1419
  d8_word_count_cv:    r = -0.0805
  d7_template_entropy: r = -0.0500
  est_words_per_page:  r = -0.0080

  d5_fill_fraction vs est_words_per_page: r = +0.1454
  (independence preserved — d5 measures spatial fill, not word density)
```

**Notes on n=150 vs n=50:**
- **Distribution broadened.** Technical grade range now includes 1 and 5;
  style range now extends to 6. The bimodal (2-3) ∪ (6) shape from n=50
  has filled in slightly with grade-4 and grade-5 examples (27 spreads
  combined). Mean dropped from 4.8 to 4.55 — batches 3 and 4 contained
  more failures than batch 2.
- **Anchor strength range jumped** from 4.79 → 16.90 max. Batches 3 and 4
  include editions with one extreme outlier item (max ratio above 16
  means one item is ~16× the mean length on its spread) — likely
  long-form items dominating brief-heavy editions.
- **d6_dead_space upper bound jumped** from 0.06 to 0.24. Batches 3 and 4
  surfaced spreads with severely under-filled columns containing trapped
  whitespace — exactly the failure modes the grading protocol calls out.
- **d2_orphans upper bound jumped** from 1 to 2 — confirming orphan
  detection picks up on actual spread defects, not just edge noise.
- **Top correlate is stable.** `d5_fill_fraction` remains the dominant
  technical-grade signal (r ≈ +0.70 at n=150). `d2_orphans` and
  `d6_dead_space` are the secondary defect signals.
- **d4_col_balance correlation flipped sign** (-0.02 at n=50, +0.18 at
  n=150). At n=50 the value was inside the noise floor; the n=150 value
  is still weak but suggests col_balance alone does not separate
  high-grade from low-grade spreads in the current data — many low-grade
  spreads have a single broken page (large fill failure) but balanced
  columns on the working page.

---

## n=200 Final-Set Summary Statistics (Task #343 — batches 2+3+4+5)

```
n_spreads:               200
train / val:             160 / 40
technical_grade mean:    4.69  (median 5, std 1.92)
style_grade mean:        3.25  (median 3, std 1.24)

Rendered-output features:
  d5_fill_fraction:      min=0.0090  max=0.8430  mean=0.4706
  d4_col_balance (pt):   min=0.00    max=700.49  mean=167.62
  d6_dead_space:         min=0.0000  max=0.2424  mean=0.0225
  d2_orphans:            min=0       max=2       mean=0.245
  d2_widows:             min=0       max=0       mean=0.000

Metadata-derived features:
  d7_template_entropy:   min=0.00    max=1.00    mean=0.22
  d8_word_count_cv:      min=0.00    max=3.99    mean=0.72
  anchor_strength:       min=1.00    max=18.40   mean=2.42
  est_items_per_page:    min=0.50    max=20.00   mean=2.70
  est_words_per_page:    min=363     max=3724    mean=1004
  page_position_frac:    min=0.00    max=1.00    mean=0.40

Technical grade distribution:
  1: 5  |  2: 37  |  3: 26  |  4: 17  |  5: 19  |  6: 54  |  7: 42

Style grade distribution:
  1: 7  |  2: 61  |  3: 54  |  4: 38  |  5: 33  |  6: 7

Per-batch row counts (with split):
  batch-002: 50 spreads (39 train / 11 val)
  batch-003: 50 spreads (40 train / 10 val)
  batch-004: 50 spreads (39 train / 11 val)
  batch-005: 50 spreads (42 train /  8 val)

Correlations with technical_grade (n=200):
  d5_fill_fraction:    r = +0.7100   (top signal, stable)
  d2_orphans:          r = -0.4709   (secondary defect signal)
  d6_dead_space:       r = -0.3045   (weakened from -0.42 at n=150)
  est_items_per_page:  r = -0.2429
  d4_col_balance:      r = +0.1695   (still weak)
  anchor_strength:     r = -0.1409
  d8_word_count_cv:    r = -0.0721
  est_words_per_page:  r = +0.0394
  d7_template_entropy: r = -0.0540

  d5_fill_fraction vs est_words_per_page: r = +0.1434
  (independence preserved)
```

**Notes on n=200 vs n=150:**
- **Distribution is now well-populated across the full grade range.**
  Technical grade now covers 1–7 with at least 5 spreads in every
  bucket; style covers 1–6 similarly. The bimodal shape is still
  visible (peaks at T:2 and T:6) but the middle is no longer thinly
  sampled. Mean technical 4.55 → 4.69; mean style 3.13 → 3.25.
- **Top signal stable.** `d5_fill_fraction` correlation with technical
  grade is r=+0.71 at n=200 (was +0.70 at n=150, +0.76 at n=50).
  Convergence across three sample sizes confirms d5 is the strongest
  metadata-derivable predictor.
- **`d6_dead_space` correlation weakened** (-0.42 → -0.30). Batch-005
  contains more masthead/terminal/empty pages where the dead-space
  signal does not separate broken from intentional sparseness; this
  matches the protocol's expectation that terminal pages have relaxed
  fill requirements.
- **`d2_orphans` correlation stable** at r=-0.47 (was -0.45 at n=150).
  Orphan detection continues to pick up real defects.
- **Anchor strength upper bound jumped** from 16.90 → 18.40 — batch-005
  surfaced one additional outlier-anchor edition.
- **`d4_col_balance` upper bound jumped** from 637 → 700pt. The signal
  remains weak overall (r=+0.17) — col-balance alone does not separate
  high from low grades because many low-grade spreads have a single
  broken page (huge fill failure) but balanced columns on the working
  side.
- **`d8_word_count_cv` upper bound jumped** from 2.91 → 3.99 — at least
  one batch-005 edition has an extreme length-mix on a single spread.
- **`est_items_per_page` upper bound** now 20 (was 19). One batch-005
  spread is dense-brief-heavy.
- **`d2_widows` remains zero across all 200 spreads.** The Typst
  layout engine still produces no widow lines under current settings.

---

## Comparison: Batch 1 vs Batch 2 vs n=150 vs n=200

| Metric | Batch 1 (n=50) | Batch 2 (n=50) | n=150 (b2-b4) | n=200 (b2-b5) | Notes |
|--------|----------------|----------------|----------------|----------------|-------|
| Technical mean | 3.0 | 4.8 | 4.55 | 4.69 | range now well-populated 1-7 |
| Technical median | 2 | 6 | 5 | 5 | middle of distribution filling in |
| Style mean | 2.6 | 2.9 | 3.13 | 3.25 | gradual improvement |
| Style median | 2 | 3 | 3 | 3 | |
| d5_fill_fraction mean | 1.55 (broken) | 0.53 | 0.47 | 0.47 | stable across n=150 and n=200 |
| d4_col_balance mean | NA | 159pt | 173pt | 168pt | |
| d6_dead_space mean | NA | 0.01 | 0.022 | 0.023 | |
| d2_orphans mean | NA | 0.16 | 0.25 | 0.25 | |
| d2_widows | NA | 0.00 | 0.00 | 0.00 | layout engine produces no widows |
| d5↔technical r | NA | +0.76 | +0.70 | +0.71 | top signal, stable |
| d6↔technical r | NA | −0.52 | −0.42 | −0.30 | weakening as terminal/masthead spreads fill in |
| d2_orphans↔technical r | NA | (sparse) | −0.45 | −0.47 | stable secondary signal |

## Follow-up surfaced in Task #343

- `edition-map.json` is missing terminal spreads from at least 6
  editions (training-028, -045, -047, -090, -093, -095). The
  feature pipeline now recovers them via ID-format fallback, but
  the map should be regenerated for use by other consumers. Light
  task — likely a missing iteration in the map-builder script.
- `batch-005.csv` ends with a stray `COMPLETED` row from a runner
  artifact. Filtered at load. Source file should be cleaned by
  whoever maintains the grading pipeline.

---

*Belle — Systems Architect & Integration Lead, IRAS*
