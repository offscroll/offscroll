# Feature Metadata — OffScroll Layout Feature Vectors

**Author:** Belle (Systems Architect, IRAS)
**Date:** 2026-05-03 (Task #329, updated from #310/#238)
**Source data:** `training/grades/batch-002.csv` (50 graded spreads, clean)
**Output:** `training/features/features-002.csv`
**Seed:** 42 | **Split:** 80% train / 20% val → 40 train, 10 val
**Previous batch:** `features.csv` (batch-001, corrupted by `}{` bug — superseded)
**Computation script:** `compute_features_002.py`

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

The 80/20 train/val split was assigned with `random.Random(42)`:
- 40 spreads in training set
- 10 spreads in validation set

The split is recorded in the `split` column. The specific assignment
is deterministic: given the same grades CSV and seed, the split will
be identical.

**Stratification note:** The bimodal technical grade distribution
(cluster at 2-3 and cluster at 6) means a random split may not
perfectly balance the failure modes across train/val. With only 50
spreads, stratification was not applied — the random split is
acceptable. Consider stratification when the graded set grows.

---

## Batch 2 Summary Statistics (Task #329 — corrected features)

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

## Comparison: Batch 1 vs Batch 2

| Metric | Batch 1 | Batch 2 (corrected) | Notes |
|--------|---------|---------------------|-------|
| Technical mean | 3.0 | 4.8 | Batch 2 is clean data; batch 1 was corrupted by `}{` bug |
| Technical median | 2 | 6 | Batch 2 has a real competent cluster |
| Style mean | 2.6 | 2.9 | Style still low — expected per pipeline spec |
| Style median | 2 | 3 | Slight improvement with clean renders |
| d5_fill_fraction mean | 1.55 (broken) | 0.53 | Was word-density proxy (r=1.0 with word count). Now spatial fill from PDF. |
| d4_col_balance | NA | 159pt mean | Computed from PDF text block positions |
| d6_dead_space | NA | 0.01 mean | Computed from PDF inter-block gaps |
| d2_orphans | NA | 0.16 mean | Detected from PDF column analysis |
| d2_widows | NA | 0.00 | Layout engine does not produce widows |

---

*Belle — Systems Architect & Integration Lead, IRAS*
