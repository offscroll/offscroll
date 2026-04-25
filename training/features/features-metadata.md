# Feature Metadata — OffScroll Layout Feature Vectors

**Author:** Belle (Systems Architect, IRAS)
**Date:** 2026-04-25 (Task #310, updated from #238)
**Source data:** `training/grades/batch-002.csv` (50 graded spreads, clean)
**Output:** `training/features/features-002.csv`
**Seed:** 42 | **Split:** 80% train / 20% val → 40 train, 10 val
**Previous batch:** `features.csv` (batch-001, corrupted by `}{` bug — superseded)

---

## Context and Limitations

These features were computed from edition metadata (`metadata.json`,
`edition.json`) without access to rendered page images. This imposes a
hard ceiling on feature quality:

- **What's computable:** structural facts about the spread (which pages,
  edition composition, item word counts, template types, image counts,
  source diversity, page position).
- **What's not computable:** any dimension that requires reading the
  rendered output — column heights, actual fill, orphan/widow detection,
  spacing measurements, white-space geometry.

Features requiring rendered output are included in the CSV with value
`NA`. When the rendering pipeline produces stable output, these should
be computed and joined to this table.

**Batch 2 quality note:** This batch uses the re-rendered training set
(#309), which fixes the `}{` rendering bug that corrupted batch-001.
These are the first usable grades. The grade distribution is bimodal:

- **Technical grade:** mean=4.8, median=6, std=1.7. Cluster at T:6
  (25 spreads — competent layouts) and cluster at T:2-3 (16 spreads —
  structural failures).
- **Style grade:** mean=2.9, median=3, std=0.9. Concentrated at 2-3
  (37 of 50 spreads). Style features are weak by design — the style
  model is expected to fail initially per the pipeline spec.

**Structural failure modes observed:**
- Empty masthead pages (~15% fill, no content below masthead) — 7
  spreads scored T:2. `d5_fill_fraction` should flag these, but the
  metadata proxy overestimates fill because it allocates by word count,
  not by rendered layout. The proxy shows d5 > 0.79 even for broken
  front pages. This is a known limitation.
- Pull-quote-only pages (~10% fill, solo pull quote) — 9 spreads
  scored T:3. Low `est_item_count` (often 1-2) combined with notes
  mentioning "pull quote only" or "section header only" pages.

**Failure mode detection note:** The metadata-only `d5_fill_fraction`
does NOT reliably flag the two structural failure modes because it
measures content *intended* for pages, not what was *rendered*. The
rendered-output features (`d2_orphans`, `d2_widows`, `d4_col_balance`,
`d6_dead_space`) are critical for capturing these failures once the
image analysis pipeline is available.

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
| `d5_fill_fraction` | float | Estimated words on spread / (WORDS_PER_PAGE_CAPACITY × n_pages). Capacity constant = 450 words/page. Values > 1.0 indicate estimated overload; values < 0.5 indicate sparse content. Proxy for D7 (page fill). **Batch-002 range:** 0.79–8.09, mean=2.29. | Capacity estimate is a fixed constant, not calibrated. Does NOT flag empty masthead or pull-quote-only pages because it measures intended content, not rendered content. |
| `d7_template_entropy` | float ≥ 0 | Shannon entropy of `layout_hint` distribution across estimated spread items. 0 = all same template; 1 = equal brief/standard mix. Proxy for template diversity. **Batch-002:** mostly 0.0 (only 6% of spreads have mixed templates in estimation). | Uses estimated item set; accuracy depends on item placement estimate. |
| `d8_word_count_cv` | float ≥ 0 | Coefficient of variation of word counts across estimated spread items. High CV = items vary widely in length → harder to achieve uniform spacing (D8). **Batch-002 range:** 0.0–1.43, mean=0.61. | Word count variation is a proxy for height variation, not a direct measure. |
| `anchor_strength` | float ≥ 1 | Max word count / mean word count across estimated spread items. 1.0 = all items same length; higher = one dominant long item. Proxy for anchor strength. **Batch-002 range:** 1.0–4.63, mean=1.91. | Word count is a proxy for rendered area. |

---

## Features Requiring Rendered Output (NA in Batch 2)

These features cannot be computed from metadata alone. They are
included as NA columns to reserve space in the schema. Once the
rendering pipeline is stable, they should be computed from the rendered
page images or layout engine output and joined to this table.

| Column | Description | How to Compute |
|--------|-------------|----------------|
| `d2_orphans` | Number of orphaned elements (headlines or captions without their body text) on the spread. Direct causal link to technical grade. | Parse rendered layout: detect headline blocks not followed by body text within the same column. |
| `d2_widows` | Number of widow lines on the spread. Per protocol: widows on page 1–2 are scored harshly. | Detect last lines of paragraphs at the top of a column without the preceding paragraph body. Requires rendered text flow. |
| `d4_col_balance` | Column height deviation. The single most visible technical defect per Neville. Measure as max column height − min column height in points. Target: ≤ 36pt (0.5 inch). | Measure rendered column heights from the layout engine or by image analysis of the column bottom edges. |
| `d6_dead_space` | Trapped white space score. Detect large gaps within the text flow that are not inter-item margins. | Analyze rendered page whitespace regions; identify gaps larger than expected inter-item spacing. |

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

## Batch 2 Summary Statistics

```
n_spreads:               50
train / val:             40 / 10
technical_grade mean:    4.8   (median 6, std 1.7)
style_grade mean:        2.9   (median 3, std 0.9)
d5_fill_fraction:        min=0.79  max=8.09  mean=2.29
d7_template_entropy:     min=0.00  max=1.00  mean=0.06
d8_word_count_cv:        min=0.00  max=1.43  mean=0.61
anchor_strength:         min=1.00  max=4.63  mean=1.91
est_items_per_page:      min=0.50  max=8.00  mean=1.78
est_words_per_page:      min=356   max=3639  mean=1031
page_position_frac:      min=0.00  max=1.00  mean=0.40

Technical grade distribution:
  2: 7  |  3: 9  |  4: 4  |  5: 2  |  6: 25  |  7: 3

Style grade distribution:
  1: 1  |  2: 17  |  3: 20  |  4: 10  |  5: 2
```

---

## Comparison: Batch 1 vs Batch 2

| Metric | Batch 1 | Batch 2 | Notes |
|--------|---------|---------|-------|
| Technical mean | 3.0 | 4.8 | Batch 2 is clean data; batch 1 was corrupted by `}{` bug |
| Technical median | 2 | 6 | Batch 2 has a real competent cluster |
| Style mean | 2.6 | 2.9 | Style still low — expected per pipeline spec |
| Style median | 2 | 3 | Slight improvement with clean renders |
| d5_fill_fraction mean | 1.55 | 2.29 | Higher content density in batch 2 editions |
| d7_template_entropy | 0.17 | 0.06 | Less template diversity in batch 2 estimated spreads |

---

*Belle — Systems Architect & Integration Lead, IRAS*
