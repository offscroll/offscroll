# Brief #312: OffScroll — Fix Empty Masthead and Pull-Quote-Only Layout Failures

**Agent:** Belle (Systems Architect)
**Date:** 2026-05-02
**Status:** Complete

## Problem

Batch-002 grading identified two structural layout engine failures
affecting ~32% of spreads:

1. **Empty masthead pages (~14%):** Front pages render masthead +
   FEATURES label but no content below. ~15% fill. Every instance
   scored T:2, S:2.

2. **Pull-quote-only pages (~18%):** Interior pages render a pull
   quote as the sole element at ~10% fill. Every instance scored
   T:3, S:2.

## Root Cause Analysis

### Empty Masthead

`_extract_front_page_feature()` returns `(None, None)` when no
edition item has `layout_hint == FEATURE`. Without a front feature,
the masthead renders alone on page 1. Section content blocks are
too large to fit the remaining space, so Typst pushes them to page
2. Result: page 1 at ~15% fill.

**All 6 failing editions confirmed:** training-032, 004, 060, 050,
096, 017 — none had any FEATURE-hinted items.

### Pull-Quote-Only Pages

`_compose_section_rows()` attaches pull quotes at the row level as
standalone `block(breakable: false)` elements. When a row fills a
page, the trailing pull quote is pushed to the next page as the
sole element.

**All 7 failing spreads confirmed:** s-009-007, s-014-004,
s-042-011, s-016-006, s-051-005, s-079-015, s-060-010.

## Fixes Applied

### 1. Front Feature Promotion (`renderer.py:_extract_front_page_feature`)

When no FEATURE item exists, the function now promotes the longest
standard article (≥300 words) to FEATURE. This ensures page 1
always has substantive content below the masthead.

- Minimum 300 words prevents tiny articles from being promoted
- BRIEFs and threads are excluded from promotion
- Explicit FEATURE items always take priority (no behavior change
  when a FEATURE exists)

### 2. Row-Level Pull Quote Suppression (`renderer.py:_compose_section_rows`)

All row-level pull quotes are now suppressed (set to `[]`). Pull
quotes render only:
- **Inline** within long articles (>1000 words, >3 paras) — existing
  behavior, unchanged
- **Notable Quotes section** at the end — expanded to catch non-inlined
  matched PQs that would previously have been row-level

### 3. Unplaced PQ Collection (`renderer.py:_collect_unplaced_pull_quotes`)

New shared helper used by both HTML and Typst renderers. Collects:
- PQs with unknown source (existing)
- PQs not matching any edition item (existing)
- **NEW:** Matched PQs whose source articles are too short to inline
  them

Front feature PQs are excluded (rendered separately after the
feature article).

## Files Changed

| File | Change |
|------|--------|
| `src/offscroll/layout/renderer.py` | Added promotion fallback in `_extract_front_page_feature`, row PQ suppression in `_compose_section_rows`, new `_collect_unplaced_pull_quotes` helper, updated `_build_html` unmatched PQ logic |
| `src/offscroll/layout/typst_renderer.py` | Import `_collect_unplaced_pull_quotes`, updated `build_typst_markup` unmatched PQ logic |
| `tests/layout/test_renderer.py` | 9 new/updated tests covering both fixes |

## Test Results

- **202 layout tests pass** (163 renderer + 39 Typst renderer)
- **9 targeted tests** for the new behavior:
  - `test_extract_front_page_feature_promotes_standard_when_no_feature`
  - `test_extract_front_page_feature_prefers_explicit_feature`
  - `test_extract_front_page_feature_skips_briefs_and_threads`
  - `test_extract_front_page_feature_requires_minimum_word_count`
  - `test_compose_section_rows_pull_quotes_suppressed`
  - `test_compose_section_rows_all_rows_have_empty_pull_quotes`
  - `test_collect_unplaced_pull_quotes_basic`
  - `test_collect_unplaced_excludes_front_feature_pqs`
  - `test_collect_unplaced_excludes_inlined_pqs`
  - `test_matched_pull_quotes_for_short_articles_go_to_notable_quotes`

## Verification on Failing Editions

All 6 masthead-only editions now produce front features:
```
training-032: feature=True, standalone_pqs=0
training-004: feature=True, standalone_pqs=0
training-060: feature=True, standalone_pqs=0
training-050: feature=True, standalone_pqs=0
training-096: feature=True, standalone_pqs=0
training-017: feature=True, standalone_pqs=0
```

All 7 pull-quote-only editions now have zero standalone PQs:
```
training-009: standalone_pqs=0
training-014: standalone_pqs=0
training-042: standalone_pqs=0
training-016: standalone_pqs=0
training-051: standalone_pqs=0
training-079: standalone_pqs=0
training-060: standalone_pqs=0
```

## Impact

Both fixes apply to **both renderers** (HTML/WeasyPrint and Typst)
since the core logic lives in `renderer.py`. The existing training
set is untouched per instructions — broken spreads remain as
negative training examples.

## Not Done

- Did NOT re-render the 100 training editions (per instructions)
- Did NOT modify grading data or batch-002 results
