# Brief #312: OffScroll Empty Masthead and Pull-Quote-Only Layout Fixes

## Root Cause

Both bugs shared a single root cause: a **logic inversion** in
`templates.typ:180`. The `standard-article` block had:

```typst
block(breakable: word-count <= 200, ...)
```

This made long articles (>200 words) **non-breakable** — the opposite
of the correct behavior. Short articles should stay together; long
articles should flow across pages. The correct condition is
`word-count > 200`.

### How this caused empty mastheads (~14% of spreads)

When no FEATURE article exists (or when the first content block is a
standard article), the masthead renders on page 1, then the first
article follows. Because long articles were non-breakable, Typst
couldn't start them on page 1 (insufficient remaining space after
the masthead), so the entire article was bumped to page 2 — leaving
page 1 with only the masthead at ~15% fill.

### How this caused pull-quote-only pages (~18% of spreads)

Standalone pull quotes were rendered as full-width blocks between
rows. When a row filled a page and a pull quote started the next
page, the following article — being non-breakable — was again bumped
to the page after that. Result: a page with only a pull quote at
~10% fill.

## Changes

### `src/offscroll/layout/typst/templates.typ`

1. **Line 180:** Fixed breakable condition: `word-count <= 200` →
   `word-count > 200`. Long articles now flow across pages. Short
   articles (<=200 words) stay together as a unit.

2. **Line 336:** Added `sticky: true` to `section-label` block.
   Section headers now stay attached to following content, preventing
   orphaned section headers (the ~3% fill "ANALYSIS header only"
   failures in batch-002).

### `src/offscroll/layout/typst_renderer.py`

3. **Suppressed standalone pull quotes** in three locations:
   - After front feature (line ~510)
   - After single-column rows (line ~560)
   - After multi-column rows (line ~610)
   
   Pull quotes now only appear **inline** within article bodies.
   This eliminates the structural cause of pull-quote-only pages
   regardless of the breakable fix.

4. **Lowered inline PQ threshold** from >1000 words / >3 paragraphs
   to >400 words / >2 paragraphs (in both `_render_feature` and
   `_render_standard`). With standalone PQs suppressed, inline
   placement is the only path — the lower threshold ensures pull
   quotes still appear in medium-length articles.

### `tests/layout/test_typst_renderer.py`

5. **Updated `test_contains_pull_quote`** → renamed to
   `test_standalone_pull_quote_suppressed` to verify the new behavior.

6. **Added 5 regression tests:**
   - `TestEmptyMastheadFix::test_no_feature_has_content_after_masthead`
   - `TestEmptyMastheadFix::test_feature_renders_on_front_page`
   - `TestPullQuoteOnlyFix::test_row_level_pull_quotes_suppressed`
   - `TestPullQuoteOnlyFix::test_inline_pull_quotes_preserved`
   - `TestPullQuoteOnlyFix::test_unmatched_pull_quotes_still_rendered`

## Verification

- **199 tests pass** (full layout test suite, 0 failures)
- **10 previously-failing editions** (032, 009, 017, 004, 050, 060,
  096, 042, 051, 079) all compile to PDF successfully
- **Zero standalone pull quotes** across all tested editions
- **Inline pull quotes preserved** (1-7 per edition depending on
  article lengths)
- **Unmatched pull quotes** in the Notable Quotes block are
  unaffected

## Scope

- WeasyPrint renderer (`renderer.py`) is **not modified** — these
  changes are Typst-specific
- Training set is **not re-rendered** per brief instructions
- No changes to shared layout composition logic
  (`_compose_section_rows`, `_will_inline_pull_quotes`)
