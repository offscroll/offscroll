# Training Set Rerender Report — Task #307

**Date:** 2026-04-24  
**Agent:** Belle  
**Purpose:** Verify `}{` Typst bug fix, re-render full training set, confirm clean output

## Summary

The `}{` rendering bug (fixed in Task #243, commit 07ad3b6) is **confirmed resolved**. The root cause was two-fold: (1) single-column rows wrapped in bare `{ }` code blocks, and (2) multi-column function calls missing the `#` prefix. Both were fixed in `typst_renderer.py` and have not regressed — no source changes since the fix.

Full re-render of all 100 editions completed with zero compilation failures and zero structural issues.

## Results

| Metric | Value |
|--------|-------|
| Editions re-rendered | 100 |
| Compiled OK | 100 |
| Failed | 0 |
| Total pages | 1,867 |
| Total spreads | 957 |
| .typ structural issues | 0 |
| PDF `}{` rendering bugs | 0 |

## Code Verification

Reviewed the fix in `src/offscroll/layout/typst_renderer.py`:

1. **Single-column rows** (lines 537-563): content emitted at top-level without `{ }` wrappers. Confirmed correct.
2. **Multi-column function calls** (lines 578-584): all function calls use `#` prefix inside content blocks. Confirmed correct.
3. **Brace escaping** (`_escape_typst`, lines 67-69): `{` and `}` in content text are escaped to `\{` and `\}`. Confirmed correct.
4. **HTML attribute stripping** (`_strip_html_attr_prefixes`): applied to all text paths. Confirmed correct.
5. No source changes to renderer since Task #243 fix (git diff is empty).

## PDF Content Scan

Scanned all 100 PDFs (1,867 pages) for `}{` text. Two editions flagged — both are **false positives** (legitimate article content, not rendering artifacts):

- **training-031, page 9:** Jinja/Vue template syntax `{% raw %}{{ todo }}{% endraw %}` in source article
- **training-042, page 26:** LaTeX math notation `\renewcommand{\arraystretch}{1.5}` in source article

No rendering bugs, function call leakage, or HTML attribute leakage detected.

## Spot-check (10 editions)

Editions checked: training-005, -015, -027, -038, -051, -060, -072, -083, -090, -100

- **Blank pages:** None
- **Delimiter leakage:** None
- **Function call leakage:** None (no `standard-article(`, `feature-article(`, etc. in PDF text)
- **HTML attribute leakage:** None
- **Layout:** Multi-column grids, section labels, mastheads, pull quotes all rendering correctly

## What was done

1. Installed Typst 0.13.1 binary
2. Verified PyMuPDF available
3. Ran `training/rerender_editions.py --editions 100 --start 1 -v`
   - All 100 editions: compiled OK, zero .typ issues
4. Updated `metadata.json` for all 100 editions with current page/spread counts
5. Regenerated `grading-manifest.json` (957 spreads, seed=44)
6. Regenerated `edition-map.json`
7. Full PDF text scan for `}{` artifacts across all 1,867 pages

## Grading manifest

`grading-manifest.json`: 957 spreads in randomized order (seed=44).  
`edition-map.json`: spread_id to edition_id mapping.

Ready for grading.
