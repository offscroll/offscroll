# Training Set Rerender Report — Task #244

**Date:** 2026-04-02  
**Agent:** Belle  
**Purpose:** Regenerate training set after #243 rendering fixes (blank pages, HTML leakage, delimiter bugs)

## Results

| Metric | Before (#243 .typ regen) | After (this run) |
|--------|--------------------------|------------------|
| Editions | 100 | 100 |
| Compiled OK | 100 | 100 |
| Failed | 0 | 0 |
| Total pages | 2,837 | 1,867 |
| Total spreads | 1,496 | 1,010 |
| Bad .typ patterns | (unknown) | 0 |

Page count reduction (2,837 → 1,867) is expected: the buggy renderer was emitting extra blank/overflow pages. Fixed renderer produces denser, correctly laid-out pages.

## What was done

1. Installed Typst 0.13.1 binary (not present in environment)
2. Installed PyMuPDF (missing from venv)
3. Ran `training/rerender_editions.py --editions 100 --start 1 -v`
   - All 100 editions: compiled OK, zero .typ issues detected
4. Rebuilt `metadata.json` for all 100 editions with correct page/spread counts
5. Regenerated `grading-manifest.json` (1,010 spreads, seed=43)
6. Regenerated `edition-map.json`

## Spot-check (10 editions)

Editions checked: training-005, training-015, training-027, training-038, training-051, training-060, training-072, training-083, training-090, training-100

Findings:
- **Blank pages:** None observed. All pages have real content.
- **Delimiter leakage:** None. No `}{` bare delimiters or `id="..."` HTML attribute remnants.
- **Layout:** Multi-column grids, section labels, mastheads, pull quotes all rendering correctly.
- **Images:** No images were downloaded during the original generation (feed items had no valid image URLs). This is expected — not a regression.
- **Typography:** Clean Typst rendering with Source Serif 4 body text, proper justification and hyphenation.

## Grading manifest

`grading-manifest.json`: 1,010 spreads in fresh randomized order (seed=43, different from original seed=42).  
`edition-map.json`: spread_id → edition_id mapping for post-grading analysis.

Ready for Neville to resume grading.
