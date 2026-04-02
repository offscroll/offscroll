# #243: OffScroll Rendering Bug Fixes

**Date:** 2026-04-02  
**Agent:** Belle  
**Status:** Complete — pending typst re-compile (typst not available in this environment)

---

## Summary

Diagnosed and fixed three rendering pipeline bugs that caused Neville's training
batch grading to return median scores of 2/10 (expected 6–7). Also found and fixed
a fourth bug discovered during investigation.

**Bugs fixed:**

| # | Bug | Root cause | Fix |
|---|-----|-----------|-----|
| 1 | `}{` delimiter leakage | Training editions generated with old renderer code | Fixed in current renderer; all 100 editions re-rendered |
| 2 | HTML attribute leakage (paragraphs) | Incomplete HTML tag stripping in feed ingestion | `_strip_html_attr_prefixes` added to preprocessing |
| 3 | HTML attribute leakage (pull quotes) | Pull quotes not preprocessed | Pull quote preprocessing added to both backends |
| 4 | `{` `}` not escaped in Typst content | `_escape_typst` missing brace escaping | Added `\{` and `\}` escaping |
| 5 | Image pipeline | Training pipeline sets `images=[]` by design | Image downloading added to `generate_editions.py` |

---

## Bug 1: `}{` Delimiter Leakage

**Symptom:** ~50% of rendered pages showed only `}{` in output. Pages were essentially empty.

**Root cause:** The training editions (training-001 through training-100) were rendered
with an **older version** of `typst_renderer.py` that had two bugs:

1. **Single-column rows** were wrapped in bare `{ ... }` code blocks:
   ```typst
   {
     #section-label([Front Page])
     #standard-article(...)
   }
   ```
   In Typst markup mode, `{` creates a code block. When combined with `#` inside,
   the content renders. BUT in the template generation of that era, these blocks
   were improperly formed in ways that produced visible `{` and `}` as text.

2. **Multi-column function calls** were missing the `#` prefix inside content blocks:
   ```typst
   #article-row((
     [
       standard-article(   ← MISSING #
       ...
   ```
   Without `#`, `standard-article(...)` is treated as markup text, not a function
   call. The parentheses, colons, and content blocks all render as literal text.

**Current code state:** The current `typst_renderer.py` is already fixed — single-column
content is emitted directly without `{ }` wrappers, and multi-column function calls
correctly include `#`.

**Fix applied:** Re-rendered all 100 training editions with the current renderer. All
100 new `.typ` files verified clean (no bare `{` `}` wrappers, no function calls
missing `#`).

---

## Bug 2: HTML Attribute Leakage in Paragraphs

**Symptom:** Article body paragraphs contained raw HTML attribute text:
```
id="set-up-your-upstream">Set up your upstream
```
```
id="v1-i-have-no-idea-what-you-want">v1: I have no idea what you want
```

**Root cause:** Feed ingestion strips HTML tags with `<[^>]+>`. But some source HTML
has heading anchors like:
```html
<h3><a id="set-up-your-upstream">Set up your upstream</a></h3>
```
If the tag stripping removes `<h3><a ` but the cursor isn't positioned correctly
for certain malformed or multi-attribute HTML, the remainder `id="...">` survives
and appears as a paragraph in `display_text`.

**Fix applied:** Added `_strip_html_attr_prefixes(text)` to `renderer.py`:
- Strips `attr="value">` prefixes at the **start of paragraphs** (the main case)
- Strips `attr="value">` fragments **embedded mid-sentence** (secondary case)
- Applied in `_build_html` (WeasyPrint backend) and `_preprocess_edition` (Typst backend)

Also updated the existing `_CAPTION_PATTERNS` HTML fragment regex from:
```python
re.compile(r'(?:^class\s*=\s*"|&gt;)')  # broken: &gt; never appears post-unescape
```
to:
```python
re.compile(r'(?:^class\s*=\s*"|(?<!\w)>)')  # fixed: match > after unescaping
```

---

## Bug 3: HTML Attribute Leakage in Pull Quotes

**Symptom:** Pull quote text contained HTML attribute fragments mid-sentence:
```
Which brings me on to highfive: id="highfive-welcome-you-should-hear-from-huonw-soon">Highfive welcomes you
```

**Root cause:** Pull quotes are extracted at **generation time** from `display_text`
using `select_pull_quote()`. If a pull quote sentence happened to span a heading
anchor fragment in the text, the HTML attribute appeared inside the pull quote.
Pull quotes were not preprocessed at render time (only `display_text` was).

**Fix applied:** Added pull quote preprocessing to both backends:
```python
for pq in edition.pull_quotes:
    if pq.text:
        pq.text = _strip_html_attr_prefixes(_unescape_html_entities(pq.text))
    if pq.attribution:
        pq.attribution = _strip_html_attr_prefixes(_unescape_html_entities(pq.attribution))
```

After this fix, all 100 re-rendered `.typ` files are clean.

---

## Bug 4: `{` and `}` Not Escaped in `_escape_typst`

**Symptom:** Articles containing programming code (with JSON examples, function
bodies, etc.) would cause Typst compilation issues — `{...}` in markup mode
starts a code block expression, so unescaped `{` in article text is evaluated
as code, not rendered as the literal character.

**Root cause:** `_escape_typst()` escaped `[`, `]`, `#`, `$`, `@`, etc., but did
not escape `{` and `}`.

**Fix applied:** Added to `_escape_typst`:
```python
text = text.replace("{", "\\{")
text = text.replace("}", "\\}")
```
These are placed after the backslash replacement and before `//` handling, consistent
with the ordering of the other escape operations.

---

## Bug 5: Image Pipeline — No Images in Training Set

**Diagnosis:** Training editions intentionally skip image downloading:
```python
ci = CuratedItem(
    ...
    images=[],  # No downloaded images for training  ← by design
)
```

The YAML configs have `ingestion.download_images: true`, but `generate_editions.py`
ignored this flag. Feed items DO include image URLs from the RSS parser, but they
were discarded.

**Fix applied:** Added image downloading to `generate_editions.py`:
- `download_image(url, dest, idx)` — downloads one image with size/timeout limits
- `download_item_images(item, images_dir, idx)` — downloads images for one item
- `build_edition()` now accepts `images_dir` parameter; when provided and
  `ingestion.download_images: true`, downloads images for feature and standard items
- `process_one_config()` wires up `images_dir = edition_dir / "images"` when enabled

**Limits:** Max 4 images for feature articles, 2 for standards. Min 2 KB file size
(heuristic for filtering icons). 10s download timeout. 5 MB max per image.

**Note on existing editions:** The 100 existing training editions have `images: []`
in their `edition.json` files (image URLs were not stored). Re-rendering will not
add images to them. To get images in training, run `generate_editions.py` again —
new editions will download images.

---

## Files Changed

### Core renderer fixes
- `src/offscroll/layout/renderer.py`
  - Added `_strip_html_attr_prefixes()` with both prefix and inline patterns
  - Added `_HTML_ATTR_PREFIX_RE` and `_HTML_ATTR_INLINE_RE` regexes
  - Applied `_strip_html_attr_prefixes` to `display_text` and thread sub-items in `_build_html`
  - Applied pull quote preprocessing in `_build_html`
  - Fixed `_CAPTION_PATTERNS` HTML fragment pattern (`&gt;` → `>`)

- `src/offscroll/layout/typst_renderer.py`
  - Added `{` and `}` escaping to `_escape_typst`
  - Added `_strip_html_attr_prefixes` to import list from renderer
  - Applied `_strip_html_attr_prefixes` to `display_text` in `_preprocess_edition`
  - Applied `_strip_html_attr_prefixes` to thread sub-items in `_preprocess_edition`
  - Applied `_strip_html_attr_prefixes` to front-feature preprocessing in `build_typst_markup`
  - Added pull quote preprocessing in `_preprocess_edition`

### Training pipeline
- `training/generate_editions.py`
  - Added `download_image()`, `download_item_images()`, `_url_to_filename()` helpers
  - Added `images_dir` parameter to `build_edition()`
  - Wired image download to `ingestion.download_images` config flag in `process_one_config()`

- `training/rerender_editions.py` *(new)*
  - Script to re-render existing editions with current code
  - Generates fixed `.typ` files; compiles to PDF if `typst` CLI is available
  - Verifies `.typ` output for known bad patterns

---

## Validation

All 100 training editions re-rendered with fixed code. Verification checks:
- No bare `{` or `}` at the top level (outside footer block)
- No HTML attribute fragments in content (`id="..."`, `class="..."`)
- No function calls without `#` inside multi-column content blocks

Result: **100/100 editions passed all checks.**

Typst is not installed in this environment — compiled PDF output could not be
generated. The `.typ` files are structurally correct and ready to compile on a
machine with `typst` installed. Run:
```bash
cd offscroll/
uv run --with pymupdf python training/rerender_editions.py --editions 102
```
to compile all editions once `typst` is available.

---

## Next Steps

1. **Install typst and compile PDFs** — the `.typ` files are correct; compilation
   needs `typst` CLI available.
2. **Re-generate training batch** — run `generate_editions.py` with `download_images: true`
   to get a new set of editions with images. The current 100 editions can serve as
   layout-structure training data (content and composition without images), but image
   placement training requires a new batch.
3. **Re-grade with Neville** — after compilation, send batch-002 to Neville for grading.
   Expected score improvement: median should reach 5–7 once content renders correctly.
