"""Typst rendering backend for OffScroll.

Generates a complete .typ file from a CuratedEdition, then compiles
it to PDF via the Typst CLI. Runs alongside the WeasyPrint backend;
selected via the ``backend`` parameter on render functions.

Architecture: Python handles all text processing (paragraph splitting,
boilerplate stripping, caption filtering) and layout composition
(row packing). The output is a self-contained .typ file that imports
template functions from templates.typ and calls them with pre-processed
data. Typst handles typography and PDF generation.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from offscroll.layout.renderer import (
    MAX_IMAGES_FEATURE,
    MAX_IMAGES_STANDARD,
    _build_pull_quote_map,
    _compose_section_rows,
    _extract_front_page_feature,
    _filter_orphaned_captions,
    _fix_subheading_concatenation,
    _generate_feature_deck,
    _has_editorial_ellipsis,
    _is_filename_caption,
    _strip_display_boilerplate,
    _unescape_html_entities,
    _will_inline_pull_quotes,
    image_insert_indices,
    split_feature_text,
    split_text_paragraphs,
)
from offscroll.models import CuratedEdition, CuratedThread, LayoutHint, PullQuote

logger = logging.getLogger(__name__)

TYPST_DIR = Path(__file__).parent / "typst"
FONTS_DIR = Path(__file__).parent / "fonts"


def _escape_typst(text: str) -> str:
    """Escape special Typst markup characters in content text.

    Typst uses # for code, @ for references, $ for math, etc.
    Content text must have these escaped so they render as literals.
    """
    if not text:
        return ""
    # Order matters: escape # first (most common in URLs/hashtags)
    text = text.replace("\\", "\\\\")
    text = text.replace("#", "\\#")
    text = text.replace("$", "\\$")
    text = text.replace("@", "\\@")
    text = text.replace("<", "\\<")
    text = text.replace(">", "\\>")
    text = text.replace("_", "\\_")
    text = text.replace("*", "\\*")
    text = text.replace("`", "\\`")
    # Typst uses // for comments — escape double slashes in URLs
    text = text.replace("//", "\\/\\/")
    return text


def _typst_string(text: str) -> str:
    """Wrap text as a Typst string literal (double-quoted)."""
    if not text:
        return '""'
    # Escape backslashes first, then quotes
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _first_alpha_index(text: str) -> int:
    """Return index of the first alphabetic character."""
    for i, ch in enumerate(text):
        if ch.isalpha():
            return i
    return 0


def _preprocess_edition(edition: CuratedEdition, config: dict) -> None:
    """Apply all text preprocessing to edition items in-place.

    This mirrors the preprocessing done in _build_html() so both
    backends produce identical text content.
    """
    for section in edition.sections:
        for item in section.items:
            if hasattr(item, "display_text") and item.display_text:
                item.display_text = _unescape_html_entities(item.display_text)
                item.display_text = _strip_display_boilerplate(item.display_text)
                item.display_text = _fix_subheading_concatenation(item.display_text)
                item._edited_for_length = _has_editorial_ellipsis(item.display_text)
            if hasattr(item, "images"):
                item_title = getattr(item, "title", None)
                for img in item.images:
                    if hasattr(img, "caption") and img.caption:
                        img.caption = _unescape_html_entities(img.caption)
                        if _is_filename_caption(img.caption, item_title):
                            img.caption = None
            if hasattr(item, "images") and hasattr(item, "layout_hint"):
                max_imgs = (
                    MAX_IMAGES_FEATURE
                    if item.layout_hint == LayoutHint.FEATURE
                    else MAX_IMAGES_STANDARD
                )
                if len(item.images) > max_imgs:
                    item.images = item.images[:max_imgs]
            if isinstance(item, CuratedThread):
                for sub in item.items:
                    if hasattr(sub, "display_text") and sub.display_text:
                        sub.display_text = _unescape_html_entities(sub.display_text)
                        sub.display_text = _strip_display_boilerplate(sub.display_text)
                        sub.display_text = _fix_subheading_concatenation(sub.display_text)
                        sub._edited_for_length = _has_editorial_ellipsis(sub.display_text)


def _resolve_image_path(local_path: str, data_dir: Path) -> str | None:
    """Resolve a relative image path to an absolute filesystem path.

    Typst needs absolute paths (not file:// URIs like WeasyPrint).
    Returns None if the image doesn't exist.
    """
    if not local_path:
        return None
    p = Path(local_path)
    if p.is_absolute():
        return str(p) if p.exists() else None
    resolved = data_dir / local_path
    if resolved.exists():
        return str(resolved)
    return None


def _render_pull_quote(pq: PullQuote) -> str:
    """Generate Typst markup for a pull quote."""
    text = _escape_typst(pq.text)
    attr = _escape_typst(pq.attribution)
    return f"pull-quote([{text}], [{attr}])\n"


def _render_image_block(img, data_dir: Path) -> str:
    """Generate Typst markup for an image block."""
    path = _resolve_image_path(getattr(img, "local_path", ""), data_dir)
    if not path:
        return ""
    caption = _escape_typst(getattr(img, "caption", None) or "")
    return f'image-block({_typst_string(path)}, caption-text: [{caption}])\n'


def _render_feature(item, pq_map: dict, data_dir: Path, debug_mode: bool) -> str:
    """Generate Typst markup for a feature article."""
    lines = []
    title = _escape_typst(getattr(item, "title", "") or "")
    author = _escape_typst(getattr(item, "author", "") or "")
    source_name = _escape_typst(getattr(item, "source_name", None) or "")
    kicker = _escape_typst(getattr(item, "kicker", "Cover Story") or "Cover Story")
    text = getattr(item, "display_text", "") or ""

    # Hero image
    hero_img = ""
    hero_caption = ""
    if getattr(item, "images", []):
        img = item.images[0]
        resolved = _resolve_image_path(getattr(img, "local_path", ""), data_dir)
        if resolved:
            hero_img = _typst_string(resolved)
            hero_caption = _escape_typst(getattr(img, "caption", None) or "")

    # Deck
    deck = _generate_feature_deck(text)
    deck_escaped = _escape_typst(deck) if deck else ""

    # Lead/body split
    lead, body_paras = split_feature_text(text, deck=deck)
    body_paras = _filter_orphaned_captions(body_paras)
    lead_escaped = _escape_typst(lead)
    fi = _first_alpha_index(lead)

    # Inline pull quote
    item_id = getattr(item, "item_id", "")
    item_pqs = pq_map.get(item_id, [])
    inline_pq_idx = -1
    inline_pq = "none"
    wc = getattr(item, "word_count", 0)
    if wc > 1000 and item_pqs and len(body_paras) > 3:
        inline_pq_idx = (len(body_paras) * 2) // 5
        pq = item_pqs[0]
        inline_pq = f"pull-quote([{_escape_typst(pq.text)}], [{_escape_typst(pq.attribution)}])"

    edited = "true" if getattr(item, "_edited_for_length", False) else "false"

    # Build body paragraphs array
    body_lines = []
    for p in body_paras:
        body_lines.append(f"  [{_escape_typst(p)}],")
    body_array = "(\n" + "\n".join(body_lines) + "\n)" if body_lines else "()"

    lines.append("feature-article(")
    lines.append(f"  title: [{title}],")
    lines.append(f"  kicker: [{kicker}],")
    lines.append(f"  author: [{author}],")
    lines.append(f"  source-name: [{source_name}],")
    if hero_img:
        lines.append(f"  hero-image: {hero_img},")
        lines.append(f"  hero-caption: [{hero_caption}],")
    if deck_escaped:
        lines.append(f"  deck: [{deck_escaped}],")
    lines.append(f"  lead-text: [{lead_escaped}],")
    lines.append(f"  lead-first-alpha: {fi},")
    lines.append(f"  body-paragraphs: {body_array},")
    if inline_pq != "none":
        lines.append(f"  inline-pq: {inline_pq},")
        lines.append(f"  inline-pq-idx: {inline_pq_idx},")
    lines.append(f"  edited-for-length: {edited},")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def _render_standard(item, pq_map: dict, data_dir: Path, debug_mode: bool) -> str:
    """Generate Typst markup for a standard article."""
    lines = []
    title = _escape_typst(getattr(item, "title", "") or "")
    author = _escape_typst(getattr(item, "author", "") or "")
    source_name = _escape_typst(getattr(item, "source_name", None) or "")
    text = getattr(item, "display_text", "") or ""
    wc = getattr(item, "word_count", 0)

    # Paragraphs
    paragraphs = _filter_orphaned_captions(split_text_paragraphs(text))

    # Images
    images_data = []
    for img in getattr(item, "images", []):
        resolved = _resolve_image_path(getattr(img, "local_path", ""), data_dir)
        if resolved:
            caption = _escape_typst(getattr(img, "caption", None) or "")
            images_data.append({"path": resolved, "caption": caption})

    # Image insert map
    extra_count = max(0, len(images_data) - 1)
    insert_map = image_insert_indices(len(paragraphs), extra_count)

    # Inline pull quote
    item_id = getattr(item, "item_id", "")
    item_pqs = pq_map.get(item_id, [])
    inline_pq_idx = -1
    inline_pq = "none"
    if wc > 1000 and item_pqs and len(paragraphs) > 3:
        inline_pq_idx = (len(paragraphs) * 2) // 5
        pq = item_pqs[0]
        inline_pq = f"pull-quote([{_escape_typst(pq.text)}], [{_escape_typst(pq.attribution)}])"

    edited = "true" if getattr(item, "_edited_for_length", False) else "false"
    editorial = _escape_typst(getattr(item, "editorial_note", None) or "")

    # Build arrays
    para_lines = []
    for p in paragraphs:
        para_lines.append(f"  [{_escape_typst(p)}],")
    para_array = "(\n" + "\n".join(para_lines) + "\n)" if para_lines else "()"

    img_lines = []
    for img in images_data:
        img_lines.append(f'  (path: {_typst_string(img["path"])}, caption: [{img["caption"]}]),')
    img_array = "(\n" + "\n".join(img_lines) + "\n)" if img_lines else "()"

    # Insert map as Typst dict
    map_entries = []
    for k, v in insert_map.items():
        map_entries.append(f'  "{k}": {v},')
    map_str = "(\n" + "\n".join(map_entries) + "\n)" if map_entries else "(:)"

    lines.append("standard-article(")
    lines.append(f"  title: [{title}],")
    lines.append(f"  author: [{author}],")
    lines.append(f"  source-name: [{source_name}],")
    lines.append(f"  images: {img_array},")
    lines.append(f"  paragraphs: {para_array},")
    lines.append(f"  insert-map: {map_str},")
    if inline_pq != "none":
        lines.append(f"  inline-pq: {inline_pq},")
        lines.append(f"  inline-pq-idx: {inline_pq_idx},")
    lines.append(f"  word-count: {wc},")
    lines.append(f"  edited-for-length: {edited},")
    if editorial and debug_mode:
        lines.append(f"  editorial-note: [{editorial}],")
    lines.append(f"  debug-mode: {'true' if debug_mode else 'false'},")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def _render_thread(item: CuratedThread, data_dir: Path) -> str:
    """Generate Typst markup for a thread."""
    headline = _escape_typst(getattr(item, "headline", "") or "")
    author = _escape_typst(getattr(item, "author", "") or "")
    source_name = _escape_typst(getattr(item, "source_name", None) or "")
    editorial = _escape_typst(getattr(item, "editorial_note", None) or "")

    posts = []
    for sub in item.items:
        text = _escape_typst(getattr(sub, "display_text", "") or "")
        posts.append(f"  [{text}],")
    posts_array = "(\n" + "\n".join(posts) + "\n)" if posts else "()"

    lines = [
        "thread-article(",
        f"  headline: [{headline}],",
        f"  author: [{author}],",
        f"  source-name: [{source_name}],",
    ]
    if editorial:
        lines.append(f"  editorial-note: [{editorial}],")
    lines.append(f"  posts: {posts_array},")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def _render_brief(item) -> str:
    """Generate Typst markup for a brief item."""
    author = _escape_typst(getattr(item, "author", "") or "")
    source_name = _escape_typst(getattr(item, "source_name", None) or "")
    text = _escape_typst(getattr(item, "display_text", "") or "")

    if source_name:
        return f"brief-item([{author}], source-name: [{source_name}], [{text}])\n"
    return f"brief-item([{author}], [{text}])\n"


def build_typst_markup(edition: CuratedEdition, config: dict) -> str:
    """Build a complete Typst document from a CuratedEdition.

    The generated .typ file is self-contained: it imports template
    functions from templates.typ and calls them with all edition data
    pre-processed by Python. Text splitting, boilerplate stripping,
    caption filtering, and row composition all happen here in Python.
    Typst handles only typography and PDF rendering.

    Returns:
        A string containing the complete .typ source.
    """
    data_dir = Path(config.get("output", {}).get("data_dir", "~/.offscroll/data"))
    if str(data_dir).startswith("~"):
        data_dir = data_dir.expanduser()

    debug_mode = config.get("newspaper", {}).get("debug_mode", False)

    # Preprocess all text (same as _build_html)
    _preprocess_edition(edition, config)

    # Extract front feature
    front_feature, _ = _extract_front_page_feature(edition)

    # Demote remaining features to standard
    for section in edition.sections:
        for item in section.items:
            if (
                not isinstance(item, CuratedThread)
                and hasattr(item, "layout_hint")
                and item.layout_hint == LayoutHint.FEATURE
            ):
                item.layout_hint = LayoutHint.STANDARD

    # Preprocess front feature
    if front_feature is not None:
        if hasattr(front_feature, "display_text") and front_feature.display_text:
            front_feature.display_text = _unescape_html_entities(front_feature.display_text)
            front_feature.display_text = _strip_display_boilerplate(front_feature.display_text)
            front_feature.display_text = _fix_subheading_concatenation(front_feature.display_text)
            front_feature._edited_for_length = _has_editorial_ellipsis(front_feature.display_text)
            if front_feature.images:
                ff_title = getattr(front_feature, "title", None)
                for img in front_feature.images:
                    if hasattr(img, "caption") and img.caption:
                        img.caption = _unescape_html_entities(img.caption)
                        if _is_filename_caption(img.caption, ff_title):
                            img.caption = None
                if len(front_feature.images) > MAX_IMAGES_FEATURE:
                    front_feature.images = front_feature.images[:MAX_IMAGES_FEATURE]
        front_feature.kicker = "Cover Story"

    # Build pull quote map
    pq_map = _build_pull_quote_map(edition.pull_quotes, edition)

    # Unmatched pull quotes
    all_item_ids: set[str] = set()
    for section in edition.sections:
        for item in section.items:
            if isinstance(item, CuratedThread):
                all_item_ids.add(item.thread_id)
                for sub in item.items:
                    all_item_ids.add(sub.item_id)
            else:
                all_item_ids.add(item.item_id)
    if front_feature is not None:
        all_item_ids.add(front_feature.item_id)

    unmatched_pqs = [
        pq
        for pq in edition.pull_quotes
        if pq.source_item_id == "unknown" or pq.source_item_id not in all_item_ids
    ]

    # Kicker labels for remaining features
    for section in edition.sections:
        for item in section.items:
            if (
                not isinstance(item, CuratedThread)
                and hasattr(item, "layout_hint")
                and item.layout_hint == LayoutHint.FEATURE
            ):
                item.kicker = section.heading

    # Compose rows for each section
    section_rows = {}
    for section in edition.sections:
        section_rows[section.heading] = _compose_section_rows(section, pq_map)

    # --- Build the .typ document ---
    out = []

    # Header: imports and page setup
    # Use relative import — the generated file is placed alongside templates
    out.append('// Generated by OffScroll Typst renderer')
    out.append(f'// Edition: {edition.edition.title} — {edition.edition.date}')
    out.append('')
    out.append('#import "templates.typ": *')
    out.append('')

    # Page setup
    ed_title = _escape_typst(edition.edition.title)
    ed_date = _escape_typst(edition.edition.date)
    footer_text = f"{ed_title} \\u{{2014}} {ed_date}"
    out.append('#set page("us-letter",')
    out.append('  margin: 0.5in,')
    out.append('  footer: context {')
    out.append('    if here().page() > 1 {')
    out.append('      set text(7pt, font: "Source Sans 3", fill: luma(153))')
    out.append('      line(length: 100%, stroke: 0.5pt + luma(204))')
    out.append('      v(0.05in)')
    out.append(f'      align(center)[{footer_text}]')
    out.append('    }')
    out.append('  }')
    out.append(')')
    out.append('')
    out.append('#set text(10pt, font: "Source Serif 4", fill: luma(26), hyphenate: true)')
    out.append('#set par(justify: true, leading: 0.52em)')
    out.append('')

    # Masthead
    ed_subtitle = _escape_typst(edition.edition.subtitle)
    editorial_note = _escape_typst(getattr(edition.edition, "editorial_note", None) or "")
    out.append(f'#masthead([{ed_title}], [{ed_subtitle}], [{ed_date}]')
    if debug_mode and editorial_note:
        out.append(f', editorial-note: [{editorial_note}], debug-mode: true')
    out.append(')')
    out.append('')

    # Curation summary
    if edition.curation_summary:
        summary = _escape_typst(edition.curation_summary)
        out.append(f'#curation-summary([{summary}])')
        out.append('')

    # Front feature
    if front_feature is not None:
        out.append('// --- Front Page Feature ---')
        out.append('#' + _render_feature(front_feature, pq_map, data_dir, debug_mode))

        # After-feature pull quotes (unless inlined)
        item_id = front_feature.item_id
        feat_pqs = pq_map.get(item_id, [])
        if feat_pqs:
            text = getattr(front_feature, "display_text", "") or ""
            _, feat_body = split_feature_text(text)
            feat_body = _filter_orphaned_captions(feat_body) if feat_body else []
            wc = getattr(front_feature, "word_count", 0)
            inline_placed = wc > 1000 and feat_pqs and len(feat_body) > 3
            if not inline_placed:
                for pq in feat_pqs:
                    out.append('#' + _render_pull_quote(pq))
        out.append('')

    # Sections
    for section in edition.sections:
        heading = section.heading
        rows = section_rows.get(heading, [])

        if not rows:
            out.append(f'#section-label([{_escape_typst(heading)}])')
            out.append('')
            continue

        for row_idx, row in enumerate(rows):
            columns = row["columns"]
            row_pqs = row.get("pull_quotes", [])
            section_heading = row.get("section_heading")

            if len(columns) == 1:
                # Single-column row
                col = columns[0]
                out.append('{')
                if section_heading:
                    out.append(f'  #section-label([{_escape_typst(section_heading)}])')

                for item in col.get("col_items", []):
                    if isinstance(item, CuratedThread):
                        out.append('  #' + _render_thread(item, data_dir))
                    elif hasattr(item, "layout_hint") and item.layout_hint == LayoutHint.FEATURE:
                        out.append('  #' + _render_feature(item, pq_map, data_dir, debug_mode))
                    else:
                        out.append('  #' + _render_standard(item, pq_map, data_dir, debug_mode))

                briefs = col.get("briefs", [])
                if briefs:
                    brief_items = []
                    for b in briefs:
                        brief_items.append('    #' + _render_brief(b))
                    out.append('  #brief-group((')
                    out.extend(brief_items)
                    out.append('  ))')

                # Single-column: pull quotes inside column
                for pq in row_pqs:
                    out.append('  #' + _render_pull_quote(pq))

                out.append('}')
                out.append('')
            else:
                # Multi-column row (grid)
                ruled_indices = []
                col_contents = []

                for ci, col in enumerate(columns):
                    if col.get("ruled", False):
                        ruled_indices.append(ci)

                    col_lines = []
                    if ci == 0 and section_heading:
                        col_lines.append(f'    section-label([{_escape_typst(section_heading)}])')

                    for item in col.get("col_items", []):
                        if isinstance(item, CuratedThread):
                            col_lines.append('    ' + _render_thread(item, data_dir))
                        elif hasattr(item, "layout_hint") and item.layout_hint == LayoutHint.FEATURE:
                            col_lines.append('    ' + _render_feature(item, pq_map, data_dir, debug_mode))
                        else:
                            col_lines.append('    ' + _render_standard(item, pq_map, data_dir, debug_mode))

                    briefs = col.get("briefs", [])
                    if briefs:
                        brief_items = []
                        for b in briefs:
                            brief_items.append('      ' + _render_brief(b))
                        col_lines.append('    brief-group((')
                        col_lines.extend(brief_items)
                        col_lines.append('    ))')

                    col_content = "\n".join(col_lines) if col_lines else ""
                    col_contents.append(col_content)

                # Build grid call
                ncols = len(columns)
                col_widths = ", ".join(["1fr"] * ncols)
                ruled_str = ", ".join(str(r) for r in ruled_indices)

                out.append(f'#article-row((')
                for ci, cc in enumerate(col_contents):
                    out.append('  [')
                    if cc:
                        out.append(cc)
                    out.append('  ],')
                out.append(f'), ruled-indices: ({ruled_str}{"," if ruled_indices else ""}))')

                # Multi-column: pull quotes after row (full width)
                for pq in row_pqs:
                    out.append('#' + _render_pull_quote(pq))

                out.append('')

    # Unmatched pull quotes
    if unmatched_pqs:
        out.append('// --- Notable Quotes ---')
        out.append('#block(above: 0.2in, stroke: (top: 1pt + luma(26)), inset: (top: 0.1in))[')
        for pq in unmatched_pqs:
            out.append('  #' + _render_pull_quote(pq))
        out.append(']')
        out.append('')

    # Colophon
    out.append(f'#colophon([{ed_title}], [{ed_subtitle}], [{ed_date}])')
    out.append('')

    return "\n".join(out)


def render_typst_pdf(
    config: dict,
    edition: CuratedEdition,
) -> Path:
    """Render a CuratedEdition to PDF via Typst CLI.

    Generates a .typ file, copies template files alongside it,
    then runs ``typst compile`` to produce PDF output.

    Args:
        config: The OffScroll config dict.
        edition: Pre-loaded CuratedEdition.

    Returns:
        Path to the generated PDF file.

    Raises:
        FileNotFoundError: If the ``typst`` CLI is not installed.
        subprocess.CalledProcessError: If Typst compilation fails.
    """
    typst_bin = shutil.which("typst")
    if typst_bin is None:
        raise FileNotFoundError(
            "Typst CLI not found. Install it: "
            "https://github.com/typst/typst#installation"
        )

    output_dir = Path(config["output"]["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    date = edition.edition.date
    typ_path = output_dir / f"newspaper-{date}.typ"
    pdf_path = output_dir / f"newspaper-{date}-typst.pdf"

    # Generate markup
    markup = build_typst_markup(edition, config)
    typ_path.write_text(markup)

    # Copy template files alongside the generated file so imports work
    templates_dest = output_dir / "templates.typ"
    shutil.copy2(TYPST_DIR / "templates.typ", templates_dest)

    # Compile
    logger.info("Compiling Typst document: %s", typ_path)
    result = subprocess.run(
        [typst_bin, "compile", "--font-path", str(FONTS_DIR), str(typ_path), str(pdf_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        logger.error("Typst compilation failed:\n%s", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )

    # Clean up template copy (keep the .typ source for debugging)
    templates_dest.unlink(missing_ok=True)

    logger.info("Typst PDF written to %s", pdf_path)
    return pdf_path
