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
    _strip_html_attr_prefixes,
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
    # Order matters: escape backslash first (it's the escape character)
    text = text.replace("\\", "\\\\")
    text = text.replace("#", "\\#")
    text = text.replace("$", "\\$")
    text = text.replace("@", "\\@")
    text = text.replace("<", "\\<")
    text = text.replace(">", "\\>")
    text = text.replace("_", "\\_")
    text = text.replace("*", "\\*")
    text = text.replace("`", "\\`")
    # Typst uses {} for code blocks in markup — escape curly braces in content
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    # Typst uses // for comments — escape double slashes in URLs
    text = text.replace("//", "\\/\\/")
    # Typst uses [] for content blocks — escape when in content text
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    # Typst treats / at line start as definition list; since source wrapping
    # is unpredictable, escape all occurrences of "/ " in content
    text = text.replace("/ ", "\\/ ")
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
                item.display_text = _strip_html_attr_prefixes(item.display_text)
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
                        sub.display_text = _strip_html_attr_prefixes(sub.display_text)
                        sub.display_text = _strip_display_boilerplate(sub.display_text)
                        sub.display_text = _fix_subheading_concatenation(sub.display_text)
                        sub._edited_for_length = _has_editorial_ellipsis(sub.display_text)
    # Preprocess pull quote text — extracted from display_text at generation
    # time, so HTML attribute leakage can appear here too.
    for pq in edition.pull_quotes:
        if pq.text:
            pq.text = _strip_html_attr_prefixes(_unescape_html_entities(pq.text))
        if pq.attribution:
            pq.attribution = _strip_html_attr_prefixes(_unescape_html_entities(pq.attribution))


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
    fi = _first_alpha_index(lead)
    # Split lead into pre-alpha, cap letter, and rest for drop cap
    # (done in Python to avoid Typst UTF-8 byte boundary issues)
    lead_pre = _escape_typst(lead[:fi]) if fi > 0 else ""
    lead_cap = _escape_typst(lead[fi]) if fi < len(lead) else ""
    lead_rest = _escape_typst(lead[fi + 1:]) if fi + 1 < len(lead) else ""

    # Inline pull quote
    item_id = getattr(item, "item_id", "")
    item_pqs = pq_map.get(item_id, [])
    inline_pq_idx = -1
    inline_pq = "none"
    wc = getattr(item, "word_count", 0)
    # Lowered from >1000/>3 to >400/>2 — with standalone row-level PQs
    # suppressed, inline placement is the only path for pull quotes.
    if wc > 400 and item_pqs and len(body_paras) > 2:
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
    lines.append(f"  lead-pre: [{lead_pre}],")
    lines.append(f"  lead-cap: [{lead_cap}],")
    lines.append(f"  lead-rest: [{lead_rest}],")
    lines.append(f"  body-paragraphs: {body_array},")
    if inline_pq != "none":
        lines.append(f"  inline-pq: {inline_pq},")
        lines.append(f"  inline-pq-idx: {inline_pq_idx},")
    lines.append(f"  edited-for-length: {edited},")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def _render_standard(
    item,
    pq_map: dict,
    data_dir: Path,
    debug_mode: bool,
    is_lead: bool = False,
) -> str:
    """Generate Typst markup for a standard article.

    is_lead: when True, the article is the page lead and receives the
    edition's configured lead amplifications (deck and/or scale-bump).
    A deck is generated from the article body and rendered above the
    byline when the edition's weight strategy permits decks on standards.
    """
    lines = []
    title = _escape_typst(getattr(item, "title", "") or "")
    author = _escape_typst(getattr(item, "author", "") or "")
    source_name = _escape_typst(getattr(item, "source_name", None) or "")
    text = getattr(item, "display_text", "") or ""
    wc = getattr(item, "word_count", 0)

    # Generate a deck only for lead items (Neville §3.4). The deck
    # generator may return None for short or unsuitable text — that's
    # fine; the template falls back to no deck.
    deck_text = _generate_feature_deck(text) if is_lead else None
    deck_escaped = _escape_typst(deck_text) if deck_text else ""

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

    # Inline pull quote — lowered threshold from >1000/>3 to match
    # _render_feature; standalone row-level PQs are now suppressed.
    item_id = getattr(item, "item_id", "")
    item_pqs = pq_map.get(item_id, [])
    inline_pq_idx = -1
    inline_pq = "none"
    if wc > 400 and item_pqs and len(paragraphs) > 2:
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
    if is_lead:
        lines.append("  is-lead: true,")
    if deck_escaped:
        lines.append(f"  deck: [{deck_escaped}],")
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


# Typographic-diversity defaults (Neville Tier 1, brief #392/#414).
# Selected per edition by config["newspaper"]["typography"]. Values are
# validated against the presets in templates.typ; unknown values fall
# back to the default.
_VALID_SCALES = ("tight", "standard", "open")
_VALID_WEIGHTS = ("two", "three")
_VALID_LEAD_AMPS = ("deck", "scale-bump")
_TYPOGRAPHY_DEFAULTS = {
    "scale": "standard",
    "weights": "three",
    "lead_amplifications": ("deck", "scale-bump"),
}


def _resolve_typography(config: dict) -> dict:
    """Pull the typography section from config and validate values.

    Returns a dict with keys scale, weights, lead_amplifications. Unknown
    values are silently replaced with defaults so a malformed config
    cannot crash rendering.
    """
    typo = config.get("newspaper", {}).get("typography", {}) or {}
    scale = typo.get("scale", _TYPOGRAPHY_DEFAULTS["scale"])
    if scale not in _VALID_SCALES:
        logger.warning("Unknown typography.scale %r; using 'standard'", scale)
        scale = _TYPOGRAPHY_DEFAULTS["scale"]
    weights = typo.get("weights", _TYPOGRAPHY_DEFAULTS["weights"])
    if weights not in _VALID_WEIGHTS:
        logger.warning("Unknown typography.weights %r; using 'three'", weights)
        weights = _TYPOGRAPHY_DEFAULTS["weights"]
    raw_amps = typo.get("lead_amplifications", _TYPOGRAPHY_DEFAULTS["lead_amplifications"])
    amps = tuple(a for a in raw_amps if a in _VALID_LEAD_AMPS)
    return {"scale": scale, "weights": weights, "lead_amplifications": amps}


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
    typography = _resolve_typography(config)

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
            front_feature.display_text = _strip_html_attr_prefixes(front_feature.display_text)
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

    # Per-edition typographic configuration (Neville Tier 1).
    # The scale governs the headline hierarchy; the weight strategy
    # governs register treatment (e.g., real small caps for kickers);
    # lead-amplifications govern how the first item in each section is
    # differentiated visually.
    amps_typst = "(" + ", ".join(
        f'"{a}"' for a in typography["lead_amplifications"]
    )
    # Typst single-element tuples need a trailing comma; multi-element
    # tuples must not double the comma. Empty tuples become "()".
    if len(typography["lead_amplifications"]) == 1:
        amps_typst += ","
    amps_typst += ")"
    out.append(
        '#set-edition-config('
        f'scale: "{typography["scale"]}", '
        f'weights: "{typography["weights"]}", '
        f'lead-amplifications: {amps_typst})'
    )
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

        # Pull quotes for the front feature are either inlined within the
        # article body (for long articles) or suppressed. Standalone pull
        # quotes after the feature cause page-break isolation — a pull quote
        # as the sole element on a page with ~10% fill.
        out.append('')

    # Sections
    for section in edition.sections:
        heading = section.heading
        rows = section_rows.get(heading, [])

        if not rows:
            out.append(f'#section-label([{_escape_typst(heading)}])')
            out.append('')
            continue

        # Lead-item differentiation (Neville §3.4): the first standard
        # article rendered in this section is marked as the section lead.
        # The template applies the edition's configured amplifications
        # (deck and/or scale-bump) to that item. Threads and briefs are
        # not eligible — they already carry distinct typographic identity.
        section_lead_taken = False

        for row_idx, row in enumerate(rows):
            columns = row["columns"]
            row_pqs = row.get("pull_quotes", [])
            section_heading = row.get("section_heading")

            if len(columns) == 1:
                # Single-column row — no wrapper needed, top-level content mode
                col = columns[0]
                if section_heading:
                    out.append(f'#section-label([{_escape_typst(section_heading)}])')

                for item in col.get("col_items", []):
                    if isinstance(item, CuratedThread):
                        out.append('#' + _render_thread(item, data_dir))
                    elif hasattr(item, "layout_hint") and item.layout_hint == LayoutHint.FEATURE:
                        out.append('#' + _render_feature(item, pq_map, data_dir, debug_mode))
                    else:
                        is_lead = not section_lead_taken
                        section_lead_taken = True
                        out.append('#' + _render_standard(
                            item, pq_map, data_dir, debug_mode, is_lead=is_lead
                        ))

                briefs = col.get("briefs", [])
                if briefs:
                    brief_items = []
                    for b in briefs:
                        brief_items.append('  [#' + _render_brief(b).rstrip() + '],')
                    out.append('#brief-group((')
                    out.extend(brief_items)
                    out.append('))')

                # Row-level pull quotes suppressed — standalone PQ blocks
                # between rows cause page-break isolation (pull-quote-only
                # pages at ~10% fill). PQs are rendered inline within
                # article bodies for articles that meet the word-count
                # threshold.

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
                        col_lines.append(f'    #section-label([{_escape_typst(section_heading)}])')

                    for item in col.get("col_items", []):
                        if isinstance(item, CuratedThread):
                            col_lines.append('    #' + _render_thread(item, data_dir))
                        elif hasattr(item, "layout_hint") and item.layout_hint == LayoutHint.FEATURE:
                            col_lines.append('    #' + _render_feature(item, pq_map, data_dir, debug_mode))
                        else:
                            is_lead = not section_lead_taken
                            section_lead_taken = True
                            col_lines.append('    #' + _render_standard(
                                item, pq_map, data_dir, debug_mode, is_lead=is_lead
                            ))

                    briefs = col.get("briefs", [])
                    if briefs:
                        brief_items = []
                        for b in briefs:
                            brief_items.append('      [#' + _render_brief(b).rstrip() + '],')
                        col_lines.append('    #brief-group((')
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

                # Row-level pull quotes suppressed (same reason as single-col).

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
    # Tested against Typst 0.13.1. Templates use content/string semantics
    # and grid API from this version. Other versions may produce different
    # output or compilation errors.
    TESTED_TYPST_VERSION = "0.13.1"

    typst_bin = shutil.which("typst")
    if typst_bin is None:
        raise FileNotFoundError(
            "Typst CLI not found. Install it: "
            "https://github.com/typst/typst#installation"
        )

    # Check installed version and warn on mismatch
    try:
        ver_result = subprocess.run(
            [typst_bin, "--version"], capture_output=True, text=True, timeout=10
        )
        if ver_result.returncode == 0:
            installed = ver_result.stdout.strip().split()
            ver_str = installed[1] if len(installed) > 1 else installed[0]
            if ver_str != TESTED_TYPST_VERSION:
                logger.warning(
                    "Typst version %s detected; templates tested against %s. "
                    "Output may differ.",
                    ver_str,
                    TESTED_TYPST_VERSION,
                )
    except (subprocess.TimeoutExpired, OSError):
        pass  # Non-fatal — proceed with compilation

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
