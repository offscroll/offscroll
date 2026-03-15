"""PDF renderer (WeasyPrint orchestration).

Renders curated editions to newspaper-style HTML and PDF output
via Jinja2 templates and WeasyPrint. Supports both CuratedEdition
and RankedEdition formats.
"""

from __future__ import annotations

import html as html_module
import logging
import re
from collections import defaultdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from offscroll.models import (
    CuratedEdition,
    CuratedThread,
    LayoutHint,
    PullQuote,
    RankedEdition,
    RankedItem,
    detect_edition_format,
)

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"

#  Render-time boilerplate truncation patterns.
# Belt-and-suspenders: catches boilerplate that leaked through
# ingestion .
_RENDER_TRUNCATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"donating\s*=\s*loving",
        r"if this labor makes",
        r"for seventeen years,? I have been spending",
        r"has a free weekly newsletter",
        r"please consider a one-time donation",
    ]
]


def _strip_display_boilerplate(text: str) -> str:
    """Truncate display_text at known boilerplate markers."""
    if not text:
        return text
    earliest = len(text)
    for pat in _RENDER_TRUNCATION_PATTERNS:
        match = pat.search(text)
        if match and match.start() < earliest:
            earliest = match.start()
    if earliest < len(text):
        text = text[:earliest].rstrip()
    return text


#  Regex to fix subheading concatenation at render time.
# Catches "word.Uppercase" patterns where a sentence-ending punctuation
# is immediately followed by an uppercase letter with no space
# ( "new.Streamlined schedulingAdding").
# This handles data already in the DB from before the ingestion fix.
_CONCAT_FIX_RE = re.compile(r'([.!?])([A-Z])')

# : Extended concatenation fix.
# Also catches colon+uppercase ("new:Include") where inline tags like
# <strong> or <h3> were stripped without whitespace insertion.
_CONCAT_COLON_RE = re.compile(r'(:)([A-Z])')


def _fix_subheading_concatenation(text: str) -> str:
    """Insert space between concatenated sentences/subheadings.

     Render-time fix for . When HTML-to-text
    conversion strips block tags without whitespace, sentences run
    together like "new.Streamlined". This inserts a space after
    sentence-ending punctuation followed immediately by an uppercase
    letter.

    Extended to also fix colon+uppercase
    ("new:Include" -> "new: Include"). This pattern arises when
    inline HTML tags (<strong>, <em>) wrapping subheadings within
    list items are stripped without whitespace insertion.
    """
    if not text:
        return text
    text = _CONCAT_FIX_RE.sub(r'\1 \2', text)
    text = _CONCAT_COLON_RE.sub(r'\1 \2', text)
    return text


#  Detect editorial ellipsis markers [\u2026] and [...].
# These are preserved (not stripped) -- they serve editorial
# transparency by showing where content was trimmed.
_EDITORIAL_ELLIPSIS_RE = re.compile(r'\[\u2026\]|\[\.{3}\]')


def _has_editorial_ellipsis(text: str) -> bool:
    """Check if text contains editorial ellipsis markers.

     Used to determine if an article has been editorially
    shortened, so we can add a "(Edited for length)" note.
    Replaces the stripping -- founder wants ellipses
    preserved for transparency.
    """
    if not text:
        return False
    return bool(_EDITORIAL_ELLIPSIS_RE.search(text))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences with smarter boundary detection.

    Handles embedded punctuation like
    "Eureka! moment" -- an exclamation or question mark followed
    by a space and a lowercase word is NOT a sentence boundary.
    Only splits on [.!?] followed by whitespace and an uppercase
    letter, a quote character, or end of string.
    """
    # First, do a naive split on sentence-ending punctuation + whitespace
    raw = re.split(r'(?<=[.!?])\s+', text)
    if not raw:
        return []

    # Rejoin fragments that start with a lowercase letter back to
    # the previous segment -- these were false splits on embedded
    # punctuation like "Eureka! moment" or "Dr. Who? really".
    merged: list[str] = [raw[0]]
    for frag in raw[1:]:
        if frag and frag[0].islower():
            merged[-1] = merged[-1] + " " + frag
        else:
            merged.append(frag)
    return merged


def _generate_feature_deck(text: str) -> str | None:
    """Generate a deck line (1-2 sentence summary) for a feature article.

    The .feature-deck CSS class exists but
    no deck was being rendered. This extracts a compelling sentence
    from the article text as a deck line, providing the reader a
    reason to engage before the lead paragraph.

    Broader fallback -- search beyond the
    first paragraph if opening text doesn't yield a suitable deck.
    Scans up to 15 sentences deep for a 10-30 word non-quoted
    sentence. This handles articles that open with quoted prose
    or long academic sentences.

    Strategy:
    1. Try the first non-quoted sentence (10-30 words).
    2. Combine short first + second sentence (<=35 words).
    3. Truncate a long first sentence at a sentence boundary.
    4. (NEW) Search deeper: scan sentences 0-14 for any suitable
       non-quoted 10-30 word sentence.

    Returns a 1-sentence deck or None if no suitable text found.
    """
    if not text or len(text) < 100:
        return None

    # Fix concatenated sentences first
    fixed = _CONCAT_FIX_RE.sub(r'\1 \2', text)

    # Find sentences.
    # : Use _split_sentences() which handles
    # embedded punctuation (e.g. "Eureka! moment") -- don't split
    # on ! or ? when followed by a lowercase word.
    sentences = _split_sentences(fixed)
    if not sentences:
        return None

    # Skip leading quoted text (pull quotes used as article openers)
    start_idx = 0
    for i, s in enumerate(sentences[:3]):
        s_stripped = s.strip()
        if s_stripped and s_stripped[0] in '\u201c"\u2018\u0027':
            start_idx = i + 1
        else:
            break
    if start_idx >= len(sentences):
        start_idx = 0  # Fall back if all sentences are quoted

    first = sentences[start_idx].strip() if start_idx < len(sentences) else ""
    if not first:
        return None

    wc = len(first.split())

    # If first sentence is between 10-30 words and ends with
    # punctuation, use it directly
    if 10 <= wc <= 30 and first[-1:] in '.!?':
        return first

    # If first sentence is short, combine with next sentence
    if wc < 10 and start_idx + 1 < len(sentences):
        second = sentences[start_idx + 1].strip()
        combined = first + " " + second
        cwc = len(combined.split())
        if cwc <= 35 and combined[-1:] in '.!?':
            return combined

    # If first sentence too long, truncate at sentence boundary
    if wc > 30:
        cut = first[:180].rfind('. ')
        if cut > 50:
            return first[:cut + 1]

    # : Broader fallback -- scan deeper into the
    # article for any suitable deck sentence. Skip quoted text and
    # very short fragments.
    search_limit = min(len(sentences), 15)
    for idx in range(start_idx + 1, search_limit):
        candidate = sentences[idx].strip()
        if not candidate:
            continue
        # Skip quoted text
        if candidate[0] in '\u201c"\u2018\u0027':
            continue
        cwc = len(candidate.split())
        if 10 <= cwc <= 30 and candidate[-1:] in '.!?':
            return candidate

    return None


# : Detect orphaned image captions.
# These are short paragraphs that describe images/illustrations
# but render as body text when the corresponding image is not
# inline. Patterns match common caption structures.
#
# : Expanded to catch art attributions
# ("Art by/from Name"), merchandise links ("Available as a print"),
# raw HTML fragments (contains class= or &gt;), and single-word
# CTA labels ("newsletter", "subscribe", "share").
_CAPTION_PATTERNS = [
    re.compile(
        r'^\(?\s*(?:Photograph|Photo|Illustration|Image|Drawing|Painting)'
        r'\s+(?:by|from|courtesy)',
        re.IGNORECASE,
    ),
    re.compile(r'^A page from\b', re.IGNORECASE),
    re.compile(
        r'(?:Photograph|Photo|Illustration|Image)'
        r'\s+(?:by|from|courtesy)\s+\w+',
        re.IGNORECASE,
    ),
    #  Art attributions ("Art by Ryoji Arai from...")
    re.compile(
        r'^Art\s+(?:by|from)\s+',
        re.IGNORECASE,
    ),
    #  Artwork/painting attributions
    # e.g. "An Erupting Volcano by Night by David Humbert de Superville."
    # or "Storm Cloud by Georgia O'Keeffe."
    # or "Karl Drais with his velocipede"
    # Pattern: starts with a capitalized word, contains " by ", short.
    re.compile(
        r'^[A-Z][^.]{5,60}\s+by\s+[A-Z][a-z]',
    ),
    #  Merchandise/print links
    re.compile(
        r'\(Available\s+as\s+a\s+print',
        re.IGNORECASE,
    ),
    #  Raw HTML fragment leaking into text
    re.compile(
        r'(?:^class\s*=\s*"|&gt;)',
    ),
    #  Single-word CTA labels
    re.compile(
        r'^(?:newsletter|subscribe|share|donate|follow)$',
        re.IGNORECASE,
    ),
    #  Standalone name attributions (1-4 capitalized words).
    # e.g. "Friedrich Nietzsche", "Karl Drais", "Georgia O'Keeffe"
    # These appear as body paragraphs when image captions leak through.
    re.compile(
        r"^(?:[A-Z][a-z']+\s+){0,3}[A-Z][a-z']+$",
    ),
]

# Maximum word count for a paragraph to be considered a caption candidate.
_CAPTION_MAX_WORDS = 30

# : Common image file extensions for filename
# caption detection.  A caption that is a raw filename (e.g.
# "framework-desktop.jpg") exposes the production pipeline to the
# reader and must be suppressed.
_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.webp', '.gif', '.svg', '.bmp',
    '.tiff', '.tif', '.ico', '.avif',
}

# : Regex for filename-like captions.
# Matches strings that look like filenames: no spaces, contains dots
# or hyphens, and ends with an image extension.
_FILENAME_CAPTION_RE = re.compile(
    r'^[^\s]+\.(?:jpe?g|png|webp|gif|svg|bmp|tiff?|ico|avif)$',
    re.IGNORECASE,
)


def _is_filename_caption(caption: str, article_title: str | None = None) -> bool:
    """Detect if an image caption is a raw filename or title echo.

    Suppresses captions that are:
    1. Raw filenames ending in image extensions (e.g. "framework-desktop.jpg",
       "lagkage-med-jordbar-og-malkechokolade.WebP")
    2. Identical (case-insensitive) to the article title -- metadata
       leaking into the visual layer

    Returns True if the caption should be suppressed.
    """
    if not caption:
        return False
    stripped = caption.strip()
    if not stripped:
        return False

    # Check for filename pattern
    if _FILENAME_CAPTION_RE.match(stripped):
        return True

    # Check if caption matches article title (title echo)
    return bool(article_title and stripped.lower() == article_title.strip().lower())


def _is_orphaned_caption(text: str) -> bool:
    """Detect if a paragraph is likely an orphaned image caption.

    Three image captions in the Marginalian
    article render as body text without their corresponding images.
    This heuristic identifies short, descriptive paragraphs that
    match common caption patterns so they can be filtered out.

    Returns True if the paragraph looks like a caption.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # Must be short
    if len(stripped.split()) > _CAPTION_MAX_WORDS:
        return False
    # Check against known caption patterns
    return any(pat.search(stripped) for pat in _CAPTION_PATTERNS)


def _filter_orphaned_captions(paragraphs: list[str]) -> list[str]:
    """Remove orphaned image captions from a list of paragraphs.

    When images exist in the DB but are not
    rendered inline, their captions appear as non sequitur body text.
    This strips them rather than confusing the reader.

    Returns the filtered list.
    """
    return [p for p in paragraphs if not _is_orphaned_caption(p)]


def _unescape_html_entities(text: str) -> str:
    """Unescape HTML character references in display text.

     Belt-and-suspenders fix for HTML entities that leaked
    through ingestion into display_text or image captions. Jinja2
    autoescape double-escapes these (& -> &amp;), making them
    appear as literal &#8220; in rendered output.
    """
    if not text:
        return text
    return html_module.unescape(text)


#  Maximum images per article to prevent clustering.
# Features get more images than standard articles.
MAX_IMAGES_FEATURE = 4
MAX_IMAGES_STANDARD = 3

STYLES_DIR = Path(__file__).parent / "styles"

def split_text_paragraphs(text: str, target_len: int = 800) -> list[str]:
    """Split text into paragraphs using a multi-tier strategy.

     Shared utility for paragraph splitting used by both
    feature and standard templates (design review
    Issue 1 -- standard template lacks paragraph splitting).

    (design review): After
    double-newline splitting, further split any large paragraph on
    single newlines. Many sources (e.g. The Marginalian) use single
    \n as paragraph separators, and the previous Strategy 1
    short-circuited before reaching Strategy 2 when \n\n splits
    produced >1 result. This caused 1,800-word text walls.

    Strategy:
    1. Split on double-newline (\n\n).
    2. Sub-split any resulting paragraph on single newline (\n)
       if the paragraph exceeds target_len chars.
    3. If still a single block, split at sentence boundaries
       every ~target_len chars.

    Returns a list of non-empty paragraph strings.
    """
    if not text:
        return []

    # Strategy 1: split on double newlines
    coarse = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Strategy 2: sub-split large paragraphs on single newlines.
    # This catches sources that use \n (not \n\n) as paragraph
    # separators within double-newline-delimited blocks.
    paras = []
    for p in coarse:
        if len(p) > target_len and "\n" in p:
            sub = [s.strip() for s in p.split("\n") if s.strip()]
            paras.extend(sub)
        else:
            paras.append(p)

    if len(paras) > 1:
        return paras

    # Strategy 3: single block -- split at sentence boundaries
    if len(text) <= target_len:
        return [text]

    result = []
    remaining = text
    while remaining:
        if len(remaining) <= target_len:
            result.append(remaining)
            break
        cut = _find_sentence_boundary(remaining, target_len)
        result.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    return result


def split_feature_text(text: str, max_lead_chars: int = 500, deck: str | None = None) -> tuple:
    """Split feature article text into (lead, body_paragraphs).

     Guarantees every feature article gets a short lead
    and a feature-body div with paragraph breaks, regardless of
    source text formatting ( + New Issue A).

     Lead is always capped at max_lead_chars, even when
    double-newline splitting succeeds (design review
    Issue 2 -- feature leads too long, e.g. Europe at 2,856 chars).
    Uses split_text_paragraphs as the shared splitting backend.

    Returns (lead_text, [body_paragraphs]).
    """
    if not text:
        return ("", [])

    # Use shared paragraph splitting to get raw paragraphs
    paras = split_text_paragraphs(text)
    if not paras:
        return ("", [])

    lead = paras[0]
    body = paras[1:]

    # : Strip the deck sentence from the lead
    # paragraph to prevent the stuttering effect where the reader
    # encounters the same sentence twice -- once as the italic deck,
    # once as the opening of the lead paragraph with a drop cap.
    if deck and lead.startswith(deck):
        lead = lead[len(deck):].lstrip()
        # If stripping the deck empties the lead, promote first body
        # paragraph to lead
        if not lead and body:
            lead = body.pop(0)

    # Cap lead at max_lead_chars, pushing overflow into body
    if len(lead) > max_lead_chars:
        cut = _find_sentence_boundary(lead, max_lead_chars)
        overflow = lead[cut:].lstrip()
        lead = lead[:cut].rstrip()
        if overflow:
            # Split the overflow into paragraphs too
            overflow_paras = split_text_paragraphs(overflow)
            body = overflow_paras + body

    return (lead, body)


def _find_sentence_boundary(text: str, target: int) -> int:
    """Find the best sentence boundary near target position.

    Looks for '. ', '! ', '? ' near the target. Searches backward
    from target first, then forward. Falls back to target if no
    boundary found within 200 chars.
    """
    # Search backward from target (prefer ending before target)
    search_start = max(0, target - 200)
    best = None
    for m in re.finditer(r'[.!?]\s', text[search_start:target + 100]):
        pos = search_start + m.start() + 1  # After the punctuation
        if pos <= target + 50:
            best = pos

    if best is not None:
        return best

    # Fallback: split at target on a space boundary
    space_pos = text.rfind(' ', max(0, target - 50), target + 50)
    if space_pos > 0:
        return space_pos + 1

    return target




def image_insert_indices(
    num_paragraphs: int,
    num_extra_images: int,
) -> dict[int, int]:
    """Compute paragraph indices at which to insert extra images.

     Distributes images evenly through the article body
    instead of clustering them all at the midpoint. Returns a dict
    mapping paragraph index (1-based, matching Jinja loop.index) to
    image index (0-based into the extra_images list).

    Example: 10 paragraphs, 2 extra images -> {3: 0, 7: 1}
    """
    if num_extra_images <= 0 or num_paragraphs <= 1:
        return {}
    # Divide the paragraph range into (num_images + 1) segments
    # and place an image at each segment boundary
    spacing = num_paragraphs / (num_extra_images + 1)
    return {int(spacing * (i + 1)): i for i in range(num_extra_images)}


def _place_ranked_items(
    ranked: RankedEdition,
    config: dict,
) -> CuratedEdition:
    """Place ranked items until the page target is met.

     The renderer is the authority on what fits. It
    processes ranked items in order, tracks estimated page fill,
    and stops when page_target is reached.

    Uses _estimate_item_height() for page-fill tracking.

    Args:
        ranked: The RankedEdition from curation.
        config: The OffScroll config dict.

    Returns:
        A CuratedEdition with only the placed items, ready for
        template rendering.
    """
    page_target = ranked.page_target
    # Reserve space for masthead (~10% of page 1) and colophon (~5%)
    masthead_overhead = 0.10
    colophon_reserve = 0.05
    available_pages = page_target - masthead_overhead - colophon_reserve

    current_fill = 0.0
    placed_items: list[RankedItem] = []

    for ri in ranked.ranked_items:
        if ri.skip:
            continue

        # Estimate how much space this item needs
        height = _estimate_ranked_item_height(ri)

        # Check if we have room. Stop if we exceed available
        # pages and are at 80%+ of target.
        if (
            current_fill + height > available_pages
            and placed_items
            and current_fill >= page_target * 0.8
        ):
            break

        placed_items.append(ri)
        current_fill += height

    logger.info(
        "Placed %d of %d items (%.1f estimated pages, target %d)",
        len(placed_items),
        len(ranked.ranked_items),
        current_fill,
        page_target,
    )

    # Select pull quotes that match placed items
    placed_ids = {ri.item_id for ri in placed_items}
    matched_pqs = [
        pq for pq in ranked.pull_quote_pool
        if pq.source_item_id in placed_ids
    ]
    # If no matched pull quotes, use unmatched ones for filler
    if not matched_pqs and ranked.pull_quote_pool:
        matched_pqs = list(ranked.pull_quote_pool[:2])

    # Convert to CuratedEdition via the model's conversion method
    # But we need to be smarter: only include placed items
    placed_count = len(placed_items)
    edition = ranked.to_curated_edition(placed_count=placed_count)

    # Override pull quotes with our placement-aware selection
    edition.pull_quotes = matched_pqs

    # Store fill info for colophon decision
    edition.estimated_content_pages = current_fill

    return edition


def _estimate_ranked_item_height(ri: RankedItem) -> float:
    """Estimate a RankedItem's height as a fraction of a page.

    Mirrors _estimate_item_height but works on RankedItem.
    """
    wc = ri.word_count
    hint = ri.layout_hint
    if isinstance(hint, str):
        hint = LayoutHint(hint)

    if hint == LayoutHint.BRIEF:
        return 0.04
    if hint == LayoutHint.FEATURE:
        # Features are bigger: full-width with drop cap, hero image
        base = wc / 300
        has_image = bool(ri.images)
        overhead = 0.20 if has_image else 0.10
        return min(1.5, base + overhead)  # Can span 1.5 pages
    # Standard
    base = wc / 385
    has_image = bool(ri.images)
    overhead = 0.12 if has_image else 0.06
    return min(1.0, base + overhead)


def _load_edition(
    config: dict,
    edition_path: Path | None = None,
    edition: CuratedEdition | None = None,
) -> CuratedEdition:
    """Load an edition from a path or use the provided one.

     Auto-detects ranked vs. curated format. If a
    RankedEdition is loaded, it is converted to CuratedEdition
    via page-fill-aware placement.

    If neither path nor edition is provided, look for the latest
    edition in the data directory.
    """
    if edition is not None:
        return edition
    if edition_path is not None:
        fmt = detect_edition_format(edition_path)
        if fmt == "ranked":
            ranked = RankedEdition.from_json(edition_path)
            return _place_ranked_items(ranked, config)
        return CuratedEdition.from_json(edition_path)
    data_dir = Path(config["output"]["data_dir"])
    editions = sorted(data_dir.glob("edition-*.json"), reverse=True)
    if not editions:
        raise FileNotFoundError(
            f"No edition files found in {data_dir}. Run \'offscroll curate\' first."
        )
    path = editions[0]
    fmt = detect_edition_format(path)
    if fmt == "ranked":
        ranked = RankedEdition.from_json(path)
        return _place_ranked_items(ranked, config)
    return CuratedEdition.from_json(path)


def _extract_front_page_feature(edition: CuratedEdition):
    """Extract the feature article from any section for page 1.

    Searches all sections (not just the first) for the first item
    with layout_hint == FEATURE. Returns (feature_item,
    feature_section_index) or (None, None) if no feature is found.
    The feature is removed from its section's items list so it is
    not rendered twice. If removing it leaves the section empty,
    the section is removed entirely.
    """
    if not edition.sections:
        return None, None
    for sec_idx, section in enumerate(edition.sections):
        for i, item in enumerate(section.items):
            if (
                not isinstance(item, CuratedThread)
                and hasattr(item, "layout_hint")
                and item.layout_hint == LayoutHint.FEATURE
            ):
                feature = section.items.pop(i)
                # Remove the section if it's now empty
                if not section.items:
                    edition.sections.pop(sec_idx)
                return feature, sec_idx
    return None, None


def _build_pull_quote_map(
    pull_quotes: list[PullQuote],
    edition: CuratedEdition,
) -> dict[str, list[PullQuote]]:
    """Build a mapping from item_id to pull quotes sourced from that item.

    For thread sub-items, maps the pull quote to the parent thread's
    thread_id so it renders after the whole thread block.
    """
    # Build a set of thread sub-item IDs -> parent thread_id
    sub_item_to_thread: dict[str, str] = {}
    for section in edition.sections:
        for item in section.items:
            if isinstance(item, CuratedThread):
                for sub in item.items:
                    sub_item_to_thread[sub.item_id] = item.thread_id

    pq_map: dict[str, list[PullQuote]] = defaultdict(list)
    for pq in pull_quotes:
        source_id = pq.source_item_id
        # If the source is a thread sub-item, attach to the thread
        if source_id in sub_item_to_thread:
            pq_map[sub_item_to_thread[source_id]].append(pq)
        else:
            pq_map[source_id].append(pq)
    return dict(pq_map)


def _estimate_item_height(item) -> float:
    """Estimate item height as a fraction of a page.

    Uses word count as a proxy. At 10pt body, ~7 words/line,
    ~55 lines/page in a single column. Multi-column layouts
    are shorter per item because they share width.

    Returns a float between 0 and 1 representing fraction of
    a full page height.
    """
    if isinstance(item, CuratedThread):
        # Thread: headline + deck + all sub-item text
        total_words = sum(sub.word_count for sub in item.items)
        # Threads have extra chrome (numbers, left border, spacing)
        return min(1.0, (total_words / 300) + 0.08)
    wc = getattr(item, "word_count", 0)
    hint = getattr(item, "layout_hint", LayoutHint.STANDARD)
    if hint == LayoutHint.BRIEF:
        return 0.04  # Briefs are very short
    # Standard/feature: ~385 words fills a full column
    base = wc / 385
    # Headlines, images, margins add ~8% overhead
    has_image = bool(getattr(item, "images", []))
    overhead = 0.12 if has_image else 0.06
    return min(1.0, base + overhead)


def _will_inline_pull_quotes(item, pull_quotes_by_item: dict[str, list[PullQuote]]) -> bool:
    """Check if an item will render its pull quotes inline in the template.

    Both the feature and standard templates
    place pull quotes inline when an article is > 1000 words and has
    > 3 paragraphs. If they will be placed inline, they should NOT
    also be rendered after the row (which causes duplication).

    This mirrors the condition in standard.html / feature.html:
        item.word_count > 1000 and item_pqs|length > 0 and paragraphs|length > 3
    """
    if isinstance(item, CuratedThread):
        return False
    wc = getattr(item, "word_count", 0)
    if wc <= 1000:
        return False
    item_id = getattr(item, "item_id", None)
    if not item_id or item_id not in pull_quotes_by_item:
        return False
    if not pull_quotes_by_item[item_id]:
        return False
    # Estimate paragraph count from display_text
    text = getattr(item, "display_text", "") or ""
    # Use same logic as split_text_paragraphs: count double-newline segments,
    # sub-split on single newlines for long ones.
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Sub-split large paragraphs
    expanded = []
    for p in paras:
        if len(p) > 800 and "\n" in p:
            expanded.extend(s.strip() for s in p.split("\n") if s.strip())
        else:
            expanded.append(p)
    return len(expanded) > 3


def _compose_section_rows(
    section,
    pull_quotes_by_item: dict[str, list[PullQuote]],
) -> list[dict]:
    """Assign items to flex rows within a section.

     Aggressive packing heuristic.

    Groups standard/feature items and threads into rows of 1-3
    columns, packing content together to maximize page fill.
    Returns a list of row dicts, each with:
      - "columns": list of column dicts with "width", "col_items",
        "ruled", and optionally "briefs"
      - "pull_quotes": list of full-width pull quotes after this row

    Packing rules :
    - Thread + standard(s) + briefs share a row when possible
    - Thread + 1 standard + briefs: 3-col row [std, thread, briefs]
    - Thread + 1 standard (no briefs): 2-col row [std, thread]
    - Thread + briefs (no standards): 2-col row [thread, briefs]
    - Thread alone: single full-width row
    - 2+ standards with briefs: [std, std, briefs] in 3-col row
    - 2 standards no briefs: [std, std] in 2-col row
    - 1 standard with briefs: [std, briefs] in 2-col row
    - Remaining standards fill 2-col rows
    """
    standards = []
    threads = []
    briefs = []

    for item in section.items:
        if isinstance(item, CuratedThread):
            threads.append(item)
        elif hasattr(item, "layout_hint") and item.layout_hint == LayoutHint.BRIEF:
            briefs.append(item)
        elif hasattr(item, "layout_hint") and item.layout_hint == LayoutHint.FEATURE:
            # Features inside content-area treated as standards
            standards.append(item)
        else:
            standards.append(item)

    rows = []
    briefs_placed = False
    standards_placed = set()  # Track which standards were packed with threads

    def _item_pqs(item):
        """Get pull quotes for an item."""
        if isinstance(item, CuratedThread):
            return pull_quotes_by_item.get(item.thread_id, [])
        return pull_quotes_by_item.get(item.item_id, [])

    #  Pack threads with standards when possible.
    # Threads and standards share rows for better page fill.
    for thread in threads:
        pqs = _item_pqs(thread)

        # Find an unplaced standard to pack with this thread
        pack_std = None
        pack_idx = None
        for si, std in enumerate(standards):
            if si not in standards_placed:
                pack_std = std
                pack_idx = si
                break

        if pack_std is not None and briefs and not briefs_placed:
            # Thread + standard + briefs in 3-col row
            pqs = pqs + _item_pqs(pack_std)
            rows.append(
                {
                    "columns": [
                        {"width": 1, "col_items": [pack_std], "ruled": False},
                        {"width": 1, "col_items": [thread], "ruled": True},
                        {"width": 1, "col_items": [], "ruled": True, "briefs": briefs},
                    ],
                    "pull_quotes": pqs,
                }
            )
            standards_placed.add(pack_idx)
            briefs_placed = True
        elif pack_std is not None:
            # Thread + standard in 2-col row
            pqs = pqs + _item_pqs(pack_std)
            rows.append(
                {
                    "columns": [
                        {"width": 1, "col_items": [pack_std], "ruled": False},
                        {"width": 1, "col_items": [thread], "ruled": True},
                    ],
                    "pull_quotes": pqs,
                }
            )
            standards_placed.add(pack_idx)
        elif briefs and not briefs_placed:
            # Thread + briefs share a 2-col row
            rows.append(
                {
                    "columns": [
                        {"width": 1, "col_items": [thread], "ruled": False},
                        {"width": 1, "col_items": [], "ruled": True, "briefs": briefs},
                    ],
                    "pull_quotes": pqs,
                }
            )
            briefs_placed = True
        else:
            rows.append(
                {
                    "columns": [
                        {"width": 1, "col_items": [thread], "ruled": False},
                    ],
                    "pull_quotes": pqs,
                }
            )

    #  Separate long articles from short ones.
    # Long articles (>500 words) get their own full-width row
    # to avoid quarter-width columns .
    remaining_stds = [s for si, s in enumerate(standards) if si not in standards_placed]
    long_stds = [s for s in remaining_stds if getattr(s, 'word_count', 0) > 500]
    short_stds = [s for s in remaining_stds if getattr(s, 'word_count', 0) <= 500]

    # Place long articles first, each in their own full-width row
    for item_a in long_stds:
        pqs = _item_pqs(item_a)
        # : Suppress row-level pull quotes when
        # the item will render them inline (word_count > 1000, > 3 paras).
        # This prevents the duplicate pull quote that appeared when the
        # Marginalian rendered through the standard template.
        if _will_inline_pull_quotes(item_a, pull_quotes_by_item):
            pqs = []
        rows.append(
            {
                "columns": [
                    {"width": 1, "col_items": [item_a], "ruled": False},
                ],
                "pull_quotes": pqs,
            }
        )

    # Place short standards in 2-column rows as before
    i = 0
    while i < len(short_stds):
        remaining = len(short_stds) - i

        if remaining >= 2:
            item_a = short_stds[i]
            item_b = short_stds[i + 1]
            pqs = _item_pqs(item_a) + _item_pqs(item_b)

            if briefs and not briefs_placed:
                # 2 standards + briefs in 3-col row
                rows.append(
                    {
                        "columns": [
                            {"width": 1, "col_items": [item_a], "ruled": False},
                            {"width": 1, "col_items": [item_b], "ruled": True},
                            {"width": 1, "col_items": [], "ruled": True, "briefs": briefs},
                        ],
                        "pull_quotes": pqs,
                    }
                )
                briefs_placed = True
            else:
                rows.append(
                    {
                        "columns": [
                            {"width": 1, "col_items": [item_a], "ruled": False},
                            {"width": 1, "col_items": [item_b], "ruled": True},
                        ],
                        "pull_quotes": pqs,
                    }
                )
            i += 2
        else:
            # Single remaining standard
            item_a = short_stds[i]
            pqs = _item_pqs(item_a)

            if briefs and not briefs_placed:
                # 1 standard + briefs in 2-col row
                rows.append(
                    {
                        "columns": [
                            {"width": 1, "col_items": [item_a], "ruled": False},
                            {"width": 1, "col_items": [], "ruled": True, "briefs": briefs},
                        ],
                        "pull_quotes": pqs,
                    }
                )
                briefs_placed = True
            else:
                rows.append(
                    {
                        "columns": [
                            {"width": 1, "col_items": [item_a], "ruled": False},
                        ],
                        "pull_quotes": pqs,
                    }
                )
            i += 1

    # If only briefs remain (no standards, no threads)
    if briefs and not briefs_placed:
        rows.append(
            {
                "columns": [
                    {"width": 1, "col_items": [], "ruled": False, "briefs": briefs},
                ],
                "pull_quotes": [],
            }
        )

    #  Inject section heading into first row for inline rendering.
    # This prevents section headers from being stranded on their own pages
    #  by making the header part of the row content.
    if rows:
        rows[0]['section_heading'] = getattr(section, 'heading', None)

    return rows


def _build_html(edition: CuratedEdition, config: dict) -> str:
    """Render a CuratedEdition to a complete HTML string.

    Loads Jinja2 templates, reads the CSS, and produces a
    self-contained HTML document with inlined styles.

    composition rules (preserved):
    1. Masthead + feature on page 1 (no isolated masthead page)
    2. Pull quotes inline after source article
    3. Sections flow continuously (no forced page breaks)
    4. Briefs clustered under "In Brief" label
    5. Thread template with numbered posts and deck line

     Flexbox layout with explicit row/column composition.
     Image path resolution, unmatched pull quotes fallback.
    """
    data_dir = Path(config.get("output", {}).get("data_dir", "~/.offscroll/data"))
    if str(data_dir).startswith("~"):
        data_dir = data_dir.expanduser()

    def first_alpha_index(text: str) -> int:
        """Return the index of the first alphabetic character in text.

        Used by drop cap logic so we skip leading punctuation like
        quotes, em dashes, etc. and always pick a letter.
        Returns 0 if no alpha character is found.
        """
        for i, ch in enumerate(text):
            if ch.isalpha():
                return i
        return 0

    def resolve_image_path(local_path: str) -> str:
        """Resolve a relative image path to an absolute file:// URI.

        WeasyPrint needs absolute paths or file:// URIs to load
        images from the local filesystem. The local_path in edition
        JSON is relative to data_dir (e.g. 'images/tag:.../file.jpg').
        """
        if not local_path:
            return ""
        p = Path(local_path)
        if p.is_absolute():
            return p.as_uri()
        resolved = data_dir / local_path
        if resolved.exists():
            return resolved.as_uri()
        return local_path

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    env.globals["first_alpha_index"] = first_alpha_index
    env.globals["resolve_image_path"] = resolve_image_path
    env.globals["split_feature_text"] = split_feature_text
    env.globals["split_text_paragraphs"] = split_text_paragraphs
    env.globals["image_insert_indices"] = image_insert_indices
    env.globals["generate_feature_deck"] = _generate_feature_deck
    env.globals["filter_orphaned_captions"] = _filter_orphaned_captions

    typography_css = (STYLES_DIR / "typography.css").read_text()
    newspaper_css = (STYLES_DIR / "newspaper.css").read_text()
    css_content = typography_css + "\n" + newspaper_css

    page_size = config.get("newspaper", {}).get("page_size", "letter")

    #  Strip boilerplate from display_text at render time
    #  Also fix subheading concatenation 
    #  Unescape HTML entities, strip editorial ellipsis,
    #            cap images per article
    for section in edition.sections:
        for item in section.items:
            if hasattr(item, 'display_text') and item.display_text:
                item.display_text = _unescape_html_entities(item.display_text)
                item.display_text = _strip_display_boilerplate(item.display_text)
                item.display_text = _fix_subheading_concatenation(item.display_text)
                item._edited_for_length = _has_editorial_ellipsis(item.display_text)
            #  Unescape image captions
            # : Suppress filename captions
            if hasattr(item, 'images'):
                item_title = getattr(item, 'title', None)
                for img in item.images:
                    if hasattr(img, 'caption') and img.caption:
                        img.caption = _unescape_html_entities(img.caption)
                        if _is_filename_caption(img.caption, item_title):
                            img.caption = None
            #  Cap images per article
            if hasattr(item, 'images') and hasattr(item, 'layout_hint'):
                max_imgs = (
                    MAX_IMAGES_FEATURE
                    if item.layout_hint == LayoutHint.FEATURE
                    else MAX_IMAGES_STANDARD
                )
                if len(item.images) > max_imgs:
                    item.images = item.images[:max_imgs]
            if isinstance(item, CuratedThread):
                for sub in item.items:
                    if hasattr(sub, 'display_text') and sub.display_text:
                        sub.display_text = _unescape_html_entities(sub.display_text)
                        sub.display_text = _strip_display_boilerplate(sub.display_text)
                        sub.display_text = _fix_subheading_concatenation(sub.display_text)
                        sub._edited_for_length = _has_editorial_ellipsis(sub.display_text)

    # Rule 1: Extract front-page feature
    front_feature, _ = _extract_front_page_feature(edition)

    # : Enforce single feature per edition.
    # After extracting the front-page feature, demote any remaining
    # FEATURE items to STANDARD to create visual hierarchy.
    for section in edition.sections:
        for item in section.items:
            if (
                not isinstance(item, CuratedThread)
                and hasattr(item, "layout_hint")
                and item.layout_hint == LayoutHint.FEATURE
            ):
                item.layout_hint = LayoutHint.STANDARD

    #  Fix concatenation on front feature too
    if (
        front_feature is not None
        and hasattr(front_feature, 'display_text')
        and front_feature.display_text
    ):
        front_feature.display_text = _unescape_html_entities(front_feature.display_text)
        front_feature.display_text = _strip_display_boilerplate(front_feature.display_text)
        front_feature.display_text = _fix_subheading_concatenation(front_feature.display_text)
        front_feature._edited_for_length = _has_editorial_ellipsis(front_feature.display_text)
        #  Unescape front feature image captions and cap images
        # : Suppress filename captions
        if front_feature.images:
            ff_title = getattr(front_feature, 'title', None)
            for img in front_feature.images:
                if hasattr(img, 'caption') and img.caption:
                    img.caption = _unescape_html_entities(img.caption)
                    if _is_filename_caption(img.caption, ff_title):
                        img.caption = None
            if len(front_feature.images) > MAX_IMAGES_FEATURE:
                front_feature.images = front_feature.images[:MAX_IMAGES_FEATURE]

    # Rule 2: Build pull quote placement map
    pq_map = _build_pull_quote_map(edition.pull_quotes, edition)

    #  Collect unmatched pull quotes for fallback rendering.
    # Pull quotes with source_item_id == "unknown" or not matching any
    # item in the edition are rendered in a Notable Quotes block.
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
        pq for pq in edition.pull_quotes
        if pq.source_item_id == "unknown"
        or pq.source_item_id not in all_item_ids
    ]

    #  Assign kicker labels to feature items.
    # Only the front-page feature (rank 1) gets "Cover Story".
    # All other features in sections get their section name as kicker.
    # ( every article labeled "Cover Story")
    if front_feature is not None:
        front_feature.kicker = "Cover Story"
    for section in edition.sections:
        for item in section.items:
            if (
                not isinstance(item, CuratedThread)
                and hasattr(item, "layout_hint")
                and item.layout_hint == LayoutHint.FEATURE
            ):
                item.kicker = section.heading

    #  Compose flex rows for each section
    section_rows = {}
    for section in edition.sections:
        section_rows[section.heading] = _compose_section_rows(section, pq_map)

    debug_mode = config.get("newspaper", {}).get("debug_mode", False)

    #  Determine if colophon is needed based on page fill
    show_colophon = True  # Always show colophon as closing element

    template = env.get_template("base.html")
    return template.render(
        edition=edition.edition,
        sections=edition.sections,
        pull_quotes=edition.pull_quotes,
        pull_quotes_by_item=pq_map,
        unmatched_pull_quotes=unmatched_pqs,
        css_content=css_content,
        page_target=edition.page_target,
        curation_summary=edition.curation_summary,
        page_size=page_size,
        front_feature=front_feature,
        section_rows=section_rows,
        debug_mode=debug_mode,
        show_colophon=show_colophon,
    )


def render_newspaper_html(
    config: dict,
    edition_path: Path | None = None,
    edition: CuratedEdition | None = None,
) -> Path:
    """Render a CuratedEdition to styled newspaper HTML.

    Args:
        config: The OffScroll config dict.
        edition_path: Path to edition.json. If None, uses the latest.
        edition: Pre-loaded CuratedEdition. If provided, edition_path
                 is ignored.

    Returns:
        Path to the generated HTML file.
    """
    ed = _load_edition(config, edition_path, edition)
    html_content = _build_html(ed, config)

    output_dir = Path(config["output"]["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"newspaper-{ed.edition.date}.html"
    output_path.write_text(html_content)

    logger.info("Newspaper HTML written to %s", output_path)
    return output_path


def render_newspaper_pdf(
    config: dict,
    edition_path: Path | None = None,
    edition: CuratedEdition | None = None,
) -> Path:
    """Render a CuratedEdition to a newspaper PDF via WeasyPrint.

    Args:
        config: The OffScroll config dict.
        edition_path: Path to edition.json. If None, uses the latest.
        edition: Pre-loaded CuratedEdition. If provided, edition_path
                 is ignored.

    Returns:
        Path to the generated PDF file.
    """
    from weasyprint import HTML

    ed = _load_edition(config, edition_path, edition)
    html_content = _build_html(ed, config)

    output_dir = Path(config["output"]["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"newspaper-{ed.edition.date}.pdf"

    HTML(string=html_content).write_pdf(str(output_path))

    logger.info("Newspaper PDF written to %s", output_path)
    return output_path


def render_newspaper(
    config: dict,
    fmt: str = "pdf",
    edition_path: Path | None = None,
    edition: CuratedEdition | None = None,
) -> Path:
    """Render a CuratedEdition to the specified format.

    Args:
        config: The OffScroll config dict.
        fmt: Output format -- "pdf" or "html".
        edition_path: Path to edition.json.
        edition: Pre-loaded CuratedEdition.

    Returns:
        Path to the generated file.

    Raises:
        ValueError: If fmt is not "pdf" or "html".
    """
    if fmt == "pdf":
        return render_newspaper_pdf(config, edition_path, edition)
    if fmt == "html":
        return render_newspaper_html(config, edition_path, edition)
    raise ValueError(f"Unknown output format: '{fmt}'. Use 'pdf' or 'html'.")
