"""LLM editorial layer (Claude + Ollama).

Ollama-powered editorial layer that refines
curation output with LLM-generated section headings, editorial notes,
pull quotes, and layout hints.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time

from offscroll.models import CuratedEdition, LayoutHint, Section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama LLM Client Wrapper
# ---------------------------------------------------------------------------


def _call_ollama(
    prompt: str,
    config: dict,
    system_prompt: str | None = None,
) -> str:
    """Call Ollama with a prompt and return the response text.

    Args:
        prompt: The user prompt.
        config: Config dict. Uses config["curation"]["ollama_model"]
            (default: "llama3.1:8b") and config["curation"]["ollama_url"]
            (default: "http://localhost:11434").
        system_prompt: Optional system prompt for context.

    Returns:
        The model's response text, stripped of whitespace.

    Raises:
        ConnectionError: If Ollama is not reachable.
        ImportError: If the ollama package is not installed.
    """
    try:
        import ollama
    except ImportError as e:
        msg = (
            "The 'ollama' package is required for editorial layer. "
            "Install it with: pip install ollama"
        )
        raise ImportError(msg) from e

    curation_config = config.get("curation", {})
    ollama_model = curation_config.get("ollama_model", "llama3.1:8b")
    ollama_url = curation_config.get("ollama_url", "http://localhost:11434")

    try:
        client = ollama.Client(host=ollama_url)

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call the model
        response = client.chat(model=ollama_model, messages=messages)

        # Extract and return response text
        return response.message.content.strip()
    except (ConnectionError, TimeoutError) as e:
        msg = f"Ollama is not reachable at {ollama_url}. Please check connection."
        raise ConnectionError(msg) from e


# ---------------------------------------------------------------------------
# Section Heading Validation
# ---------------------------------------------------------------------------

_REFUSAL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"i don'?t see",
        r"please provide",
        r"i cannot",
        r"i can'?t",
        r"sorry",
        r"i'?m not able",
        r"i'?m unable",
        r"as an ai",
        r"i don'?t have",
        r"no (?:content|text|information)",
    ]
]


def _is_valid_heading(heading: str) -> bool:
    """Check if an LLM-generated heading is valid.

    Rejects empty strings, headings over 60 characters,
    and common LLM refusal patterns.
    """
    if not heading or not heading.strip():
        return False
    stripped = heading.strip()
    if len(stripped) > 60:
        return False
    return not any(pat.search(stripped) for pat in _REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# Section Heading Generation
# ---------------------------------------------------------------------------


def generate_section_headings(
    sections: list[Section],
    config: dict,
) -> list[Section]:
    """Replace generic section headings with LLM-generated ones.

    Takes sections with headings like "Topic 1" and replaces them
    with descriptive, newspaper-style section names based on the
    content of the items in each section.

    Args:
        sections: List of Section objects with items.
        config: The OffScroll config dict.

    Returns:
        The same sections with updated headings.
    """
    if not sections:
        return sections

    updated_sections = []
    for section in sections:
        if not section.items:
            # Keep empty sections as-is
            updated_sections.append(section)
            continue

        # : Include article titles and longer
        # content snippets so the LLM can classify by actual topic,
        # not just the source feed.  Previously only 200 chars from
        # 2 items were sent, causing a Nietzsche philosophy essay
        # from a tech-adjacent feed to be labeled "Technology News."
        sample_items = section.items[:3]  # Use up to 3 items for context
        content_samples = []
        for item in sample_items:
            title = getattr(item, "title", "") or ""
            text = getattr(item, "display_text", "") or ""
            # Include title prominently, plus 300 chars of body
            entry = f"Title: {title}\nContent: {text[:300]}"
            content_samples.append(entry)

        content_context = "\n---\n".join(content_samples)

        # : Improved prompt for newspaper-
        # convention section headings.  Prior prompt produced LLM
        # marketing slogans ("Streamline Your Workflow").  New prompt
        # explicitly requests category labels or thematic descriptors.
        #
        # : Prompt now emphasizes classifying
        # by actual article content, not by source/feed origin.
        # A philosophy essay from a tech blog is philosophy, not tech.
        prompt = (
            f"Given the following article titles and content snippets, "
            f"suggest a newspaper section heading (2-4 words). "
            f"Classify based on the ACTUAL TOPIC of the articles, "
            f"not the source website or feed. For example, a philosophy "
            f"essay is 'Philosophy' or 'Ideas', not 'Technology' just "
            f"because it comes from a tech blog. "
            f"The heading should be a category label (like 'Technology', "
            f"'Science', 'Culture', 'Philosophy', 'Ideas') or a thematic "
            f"descriptor (like 'The Digital Frontier', 'Making Things'). "
            f"Do NOT use marketing slogans, imperative verbs, or quoted "
            f"phrases. Return only the heading text, nothing else.\n\n"
            f"Articles:\n{content_context}"
        )

        try:
            new_heading = _call_ollama(prompt, config)
            # : Strip quotation marks that LLMs
            # sometimes wrap headings in.
            if new_heading:
                new_heading = new_heading.strip()
                # Remove leading/trailing quote characters one at a time
                _QUOTE_CHARS = set('"\'"\u201c\u201d\u2018\u2019')
                while new_heading and new_heading[0] in _QUOTE_CHARS:
                    new_heading = new_heading[1:]
                while new_heading and new_heading[-1] in _QUOTE_CHARS:
                    new_heading = new_heading[:-1]
            if _is_valid_heading(new_heading):
                section.heading = new_heading.strip()
            else:
                # Fallback: use first item's title, truncated
                titles = [
                    item.title for item in section.items if hasattr(item, "title") and item.title
                ]
                if titles:
                    section.heading = titles[0][:40]
                # else: keep existing generic heading
        except (ConnectionError, ImportError):
            # Keep existing heading on error
            pass

        updated_sections.append(section)

    return updated_sections


# ---------------------------------------------------------------------------
# Editorial Notes
# ---------------------------------------------------------------------------


def generate_editorial_notes(
    edition: CuratedEdition,
    config: dict,
) -> CuratedEdition:
    """Generate editorial notes for the edition and individual items.

    - Edition-level editorial_note: a brief opening editorial
      summarizing the week's themes.
    - Per-item editorial_note: one-sentence context for featured
      and standard items (skip briefs).

    Args:
        edition: The CuratedEdition to enhance.
        config: The OffScroll config dict.

    Returns:
        The edition with editorial notes populated.
    """
    if not edition.sections:
        return edition

    # Generate edition-level editorial note
    section_headings = [s.heading for s in edition.sections if s.heading]
    themes = ", ".join(section_headings[:5])  # Summarize first 5 sections

    edition_prompt = (
        f"Write a 1-2 sentence opening editorial note for a newspaper "
        f"edition with these main themes: {themes}. "
        f"Be warm and inviting. Return only the note, no attribution."
    )

    with contextlib.suppress(ConnectionError, ImportError):
        edition.edition.editorial_note = _call_ollama(edition_prompt, config)

    # Generate per-item editorial notes for featured and standard items
    for section in edition.sections:
        for item in section.items:
            # Skip threads and briefs
            if not hasattr(item, "display_text"):
                continue
            if item.layout_hint == LayoutHint.BRIEF:
                continue

            # Generate context note for this item
            display_text = item.display_text[:300] if item.display_text else ""
            item_prompt = (
                f"Provide 1-2 sentence context or commentary for this article: "
                f'"{display_text}...". Be concise and neutral. '
                f"Return only the note."
            )

            with contextlib.suppress(ConnectionError, ImportError):
                item.editorial_note = _call_ollama(item_prompt, config)

    return edition


# ---------------------------------------------------------------------------
# Pull Quote Extraction
# ---------------------------------------------------------------------------


_LLM_PREAMBLE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^here (?:are|is)\b",
        r"^sure[!,.]",
        r"^the following\b",
        r"^certainly[!,.]",
        r"^of course[!,.]",
        r"^\d+\.\s",
        r"^-\s",
        r"^\*\s",
        r'^["\u201c]?quote\s+\d',
    ]
]


def _clean_pull_quote(text: str) -> str | None:
    """Clean a pull quote, removing LLM preamble and numbering.

    Returns None if the text is pure preamble and should be discarded.
    """
    text = text.strip()
    if not text:
        return None

    # Strip leading numbering like "1. ", "2) ", "- ", "* "
    text = re.sub(r"^\d+[.)]\s*", "", text).strip()
    text = re.sub(r"^[-*]\s*", "", text).strip()

    # Strip surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("\u201c") and text.endswith("\u201d")
    ):
        text = text[1:-1].strip()

    if not text:
        return None

    # Reject if the entire line is LLM preamble
    if any(pat.search(text) for pat in _LLM_PREAMBLE_PATTERNS):
        return None

    return text


def _find_source_item(
    quote_text: str,
    items_with_text: list[tuple[str, str]],
) -> str:
    """Find which item a pull quote came from.

    Uses substring containment search across the full display_text
    of each item, not just the first 100 characters.

    Args:
        quote_text: The pull quote text.
        items_with_text: List of (item_id, display_text) tuples.

    Returns:
        The item_id of the matching item, or "unknown".
    """
    # Try exact substring match first
    for item_id, display_text in items_with_text:
        if quote_text in display_text:
            return item_id

    # Try fuzzy: check if most words (80%+) of the quote appear in the item
    quote_words = set(quote_text.lower().split())
    if len(quote_words) < 3:
        return "unknown"
    best_match = "unknown"
    best_overlap = 0.0
    for item_id, display_text in items_with_text:
        item_words = set(display_text.lower().split())
        overlap = len(quote_words & item_words) / len(quote_words)
        if overlap > best_overlap and overlap >= 0.8:
            best_overlap = overlap
            best_match = item_id
    return best_match


def extract_pull_quotes(
    edition: CuratedEdition,
    config: dict,
) -> CuratedEdition:
    """Extract 2-3 pull quotes from the edition content.

     improved source attribution and LLM preamble cleaning.
     If the edition already has quality pull quotes from
    the heuristic selector (which applies skip-first-sentence,
    min/max word count, and strong-claim scoring), keep those
    rather than replacing with LLM-generated ones. The heuristic
    selector is more reliable for pull quote quality than the LLM.

    Args:
        edition: The CuratedEdition to enhance.
        config: The OffScroll config dict.

    Returns:
        The edition with pull_quotes updated.
    """
    #  Preserve heuristic pull quotes when they exist.
    # The heuristic _select_pull_quote in selection.py applies quality
    # rules (skip first sentence, min/max word count, strong claim
    # scoring) that the LLM does not consistently respect.
    if edition.pull_quotes:
        return edition

    if not edition.sections:
        return edition

    # Collect content from all items with full text for attribution
    all_content = []
    items_with_text: list[tuple[str, str]] = []

    for section in edition.sections:
        for item in section.items:
            if not hasattr(item, "display_text"):
                continue
            text = item.display_text or ""
            if text:
                all_content.append(text[:500])
                item_id = getattr(item, "item_id", "unknown")
                items_with_text.append((item_id, text))

    if not all_content:
        return edition

    content_block = "\n---\n".join(all_content[:10])  # Limit to first 10 items

    prompt = (
        f"From the following article snippets, extract 2-3 striking "
        f"quotes (full sentences). Each quote should be compelling and "
        f"representative. Return only the quotes, one per line, "
        f"without attribution or numbering.\n\n{content_block}"
    )

    try:
        response = _call_ollama(prompt, config)
        raw_lines = [q.strip() for q in response.split("\n") if q.strip()]

        # Clean quotes and filter out LLM preamble
        cleaned_quotes = []
        for line in raw_lines:
            cleaned = _clean_pull_quote(line)
            if cleaned:
                cleaned_quotes.append(cleaned)

        # Limit to 3 quotes
        edition.pull_quotes = []
        for quote_text in cleaned_quotes[:3]:
            # Find source item using full-text search
            source_item_id = _find_source_item(quote_text, items_with_text)

            # Determine attribution from source item
            attribution = "Editor"
            if source_item_id != "unknown":
                for section in edition.sections:
                    for item in section.items:
                        if hasattr(item, "item_id") and item.item_id == source_item_id:
                            attribution = getattr(item, "author", "Editor")
                            break

            from offscroll.models import PullQuote

            edition.pull_quotes.append(
                PullQuote(
                    text=quote_text,
                    attribution=attribution,
                    source_item_id=source_item_id,
                )
            )
    except (ConnectionError, ImportError):
        # Keep existing pull quotes
        pass

    return edition


# ---------------------------------------------------------------------------
# Layout Hint Refinement
# ---------------------------------------------------------------------------


def assign_layout_hints(
    edition: CuratedEdition,
    config: dict,
) -> CuratedEdition:
    """Refine layout hints using LLM judgment.

    The heuristic assigns hints by word count.
    The LLM can adjust hints for non-cover articles: promote a
    short but important item to STANDARD, or mark a long but
    shallow item as BRIEF. The LLM sees the full edition context
    and can make better decisions.

    The LLM must NOT override the curation
    optimizer's FEATURE (cover story) assignment. The optimizer
    selects the cover story based on quality score, word count, and
    content depth. The LLM previously promoted a 174-word product
    update to feature status, displacing a 2,800-word essay from
    the cover. The cover story assignment is now authoritative.

    Args:
        edition: The CuratedEdition to enhance.
        config: The OffScroll config dict.

    Returns:
        The edition with refined layout_hint values.
    """
    if not edition.sections:
        return edition

    # Process items one at a time
    for section in edition.sections:
        for item in section.items:
            if not hasattr(item, "display_text"):
                continue

            # : Never override FEATURE assignment.
            # The curation optimizer's cover story selection is authoritative.
            # The LLM can only adjust STANDARD <-> BRIEF hints.
            if item.layout_hint == LayoutHint.FEATURE:
                continue

            display_text = item.display_text[:300] if item.display_text else ""
            current_hint = (
                item.layout_hint.value
                if hasattr(item.layout_hint, "value")
                else str(item.layout_hint)
            )

            prompt = (
                f"A newspaper article has {item.word_count} words. "
                f"Current layout hint is '{current_hint}'. "
                f'Here\'s the content: "{display_text}..."\n\n'
                f"Based on importance and readability, should this be "
                f"'standard' (regular column) or 'brief' (short mention)? "
                f"Return only one word: standard or brief."
            )

            try:
                response = _call_ollama(prompt, config)
                hint_str = response.strip().lower()
                #  Only allow standard or brief from LLM.
                # The LLM cannot promote articles to feature status.
                if hint_str in ("standard", "brief"):
                    item.layout_hint = LayoutHint(hint_str)
            except (ConnectionError, ImportError):
                # Keep existing hint
                pass

    return edition


# ---------------------------------------------------------------------------
# Curation Metadata
# ---------------------------------------------------------------------------


def generate_curation_metadata(
    edition: CuratedEdition,
    config: dict,
) -> CuratedEdition:
    """Generate human-readable curation rationale for each item.

    Adds a selection_rationale field explaining why each item was
    selected -- in human terms, not optimization math.

    Args:
        edition: The CuratedEdition to enhance.
        config: The OffScroll config dict.

    Returns:
        The edition with curation metadata populated.
    """
    if not edition.sections:
        return edition

    for section in edition.sections:
        for item in section.items:
            if not hasattr(item, "display_text"):
                continue

            display_text = item.display_text[:200] if item.display_text else ""
            author = getattr(item, "author", "Unknown")

            prompt = (
                f"Explain in 1-2 sentences why this article by {author} "
                f"was selected for the edition. Be conversational. "
                f'Content: "{display_text}..." '
                f"Return only the explanation."
            )

            try:
                rationale = _call_ollama(prompt, config)
                # Set the declared field on CuratedItem
                item.selection_rationale = rationale
            except (ConnectionError, ImportError):
                # Set a default if LLM fails
                item.selection_rationale = "Selected for topical relevance."

    return edition


# ---------------------------------------------------------------------------
# Editorial Orchestrator
# ---------------------------------------------------------------------------


def run_editorial(
    edition: CuratedEdition,
    config: dict,
) -> CuratedEdition:
    """Run all editorial sub-tasks on a CuratedEdition.

    Orchestrates: section headings -> editorial notes ->
    pull quotes -> layout hints -> curation metadata.

    If Ollama is not available, logs a warning and returns the
    edition unchanged (graceful degradation).

    Args:
        edition: The CuratedEdition from the optimizer.
        config: The OffScroll config dict.

    Returns:
        The editorially polished CuratedEdition.
    """
    try:
        logger.info("Starting editorial layer processing")
        start_time = time.time()

        # Section headings
        t0 = time.time()
        edition.sections = generate_section_headings(edition.sections, config)
        logger.info(f"Generated section headings in {time.time() - t0:.2f}s")

        # Editorial notes
        t0 = time.time()
        edition = generate_editorial_notes(edition, config)
        logger.info(f"Generated editorial notes in {time.time() - t0:.2f}s")

        # Pull quotes
        t0 = time.time()
        edition = extract_pull_quotes(edition, config)
        logger.info(f"Extracted pull quotes in {time.time() - t0:.2f}s")

        # Layout hints
        t0 = time.time()
        edition = assign_layout_hints(edition, config)
        logger.info(f"Assigned layout hints in {time.time() - t0:.2f}s")

        # Curation metadata
        t0 = time.time()
        edition = generate_curation_metadata(edition, config)
        logger.info(f"Generated curation metadata in {time.time() - t0:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"Editorial layer complete in {total_time:.2f}s")

        return edition

    except ConnectionError:
        logger.warning("Ollama is not available. Returning edition without editorial polish.")
        return edition
