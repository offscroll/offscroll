"""Tests for LLM-powered editorial layer.

Editorial layer with Ollama integration.
All tests mock the Ollama client. No test requires a running Ollama instance.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from offscroll.curation.editorial import (
    _call_ollama,
    _is_valid_heading,
    assign_layout_hints,
    extract_pull_quotes,
    generate_curation_metadata,
    generate_editorial_notes,
    generate_section_headings,
    run_editorial,
)
from offscroll.models import (
    CuratedEdition,
    CuratedItem,
    EditionMeta,
    LayoutHint,
    Section,
)


def _make_test_edition() -> CuratedEdition:
    """Helper to create a test CuratedEdition."""
    return CuratedEdition(
        edition=EditionMeta(
            date="2026-03-04",
            title="Test Gazette",
            subtitle="Vol. 1, No. 1",
            editorial_note=None,
        ),
        sections=[
            Section(
                heading="Topic 1",
                items=[
                    CuratedItem(
                        item_id="item-1",
                        display_text="Climate change threatens food security. "
                        "Rising temperatures are affecting crop yields worldwide.",
                        author="Alice",
                        title="Climate Crisis Deepens",
                        layout_hint=LayoutHint.FEATURE,
                        word_count=500,
                    ),
                    CuratedItem(
                        item_id="item-2",
                        display_text="Electric vehicles continue gaining market share.",
                        author="Bob",
                        title="EV Market Growth",
                        layout_hint=LayoutHint.STANDARD,
                        word_count=150,
                    ),
                ],
            ),
            Section(
                heading="Topic 2",
                items=[
                    CuratedItem(
                        item_id="item-3",
                        display_text="Brief update.",
                        author="Carol",
                        layout_hint=LayoutHint.BRIEF,
                        word_count=20,
                    ),
                ],
            ),
        ],
        pull_quotes=[],
        page_target=10,
        estimated_content_pages=2.0,
    )


def _make_test_config() -> dict:
    """Helper to create a test config."""
    return {
        "curation": {
            "ollama_model": "llama3.2:3b",
            "ollama_url": "http://localhost:11434",
        }
    }


# ---------------------------------------------------------------------------
# _call_ollama Tests
# ---------------------------------------------------------------------------


def test_call_ollama_basic():
    """_call_ollama returns response text from mocked client."""
    config = _make_test_config()

    mock_response = MagicMock()
    mock_response.message.content = "   Response text   "

    mock_client = MagicMock()
    mock_client.chat.return_value = mock_response

    mock_ollama = MagicMock()
    mock_ollama.Client.return_value = mock_client

    with patch.dict("sys.modules", {"ollama": mock_ollama}):
        result = _call_ollama("Test prompt", config)

        assert result == "Response text"
        mock_ollama.Client.assert_called_once_with(host="http://localhost:11434")
        mock_client.chat.assert_called_once()


def test_call_ollama_connection_error():
    """_call_ollama raises ConnectionError when Ollama is unreachable."""
    config = _make_test_config()

    mock_client = MagicMock()
    mock_client.chat.side_effect = ConnectionError("Connection failed")

    mock_ollama = MagicMock()
    mock_ollama.Client.return_value = mock_client

    with (
        patch.dict("sys.modules", {"ollama": mock_ollama}),
        pytest.raises(ConnectionError, match="not reachable"),
    ):
        _call_ollama("Test prompt", config)


def test_call_ollama_system_prompt():
    """_call_ollama includes system prompt in messages when provided."""
    config = _make_test_config()

    mock_response = MagicMock()
    mock_response.message.content = "Response"

    mock_client = MagicMock()
    mock_client.chat.return_value = mock_response

    mock_ollama = MagicMock()
    mock_ollama.Client.return_value = mock_client

    with patch.dict("sys.modules", {"ollama": mock_ollama}):
        _call_ollama("User prompt", config, system_prompt="System context")

        # Check that chat was called with both system and user messages
        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System context"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User prompt"


def test_call_ollama_import_error():
    """_call_ollama raises ImportError if ollama package not installed."""
    config = _make_test_config()

    # Simulate that ollama is not installed
    with (
        patch("sys.modules", {"ollama": None}),
        pytest.raises(ImportError, match="Install it with"),
    ):
        _call_ollama("Test prompt", config)


# ---------------------------------------------------------------------------
# generate_section_headings Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _is_valid_heading Tests
# ---------------------------------------------------------------------------


def test_is_valid_heading_accepts_short():
    """Short, valid headings are accepted."""
    assert _is_valid_heading("Climate Watch") is True
    assert _is_valid_heading("Tech Trends") is True


def test_is_valid_heading_rejects_long():
    """Headings over 60 characters are rejected."""
    long_heading = "A" * 61
    assert _is_valid_heading(long_heading) is False


def test_is_valid_heading_rejects_refusals():
    """LLM refusal patterns are rejected."""
    assert _is_valid_heading("I don't see any content to summarize") is False
    assert _is_valid_heading("Please provide more context") is False
    assert _is_valid_heading("I cannot generate a heading") is False
    assert _is_valid_heading("Sorry, I need more information") is False
    assert _is_valid_heading("I'm not able to determine the topic") is False
    assert _is_valid_heading("As an AI, I can't determine that") is False


def test_is_valid_heading_rejects_empty():
    """Empty strings are rejected."""
    assert _is_valid_heading("") is False
    assert _is_valid_heading("   ") is False


def test_generate_section_headings_refusal_fallback():
    """Refusal heading falls back to first item title."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        mock_llm.side_effect = [
            "I don't see any content here",  # Refusal for section 1
            "Tech Roundup",  # Valid for section 2
        ]

        result = generate_section_headings(edition.sections, config)

        # Section 1 should fall back to first item's title
        assert result[0].heading == "Climate Crisis Deepens"
        # Section 2 should use the valid LLM heading
        assert result[1].heading == "Tech Roundup"


def test_generate_section_headings_basic():
    """generate_section_headings replaces generic headings with LLM output."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        mock_llm.side_effect = [
            "Climate and Environment",
            "Technology Trends",
        ]

        result = generate_section_headings(edition.sections, config)

        assert result[0].heading == "Climate and Environment"
        assert result[1].heading == "Technology Trends"
        assert mock_llm.call_count == 2


def test_generate_section_headings_fallback_on_empty_response():
    """generate_section_headings falls back to item title if LLM returns empty."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        mock_llm.return_value = "   "  # Empty after strip

        result = generate_section_headings(edition.sections, config)

        # First section should fall back to first item's title
        assert result[0].heading == "Climate Crisis Deepens"


def test_generate_section_headings_empty_edition():
    """generate_section_headings handles empty section list."""
    result = generate_section_headings([], _make_test_config())
    assert result == []


def test_generate_section_headings_connection_error():
    """generate_section_headings preserves headings on connection error."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        mock_llm.side_effect = ConnectionError("Ollama down")

        result = generate_section_headings(edition.sections, config)

        # Headings should be unchanged
        assert result[0].heading == "Topic 1"
        assert result[1].heading == "Topic 2"


# ---------------------------------------------------------------------------
# generate_editorial_notes Tests
# ---------------------------------------------------------------------------


def test_generate_editorial_notes_full():
    """generate_editorial_notes populates edition and item notes."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        responses = [
            "Welcome to this week's edition.",  # Edition note
            "Commentary on climate issue.",  # Item 1 note
            "EV market commentary.",  # Item 2 note
            # Item 3 is BRIEF, skipped
        ]
        mock_llm.side_effect = responses

        result = generate_editorial_notes(edition, config)

        assert result.edition.editorial_note == "Welcome to this week's edition."
        assert result.sections[0].items[0].editorial_note == "Commentary on climate issue."
        assert result.sections[0].items[1].editorial_note == "EV market commentary."


def test_generate_editorial_notes_skips_briefs():
    """generate_editorial_notes skips BRIEF layout items."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        mock_llm.side_effect = [
            "Edition note.",
            "Feature item note.",
            "Standard item note.",
        ]

        result = generate_editorial_notes(edition, config)

        # Brief item (item-3) should not have been passed to LLM
        brief_item = result.sections[1].items[0]
        assert not hasattr(brief_item, "editorial_note") or brief_item.editorial_note is None


def test_generate_editorial_notes_empty_edition():
    """generate_editorial_notes handles empty edition."""
    edition = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-04",
            title="Empty",
            subtitle="Vol. 1",
        ),
        sections=[],
    )
    result = generate_editorial_notes(edition, _make_test_config())
    assert result.edition.editorial_note is None


# ---------------------------------------------------------------------------
# extract_pull_quotes Tests
# ---------------------------------------------------------------------------


def test_extract_pull_quotes_basic():
    """extract_pull_quotes extracts pull quotes from content.

    update: improved source attribution finds the actual
    author when the quote text matches an item's display_text.
    Quotes not matching any item still get "Editor" attribution.
    """
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        mock_llm.return_value = (
            "Climate change threatens food security.\n"
            "Rising temperatures affect crop yields.\n"
            "Action must be taken immediately."
        )

        result = extract_pull_quotes(edition, config)

        assert len(result.pull_quotes) == 3
        assert "Climate change" in result.pull_quotes[0].text
        # First two quotes match item-1 (Alice's content), so they
        # get the actual author attribution
        assert result.pull_quotes[0].attribution == "Alice"
        # Third quote doesn't match any item
        assert result.pull_quotes[2].attribution == "Editor"


def test_extract_pull_quotes_count():
    """extract_pull_quotes limits to 2-3 quotes."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        # Return 5 quotes; should be limited to 3
        mock_llm.return_value = "\n".join([f"Quote {i}." for i in range(5)])

        result = extract_pull_quotes(edition, config)

        assert len(result.pull_quotes) <= 3


def test_extract_pull_quotes_empty_edition():
    """extract_pull_quotes handles empty edition."""
    edition = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-04",
            title="Empty",
            subtitle="Vol. 1",
        ),
        sections=[],
    )
    config = _make_test_config()

    result = extract_pull_quotes(edition, config)
    assert result.pull_quotes == []


# ---------------------------------------------------------------------------
# assign_layout_hints Tests
# ---------------------------------------------------------------------------


def test_assign_layout_hints_override():
    """assign_layout_hints can override non-FEATURE heuristic hints.

    FEATURE items are now protected from
    LLM override. The curation optimizer's cover story assignment
    is authoritative. The LLM can only adjust STANDARD <-> BRIEF.
    """
    edition = _make_test_edition()
    config = _make_test_config()

    original_hint = edition.sections[0].items[0].layout_hint
    assert original_hint == LayoutHint.FEATURE

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        # Section 0: 2 items (FEATURE -- skipped, STANDARD)
        # Section 1: 1 item (BRIEF)
        # Total: 2 items to process (FEATURE is skipped)
        mock_llm.side_effect = ["standard", "brief"]

        result = assign_layout_hints(edition, config)

        #  FEATURE is preserved -- LLM cannot override cover story
        assert result.sections[0].items[0].layout_hint == LayoutHint.FEATURE
        assert result.sections[0].items[1].layout_hint == LayoutHint.STANDARD
        assert result.sections[1].items[0].layout_hint == LayoutHint.BRIEF


def test_assign_layout_hints_invalid_response():
    """assign_layout_hints preserves hint on invalid LLM response."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        mock_llm.return_value = "invalid_hint"

        result = assign_layout_hints(edition, config)

        # Should keep original hint
        assert result.sections[0].items[0].layout_hint == LayoutHint.FEATURE


# ---------------------------------------------------------------------------
# generate_curation_metadata Tests
# ---------------------------------------------------------------------------


def test_generate_curation_metadata_populates_rationale():
    """generate_curation_metadata adds selection_rationale to items."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        mock_llm.side_effect = [
            "Selected for climate impact.",
            "Included for technology relevance.",
            "Timely brief mention.",
        ]

        result = generate_curation_metadata(edition, config)

        # All items should have rationale added
        assert hasattr(result.sections[0].items[0], "selection_rationale")
        assert "climate impact" in result.sections[0].items[0].selection_rationale


def test_generate_curation_metadata_fallback():
    """generate_curation_metadata falls back on connection error."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        mock_llm.side_effect = ConnectionError("Ollama down")

        result = generate_curation_metadata(edition, config)

        # All items should have default rationale
        assert result.sections[0].items[0].selection_rationale == "Selected for topical relevance."


# ---------------------------------------------------------------------------
# run_editorial Tests
# ---------------------------------------------------------------------------


def test_run_editorial_full_pipeline():
    """run_editorial orchestrates all sub-tasks."""
    edition = _make_test_edition()
    config = _make_test_config()

    with (
        patch("offscroll.curation.editorial.generate_section_headings") as mock_headings,
        patch("offscroll.curation.editorial.generate_editorial_notes") as mock_notes,
        patch("offscroll.curation.editorial.extract_pull_quotes") as mock_quotes,
        patch("offscroll.curation.editorial.assign_layout_hints") as mock_hints,
        patch("offscroll.curation.editorial.generate_curation_metadata") as mock_meta,
    ):
        # Setup return values to match input
        mock_headings.return_value = edition.sections
        mock_notes.return_value = edition
        mock_quotes.return_value = edition
        mock_hints.return_value = edition
        mock_meta.return_value = edition

        result = run_editorial(edition, config)

        # Verify all functions were called
        mock_headings.assert_called_once()
        mock_notes.assert_called_once()
        mock_quotes.assert_called_once()
        mock_hints.assert_called_once()
        mock_meta.assert_called_once()
        assert result == edition


def test_run_editorial_ollama_down():
    """run_editorial returns edition unchanged when Ollama is unavailable."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial.generate_section_headings") as mock_headings:
        mock_headings.side_effect = ConnectionError("Ollama not reachable")

        result = run_editorial(edition, config)

        # Should return edition unchanged (graceful degradation)
        assert result == edition


def test_run_editorial_empty_edition():
    """run_editorial handles empty edition."""
    edition = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-04",
            title="Empty",
            subtitle="Vol. 1",
        ),
        sections=[],
    )
    config = _make_test_config()

    with patch("offscroll.curation.editorial.generate_section_headings") as mock_headings:
        mock_headings.return_value = []

        result = run_editorial(edition, config)

        # Should return empty edition
        assert result.sections == []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clean_pull_quote_strips_numbering():
    """12.2: _clean_pull_quote strips leading numbering."""
    from offscroll.curation.editorial import _clean_pull_quote

    assert _clean_pull_quote("1. A great sentence.") == "A great sentence."
    assert _clean_pull_quote("2) Another quote.") == "Another quote."
    assert _clean_pull_quote("- A dashed quote.") == "A dashed quote."
    assert _clean_pull_quote("* A starred quote.") == "A starred quote."


def test_clean_pull_quote_strips_surrounding_quotes():
    """12.2: _clean_pull_quote strips surrounding quotation marks."""
    from offscroll.curation.editorial import _clean_pull_quote

    assert _clean_pull_quote('"A quoted sentence."') == "A quoted sentence."
    assert _clean_pull_quote("\u201cA curly-quoted sentence.\u201d") == "A curly-quoted sentence."


def test_clean_pull_quote_rejects_preamble():
    """12.2: _clean_pull_quote rejects LLM preamble lines."""
    from offscroll.curation.editorial import _clean_pull_quote

    assert _clean_pull_quote("Here are 3 striking quotes:") is None
    assert _clean_pull_quote("Sure! Here you go.") is None
    assert _clean_pull_quote("The following are notable:") is None
    assert _clean_pull_quote("Certainly! Let me extract those.") is None
    assert _clean_pull_quote("Of course! I found these.") is None


def test_clean_pull_quote_accepts_valid():
    """12.2: _clean_pull_quote accepts valid quote text."""
    from offscroll.curation.editorial import _clean_pull_quote

    assert _clean_pull_quote("The river runs deep and fast.") == "The river runs deep and fast."
    result = _clean_pull_quote("  Life is what happens when you're busy.  ")
    assert result == "Life is what happens when you're busy."


def test_find_source_item_exact_match():
    """12.2: _find_source_item matches quotes to items by substring."""
    from offscroll.curation.editorial import _find_source_item

    items_with_text = [
        ("item-1", "Climate change threatens food security in many regions."),
        ("item-2", "Electric vehicles continue gaining market share worldwide."),
    ]
    assert _find_source_item("Climate change threatens food security", items_with_text) == "item-1"
    assert _find_source_item("Electric vehicles continue gaining", items_with_text) == "item-2"
    assert _find_source_item("A completely unrelated quote.", items_with_text) == "unknown"


def test_find_source_item_fuzzy_match():
    """12.2: _find_source_item uses fuzzy match when exact fails."""
    from offscroll.curation.editorial import _find_source_item

    items_with_text = [
        ("item-1", "Climate change threatens food security in many regions around the world."),
    ]
    # LLM reformulates slightly: different prepositions but 80%+ word overlap
    result = _find_source_item(
        "Climate change threatens food security in many regions around the globe.",
        items_with_text,
    )
    assert result == "item-1"


def test_extract_pull_quotes_no_preamble():
    """12.2: extract_pull_quotes filters out LLM preamble from output."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        mock_llm.return_value = (
            "Here are 3 striking quotes:\n"
            "1. Climate change threatens food security.\n"
            "2. Rising temperatures affect crop yields.\n"
            "3. Action must be taken immediately."
        )

        result = extract_pull_quotes(edition, config)

        # The preamble "Here are 3 striking quotes:" should be filtered
        assert len(result.pull_quotes) == 3
        assert not any("Here are" in pq.text for pq in result.pull_quotes)
        # The numbering should be stripped
        assert not any(pq.text.startswith("1.") for pq in result.pull_quotes)


def test_extract_pull_quotes_source_attribution():
    """12.2: extract_pull_quotes attributes quotes to correct source items."""
    edition = _make_test_edition()
    config = _make_test_config()

    with patch("offscroll.curation.editorial._call_ollama") as mock_llm:
        # Return a quote that exactly matches text from item-1
        mock_llm.return_value = "Climate change threatens food security."

        result = extract_pull_quotes(edition, config)

        assert len(result.pull_quotes) == 1
        assert result.pull_quotes[0].source_item_id == "item-1"
        assert result.pull_quotes[0].attribution == "Alice"
