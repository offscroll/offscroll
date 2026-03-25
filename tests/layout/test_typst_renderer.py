"""Tests for the Typst rendering backend.

Tests Typst markup generation from CuratedEdition data.
PDF compilation tests are marked slow (require Typst CLI).
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest

from offscroll.layout.typst_renderer import (
    _escape_typst,
    _first_alpha_index,
    _typst_string,
    build_typst_markup,
)
from offscroll.models import (
    CuratedEdition,
    CuratedImage,
    CuratedItem,
    CuratedThread,
    EditionMeta,
    LayoutHint,
    PullQuote,
    Section,
)


# --- Unit tests for helper functions ---


class TestEscapeTypst:
    def test_empty_string(self):
        assert _escape_typst("") == ""

    def test_no_special_chars(self):
        assert _escape_typst("Hello world") == "Hello world"

    def test_hash(self):
        assert _escape_typst("#hashtag") == "\\#hashtag"

    def test_at_sign(self):
        assert _escape_typst("@user") == "\\@user"

    def test_dollar(self):
        assert _escape_typst("$100") == "\\$100"

    def test_angle_brackets(self):
        assert _escape_typst("<html>") == "\\<html\\>"

    def test_underscores(self):
        assert _escape_typst("snake_case") == "snake\\_case"

    def test_asterisks(self):
        assert _escape_typst("*bold*") == "\\*bold\\*"

    def test_backticks(self):
        assert _escape_typst("`code`") == "\\`code\\`"

    def test_double_slash(self):
        assert _escape_typst("https://example.com") == "https:\\/\\/example.com"

    def test_mixed(self):
        result = _escape_typst("#tag @user $5")
        assert "\\#" in result
        assert "\\@" in result
        assert "\\$" in result


class TestFirstAlphaIndex:
    def test_starts_with_letter(self):
        assert _first_alpha_index("Hello") == 0

    def test_starts_with_quote(self):
        assert _first_alpha_index('"Hello') == 1

    def test_no_alpha(self):
        assert _first_alpha_index("123") == 0

    def test_em_dash_prefix(self):
        assert _first_alpha_index("\u2014Hello") == 1


class TestTypstString:
    def test_empty(self):
        assert _typst_string("") == '""'

    def test_simple(self):
        assert _typst_string("hello") == '"hello"'

    def test_quotes(self):
        assert _typst_string('say "hi"') == '"say \\"hi\\""'

    def test_backslash(self):
        assert _typst_string("a\\b") == '"a\\\\b"'


# --- Integration tests for markup generation ---


class TestBuildTypstMarkup:
    @pytest.fixture
    def edition(self) -> CuratedEdition:
        """Minimal edition for markup generation tests."""
        return CuratedEdition(
            edition=EditionMeta(
                date="2026-03-22",
                title="Test Gazette",
                subtitle="Daily Edition",
                editorial_note="Test note",
            ),
            sections=[
                Section(
                    heading="Top Stories",
                    items=[
                        CuratedItem(
                            item_id="feat-001",
                            display_text="This is the lead feature story. " * 15,
                            author="Alice",
                            title="Feature Story Title",
                            layout_hint=LayoutHint.FEATURE,
                            word_count=90,
                        ),
                        CuratedItem(
                            item_id="std-001",
                            display_text="A standard article. " * 10,
                            author="Bob",
                            title="Standard Article",
                            layout_hint=LayoutHint.STANDARD,
                            word_count=30,
                        ),
                    ],
                ),
                Section(
                    heading="In Brief",
                    items=[
                        CuratedItem(
                            item_id="brief-001",
                            display_text="Brief news item one.",
                            author="Carol",
                            layout_hint=LayoutHint.BRIEF,
                            word_count=4,
                        ),
                    ],
                ),
            ],
            pull_quotes=[
                PullQuote(
                    text="A notable quote.",
                    attribution="Alice",
                    source_item_id="feat-001",
                ),
            ],
            page_target=4,
            estimated_content_pages=1.5,
        )

    @pytest.fixture
    def config(self, tmp_path) -> dict:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        return {
            "output": {"data_dir": str(data_dir)},
            "newspaper": {"debug_mode": False},
        }

    def test_returns_string(self, edition, config):
        markup = build_typst_markup(edition, config)
        assert isinstance(markup, str)
        assert len(markup) > 0

    def test_contains_import(self, edition, config):
        markup = build_typst_markup(edition, config)
        assert '#import "templates.typ"' in markup

    def test_contains_page_setup(self, edition, config):
        markup = build_typst_markup(edition, config)
        assert '#set page("us-letter"' in markup

    def test_contains_masthead(self, edition, config):
        markup = build_typst_markup(edition, config)
        assert "#masthead(" in markup
        assert "Test Gazette" in markup

    def test_contains_feature(self, edition, config):
        markup = build_typst_markup(edition, config)
        assert "feature-article(" in markup
        assert "Feature Story Title" in markup

    def test_contains_standard(self, edition, config):
        markup = build_typst_markup(edition, config)
        assert "standard-article(" in markup
        assert "Standard Article" in markup

    def test_contains_brief(self, edition, config):
        markup = build_typst_markup(edition, config)
        assert "brief-item(" in markup
        assert "Carol" in markup

    def test_contains_colophon(self, edition, config):
        markup = build_typst_markup(edition, config)
        assert "#colophon(" in markup

    def test_contains_pull_quote(self, edition, config):
        markup = build_typst_markup(edition, config)
        assert "A notable quote" in markup

    def test_section_label(self, edition, config):
        markup = build_typst_markup(edition, config)
        assert "section-label(" in markup

    def test_escapes_special_chars(self, config):
        """Special Typst characters in content are escaped."""
        edition = CuratedEdition(
            edition=EditionMeta(
                date="2026-03-22",
                title="Test $pecial",
                subtitle="@Edition #1",
            ),
            sections=[
                Section(
                    heading="News",
                    items=[
                        CuratedItem(
                            item_id="esc-001",
                            display_text="Price is $5 @user #tag",
                            author="Test",
                            title="Escaped",
                            layout_hint=LayoutHint.STANDARD,
                            word_count=5,
                        ),
                    ],
                ),
            ],
            pull_quotes=[],
            page_target=2,
        )
        markup = build_typst_markup(edition, config)
        # $ should be escaped in content
        assert "\\$" in markup

    def test_thread_rendering(self, config):
        """Thread articles render with thread-article template."""
        edition = CuratedEdition(
            edition=EditionMeta(
                date="2026-03-22",
                title="Test",
                subtitle="Test",
            ),
            sections=[
                Section(
                    heading="Threads",
                    items=[
                        CuratedThread(
                            thread_id="thread-001",
                            headline="A Discussion",
                            author="Dan",
                            editorial_note="Context for the thread.",
                            items=[
                                CuratedItem(
                                    item_id="t-sub-001",
                                    display_text="First post in thread.",
                                    author="Dan",
                                    word_count=4,
                                ),
                                CuratedItem(
                                    item_id="t-sub-002",
                                    display_text="Second post in thread.",
                                    author="Dan",
                                    word_count=4,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            pull_quotes=[],
            page_target=2,
        )
        markup = build_typst_markup(edition, config)
        assert "thread-article(" in markup
        assert "A Discussion" in markup
        assert "First post in thread" in markup

    def test_debug_mode_editorial_note(self, edition, config):
        """Editorial notes only appear in debug mode."""
        markup_prod = build_typst_markup(edition, config)
        assert "debug-mode: true" not in markup_prod

        debug_config = copy.deepcopy(config)
        debug_config["newspaper"]["debug_mode"] = True
        markup_debug = build_typst_markup(edition, debug_config)
        assert "debug-mode: true" in markup_debug


# --- Typst CLI compilation test (requires typst binary) ---


@pytest.mark.slow
class TestTypstPdfCompilation:
    """Tests that require the Typst CLI to be installed."""

    @pytest.fixture(autouse=True)
    def check_typst(self):
        if shutil.which("typst") is None:
            pytest.skip("Typst CLI not installed")

    @pytest.fixture
    def config(self, tmp_path) -> dict:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        return {
            "output": {"data_dir": str(data_dir)},
            "newspaper": {"debug_mode": False},
        }

    @pytest.fixture
    def simple_edition(self) -> CuratedEdition:
        return CuratedEdition(
            edition=EditionMeta(
                date="2026-03-22",
                title="Typst Test",
                subtitle="Compilation Check",
            ),
            sections=[
                Section(
                    heading="News",
                    items=[
                        CuratedItem(
                            item_id="s-001",
                            display_text="A simple test article for Typst compilation.",
                            author="Tester",
                            title="Test Article",
                            layout_hint=LayoutHint.STANDARD,
                            word_count=7,
                        ),
                    ],
                ),
            ],
            pull_quotes=[],
            page_target=2,
        )

    def test_pdf_generated(self, simple_edition, config):
        """Typst CLI produces a PDF file from generated markup."""
        from offscroll.layout.typst_renderer import render_typst_pdf

        pdf_path = render_typst_pdf(config, simple_edition)
        assert pdf_path.exists()
        assert pdf_path.suffix == ".pdf"
        # PDF should have a non-trivial size
        assert pdf_path.stat().st_size > 1000

    def test_typ_source_preserved(self, simple_edition, config):
        """The .typ source file is preserved for debugging."""
        from offscroll.layout.typst_renderer import render_typst_pdf

        render_typst_pdf(config, simple_edition)
        data_dir = Path(config["output"]["data_dir"])
        typ_files = list(data_dir.glob("*.typ"))
        assert len(typ_files) >= 1
