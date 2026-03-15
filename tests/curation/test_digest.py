"""Tests for email digest rendering.

Tests cover Jinja2 template rendering, HTML generation, and email delivery.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from offscroll.curation.digest import render_digest


def test_render_digest_creates_file(sample_curated_edition, sample_config, tmp_path):
    """HTML file is written to disk."""
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    path = render_digest(config, edition=sample_curated_edition)
    assert path.exists()
    assert path.suffix == ".html"
    assert path.stat().st_size > 0


def test_render_digest_valid_html(sample_curated_edition, sample_config, tmp_path):
    """Output contains proper HTML structure."""
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    path = render_digest(config, edition=sample_curated_edition)
    content = path.read_text()
    assert "<html>" in content.lower()
    assert "</html>" in content.lower()
    assert "<head>" in content
    assert "</head>" in content
    assert "<body>" in content
    assert "</body>" in content


def test_render_digest_includes_title(sample_curated_edition, sample_config, tmp_path):
    """Edition title appears in output."""
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    path = render_digest(config, edition=sample_curated_edition)
    content = path.read_text()
    assert sample_curated_edition.edition.title in content


def test_render_digest_includes_sections(sample_curated_edition, sample_config, tmp_path):
    """All section headings present in output."""
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    path = render_digest(config, edition=sample_curated_edition)
    content = path.read_text()
    for section in sample_curated_edition.sections:
        assert section.heading in content


def test_render_digest_includes_items(sample_curated_edition, sample_config, tmp_path):
    """Item content present in output."""
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    path = render_digest(config, edition=sample_curated_edition)
    content = path.read_text()
    first_item = sample_curated_edition.sections[0].items[0]
    # Check a substring of the display text
    assert first_item.display_text[:50] in content


def test_render_digest_includes_authors(sample_curated_edition, sample_config, tmp_path):
    """Author names present in output."""
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    path = render_digest(config, edition=sample_curated_edition)
    content = path.read_text()
    first_item = sample_curated_edition.sections[0].items[0]
    assert first_item.author in content


def test_render_digest_includes_quotes(sample_curated_edition, sample_config, tmp_path):
    """Pull quotes rendered in output."""
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    path = render_digest(config, edition=sample_curated_edition)
    content = path.read_text()
    first_quote = sample_curated_edition.pull_quotes[0]
    assert first_quote.text in content
    assert first_quote.attribution in content


def test_render_digest_editorial_notes(sample_curated_edition, sample_config, tmp_path):
    """Editorial notes rendered when present."""
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    path = render_digest(config, edition=sample_curated_edition)
    content = path.read_text()
    # Check that edition-level editorial note is present
    if sample_curated_edition.edition.editorial_note:
        assert sample_curated_edition.edition.editorial_note in content


def test_render_digest_no_editorial_notes(sample_config, tmp_path):
    """Works when editorial notes are None."""
    from offscroll.models import CuratedEdition, EditionMeta, Section

    edition = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-01",
            title="Test",
            subtitle="Vol. 1",
            editorial_note=None,
        ),
        sections=[Section(heading="Test", items=[])],
        pull_quotes=[],
    )
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    path = render_digest(config, edition=edition)
    assert path.exists()
    content = path.read_text()
    assert "<html>" in content.lower()


def test_render_digest_from_json_path(sample_config, tmp_path):
    """Loads from edition JSON path."""
    json_path = Path(__file__).parent.parent / "sample_data" / "editions" / "sample_edition.json"
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    path = render_digest(config, edition_path=json_path)
    assert path.exists()
    content = path.read_text()
    assert "The Test Gazette" in content


def test_render_digest_from_edition_obj(sample_curated_edition, sample_config, tmp_path):
    """Works with CuratedEdition object."""
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    path = render_digest(config, edition=sample_curated_edition)
    assert path.exists()
    assert path.stat().st_size > 0


def test_render_digest_no_edition_error(sample_config, tmp_path):
    """Raises FileNotFoundError when no edition found."""
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    with pytest.raises(FileNotFoundError):
        render_digest(config)


def test_render_digest_send_false(sample_curated_edition, sample_config, tmp_path):
    """send=False does not attempt SMTP."""
    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    # Should not raise or attempt network operations
    path = render_digest(config, edition=sample_curated_edition, send=False)
    assert path.exists()


def test_render_digest_send_smtp_mock(sample_curated_edition, sample_config, tmp_path):
    """send=True with mocked SMTP sends email."""
    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
        "email": {
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_user_env": "SMTP_USER",
            "smtp_password_env": "SMTP_PASS",
            "from_address": "test@example.com",
            "to_addresses": ["reader@example.com"],
        },
    }
    with (
        patch.dict(os.environ, {"SMTP_USER": "user", "SMTP_PASS": "pass"}),
        patch("offscroll.curation.digest.smtplib.SMTP") as mock_smtp,
    ):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
        path = render_digest(config, edition=sample_curated_edition, send=True)
    assert path.exists()
    mock_server.send_message.assert_called_once()


def test_render_digest_send_smtp_failure(sample_curated_edition, sample_config, tmp_path):
    """SMTP error logged, HTML still returned."""
    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
        "email": {
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_user_env": "SMTP_USER",
            "smtp_password_env": "SMTP_PASS",
            "from_address": "test@example.com",
            "to_addresses": ["reader@example.com"],
        },
    }
    with (
        patch.dict(os.environ, {"SMTP_USER": "user", "SMTP_PASS": "pass"}),
        patch("offscroll.curation.digest.smtplib.SMTP") as mock_smtp,
    ):
        mock_smtp.side_effect = Exception("Connection failed")
        # Should not raise, just log the error
        path = render_digest(config, edition=sample_curated_edition, send=True)
    # HTML file should still be created
    assert path.exists()
