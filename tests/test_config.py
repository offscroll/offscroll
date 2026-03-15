"""Tests for configuration loading."""

from offscroll.config import DEFAULTS, _deep_merge, _validate


def test_deep_merge_nested():
    """Deep merge preserves nested structure."""
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}}
    result = _deep_merge(base, override)
    assert result == {"a": {"b": 10, "c": 2}, "d": 3}


def test_validate_no_feeds():
    """Validation catches missing feed sources."""
    config = _deep_merge(
        DEFAULTS,
        {"feeds": {"rss": [], "mastodon": [], "bluesky": [], "opml_files": []}},
    )
    errors = _validate(config)
    assert any("No feed sources" in e for e in errors)


def test_validate_valid_config(sample_config):
    """A valid config passes validation."""
    errors = _validate(sample_config)
    assert errors == []


def test_default_embedding_provider_ollama():
    """DEFAULTS["embedding"]["provider"] == "ollama"."""
    assert DEFAULTS["embedding"]["provider"] == "ollama"


def test_default_curation_model_ollama():
    """DEFAULTS["curation"]["model"] == "ollama"."""
    assert DEFAULTS["curation"]["model"] == "ollama"
