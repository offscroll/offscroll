"""Tests for embedding generation.

embedding provider interface with stub.
"""

from __future__ import annotations

import math

import pytest

from offscroll.ingestion.embeddings import _embed_stub, _embed_texts, embed_items
from offscroll.models import FeedItem, SourceType


def _item(
    item_id: str = "test",
    content_text: str = "Test content",
    title: str | None = None,
    embedding: list[float] | None = None,
) -> FeedItem:
    """Create a minimal FeedItem for testing."""
    return FeedItem(
        item_id=item_id,
        source_type=SourceType.RSS,
        feed_url="https://example.com/feed.xml",
        content_text=content_text,
        title=title,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# _embed_stub
# ---------------------------------------------------------------------------


def test_embed_stub_deterministic():
    """Same text produces same embedding twice."""
    result1 = _embed_stub(["hello world"])
    result2 = _embed_stub(["hello world"])
    assert result1 == result2


def test_embed_stub_different_texts():
    """Different texts produce different embeddings."""
    results = _embed_stub(["hello", "goodbye"])
    assert results[0] != results[1]


def test_embed_stub_dimension():
    """Embedding has 8 dimensions."""
    results = _embed_stub(["test text"])
    assert len(results[0]) == 8


def test_embed_stub_normalized():
    """Embedding is unit-length (norm ~= 1.0)."""
    results = _embed_stub(["some text to embed"])
    norm = math.sqrt(sum(x * x for x in results[0]))
    assert abs(norm - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# embed_items
# ---------------------------------------------------------------------------


def test_embed_items_populates_embeddings():
    """Items get embedding field set."""
    items = [_item(item_id="a", content_text="First item")]
    config = {"embedding": {"provider": "stub"}}
    result = embed_items(items, config)
    assert len(result) == 1
    assert result[0].embedding is not None
    assert len(result[0].embedding) == 8


def test_embed_items_skips_already_embedded():
    """Items with existing embeddings unchanged."""
    existing_emb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    items = [
        _item(item_id="a", content_text="First", embedding=existing_emb),
        _item(item_id="b", content_text="Second"),
    ]
    config = {"embedding": {"provider": "stub"}}
    result = embed_items(items, config)
    # First item should keep its original embedding
    assert result[0].embedding == existing_emb
    # Second item should get a new embedding
    assert result[1].embedding is not None
    assert result[1].embedding != existing_emb


def test_embed_items_prepends_title():
    """Items with title get 'title: text' input."""
    items_with_title = [_item(item_id="a", content_text="body", title="My Title")]
    items_without_title = [_item(item_id="b", content_text="body", title=None)]
    config = {"embedding": {"provider": "stub"}}

    embed_items(items_with_title, config)
    embed_items(items_without_title, config)

    # Embeddings should be different because the input text differs
    # ("My Title: body" vs "body")
    assert items_with_title[0].embedding != items_without_title[0].embedding


def test_embed_items_empty_list():
    """Empty items list returns empty list."""
    config = {"embedding": {"provider": "stub"}}
    result = embed_items([], config)
    assert result == []


# ---------------------------------------------------------------------------
# _embed_texts
# ---------------------------------------------------------------------------


def test_embed_texts_stub_provider():
    """Config with provider='stub' works."""
    config = {"embedding": {"provider": "stub"}}
    results = _embed_texts(["test"], config)
    assert len(results) == 1
    assert len(results[0]) == 8


def test_embed_texts_unknown_provider():
    """Unknown provider raises NotImplementedError."""
    config = {"embedding": {"provider": "openai"}}
    with pytest.raises(NotImplementedError, match="openai"):
        _embed_texts(["test"], config)


# ---------------------------------------------------------------------------
# _embed_ollama tests
# ---------------------------------------------------------------------------


def test_embed_ollama_import_error():
    """ollama not installed raises ImportError."""
    from unittest.mock import patch

    from offscroll.ingestion.embeddings import _embed_ollama

    config = {
        "embedding": {
            "provider": "ollama",
            "ollama_model": "nomic-embed-text",
            "ollama_url": "http://localhost:11434",
        }
    }

    # Mock the import to fail
    with (
        patch.dict("sys.modules", {"ollama": None}),
        pytest.raises(ImportError, match="not installed"),
    ):
        _embed_ollama(["test"], config)


def test_embed_texts_ollama_dispatch():
    """config with provider='ollama' dispatches correctly."""
    from unittest.mock import MagicMock, patch

    config = {
        "embedding": {
            "provider": "ollama",
            "ollama_model": "nomic-embed-text",
            "ollama_url": "http://localhost:11434",
        }
    }

    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4]]}

    # Mock the ollama module in the function's local scope
    with patch(
        "builtins.__import__",
        side_effect=lambda name, *args, **kwargs: (
            MagicMock(Client=MagicMock(return_value=mock_client))
            if name == "ollama"
            else __import__(name, *args, **kwargs)
        ),
    ):
        results = _embed_texts(["test text"], config)

    assert len(results) == 1
    assert results[0] == [0.1, 0.2, 0.3, 0.4]


def test_embed_ollama_connection_error():
    """ollama unreachable raises ConnectionError."""
    from unittest.mock import MagicMock, patch

    from offscroll.ingestion.embeddings import _embed_ollama

    config = {
        "embedding": {
            "provider": "ollama",
            "ollama_model": "nomic-embed-text",
            "ollama_url": "http://localhost:11434",
        }
    }

    # Mock the ollama module to raise an error on Client instantiation
    def mock_import(name, *args, **kwargs):
        if name == "ollama":
            mock_mod = MagicMock()
            mock_mod.Client.side_effect = Exception("Connection refused")
            return mock_mod
        return __import__(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        pytest.raises(ConnectionError, match="Cannot connect"),
    ):
        _embed_ollama(["test"], config)


def test_embed_ollama_returns_vectors():
    """mocked ollama returns proper embeddings."""
    from unittest.mock import MagicMock, patch

    from offscroll.ingestion.embeddings import _embed_ollama

    config = {
        "embedding": {
            "provider": "ollama",
            "ollama_model": "nomic-embed-text",
            "ollama_url": "http://localhost:11434",
        }
    }

    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]}

    def mock_import(name, *args, **kwargs):
        if name == "ollama":
            mock_mod = MagicMock()
            mock_mod.Client.return_value = mock_client
            return mock_mod
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        result = _embed_ollama(["test text"], config)

    assert len(result) == 1
    assert result[0] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def test_embed_ollama_truncates_long_text():
    """Text longer than 6000 characters is truncated before sending."""
    from unittest.mock import MagicMock, patch

    from offscroll.ingestion.embeddings import _embed_ollama

    config = {
        "embedding": {
            "provider": "ollama",
            "ollama_model": "nomic-embed-text",
            "ollama_url": "http://localhost:11434",
        }
    }

    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4]]}

    def mock_import(name, *args, **kwargs):
        if name == "ollama":
            mock_mod = MagicMock()
            mock_mod.Client.return_value = mock_client
            return mock_mod
        return __import__(name, *args, **kwargs)

    long_text = "x" * 7000  # Exceeds the 6000-character limit

    with patch("builtins.__import__", side_effect=mock_import):
        result = _embed_ollama([long_text], config)

    # The embedding call should have received truncated text
    assert len(result) == 1
    actual_input = mock_client.embed.call_args[1]["input"]
    assert len(actual_input) == 6000


def test_embed_ollama_does_not_truncate_short_text():
    """Text at or under 6000 characters is sent unchanged."""
    from unittest.mock import MagicMock, patch

    from offscroll.ingestion.embeddings import _embed_ollama

    config = {
        "embedding": {
            "provider": "ollama",
            "ollama_model": "nomic-embed-text",
            "ollama_url": "http://localhost:11434",
        }
    }

    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4]]}

    def mock_import(name, *args, **kwargs):
        if name == "ollama":
            mock_mod = MagicMock()
            mock_mod.Client.return_value = mock_client
            return mock_mod
        return __import__(name, *args, **kwargs)

    short_text = "x" * 5999  # Under the 6000-character limit

    with patch("builtins.__import__", side_effect=mock_import):
        result = _embed_ollama([short_text], config)

    assert len(result) == 1
    actual_input = mock_client.embed.call_args[1]["input"]
    assert len(actual_input) == 5999
