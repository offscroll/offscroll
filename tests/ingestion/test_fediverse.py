"""Tests for Fediverse ingestion (Mastodon, Bluesky).

comprehensive tests for Mastodon and Bluesky API integrations.
All tests use mocks (no real API calls).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from offscroll.models import SourceType

SAMPLE_DATA = Path(__file__).parent.parent / "sample_data"


# ============================================================================
# Mastodon Mock Helpers
# ============================================================================


def _make_mock_status(
    id: int = 1,
    content: str = "<p>Test post</p>",
    acct: str = "user",
    display_name: str = "Test User",
    url: str = "https://mastodon.social/@user/1",
    reblog: dict | None = None,
    in_reply_to_id: int | None = None,
    media_attachments: list | None = None,
    created_at: datetime | None = None,
) -> dict:
    """Build a mock Mastodon status dict."""
    return {
        "id": id,
        "content": content,
        "url": url,
        "reblog": reblog,
        "in_reply_to_id": in_reply_to_id,
        "created_at": created_at or datetime.now(UTC),
        "media_attachments": media_attachments or [],
        "account": {
            "acct": acct,
            "display_name": display_name,
            "url": f"https://mastodon.social/@{acct}",
        },
    }


# ============================================================================
# Mastodon: _status_to_feed_item() tests
# ============================================================================


def test_status_to_feed_item_basic():
    """Standard status converts correctly to FeedItem."""
    from offscroll.ingestion.fediverse import _status_to_feed_item

    status = _make_mock_status(
        id=123,
        content="<p>Hello world</p>",
        acct="alice",
        display_name="Alice Smith",
        url="https://mastodon.social/@alice/123",
    )
    item = _status_to_feed_item(status, "https://mastodon.social")

    assert item.item_id == "123"
    assert item.source_type == SourceType.MASTODON
    assert item.feed_url == "https://mastodon.social"
    assert item.item_url == "https://mastodon.social/@alice/123"
    assert item.author == "Alice Smith"
    assert item.author_url == "https://mastodon.social/@alice"
    assert item.content_text == "Hello world"
    assert item.content_html == "<p>Hello world</p>"
    assert item.title is None
    assert item.is_thread is False
    assert item.thread_id is None


def test_status_to_feed_item_boost():
    """Boosted status uses original content with booster context."""
    from offscroll.ingestion.fediverse import _status_to_feed_item

    original = _make_mock_status(
        id=100,
        content="<p>Original content</p>",
        acct="bob",
        display_name="Bob Jones",
    )
    boost = _make_mock_status(
        id=200,
        content="ignored",  # This should be ignored for boosted posts
        acct="alice",
        display_name="Alice Smith",
        reblog=original,
    )

    item = _status_to_feed_item(boost, "https://mastodon.social")

    # Should use original content
    assert "Original content" in item.content_text
    # Should include booster mention
    assert "Alice Smith" in item.content_text
    # Author should be original author
    assert item.author == "Bob Jones"


def test_status_to_feed_item_thread():
    """Reply sets is_thread=True and thread_id."""
    from offscroll.ingestion.fediverse import _status_to_feed_item

    status = _make_mock_status(
        id=456,
        in_reply_to_id=450,
    )
    item = _status_to_feed_item(status, "https://mastodon.social")

    assert item.is_thread is True
    assert item.thread_id == "450"


def test_status_to_feed_item_images():
    """Media attachments produce ImageContent objects."""
    from offscroll.ingestion.fediverse import _status_to_feed_item

    status = _make_mock_status(
        id=789,
        media_attachments=[
            {
                "type": "image",
                "url": "https://example.com/img1.jpg",
                "description": "First image",
            },
            {
                "type": "image",
                "url": "https://example.com/img2.jpg",
                "description": None,
            },
        ],
    )
    item = _status_to_feed_item(status, "https://mastodon.social")

    assert len(item.images) == 2
    assert item.images[0].url == "https://example.com/img1.jpg"
    assert item.images[0].alt_text == "First image"
    assert item.images[1].url == "https://example.com/img2.jpg"
    assert item.images[1].alt_text is None


def test_status_to_feed_item_no_display_name():
    """Falls back to @acct when display_name is empty."""
    from offscroll.ingestion.fediverse import _status_to_feed_item

    status = _make_mock_status(
        id=999,
        acct="noname",
        display_name="",  # Empty display name
    )
    item = _status_to_feed_item(status, "https://mastodon.social")

    assert item.author == "@noname"


# ============================================================================
# Mastodon: ingest_mastodon() tests
# ============================================================================


def test_ingest_mastodon_basic(sample_config, tmp_path):
    """Mocked API returns items, stored in DB."""
    from offscroll.ingestion.fediverse import ingest_mastodon
    from offscroll.ingestion.store import init_db

    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
        "feeds": {
            **sample_config["feeds"],
            "mastodon": [
                {
                    "instance": "https://mastodon.social",
                    "access_token_env": "MASTO_TOKEN",
                    "timeline": "home",
                }
            ],
        },
    }

    mock_statuses = [
        _make_mock_status(id=1, content="<p>First post</p>"),
        _make_mock_status(id=2, content="<p>Second post</p>", acct="other"),
    ]

    mock_mastodon_mod = MagicMock()
    mock_api = MagicMock()
    mock_api.timeline_home.return_value = mock_statuses
    mock_mastodon_mod.Mastodon.return_value = mock_api

    with (
        patch.dict(os.environ, {"MASTO_TOKEN": "test-token"}),
        patch.dict("sys.modules", {"mastodon": mock_mastodon_mod}),
    ):
        init_db(config)
        count = ingest_mastodon(config)

    assert count == 2
    mock_mastodon_mod.Mastodon.assert_called_with(
        access_token="test-token",
        api_base_url="https://mastodon.social",
    )
    mock_api.timeline_home.assert_called_once_with(limit=40)


def test_ingest_mastodon_no_config(sample_config):
    """Empty mastodon config returns 0."""
    from offscroll.ingestion.fediverse import ingest_mastodon

    config = {
        **sample_config,
        "feeds": {
            **sample_config["feeds"],
            "mastodon": [],
        },
    }

    count = ingest_mastodon(config)
    assert count == 0


def test_ingest_mastodon_import_error(sample_config):
    """Missing mastodon.py raises ImportError with helpful message."""
    from offscroll.ingestion.fediverse import ingest_mastodon

    config = {
        **sample_config,
        "feeds": {
            **sample_config["feeds"],
            "mastodon": [
                {
                    "instance": "https://mastodon.social",
                    "access_token_env": "MASTO_TOKEN",
                    "timeline": "home",
                }
            ],
        },
    }

    with (
        patch.dict("sys.modules", {"mastodon": None}),
        pytest.raises(ImportError, match="Mastodon.py not installed"),
    ):
        ingest_mastodon(config)


def test_ingest_mastodon_missing_token(sample_config, tmp_path):
    """Missing env var for access token logs error and skips."""
    from offscroll.ingestion.fediverse import ingest_mastodon
    from offscroll.ingestion.store import init_db

    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
        "feeds": {
            **sample_config["feeds"],
            "mastodon": [
                {
                    "instance": "https://mastodon.social",
                    "access_token_env": "MISSING_VAR",
                    "timeline": "home",
                }
            ],
        },
    }

    # Ensure env var is NOT set
    if "MISSING_VAR" in os.environ:
        del os.environ["MISSING_VAR"]

    mock_mastodon_mod = MagicMock()
    with patch.dict("sys.modules", {"mastodon": mock_mastodon_mod}):
        init_db(config)
        count = ingest_mastodon(config)
    assert count == 0


def test_ingest_mastodon_public_timeline(sample_config, tmp_path):
    """Fetches public timeline when configured."""
    from offscroll.ingestion.fediverse import ingest_mastodon
    from offscroll.ingestion.store import init_db

    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
        "feeds": {
            **sample_config["feeds"],
            "mastodon": [
                {
                    "instance": "https://mastodon.social",
                    "access_token_env": "MASTO_TOKEN",
                    "timeline": "public",
                }
            ],
        },
    }

    mock_statuses = [_make_mock_status(id=1)]

    mock_mastodon_mod = MagicMock()
    mock_api = MagicMock()
    mock_api.timeline_public.return_value = mock_statuses
    mock_mastodon_mod.Mastodon.return_value = mock_api

    with (
        patch.dict(os.environ, {"MASTO_TOKEN": "test-token"}),
        patch.dict("sys.modules", {"mastodon": mock_mastodon_mod}),
    ):
        init_db(config)
        count = ingest_mastodon(config)

    assert count == 1
    mock_api.timeline_public.assert_called_once_with(limit=40)


def test_ingest_mastodon_list_timeline(sample_config, tmp_path):
    """Fetches list timeline when configured."""
    from offscroll.ingestion.fediverse import ingest_mastodon
    from offscroll.ingestion.store import init_db

    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
        "feeds": {
            **sample_config["feeds"],
            "mastodon": [
                {
                    "instance": "https://mastodon.social",
                    "access_token_env": "MASTO_TOKEN",
                    "timeline": "list:12345",
                }
            ],
        },
    }

    mock_statuses = [_make_mock_status(id=1)]

    mock_mastodon_mod = MagicMock()
    mock_api = MagicMock()
    mock_api.timeline_list.return_value = mock_statuses
    mock_mastodon_mod.Mastodon.return_value = mock_api

    with (
        patch.dict(os.environ, {"MASTO_TOKEN": "test-token"}),
        patch.dict("sys.modules", {"mastodon": mock_mastodon_mod}),
    ):
        init_db(config)
        count = ingest_mastodon(config)

    assert count == 1
    mock_api.timeline_list.assert_called_once_with("12345", limit=40)


def test_ingest_mastodon_skips_empty_content(sample_config, tmp_path):
    """Statuses with no content are skipped."""
    from offscroll.ingestion.fediverse import ingest_mastodon
    from offscroll.ingestion.store import init_db

    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
        "feeds": {
            **sample_config["feeds"],
            "mastodon": [
                {
                    "instance": "https://mastodon.social",
                    "access_token_env": "MASTO_TOKEN",
                    "timeline": "home",
                }
            ],
        },
    }

    mock_statuses = [
        _make_mock_status(id=1, content=""),  # Empty
        _make_mock_status(id=2, content="<p>Good</p>"),
    ]

    mock_mastodon_mod = MagicMock()
    mock_api = MagicMock()
    mock_api.timeline_home.return_value = mock_statuses
    mock_mastodon_mod.Mastodon.return_value = mock_api

    with (
        patch.dict(os.environ, {"MASTO_TOKEN": "test-token"}),
        patch.dict("sys.modules", {"mastodon": mock_mastodon_mod}),
    ):
        init_db(config)
        count = ingest_mastodon(config)

    assert count == 1


# ============================================================================
# Bluesky Mock Helpers
# ============================================================================


def _make_mock_post(
    uri: str = "at://did:plc:user/app.bsky.feed.post/abc123",
    handle: str = "user.bsky.social",
    display_name: str = "User Name",
    text: str = "Test post",
    created_at: str = "2026-03-01T10:00:00.000Z",
    reply: dict | None = None,
    embed: dict | None = None,
) -> dict:
    """Build a mock Bluesky post dict."""
    return {
        "post": {
            "uri": uri,
            "author": {
                "handle": handle,
                "display_name": display_name,
            },
            "record": {
                "text": text,
                "created_at": created_at,
                "reply": reply,
            },
            "embed": embed,
        }
    }


# ============================================================================
# Bluesky: _bsky_post_to_feed_item() tests
# ============================================================================


def test_bsky_post_to_feed_item_basic():
    """Standard post converts correctly to FeedItem."""
    from offscroll.ingestion.fediverse import _bsky_post_to_feed_item

    feed_view = _make_mock_post(
        uri="at://did:plc:alice/app.bsky.feed.post/123",
        handle="alice.bsky.social",
        display_name="Alice",
        text="Hello Bluesky",
    )
    item = _bsky_post_to_feed_item(feed_view)

    assert item.item_id == "at://did:plc:alice/app.bsky.feed.post/123"
    assert item.source_type == SourceType.BLUESKY
    assert item.feed_url == "https://bsky.app"
    assert item.author == "Alice"
    assert item.author_url == "https://bsky.app/profile/alice.bsky.social"
    assert item.content_text == "Hello Bluesky"
    assert item.content_html is None
    assert item.title is None
    assert item.is_thread is False


def test_bsky_post_to_feed_item_quote():
    """Quote post appends quoted text."""
    from offscroll.ingestion.fediverse import _bsky_post_to_feed_item

    feed_view = _make_mock_post(
        text="Agreed!",
        embed={
            "$type": "app.bsky.embed.record#view",
            "record": {
                "author": {"handle": "bob.bsky.social"},
                "record": {"text": "This is important"},
            },
        },
    )
    item = _bsky_post_to_feed_item(feed_view)

    assert "Agreed!" in item.content_text
    assert "[Quoting @bob.bsky.social]" in item.content_text
    assert "This is important" in item.content_text


def test_bsky_post_to_feed_item_thread():
    """Reply sets is_thread=True and thread_id."""
    from offscroll.ingestion.fediverse import _bsky_post_to_feed_item

    feed_view = _make_mock_post(
        text="Reply text",
        reply={
            "root": {
                "uri": "at://did:plc:author/app.bsky.feed.post/root123",
            }
        },
    )
    item = _bsky_post_to_feed_item(feed_view)

    assert item.is_thread is True
    assert item.thread_id == "at://did:plc:author/app.bsky.feed.post/root123"


def test_bsky_post_to_feed_item_images():
    """Embed images produce ImageContent objects."""
    from offscroll.ingestion.fediverse import _bsky_post_to_feed_item

    feed_view = _make_mock_post(
        text="With images",
        embed={
            "$type": "app.bsky.embed.images#view",
            "images": [
                {
                    "thumb": "https://example.com/img1_thumb.jpg",
                    "alt": "First image",
                },
                {
                    "thumb": "https://example.com/img2_thumb.jpg",
                    "alt": None,
                },
            ],
        },
    )
    item = _bsky_post_to_feed_item(feed_view)

    assert len(item.images) == 2
    assert item.images[0].url == "https://example.com/img1_thumb.jpg"
    assert item.images[0].alt_text == "First image"
    assert item.images[1].url == "https://example.com/img2_thumb.jpg"


# ============================================================================
# Bluesky: ingest_bluesky() tests
# ============================================================================


def test_ingest_bluesky_basic(sample_config, tmp_path):
    """Mocked client returns items, stored in DB."""
    from offscroll.ingestion.fediverse import ingest_bluesky
    from offscroll.ingestion.store import init_db

    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
        "feeds": {
            **sample_config["feeds"],
            "bluesky": [
                {
                    "handle": "user.bsky.social",
                    "app_password_env": "BSKY_PASSWORD",
                    "feed": "timeline",
                }
            ],
        },
    }

    mock_posts = [
        _make_mock_post(uri="at://1", text="First post"),
        _make_mock_post(uri="at://2", text="Second post"),
    ]

    mock_atproto_mod = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.feed = mock_posts
    mock_client.get_timeline.return_value = mock_response
    mock_atproto_mod.Client.return_value = mock_client

    with (
        patch.dict(os.environ, {"BSKY_PASSWORD": "test-pass"}),
        patch.dict("sys.modules", {"atproto": mock_atproto_mod}),
    ):
        init_db(config)
        count = ingest_bluesky(config)

    assert count == 2
    mock_client.login.assert_called_once_with("user.bsky.social", "test-pass")
    mock_client.get_timeline.assert_called_once_with(limit=50)


def test_ingest_bluesky_no_config(sample_config):
    """Empty bluesky config returns 0."""
    from offscroll.ingestion.fediverse import ingest_bluesky

    config = {
        **sample_config,
        "feeds": {
            **sample_config["feeds"],
            "bluesky": [],
        },
    }

    count = ingest_bluesky(config)
    assert count == 0


def test_ingest_bluesky_import_error(sample_config):
    """Missing atproto raises ImportError with helpful message."""
    from offscroll.ingestion.fediverse import ingest_bluesky

    config = {
        **sample_config,
        "feeds": {
            **sample_config["feeds"],
            "bluesky": [
                {
                    "handle": "user.bsky.social",
                    "app_password_env": "BSKY_PASSWORD",
                    "feed": "timeline",
                }
            ],
        },
    }

    with (
        patch.dict("sys.modules", {"atproto": None}),
        pytest.raises(ImportError, match="atproto not installed"),
    ):
        ingest_bluesky(config)


def test_ingest_bluesky_missing_password(sample_config, tmp_path):
    """Missing env var for password logs error and skips."""
    from offscroll.ingestion.fediverse import ingest_bluesky
    from offscroll.ingestion.store import init_db

    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
        "feeds": {
            **sample_config["feeds"],
            "bluesky": [
                {
                    "handle": "user.bsky.social",
                    "app_password_env": "MISSING_VAR",
                    "feed": "timeline",
                }
            ],
        },
    }

    # Ensure env var is NOT set
    if "MISSING_VAR" in os.environ:
        del os.environ["MISSING_VAR"]

    mock_atproto_mod = MagicMock()
    with patch.dict("sys.modules", {"atproto": mock_atproto_mod}):
        init_db(config)
        count = ingest_bluesky(config)
    assert count == 0


def test_ingest_bluesky_author_feed(sample_config, tmp_path):
    """Fetches author feed when configured."""
    from offscroll.ingestion.fediverse import ingest_bluesky
    from offscroll.ingestion.store import init_db

    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
        "feeds": {
            **sample_config["feeds"],
            "bluesky": [
                {
                    "handle": "user.bsky.social",
                    "app_password_env": "BSKY_PASSWORD",
                    "feed": "author:did:plc:someid",
                }
            ],
        },
    }

    mock_posts = [_make_mock_post(uri="at://1", text="Author post")]

    mock_atproto_mod = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.feed = mock_posts
    mock_client.get_author_feed.return_value = mock_response
    mock_atproto_mod.Client.return_value = mock_client

    with (
        patch.dict(os.environ, {"BSKY_PASSWORD": "test-pass"}),
        patch.dict("sys.modules", {"atproto": mock_atproto_mod}),
    ):
        init_db(config)
        count = ingest_bluesky(config)

    assert count == 1
    mock_client.get_author_feed.assert_called_once_with(actor="did:plc:someid", limit=50)


def test_ingest_bluesky_skips_empty_content(sample_config, tmp_path):
    """Posts with no content are skipped."""
    from offscroll.ingestion.fediverse import ingest_bluesky
    from offscroll.ingestion.store import init_db

    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
        "feeds": {
            **sample_config["feeds"],
            "bluesky": [
                {
                    "handle": "user.bsky.social",
                    "app_password_env": "BSKY_PASSWORD",
                    "feed": "timeline",
                }
            ],
        },
    }

    mock_posts = [
        _make_mock_post(uri="at://1", text=""),  # Empty
        _make_mock_post(uri="at://2", text="Good post"),
    ]

    mock_atproto_mod = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.feed = mock_posts
    mock_client.get_timeline.return_value = mock_response
    mock_atproto_mod.Client.return_value = mock_client

    with (
        patch.dict(os.environ, {"BSKY_PASSWORD": "test-pass"}),
        patch.dict("sys.modules", {"atproto": mock_atproto_mod}),
    ):
        init_db(config)
        count = ingest_bluesky(config)

    assert count == 1
