"""Shared test fixtures.

These are available to all test files automatically.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import pytest

from offscroll.config import DEFAULTS
from offscroll.models import (
    CuratedEdition,
    CuratedImage,
    CuratedItem,
    EditionMeta,
    FeedItem,
    ImageContent,
    LayoutHint,
    PullQuote,
    Section,
    SourceType,
)

SAMPLE_DATA = Path(__file__).parent / "sample_data"


@pytest.fixture
def sample_config() -> dict:
    """A valid config dict for testing. No file I/O needed."""
    config = copy.deepcopy(DEFAULTS)
    config["feeds"] = {
        "rss": [{"url": "https://example.com/feed.xml", "name": "Test Feed"}],
        "mastodon": [],
        "bluesky": [],
        "opml_files": [],
    }
    config["output"]["data_dir"] = "/tmp/offscroll-test-data"
    return config


@pytest.fixture
def sample_feed_items() -> list[FeedItem]:
    """A list of FeedItems for testing curation and layout."""
    return [
        FeedItem(
            item_id="rss-001",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            item_url="https://example.com/post-1",
            author="Alice",
            title="First Post",
            content_text="This is the first post. It has enough words to be interesting "
            "but not so many that it becomes unwieldy for testing purposes.",
            published_at=datetime(2026, 3, 1, 10, 0, 0),
        ),
        FeedItem(
            item_id="rss-002",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            item_url="https://example.com/post-2",
            author="Bob",
            title="Second Post",
            content_text="Another post with different content about a completely different topic.",
            published_at=datetime(2026, 3, 1, 12, 0, 0),
            images=[ImageContent(url="https://example.com/image.jpg", alt_text="A test image")],
        ),
        FeedItem(
            item_id="mastodon-001",
            source_type=SourceType.MASTODON,
            feed_url="https://mastodon.social",
            item_url="https://mastodon.social/@carol/12345",
            author="@carol@mastodon.social",
            content_text=(
                "A toot from the fediverse. No title because Mastodon posts don't have titles."
            ),
            published_at=datetime(2026, 3, 1, 14, 0, 0),
        ),
    ]


@pytest.fixture
def sample_curated_edition() -> CuratedEdition:
    """A complete CuratedEdition for testing the layout renderer."""
    return CuratedEdition(
        edition=EditionMeta(
            date="2026-03-01",
            title="The Test Gazette",
            subtitle="Vol. 1, No. 1",
            editorial_note="Welcome to the first test edition.",
        ),
        sections=[
            Section(
                heading="Top Stories",
                items=[
                    CuratedItem(
                        item_id="rss-001",
                        display_text="This is the lead story. " * 20,
                        author="Alice",
                        title="A Feature Story",
                        layout_hint=LayoutHint.FEATURE,
                        word_count=100,
                    ),
                    CuratedItem(
                        item_id="rss-002",
                        display_text="A standard item with moderate length content. " * 10,
                        author="Bob",
                        title="Standard Item",
                        images=[
                            CuratedImage(
                                local_path="images/placeholder.jpg",
                                caption="A test image caption",
                                width=800,
                                height=600,
                            )
                        ],
                        layout_hint=LayoutHint.STANDARD,
                        word_count=60,
                    ),
                ],
            ),
            Section(
                heading="Around the Web",
                items=[
                    CuratedItem(
                        item_id="mastodon-001",
                        display_text="A brief item from Mastodon.",
                        author="@carol@mastodon.social",
                        layout_hint=LayoutHint.BRIEF,
                        word_count=6,
                    ),
                ],
            ),
        ],
        pull_quotes=[
            PullQuote(
                text="The best test is the one that catches the bug you didn't expect.",
                attribution="Alice",
                source_item_id="rss-001",
            ),
        ],
        page_target=10,
        estimated_content_pages=2.5,
    )


@pytest.fixture
def sample_rss_xml() -> str:
    """Raw RSS 2.0 XML for testing the feed parser."""
    return (SAMPLE_DATA / "feeds" / "sample_rss.xml").read_text()


@pytest.fixture
def sample_edition_json() -> dict:
    """Raw JSON dict for testing CuratedEdition.from_json()."""
    return json.loads((SAMPLE_DATA / "editions" / "sample_edition.json").read_text())


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    """Path to a temporary SQLite database."""
    return tmp_path / "test.db"


@pytest.fixture
def sample_feed_items_with_edge_cases() -> list[FeedItem]:
    """FeedItems covering edge cases: no title, very long content, multiple images."""
    return [
        # Item with no title (common in Mastodon)
        FeedItem(
            item_id="mastodon-notitle",
            source_type=SourceType.MASTODON,
            feed_url="https://mastodon.social",
            author="@user@mastodon.social",
            content_text="A post without a title. Just content. This is normal for social media.",
            published_at=datetime(2026, 3, 1, 8, 0, 0),
        ),
        # Item with very long content (2000+ words simulated)
        FeedItem(
            item_id="rss-longform",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            author="Longform Author",
            title="A Deep Dive",
            content_text=" ".join(["word"] * 2500),
            published_at=datetime(2026, 3, 1, 9, 0, 0),
        ),
        # Item with multiple images
        FeedItem(
            item_id="rss-multiimage",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            author="Photog",
            title="Photo Essay",
            content_text="A visual story told in multiple images.",
            images=[
                ImageContent(url="https://example.com/img1.jpg", alt_text="First image"),
                ImageContent(url="https://example.com/img2.jpg", alt_text="Second image"),
                ImageContent(url="https://example.com/img3.jpg", alt_text="Third image"),
            ],
            published_at=datetime(2026, 3, 1, 11, 0, 0),
        ),
    ]


@pytest.fixture
def sample_edition_with_empty_section() -> CuratedEdition:
    """CuratedEdition with an empty section (edge case)."""
    return CuratedEdition.from_json(SAMPLE_DATA / "editions" / "sample_edition_empty_section.json")


@pytest.fixture
def sample_edition_briefs_only() -> CuratedEdition:
    """CuratedEdition with only brief items."""
    return CuratedEdition.from_json(SAMPLE_DATA / "editions" / "sample_edition_briefs_only.json")
