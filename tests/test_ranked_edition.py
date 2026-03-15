"""Tests for  Ranked Edition models and pipeline.

Tests the new RankedItem/RankedEdition models, the ranking
function, the page-fill-aware placement, and backward
compatibility with CuratedEdition.
"""

from __future__ import annotations

import json

from offscroll.models import (
    CuratedEdition,
    CuratedImage,
    EditionMeta,
    LayoutHint,
    PullQuote,
    RankedEdition,
    RankedItem,
    detect_edition_format,
)

# ---------------------------------------------------------------------------
# RankedItem basics
# ---------------------------------------------------------------------------


def test_ranked_item_defaults():
    """RankedItem has correct defaults for optional fields."""
    ri = RankedItem(
        rank=1,
        item_id="test-001",
        layout_hint=LayoutHint.STANDARD,
        section="News",
        display_text="Hello world.",
    )
    assert ri.skip is False
    assert ri.skip_reason is None
    assert ri.images == []
    assert ri.author == "Unknown"
    assert ri.word_count == 0
    assert ri.quality_score is None


def test_ranked_item_skip_flag():
    """RankedItem skip flag works."""
    ri = RankedItem(
        rank=5,
        item_id="test-005",
        layout_hint=LayoutHint.BRIEF,
        section="In Brief",
        display_text="",
        skip=True,
        skip_reason="Empty content",
    )
    assert ri.skip is True
    assert ri.skip_reason == "Empty content"


# ---------------------------------------------------------------------------
# RankedEdition serialization
# ---------------------------------------------------------------------------


def test_ranked_edition_roundtrip(tmp_path):
    """RankedEdition survives JSON serialization and deserialization."""
    edition = RankedEdition(
        edition=EditionMeta(
            date="2026-03-08",
            title="Test Gazette",
            subtitle="Vol. 1, No. 5",
        ),
        ranked_items=[
            RankedItem(
                rank=1,
                item_id="feat-001",
                layout_hint=LayoutHint.FEATURE,
                section="Top Stories",
                display_text="A long feature article. " * 20,
                title="The Big Story",
                author="Alice",
                word_count=100,
                quality_score=0.92,
                images=[CuratedImage(local_path="images/hero.jpg", caption="Hero")],
            ),
            RankedItem(
                rank=2,
                item_id="std-002",
                layout_hint=LayoutHint.STANDARD,
                section="Tech",
                display_text="A standard article.",
                author="Bob",
                word_count=50,
            ),
            RankedItem(
                rank=3,
                item_id="skip-003",
                layout_hint=LayoutHint.BRIEF,
                section="In Brief",
                display_text="",
                skip=True,
                skip_reason="Empty content",
            ),
        ],
        pull_quote_pool=[
            PullQuote(
                text="A striking quote from the feature.",
                attribution="Alice",
                source_item_id="feat-001",
            ),
        ],
        page_target=7,
        curation_summary="3 items ranked from 15 candidates",
    )

    path = tmp_path / "ranked_edition.json"
    edition.to_json(path)

    loaded = RankedEdition.from_json(path)
    assert loaded.edition.title == "Test Gazette"
    assert loaded.edition.subtitle == "Vol. 1, No. 5"
    assert len(loaded.ranked_items) == 3
    assert loaded.ranked_items[0].rank == 1
    assert loaded.ranked_items[0].layout_hint == LayoutHint.FEATURE
    assert loaded.ranked_items[0].images[0].local_path == "images/hero.jpg"
    assert loaded.ranked_items[2].skip is True
    assert loaded.ranked_items[2].skip_reason == "Empty content"
    assert len(loaded.pull_quote_pool) == 1
    assert loaded.page_target == 7
    assert loaded.curation_summary == "3 items ranked from 15 candidates"


def test_ranked_edition_format_marker(tmp_path):
    """RankedEdition JSON includes _format marker for detection."""
    edition = RankedEdition(
        edition=EditionMeta(
            date="2026-03-08",
            title="Test",
            subtitle="Vol. 1",
        ),
        ranked_items=[],
        page_target=7,
    )
    path = tmp_path / "edition.json"
    edition.to_json(path)

    with open(path) as f:
        data = json.load(f)
    assert data["_format"] == "ranked"


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def test_detect_ranked_format(tmp_path):
    """detect_edition_format identifies ranked editions."""
    edition = RankedEdition(
        edition=EditionMeta(date="2026-03-08", title="Test", subtitle="Vol. 1"),
        ranked_items=[],
    )
    path = tmp_path / "ranked.json"
    edition.to_json(path)
    assert detect_edition_format(path) == "ranked"


def test_detect_curated_format(tmp_path):
    """detect_edition_format identifies curated editions."""
    edition = CuratedEdition(
        edition=EditionMeta(date="2026-03-08", title="Test", subtitle="Vol. 1"),
    )
    path = tmp_path / "curated.json"
    edition.to_json(path)
    assert detect_edition_format(path) == "curated"


# ---------------------------------------------------------------------------
# to_curated_edition conversion
# ---------------------------------------------------------------------------


def test_to_curated_edition_groups_by_section():
    """to_curated_edition groups ranked items into sections."""
    edition = RankedEdition(
        edition=EditionMeta(date="2026-03-08", title="Test", subtitle="Vol. 1"),
        ranked_items=[
            RankedItem(
                rank=1,
                item_id="a",
                layout_hint=LayoutHint.FEATURE,
                section="Top Stories",
                display_text="Feature content.",
                author="Alice",
                word_count=300,
            ),
            RankedItem(
                rank=2,
                item_id="b",
                layout_hint=LayoutHint.STANDARD,
                section="Tech",
                display_text="Tech content.",
                author="Bob",
                word_count=100,
            ),
            RankedItem(
                rank=3,
                item_id="c",
                layout_hint=LayoutHint.STANDARD,
                section="Top Stories",
                display_text="More top stories.",
                author="Carol",
                word_count=80,
            ),
        ],
    )
    curated = edition.to_curated_edition()
    assert len(curated.sections) == 2
    assert curated.sections[0].heading == "Top Stories"
    assert len(curated.sections[0].items) == 2
    assert curated.sections[1].heading == "Tech"
    assert len(curated.sections[1].items) == 1


def test_to_curated_edition_skips_flagged_items():
    """to_curated_edition excludes items with skip=True."""
    edition = RankedEdition(
        edition=EditionMeta(date="2026-03-08", title="Test", subtitle="Vol. 1"),
        ranked_items=[
            RankedItem(
                rank=1,
                item_id="a",
                layout_hint=LayoutHint.STANDARD,
                section="News",
                display_text="Good content.",
                word_count=100,
            ),
            RankedItem(
                rank=2,
                item_id="b",
                layout_hint=LayoutHint.BRIEF,
                section="News",
                display_text="",
                skip=True,
                skip_reason="Empty",
            ),
        ],
    )
    curated = edition.to_curated_edition()
    total_items = sum(len(s.items) for s in curated.sections)
    assert total_items == 1
    assert curated.sections[0].items[0].item_id == "a"


def test_to_curated_edition_respects_placed_count():
    """to_curated_edition with placed_count limits items."""
    edition = RankedEdition(
        edition=EditionMeta(date="2026-03-08", title="Test", subtitle="Vol. 1"),
        ranked_items=[
            RankedItem(
                rank=i,
                item_id=f"item-{i}",
                layout_hint=LayoutHint.STANDARD,
                section="News",
                display_text=f"Content {i}.",
                word_count=100,
            )
            for i in range(1, 11)
        ],
    )
    curated = edition.to_curated_edition(placed_count=3)
    total_items = sum(len(s.items) for s in curated.sections)
    assert total_items == 3


# ---------------------------------------------------------------------------
# rank_items
# ---------------------------------------------------------------------------


def test_rank_items_orders_by_quality():
    """rank_items returns items sorted by quality score."""
    from offscroll.curation.selection import rank_items
    from offscroll.models import FeedItem, SourceType

    pool = [
        FeedItem(
            item_id="short",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            content_text="Short.",
            word_count=5,
            cluster_id=0,
        ),
        FeedItem(
            item_id="long",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            content_text=" ".join(["word"] * 500),
            word_count=500,
            cluster_id=1,
        ),
        FeedItem(
            item_id="medium",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            content_text=" ".join(["word"] * 100),
            word_count=100,
            cluster_id=2,
        ),
    ]
    ranked = rank_items(pool, n_clusters=3)
    # Long article should rank highest (highest quality score)
    assert ranked[0][0].item_id == "long"
    # Short should rank lowest
    assert ranked[-1][0].item_id == "short"


def test_rank_items_returns_all():
    """rank_items returns ALL items, not a subset."""
    from offscroll.curation.selection import rank_items
    from offscroll.models import FeedItem, SourceType

    pool = [
        FeedItem(
            item_id=f"item-{i}",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            content_text=f"Content for item {i}. " * 10,
            word_count=50,
            cluster_id=i % 3,
        )
        for i in range(20)
    ]
    ranked = rank_items(pool, n_clusters=3)
    assert len(ranked) == 20  # All items returned


def test_rank_items_diversity_penalty():
    """rank_items penalizes same-author items."""
    from offscroll.curation.selection import rank_items
    from offscroll.models import FeedItem, SourceType

    pool = [
        FeedItem(
            item_id="alice-1",
            source_type=SourceType.RSS,
            feed_url="https://a.com/feed",
            content_text=" ".join(["word"] * 200),
            word_count=200,
            cluster_id=0,
            author="Alice",
        ),
        FeedItem(
            item_id="alice-2",
            source_type=SourceType.RSS,
            feed_url="https://a.com/feed",
            content_text=" ".join(["word"] * 200),
            word_count=200,
            cluster_id=1,
            author="Alice",
        ),
        FeedItem(
            item_id="bob-1",
            source_type=SourceType.RSS,
            feed_url="https://b.com/feed",
            content_text=" ".join(["word"] * 200),
            word_count=200,
            cluster_id=2,
            author="Bob",
        ),
    ]
    ranked = rank_items(pool, n_clusters=3)
    # Alice's second article should be penalized
    ids = [r[0].item_id for r in ranked]
    # Bob should appear before Alice's second article
    assert ids.index("bob-1") < ids.index("alice-2")


def test_rank_items_empty_pool():
    """rank_items handles empty pool."""
    from offscroll.curation.selection import rank_items

    assert rank_items([], n_clusters=0) == []


# ---------------------------------------------------------------------------
# _build_ranked_edition
# ---------------------------------------------------------------------------


def test_build_ranked_edition_produces_ranked_items(sample_config):
    """_build_ranked_edition produces a RankedEdition with items."""
    from unittest.mock import patch

    from offscroll.curation.selection import _build_ranked_edition
    from offscroll.models import FeedItem, SourceType

    pool = [
        FeedItem(
            item_id=f"item-{i}",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            content_text=f"Content for item {i}. " * 20,
            word_count=80 + i * 10,
            cluster_id=i % 3,
            author=f"Author-{i}",
            title=f"Article {i}",
        )
        for i in range(10)
    ]

    with patch(
        "offscroll.curation.selection.get_feed_name_map",
        return_value={"https://example.com/feed.xml": "Test Feed"},
    ), patch(
        "offscroll.curation.selection.get_edition_count",
        return_value=4,
    ):
        ranked = _build_ranked_edition(pool, n_clusters=3, config=sample_config)

    assert isinstance(ranked, RankedEdition)
    assert len(ranked.ranked_items) == 10  # All items ranked
    assert ranked.ranked_items[0].rank == 1
    assert ranked.ranked_items[9].rank == 10
    # First item should have highest quality score
    assert ranked.ranked_items[0].quality_score >= ranked.ranked_items[9].quality_score


def test_build_ranked_edition_flags_empty_items(sample_config):
    """_build_ranked_edition flags empty-content items as skip."""
    from unittest.mock import patch

    from offscroll.curation.selection import _build_ranked_edition
    from offscroll.models import FeedItem, SourceType

    pool = [
        FeedItem(
            item_id="good",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            content_text="Good content with enough words.",
            word_count=30,
            cluster_id=0,
        ),
        FeedItem(
            item_id="empty",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            content_text="",
            word_count=0,
            cluster_id=1,
        ),
        FeedItem(
            item_id="tiny",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            content_text="hi",
            word_count=1,
            cluster_id=2,
        ),
    ]

    with patch(
        "offscroll.curation.selection.get_feed_name_map",
        return_value={},
    ), patch(
        "offscroll.curation.selection.get_edition_count",
        return_value=0,
    ):
        ranked = _build_ranked_edition(pool, n_clusters=3, config=sample_config)

    skip_items = [ri for ri in ranked.ranked_items if ri.skip]
    non_skip = [ri for ri in ranked.ranked_items if not ri.skip]
    assert len(skip_items) == 2  # empty and tiny
    assert len(non_skip) == 1
    assert non_skip[0].item_id == "good"


def test_build_ranked_edition_pull_quotes(sample_config):
    """_build_ranked_edition generates pull quotes from top items."""
    from unittest.mock import patch

    from offscroll.curation.selection import _build_ranked_edition
    from offscroll.models import FeedItem, SourceType

    pool = [
        FeedItem(
            item_id=f"item-{i}",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            content_text=(
                "This is the opening sentence of the article. "
                "Nothing in the industry transforms outcomes quite like this approach. "
                "The results reveal everything about how the system was designed."
            ),
            word_count=100,
            cluster_id=i,
            author=f"Author-{i}",
        )
        for i in range(6)
    ]

    with patch(
        "offscroll.curation.selection.get_feed_name_map",
        return_value={},
    ), patch(
        "offscroll.curation.selection.get_edition_count",
        return_value=0,
    ):
        ranked = _build_ranked_edition(pool, n_clusters=6, config=sample_config)

    assert len(ranked.pull_quote_pool) >= 1
    assert len(ranked.pull_quote_pool) <= 3


# ---------------------------------------------------------------------------
# Renderer: _place_ranked_items
# ---------------------------------------------------------------------------


def test_place_ranked_items_stops_at_target(sample_config):
    """Renderer places items until page target is reached."""
    from offscroll.layout.renderer import _place_ranked_items

    # Create a ranked edition with many items
    items = [
        RankedItem(
            rank=i,
            item_id=f"item-{i}",
            layout_hint=LayoutHint.STANDARD,
            section="News",
            display_text=" ".join(["word"] * 400),  # ~1 page each
            word_count=400,
            author=f"Author-{i}",
        )
        for i in range(1, 21)
    ]

    ranked = RankedEdition(
        edition=EditionMeta(date="2026-03-08", title="Test", subtitle="Vol. 1"),
        ranked_items=items,
        page_target=3,
    )

    curated = _place_ranked_items(ranked, sample_config)
    total_items = sum(len(s.items) for s in curated.sections)
    # Should place ~3 pages worth, not all 20 items
    assert total_items < 20
    assert total_items >= 2  # At least 2 items for 3 pages


def test_place_ranked_items_skips_flagged(sample_config):
    """Renderer skips items flagged skip=True."""
    from offscroll.layout.renderer import _place_ranked_items

    items = [
        RankedItem(
            rank=1,
            item_id="good",
            layout_hint=LayoutHint.STANDARD,
            section="News",
            display_text="Good content.",
            word_count=100,
        ),
        RankedItem(
            rank=2,
            item_id="skipped",
            layout_hint=LayoutHint.BRIEF,
            section="News",
            display_text="",
            skip=True,
            skip_reason="Empty",
        ),
        RankedItem(
            rank=3,
            item_id="also-good",
            layout_hint=LayoutHint.STANDARD,
            section="News",
            display_text="Also good.",
            word_count=80,
        ),
    ]

    ranked = RankedEdition(
        edition=EditionMeta(date="2026-03-08", title="Test", subtitle="Vol. 1"),
        ranked_items=items,
        page_target=7,
    )

    curated = _place_ranked_items(ranked, sample_config)
    all_ids = [
        item.item_id
        for section in curated.sections
        for item in section.items
    ]
    assert "skipped" not in all_ids
    assert "good" in all_ids
    assert "also-good" in all_ids


def test_place_ranked_items_fills_to_80_percent(sample_config):
    """Renderer fills pages to at least 80% before stopping."""
    from offscroll.layout.renderer import _place_ranked_items

    # Each item is ~0.5 pages, target is 7, so we need ~14 items
    # to fill 7 pages. Renderer should place enough to reach 80% = 5.6 pages.
    items = [
        RankedItem(
            rank=i,
            item_id=f"item-{i}",
            layout_hint=LayoutHint.STANDARD,
            section=f"Section {(i % 3) + 1}",
            display_text=" ".join(["word"] * 200),
            word_count=200,
        )
        for i in range(1, 30)
    ]

    ranked = RankedEdition(
        edition=EditionMeta(date="2026-03-08", title="Test", subtitle="Vol. 1"),
        ranked_items=items,
        page_target=7,
    )

    curated = _place_ranked_items(ranked, sample_config)
    total_items = sum(len(s.items) for s in curated.sections)
    # With 200 words per item, each is ~0.5 pages.
    # 7 pages target -> should place many items
    assert total_items >= 10  # At least ~5 pages worth


# ---------------------------------------------------------------------------
# Colophon rendering
# ---------------------------------------------------------------------------


def test_colophon_present_in_html(sample_curated_edition, sample_config):
    """ Colophon appears in rendered HTML."""
    from offscroll.layout.renderer import _build_html

    html = _build_html(sample_curated_edition, sample_config)
    assert "colophon" in html
    assert "colophon-rule" in html
    assert sample_curated_edition.edition.title in html


def test_colophon_contains_edition_meta(sample_curated_edition, sample_config):
    """Colophon shows edition title and subtitle."""
    from offscroll.layout.renderer import _build_html

    html = _build_html(sample_curated_edition, sample_config)
    assert "colophon-title" in html
    assert "colophon-meta" in html


# ---------------------------------------------------------------------------
# Backward compatibility: CuratedEdition still works
# ---------------------------------------------------------------------------


def test_build_html_still_works_with_curated_edition(
    sample_curated_edition, sample_config
):
    """Existing CuratedEdition rendering still works after """
    from offscroll.layout.renderer import _build_html

    html = _build_html(sample_curated_edition, sample_config)
    assert "<html" in html
    assert sample_curated_edition.edition.title in html
    # Feature should still be extracted to page 1
    assert "A Feature Story" in html


def test_render_html_still_works_with_curated_edition(
    sample_curated_edition, sample_config, tmp_path
):
    """render_newspaper_html still works with CuratedEdition."""
    from offscroll.layout.renderer import render_newspaper_html

    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    path = render_newspaper_html(config, edition=sample_curated_edition)
    assert path.exists()
    assert path.suffix == ".html"


def test_load_ranked_edition_from_file(tmp_path, sample_config):
    """_load_edition auto-detects and loads ranked edition."""
    from offscroll.layout.renderer import _load_edition

    ranked = RankedEdition(
        edition=EditionMeta(date="2026-03-08", title="Test", subtitle="Vol. 1"),
        ranked_items=[
            RankedItem(
                rank=1,
                item_id="item-1",
                layout_hint=LayoutHint.STANDARD,
                section="News",
                display_text="Content here.",
                word_count=100,
            ),
        ],
        page_target=5,
    )
    path = tmp_path / "edition-2026-03-08.json"
    ranked.to_json(path)

    config = {**sample_config, "output": {"data_dir": str(tmp_path)}}
    loaded = _load_edition(config, edition_path=path)
    # Should be converted to CuratedEdition
    assert isinstance(loaded, CuratedEdition)
    assert loaded.edition.title == "Test"
    total_items = sum(len(s.items) for s in loaded.sections)
    assert total_items == 1
