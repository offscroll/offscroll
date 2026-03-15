"""Tests for shared data models."""

from dataclasses import asdict

from offscroll.models import (
    CuratedEdition,
    CuratedItem,
    FeedItem,
    LayoutHint,
    SourceType,
)


def test_feed_item_word_count_auto():
    """word_count is computed from content_text if not provided."""
    item = FeedItem(
        item_id="test-1",
        source_type=SourceType.RSS,
        feed_url="https://example.com/feed.xml",
        content_text="one two three four five",
    )
    assert item.word_count == 5


def test_feed_item_word_count_explicit():
    """Explicit word_count overrides auto-computation."""
    item = FeedItem(
        item_id="test-1",
        source_type=SourceType.RSS,
        feed_url="https://example.com/feed.xml",
        content_text="one two three",
        word_count=99,
    )
    assert item.word_count == 99


def test_curated_edition_roundtrip(sample_curated_edition, tmp_path):
    """CuratedEdition survives JSON serialization and deserialization."""
    path = tmp_path / "edition.json"
    sample_curated_edition.to_json(path)
    loaded = CuratedEdition.from_json(path)
    assert loaded.edition.title == sample_curated_edition.edition.title
    assert len(loaded.sections) == len(sample_curated_edition.sections)
    assert loaded.sections[0].items[0].layout_hint == LayoutHint.FEATURE
    assert len(loaded.pull_quotes) == 1


def test_curated_item_metadata_fields():
    """CuratedItem accepts cluster_id, quality_score, selection_rationale."""
    item = CuratedItem(
        item_id="test-001",
        display_text="Test content.",
        author="Tester",
        cluster_id=3,
        quality_score=0.87,
        selection_rationale="Top item in cluster",
    )
    assert item.cluster_id == 3
    assert item.quality_score == 0.87
    assert item.selection_rationale == "Top item in cluster"


def test_curated_item_metadata_defaults():
    """Metadata fields default to None."""
    item = CuratedItem(
        item_id="test-002",
        display_text="Test content.",
        author="Tester",
    )
    assert item.cluster_id is None
    assert item.quality_score is None
    assert item.selection_rationale is None


def test_curated_item_to_json_metadata():
    """Metadata fields appear in asdict() / JSON output."""
    item = CuratedItem(
        item_id="test-001",
        display_text="Test content.",
        author="Tester",
        cluster_id=3,
        quality_score=0.87,
        selection_rationale="Top item in cluster",
    )
    d = asdict(item)
    assert d["cluster_id"] == 3
    assert d["quality_score"] == 0.87
    assert d["selection_rationale"] == "Top item in cluster"


def test_curated_edition_curation_summary():
    """CuratedEdition accepts and serializes curation_summary."""
    from offscroll.models import EditionMeta

    edition = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-01",
            title="Test",
            subtitle="Vol. 1, No. 1",
        ),
        curation_summary="5 items selected from 20 candidates. Loss: 0.123",
    )
    assert edition.curation_summary == "5 items selected from 20 candidates. Loss: 0.123"
    d = asdict(edition)
    assert d["curation_summary"] == "5 items selected from 20 candidates. Loss: 0.123"


def test_curated_edition_from_json_metadata(tmp_path):
    """from_json() round-trips metadata fields."""
    from offscroll.models import EditionMeta, Section

    original = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-01",
            title="Test",
            subtitle="Vol. 1, No. 1",
        ),
        sections=[
            Section(
                heading="Test Section",
                items=[
                    CuratedItem(
                        item_id="test-001",
                        display_text="Test content.",
                        author="Tester",
                        cluster_id=2,
                        quality_score=0.95,
                        selection_rationale="Best item",
                    )
                ],
            )
        ],
        curation_summary="1 item selected from 10 candidates. Loss: 0.050",
    )
    path = tmp_path / "edition_metadata.json"
    original.to_json(path)
    loaded = CuratedEdition.from_json(path)
    assert loaded.curation_summary == "1 item selected from 10 candidates. Loss: 0.050"
    assert loaded.sections[0].items[0].cluster_id == 2
    assert loaded.sections[0].items[0].quality_score == 0.95
    assert loaded.sections[0].items[0].selection_rationale == "Best item"


def test_curated_edition_from_json_no_metadata(tmp_path):
    """from_json() handles JSON without metadata fields (backward compat)."""
    import json

    # Create minimal JSON without new fields
    json_data = {
        "edition": {
            "date": "2026-03-01",
            "title": "Test",
            "subtitle": "Vol. 1, No. 1",
        },
        "sections": [
            {
                "heading": "Test Section",
                "items": [
                    {
                        "item_id": "test-001",
                        "display_text": "Test content.",
                        "author": "Tester",
                    }
                ],
            }
        ],
        "pull_quotes": [],
        "page_target": 10,
        "estimated_content_pages": 0.0,
    }
    path = tmp_path / "edition_no_metadata.json"
    with open(path, "w") as f:
        json.dump(json_data, f)
    loaded = CuratedEdition.from_json(path)
    assert loaded.curation_summary is None
    assert loaded.sections[0].items[0].cluster_id is None
    assert loaded.sections[0].items[0].quality_score is None
    assert loaded.sections[0].items[0].selection_rationale is None


# ---------------------------------------------------------------------------
# CuratedItem source attribution fields
# ---------------------------------------------------------------------------


def test_curated_item_source_name():
    """CuratedItem accepts and stores source_name."""
    item = CuratedItem(
        item_id="src-001",
        display_text="Content.",
        author="Author",
        source_name="Tech Blog",
    )
    assert item.source_name == "Tech Blog"


def test_curated_item_item_url():
    """CuratedItem accepts and stores item_url."""
    item = CuratedItem(
        item_id="url-001",
        display_text="Content.",
        author="Author",
        item_url="https://example.com/post-1",
    )
    assert item.item_url == "https://example.com/post-1"


def test_curated_item_source_defaults():
    """source_name and item_url default to None."""
    item = CuratedItem(
        item_id="def-001",
        display_text="Content.",
        author="Author",
    )
    assert item.source_name is None
    assert item.item_url is None


def test_curated_item_item_url_roundtrip(tmp_path):
    """item_url survives JSON serialization and deserialization."""
    from offscroll.models import EditionMeta, Section

    original = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-07",
            title="Test",
            subtitle="Vol. 1, No. 1",
        ),
        sections=[
            Section(
                heading="Section",
                items=[
                    CuratedItem(
                        item_id="rt-001",
                        display_text="Content.",
                        author="Author",
                        source_name="My Blog",
                        item_url="https://example.com/article",
                    )
                ],
            )
        ],
    )
    path = tmp_path / "edition_url.json"
    original.to_json(path)
    loaded = CuratedEdition.from_json(path)
    item = loaded.sections[0].items[0]
    assert item.source_name == "My Blog"
    assert item.item_url == "https://example.com/article"


# ---------------------------------------------------------------------------
#  CuratedImage Optional Dimensions
# ---------------------------------------------------------------------------


def test_curated_image_optional_dimensions():
    """(11.6): CuratedImage width/height are optional."""
    from offscroll.models import CuratedImage

    # No dimensions
    img = CuratedImage(local_path="images/test.jpg", caption="Test")
    assert img.width is None
    assert img.height is None

    # With dimensions
    img2 = CuratedImage(
        local_path="images/test.jpg", caption="Test", width=800, height=600
    )
    assert img2.width == 800
    assert img2.height == 600


def test_curated_image_roundtrip_with_none_dimensions(tmp_path):
    """(11.6): CuratedImage with None dimensions survives JSON roundtrip."""
    from offscroll.models import (
        CuratedEdition,
        CuratedImage,
        CuratedItem,
        EditionMeta,
        Section,
    )

    edition = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-08",
            title="Test",
            subtitle="Vol. 1, No. 1",
        ),
        sections=[
            Section(
                heading="News",
                items=[
                    CuratedItem(
                        item_id="img-rt",
                        display_text="Article.",
                        author="Writer",
                        images=[
                            CuratedImage(
                                local_path="images/test.jpg",
                                caption="Test image",
                                width=None,
                                height=None,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    path = tmp_path / "edition_img.json"
    edition.to_json(path)
    loaded = CuratedEdition.from_json(path)
    img = loaded.sections[0].items[0].images[0]
    assert img.local_path == "images/test.jpg"
    assert img.caption == "Test image"
    assert img.width is None
    assert img.height is None
