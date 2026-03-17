"""Tests for SQLite persistence layer.

store functions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from offscroll.ingestion.store import (
    get_cluster_count,
    get_edition_count,
    get_items_for_curation,
    init_db,
    record_edition,
    store_item,
    update_image_paths,
)
from offscroll.models import FeedItem, ImageContent, SourceType


def _make_config(tmp_path) -> dict:
    """Create a test config pointing at tmp_path."""
    return {"output": {"data_dir": str(tmp_path)}}


def _make_item(
    item_id: str = "test-001",
    feed_url: str = "https://example.com/feed.xml",
    content_text: str = "Test content for storage",
    author: str = "TestAuthor",
    embedding: list[float] | None = None,
    cluster_id: int | None = None,
    images: list[ImageContent] | None = None,
    ingested_at: datetime | None = None,
) -> FeedItem:
    """Create a minimal FeedItem for testing."""
    return FeedItem(
        item_id=item_id,
        source_type=SourceType.RSS,
        feed_url=feed_url,
        content_text=content_text,
        author=author,
        embedding=embedding,
        cluster_id=cluster_id,
        images=images or [],
        ingested_at=ingested_at or datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# store_item
# ---------------------------------------------------------------------------


def test_store_item_insert(tmp_path):
    """Store a new item, returns True."""
    config = _make_config(tmp_path)
    init_db(config)
    item = _make_item()
    assert store_item(config, item) is True


def test_store_item_duplicate(tmp_path):
    """Store same item_id twice, second returns False."""
    config = _make_config(tmp_path)
    init_db(config)
    item = _make_item()
    assert store_item(config, item) is True
    assert store_item(config, item) is False


def test_store_item_roundtrip(tmp_path):
    """Store then retrieve, all fields match."""
    config = _make_config(tmp_path)
    init_db(config)
    now = datetime.now(UTC)
    item = _make_item(
        item_id="roundtrip-001",
        embedding=[0.1, 0.2, 0.3],
        cluster_id=2,
        ingested_at=now,
    )
    store_item(config, item)

    results = get_items_for_curation(
        config, since=now - timedelta(hours=1), exclude_previous_editions=False
    )
    assert len(results) == 1
    retrieved = results[0]
    assert retrieved.item_id == "roundtrip-001"
    assert retrieved.source_type == SourceType.RSS
    assert retrieved.feed_url == "https://example.com/feed.xml"
    assert retrieved.author == "TestAuthor"
    assert retrieved.content_text == "Test content for storage"
    assert retrieved.cluster_id == 2
    assert retrieved.embedding is not None
    assert len(retrieved.embedding) == 3
    assert abs(retrieved.embedding[0] - 0.1) < 1e-9


def test_store_item_with_images(tmp_path):
    """Images are serialized and deserialized correctly."""
    config = _make_config(tmp_path)
    init_db(config)
    images = [
        ImageContent(url="https://example.com/img1.jpg", alt_text="First"),
        ImageContent(url="https://example.com/img2.jpg", alt_text="Second"),
    ]
    now = datetime.now(UTC)
    item = _make_item(
        item_id="images-001",
        images=images,
        embedding=[1.0],
        cluster_id=0,
        ingested_at=now,
    )
    store_item(config, item)

    results = get_items_for_curation(
        config, since=now - timedelta(hours=1), exclude_previous_editions=False
    )
    assert len(results) == 1
    assert len(results[0].images) == 2
    assert results[0].images[0].url == "https://example.com/img1.jpg"
    assert results[0].images[0].alt_text == "First"
    assert results[0].images[1].url == "https://example.com/img2.jpg"


def test_store_item_with_embedding(tmp_path):
    """Embedding is stored as BLOB and retrieved as list[float]."""
    config = _make_config(tmp_path)
    init_db(config)
    emb = [0.5, -0.3, 0.7, 0.0, 1.0]
    now = datetime.now(UTC)
    item = _make_item(
        item_id="emb-001",
        embedding=emb,
        cluster_id=1,
        ingested_at=now,
    )
    store_item(config, item)

    results = get_items_for_curation(
        config, since=now - timedelta(hours=1), exclude_previous_editions=False
    )
    assert len(results) == 1
    assert results[0].embedding is not None
    assert len(results[0].embedding) == 5
    for i in range(5):
        assert abs(results[0].embedding[i] - emb[i]) < 1e-9


# ---------------------------------------------------------------------------
# get_items_for_curation
# ---------------------------------------------------------------------------


def test_get_items_for_curation_basic(tmp_path):
    """Returns items with embedding and cluster_id set."""
    config = _make_config(tmp_path)
    init_db(config)
    now = datetime.now(UTC)
    item = _make_item(
        item_id="curation-001",
        embedding=[1.0, 0.0],
        cluster_id=0,
        ingested_at=now,
    )
    store_item(config, item)

    results = get_items_for_curation(config, since=now - timedelta(hours=1))
    assert len(results) == 1
    assert results[0].item_id == "curation-001"


def test_get_items_for_curation_excludes_unembedded(tmp_path):
    """Items without embedding are excluded."""
    config = _make_config(tmp_path)
    init_db(config)
    now = datetime.now(UTC)
    # Item with embedding
    store_item(
        config,
        _make_item(
            item_id="embedded",
            embedding=[1.0],
            cluster_id=0,
            ingested_at=now,
        ),
    )
    # Item without embedding
    store_item(
        config,
        _make_item(item_id="unembedded", ingested_at=now),
    )

    results = get_items_for_curation(config, since=now - timedelta(hours=1))
    assert len(results) == 1
    assert results[0].item_id == "embedded"


def test_get_items_for_curation_since(tmp_path):
    """Only returns items after since datetime."""
    config = _make_config(tmp_path)
    init_db(config)
    old_time = datetime.now(UTC) - timedelta(days=30)
    new_time = datetime.now(UTC)

    store_item(
        config,
        _make_item(
            item_id="old",
            embedding=[1.0],
            cluster_id=0,
            ingested_at=old_time,
        ),
    )
    store_item(
        config,
        _make_item(
            item_id="new",
            embedding=[1.0],
            cluster_id=0,
            ingested_at=new_time,
        ),
    )

    # Only get items from the last day
    results = get_items_for_curation(
        config,
        since=datetime.now(UTC) - timedelta(days=1),
        exclude_previous_editions=False,
    )
    assert len(results) == 1
    assert results[0].item_id == "new"


def test_get_items_for_curation_excludes_editions(tmp_path):
    """Items in previous editions are excluded when flag is True."""
    config = _make_config(tmp_path)
    init_db(config)
    now = datetime.now(UTC)

    store_item(
        config,
        _make_item(
            item_id="used",
            embedding=[1.0],
            cluster_id=0,
            ingested_at=now,
        ),
    )
    store_item(
        config,
        _make_item(
            item_id="fresh",
            embedding=[1.0],
            cluster_id=1,
            ingested_at=now,
        ),
    )

    # Record an edition using the first item
    record_edition(config, "edition-1", ["used"], "/tmp/edition-1.json")

    results = get_items_for_curation(
        config,
        since=now - timedelta(hours=1),
        exclude_previous_editions=True,
    )
    assert len(results) == 1
    assert results[0].item_id == "fresh"


# ---------------------------------------------------------------------------
# get_cluster_count
# ---------------------------------------------------------------------------


def test_get_cluster_count_basic(tmp_path):
    """Counts distinct non-noise clusters."""
    config = _make_config(tmp_path)
    init_db(config)
    for i in range(3):
        store_item(
            config,
            _make_item(
                item_id=f"cluster-{i}",
                embedding=[float(i)],
                cluster_id=i,
            ),
        )
    assert get_cluster_count(config) == 3


def test_get_cluster_count_excludes_noise(tmp_path):
    """cluster_id -1 (noise) is not counted."""
    config = _make_config(tmp_path)
    init_db(config)
    store_item(
        config,
        _make_item(item_id="noise", embedding=[1.0], cluster_id=-1),
    )
    store_item(
        config,
        _make_item(item_id="real", embedding=[1.0], cluster_id=0),
    )
    assert get_cluster_count(config) == 1


# ---------------------------------------------------------------------------
# record_edition
# ---------------------------------------------------------------------------


def test_record_edition_basic(tmp_path):
    """Creates edition and edition_items rows."""
    config = _make_config(tmp_path)
    init_db(config)
    now = datetime.now(UTC)
    store_item(
        config,
        _make_item(item_id="item-a", embedding=[1.0], cluster_id=0, ingested_at=now),
    )
    store_item(
        config,
        _make_item(item_id="item-b", embedding=[1.0], cluster_id=1, ingested_at=now),
    )

    record_edition(config, "edition-test", ["item-a", "item-b"], "/tmp/edition.json")

    # Verify by checking that items are now excluded from curation
    results = get_items_for_curation(
        config,
        since=now - timedelta(hours=1),
        exclude_previous_editions=True,
    )
    assert len(results) == 0


def test_record_edition_query(tmp_path):
    """Items from recorded edition excluded from next curation."""
    config = _make_config(tmp_path)
    init_db(config)
    now = datetime.now(UTC)

    for i in range(4):
        store_item(
            config,
            _make_item(
                item_id=f"item-{i}",
                embedding=[float(i)],
                cluster_id=i % 2,
                ingested_at=now,
            ),
        )

    # Record edition with first two items
    record_edition(config, "ed-1", ["item-0", "item-1"], "/tmp/ed-1.json")

    # Only the other two should be available
    results = get_items_for_curation(
        config,
        since=now - timedelta(hours=1),
        exclude_previous_editions=True,
    )
    result_ids = {r.item_id for r in results}
    assert result_ids == {"item-2", "item-3"}


# ---------------------------------------------------------------------------
# get_edition_count
# ---------------------------------------------------------------------------


def test_get_edition_count_empty(tmp_path):
    """Empty DB returns 0 editions."""
    config = _make_config(tmp_path)
    init_db(config)
    assert get_edition_count(config) == 0


def test_get_edition_count_after_editions(tmp_path):
    """Count increments with each recorded edition."""
    config = _make_config(tmp_path)
    init_db(config)
    now = datetime.now(UTC)
    for i in range(3):
        store_item(
            config,
            _make_item(
                item_id=f"item-{i}",
                embedding=[float(i)],
                cluster_id=0,
                ingested_at=now,
            ),
        )

    assert get_edition_count(config) == 0

    record_edition(config, "ed-1", ["item-0"], "/tmp/ed-1.json")
    assert get_edition_count(config) == 1

    record_edition(config, "ed-2", ["item-1"], "/tmp/ed-2.json")
    assert get_edition_count(config) == 2


# ---------------------------------------------------------------------------
# update_image_paths
# ---------------------------------------------------------------------------


def test_update_image_paths(tmp_path):
    """Image paths updated after download."""
    config = _make_config(tmp_path)
    init_db(config)
    images = [
        ImageContent(url="https://example.com/img.jpg"),
    ]
    item = _make_item(item_id="img-001", images=images)
    store_item(config, item)

    # Simulate download by setting local_path
    updated_images = [
        ImageContent(
            url="https://example.com/img.jpg",
            local_path="images/img-001/abc123.jpg",
        ),
    ]
    update_image_paths(config, "img-001", updated_images)

    # Verify by retrieving (need embedding/cluster for curation view)
    # Use direct DB query instead
    import sqlite3
    from pathlib import Path

    db_path = Path(config["output"]["data_dir"]) / "offscroll.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT images_json FROM feed_items WHERE item_id = ?",
        ("img-001",),
    ).fetchone()
    conn.close()

    import json

    images_data = json.loads(row["images_json"])
    assert len(images_data) == 1
    assert images_data[0]["local_path"] == "images/img-001/abc123.jpg"


# ---------------------------------------------------------------------------
# _ensure_feed_source upsert (source attribution pipeline)
# ---------------------------------------------------------------------------


def test_ensure_feed_source_upsert_name(tmp_path):
    """feed_name is upserted when existing row has NULL name."""
    from offscroll.ingestion.store import get_feed_name_map

    config = _make_config(tmp_path)
    init_db(config)

    # First store: name will be NULL (no feed_name passed)
    item = _make_item(item_id="upsert-001")
    store_item(config, item)

    # Verify name is NULL
    name_map = get_feed_name_map(config)
    assert item.feed_url not in name_map

    # Second store with feed_name: should upsert
    item2 = _make_item(item_id="upsert-002")
    store_item(config, item2, feed_name="Test Blog")

    name_map = get_feed_name_map(config)
    assert name_map[item.feed_url] == "Test Blog"


def test_ensure_feed_source_does_not_overwrite_name(tmp_path):
    """feed_name upsert does not overwrite an existing non-NULL name."""
    from offscroll.ingestion.store import get_feed_name_map

    config = _make_config(tmp_path)
    init_db(config)

    # First store with a name
    item = _make_item(item_id="keep-001")
    store_item(config, item, feed_name="Original Name")

    # Second store with different name: should NOT overwrite
    item2 = _make_item(item_id="keep-002")
    store_item(config, item2, feed_name="New Name")

    name_map = get_feed_name_map(config)
    assert name_map[item.feed_url] == "Original Name"


# ---------------------------------------------------------------------------
# get_feed_name_map (source attribution pipeline)
# ---------------------------------------------------------------------------


def test_get_feed_name_map_empty(tmp_path):
    """Empty DB returns empty dict."""
    from offscroll.ingestion.store import get_feed_name_map

    config = _make_config(tmp_path)
    init_db(config)
    assert get_feed_name_map(config) == {}


def test_get_feed_name_map_with_named_feeds(tmp_path):
    """Returns only feeds with non-NULL names."""
    from offscroll.ingestion.store import get_feed_name_map

    config = _make_config(tmp_path)
    init_db(config)

    # Store items from two different feeds
    item1 = _make_item(
        item_id="map-001",
        feed_url="https://blog-a.com/feed.xml",
    )
    store_item(config, item1, feed_name="Blog A")

    item2 = _make_item(
        item_id="map-002",
        feed_url="https://blog-b.com/feed.xml",
    )
    store_item(config, item2)  # No feed_name

    name_map = get_feed_name_map(config)
    assert name_map == {"https://blog-a.com/feed.xml": "Blog A"}


def test_get_feed_name_map_multiple(tmp_path):
    """Multiple named feeds are all returned."""
    from offscroll.ingestion.store import get_feed_name_map

    config = _make_config(tmp_path)
    init_db(config)

    for i, (url, name) in enumerate(
        [
            ("https://a.com/feed", "Alpha"),
            ("https://b.com/feed", "Beta"),
            ("https://c.com/feed", "Gamma"),
        ]
    ):
        item = _make_item(item_id=f"multi-{i}", feed_url=url)
        store_item(config, item, feed_name=name)

    name_map = get_feed_name_map(config)
    assert len(name_map) == 3
    assert name_map["https://a.com/feed"] == "Alpha"
    assert name_map["https://b.com/feed"] == "Beta"
    assert name_map["https://c.com/feed"] == "Gamma"


def test_repair_missing_images(tmp_path):
    """repair_missing_images re-extracts images from content_html."""
    from offscroll.ingestion.store import init_db, repair_missing_images, store_item
    from offscroll.models import FeedItem, SourceType

    config = {"output": {"data_dir": str(tmp_path)}}
    init_db(config)

    # Store an item with content_html containing img tags but no images
    item = FeedItem(
        item_id="repair-test-1",
        source_type=SourceType.RSS,
        feed_url="https://example.com/feed",
        item_url="https://example.com/post/1",
        author="Test Author",
        title="Test Post",
        content_text="Some text content.",
        content_html='<p>Text with <img src="https://example.com/img.jpg" alt="photo"> inline.</p>',
        images=[],  # No images extracted initially
    )
    store_item(config, item)

    # Repair should find the img tag and add it
    repaired = repair_missing_images(config)
    assert repaired == 1

    # Running again should repair 0 (already fixed)
    repaired2 = repair_missing_images(config)
    assert repaired2 == 0
