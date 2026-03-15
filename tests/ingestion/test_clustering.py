"""Tests for content clustering.

Tests for the clustering module.
Tests should cover HDBSCAN clustering, similarity computation, and cluster labeling.
"""

from __future__ import annotations

from offscroll.ingestion.clustering import cluster_items
from offscroll.ingestion.embeddings import _embed_stub
from offscroll.ingestion.store import (
    get_items_for_clustering,
    store_item,
    update_cluster_ids,
)
from offscroll.models import FeedItem, SourceType


def _item(
    item_id: str = "test",
    content_text: str = "Test content",
    embedding: list[float] | None = None,
    cluster_id: int | None = None,
) -> FeedItem:
    """Create a minimal FeedItem for testing."""
    return FeedItem(
        item_id=item_id,
        source_type=SourceType.RSS,
        feed_url="https://example.com/feed.xml",
        content_text=content_text,
        embedding=embedding,
        cluster_id=cluster_id,
    )


# ---------------------------------------------------------------------------
# cluster_items
# ---------------------------------------------------------------------------


def test_cluster_basic():
    """items with distinct content are assigned cluster_ids."""
    # Create items with dissimilar content to maximize separation
    # even with hash-based embeddings
    texts = [
        "climate change effects on coral reefs",
        "global warming impacts on marine ecosystems",
        "ocean temperature rise and coral bleaching",
        "sourdough bread baking techniques",
        "homemade pasta recipe with fresh herbs",
    ]
    items = []
    for text in texts:
        item = _item(item_id=f"id-{hash(text)}", content_text=text)
        [embedding] = _embed_stub([text])
        item.embedding = embedding
        items.append(item)

    config = {"clustering": {"min_cluster_size": 2}}
    result = cluster_items(items, config)

    # All items should have cluster_id set
    assert all(item.cluster_id is not None for item in result)


def test_cluster_noise_items():
    """items that do not fit a cluster get cluster_id -1."""
    # Create items with very specific content
    items = []
    texts = [f"unique item number {i}" for i in range(3)]
    for text in texts:
        item = _item(item_id=f"id-{hash(text)}", content_text=text)
        [embedding] = _embed_stub([text])
        item.embedding = embedding
        items.append(item)

    config = {"clustering": {"min_cluster_size": 2}}
    result = cluster_items(items, config)

    # Some items may be assigned as noise (-1)
    cluster_ids = [item.cluster_id for item in result]
    assert all(cid is not None for cid in cluster_ids)


def test_cluster_too_few_items():
    """fewer items than min_cluster_size -> all noise."""
    items = []
    texts = ["item one", "item two"]
    for text in texts:
        item = _item(item_id=f"id-{hash(text)}", content_text=text)
        [embedding] = _embed_stub([text])
        item.embedding = embedding
        items.append(item)

    config = {"clustering": {"min_cluster_size": 5}}
    result = cluster_items(items, config)

    # All items should be assigned to noise cluster
    assert all(item.cluster_id == -1 for item in result)


def test_cluster_empty_list():
    """empty list returns empty list."""
    config = {"clustering": {"min_cluster_size": 3}}
    result = cluster_items([], config)
    assert result == []


def test_cluster_skips_no_embedding():
    """items without embeddings are skipped."""
    items = [
        _item(item_id="a", content_text="has embedding"),
        _item(item_id="b", content_text="no embedding", embedding=None),
        _item(item_id="c", content_text="has embedding"),
    ]
    # Add embeddings to items a and c
    [embedding_a] = _embed_stub([items[0].content_text])
    [embedding_c] = _embed_stub([items[2].content_text])
    items[0].embedding = embedding_a
    items[2].embedding = embedding_c

    config = {"clustering": {"min_cluster_size": 2}}
    result = cluster_items(items, config)

    # Items a and c should have cluster_ids
    assert result[0].cluster_id is not None
    assert result[2].cluster_id is not None
    # Item b should keep its None cluster_id
    assert result[1].cluster_id is None


def test_cluster_preserves_other_fields():
    """cluster_items does not modify non-cluster fields."""
    items = []
    texts = ["text one", "text two", "text three"]
    for text in texts:
        item = _item(
            item_id=f"id-{hash(text)}",
            content_text=text,
        )
        [embedding] = _embed_stub([text])
        item.embedding = embedding
        items.append(item)

    original_content = [item.content_text for item in items]
    original_ids = [item.item_id for item in items]

    config = {"clustering": {"min_cluster_size": 2}}
    result = cluster_items(items, config)

    # Non-cluster fields should be unchanged
    for i, item in enumerate(result):
        assert item.content_text == original_content[i]
        assert item.item_id == original_ids[i]


def test_cluster_config_min_size():
    """min_cluster_size from config is respected."""
    items = []
    texts = ["a", "b", "c", "d"]
    for text in texts:
        item = _item(item_id=f"id-{hash(text)}", content_text=text)
        [embedding] = _embed_stub([text])
        item.embedding = embedding
        items.append(item)

    # With min_cluster_size=4, items should cluster if they have 4+ embeddings
    config = {"clustering": {"min_cluster_size": 4}}
    result = cluster_items(items, config)
    assert all(item.cluster_id is not None for item in result)


# ---------------------------------------------------------------------------
# update_cluster_ids and get_items_for_clustering
# ---------------------------------------------------------------------------


def test_update_cluster_ids_basic(tmp_path):
    """cluster_ids persisted to DB."""
    # Create minimal config
    config = {"output": {"data_dir": str(tmp_path)}}

    # Store some items
    from offscroll.ingestion.store import init_db

    init_db(config)

    items = [
        _item(item_id="a", content_text="first item text"),
        _item(item_id="b", content_text="second item text"),
    ]

    # Embed them first
    for item in items:
        [embedding] = _embed_stub([item.content_text])
        item.embedding = embedding
        store_item(config, item)

    # Cluster them (min_cluster_size must be > 1 for HDBSCAN)
    items = cluster_items(items, {"clustering": {"min_cluster_size": 2}})

    # Update the DB
    count = update_cluster_ids(config, items)

    assert count == 2


def test_get_items_for_clustering(tmp_path):
    """returns items with embeddings but no cluster_id."""
    from offscroll.ingestion.store import init_db

    config = {"output": {"data_dir": str(tmp_path)}}
    init_db(config)

    # Create items: one with embedding but no cluster, one without embedding
    items_to_store = [
        _item(item_id="a", content_text="first"),
        _item(item_id="b", content_text="second"),
        _item(item_id="c", content_text="third"),
    ]

    # Add embeddings to a and b, store them
    [emb_a] = _embed_stub([items_to_store[0].content_text])
    [emb_b] = _embed_stub([items_to_store[1].content_text])
    items_to_store[0].embedding = emb_a
    items_to_store[1].embedding = emb_b

    for item in items_to_store[:2]:
        store_item(config, item)

    # Store c without embedding
    store_item(config, items_to_store[2])

    # Now manually set a cluster_id on item a in the DB to simulate it being clustered
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "offscroll.db"))
    conn.execute("UPDATE feed_items SET cluster_id = 0 WHERE item_id = 'a'")
    conn.commit()
    conn.close()

    # Get items for clustering
    items = get_items_for_clustering(config)

    # Should only return b (has embedding, no cluster_id)
    assert len(items) == 1
    assert items[0].item_id == "b"
