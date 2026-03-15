"""SQLite persistence for ingested content.

It defines the database schema and provides
functions for curation queries.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from offscroll.models import FeedItem, ImageContent, SourceType


def _get_db_path(config: dict) -> Path:
    """Get the path to the SQLite database."""
    data_dir = Path(config["output"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "offscroll.db"


def _get_connection(config: dict) -> sqlite3.Connection:
    """Get a database connection with foreign keys enabled."""
    db_path = _get_db_path(config)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(config: dict) -> None:
    """Initialize the SQLite database with the full schema.

    Creates all tables (feed_sources, feed_items, editions, edition_items),
    the items_ready_for_curation view, indexes, and enables foreign keys.
    """
    conn = _get_connection(config)
    cursor = conn.cursor()

    # Create feed_sources table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feed_sources (
            url             TEXT PRIMARY KEY,
            source_type     TEXT NOT NULL CHECK(
                source_type IN ('rss', 'atom', 'mastodon', 'bluesky')
            ),
            name            TEXT,
            last_polled     TEXT,
            last_item_id    TEXT
        )
    """)

    # Create feed_items table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feed_items (
            item_id         TEXT PRIMARY KEY,
            source_type     TEXT NOT NULL CHECK(
                source_type IN ('rss', 'atom', 'mastodon', 'bluesky')
            ),
            feed_url        TEXT NOT NULL,
            item_url        TEXT,
            author          TEXT,
            author_url      TEXT,
            title           TEXT,
            content_text    TEXT NOT NULL,
            content_html    TEXT,
            published_at    TEXT,
            ingested_at     TEXT NOT NULL,
            images_json     TEXT,
            is_thread       INTEGER DEFAULT 0,
            thread_id       TEXT,
            thread_position INTEGER,
            word_count      INTEGER NOT NULL,
            embedding       BLOB,
            cluster_id      INTEGER,
            FOREIGN KEY (feed_url) REFERENCES feed_sources(url)
        )
    """)

    # Create editions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS editions (
            edition_id    TEXT PRIMARY KEY,
            created_at    TEXT NOT NULL,
            json_path     TEXT,
            status        TEXT DEFAULT 'draft' CHECK(status IN ('draft', 'published'))
        )
    """)

    # Create edition_items table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS edition_items (
            edition_id    TEXT NOT NULL,
            item_id       TEXT NOT NULL,
            PRIMARY KEY (edition_id, item_id),
            FOREIGN KEY (edition_id) REFERENCES editions(edition_id),
            FOREIGN KEY (item_id) REFERENCES feed_items(item_id)
        )
    """)

    # Create view for items ready for curation
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS items_ready_for_curation AS
            SELECT * FROM feed_items
            WHERE embedding IS NOT NULL AND cluster_id IS NOT NULL
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_items_feed ON feed_items(feed_url)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_items_cluster ON feed_items(cluster_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_items_ingested ON feed_items(ingested_at)")

    conn.commit()
    conn.close()


def _serialize_images(images: list[ImageContent]) -> str:
    """Serialize images list to JSON string for storage."""
    return json.dumps([asdict(img) for img in images])


def _deserialize_images(images_json: str | None) -> list[ImageContent]:
    """Deserialize images JSON string back to list of ImageContent."""
    if not images_json:
        return []
    data = json.loads(images_json)
    return [ImageContent(**d) for d in data]


def _serialize_embedding(embedding: list[float] | None) -> bytes | None:
    """Serialize embedding to BLOB for storage."""
    if embedding is None:
        return None
    return json.dumps(embedding).encode("utf-8")


def _deserialize_embedding(blob: bytes | None) -> list[float] | None:
    """Deserialize embedding BLOB back to list of floats."""
    if blob is None:
        return None
    return json.loads(blob.decode("utf-8"))


def _ensure_feed_source(
    conn: sqlite3.Connection,
    item: FeedItem,
    feed_name: str | None = None,
) -> None:
    """Ensure the feed source exists in feed_sources table.

    If feed_name is provided and the existing row has name=NULL,
    update it via upsert.
    """
    conn.execute(
        """INSERT INTO feed_sources (url, source_type, name)
           VALUES (?, ?, ?)
           ON CONFLICT(url) DO UPDATE SET name = excluded.name
           WHERE feed_sources.name IS NULL AND excluded.name IS NOT NULL""",
        (item.feed_url, item.source_type.value, feed_name),
    )


def register_feed_source(
    config: dict,
    url: str,
    source_type: str,
    name: str | None = None,
) -> bool:
    """Register a feed source in the database.

    If the source already exists and has no name, the provided name
    is set via upsert.

    Args:
        config: The OffScroll config dict.
        url: Feed URL.
        source_type: "rss", "atom", "mastodon", or "bluesky".
        name: Human-readable name (optional).

    Returns:
        True if inserted (new), False if already existed.
    """
    conn = _get_connection(config)
    try:
        cursor = conn.execute(
            """INSERT INTO feed_sources (url, source_type, name)
               VALUES (?, ?, ?)
               ON CONFLICT(url) DO UPDATE SET name = excluded.name
               WHERE feed_sources.name IS NULL AND excluded.name IS NOT NULL""",
            (url, source_type, name),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _row_to_feed_item(row: sqlite3.Row) -> FeedItem:
    """Convert a database row to a FeedItem instance."""
    published_at = None
    if row["published_at"]:
        published_at = datetime.fromisoformat(row["published_at"])

    ingested_at = datetime.fromisoformat(row["ingested_at"])

    return FeedItem(
        item_id=row["item_id"],
        source_type=SourceType(row["source_type"]),
        feed_url=row["feed_url"],
        item_url=row["item_url"],
        author=row["author"],
        author_url=row["author_url"],
        title=row["title"],
        content_text=row["content_text"],
        content_html=row["content_html"],
        published_at=published_at,
        ingested_at=ingested_at,
        images=_deserialize_images(row["images_json"]),
        is_thread=bool(row["is_thread"]),
        thread_id=row["thread_id"],
        thread_position=row["thread_position"],
        word_count=row["word_count"],
        embedding=_deserialize_embedding(row["embedding"]),
        cluster_id=row["cluster_id"],
    )


def store_item(
    config: dict,
    item: FeedItem,
    feed_name: str | None = None,
) -> bool:
    """Store a FeedItem in the database.

    Args:
        config: The OffScroll config dict.
        item: The FeedItem to store.
        feed_name: Optional feed-level title to store on the
            feed source row (upserted if the existing name is NULL).

    Returns:
        True if the item was inserted (new), False if it already
        existed (duplicate item_id -- skip silently).
    """
    conn = _get_connection(config)
    try:
        _ensure_feed_source(conn, item, feed_name=feed_name)

        cursor = conn.execute(
            """INSERT OR IGNORE INTO feed_items (
                item_id, source_type, feed_url, item_url, author,
                author_url, title, content_text, content_html,
                published_at, ingested_at, images_json, is_thread,
                thread_id, thread_position, word_count, embedding,
                cluster_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.item_id,
                item.source_type.value,
                item.feed_url,
                item.item_url,
                item.author,
                item.author_url,
                item.title,
                item.content_text,
                item.content_html,
                item.published_at.isoformat() if item.published_at else None,
                item.ingested_at.isoformat(),
                _serialize_images(item.images),
                int(item.is_thread),
                item.thread_id,
                item.thread_position,
                item.word_count,
                _serialize_embedding(item.embedding),
                item.cluster_id,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_items_for_curation(
    config: dict,
    since: datetime | None = None,
    exclude_previous_editions: bool = True,
) -> list[FeedItem]:
    """Return all FeedItems with embeddings and cluster assignments
    since the given datetime (default: last 7 days).

    If exclude_previous_editions is True (default), items already
    assigned to a previous edition are excluded. This uses the
    items_ready_for_curation view joined against edition_items.

    Items are returned as FeedItem dataclass instances with
    embedding and cluster_id populated.
    """
    if since is None:
        since = datetime.now(UTC) - timedelta(days=7)

    conn = _get_connection(config)
    conn.row_factory = sqlite3.Row
    try:
        if exclude_previous_editions:
            query = """
                SELECT irc.* FROM items_ready_for_curation irc
                LEFT JOIN edition_items ei ON irc.item_id = ei.item_id
                WHERE ei.item_id IS NULL
                  AND irc.ingested_at >= ?
            """
        else:
            query = """
                SELECT * FROM items_ready_for_curation
                WHERE ingested_at >= ?
            """

        rows = conn.execute(query, (since.isoformat(),)).fetchall()
        return [_row_to_feed_item(row) for row in rows]
    finally:
        conn.close()


def get_cluster_count(config: dict) -> int:
    """Return the number of distinct non-noise clusters.

    Counts distinct cluster_id values from feed_items WHERE
    cluster_id IS NOT NULL AND cluster_id != -1 AND
    embedding IS NOT NULL.
    """
    conn = _get_connection(config)
    try:
        row = conn.execute(
            """SELECT COUNT(DISTINCT cluster_id) FROM feed_items
               WHERE cluster_id IS NOT NULL
                 AND cluster_id != -1
                 AND embedding IS NOT NULL"""
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def record_edition(
    config: dict,
    edition_id: str,
    item_ids: list[str],
    json_path: str,
) -> None:
    """Record an edition and its item assignments in the DB.

    Creates an entry in editions and edition_items.
    """
    conn = _get_connection(config)
    try:
        conn.execute(
            """INSERT OR REPLACE INTO editions (edition_id, created_at, json_path, status)
               VALUES (?, ?, ?, 'draft')""",
            (edition_id, datetime.now(UTC).isoformat(), json_path),
        )
        # Clear previous item assignments for this edition
        conn.execute(
            "DELETE FROM edition_items WHERE edition_id = ?",
            (edition_id,),
        )
        for item_id in item_ids:
            conn.execute(
                """INSERT INTO edition_items (edition_id, item_id)
                   VALUES (?, ?)""",
                (edition_id, item_id),
            )
        conn.commit()
    finally:
        conn.close()


def update_image_paths(
    config: dict,
    item_id: str,
    images: list[ImageContent],
) -> None:
    """Update the images_json for an item after image download.

    Args:
        config: The OffScroll config dict.
        item_id: The item whose images to update.
        images: Updated ImageContent list with local_path populated.
    """
    conn = _get_connection(config)
    try:
        conn.execute(
            "UPDATE feed_items SET images_json = ? WHERE item_id = ?",
            (_serialize_images(images), item_id),
        )
        conn.commit()
    finally:
        conn.close()


def repair_missing_images(config: dict) -> int:
    """Re-extract images from content_html for items with zero images.

     Items ingested before the HTML image extraction
    fallback  may have empty images_json despite having
    img tags in their content_html. This one-time repair scans for
    such items and populates their images from HTML.

    Args:
        config: The OffScroll config dict.

    Returns:
        The count of items repaired.
    """
    from offscroll.ingestion.feeds import _extract_images_from_html

    conn = _get_connection(config)
    try:
        cursor = conn.execute(
            """SELECT item_id, content_html, images_json
               FROM feed_items
               WHERE content_html IS NOT NULL
                 AND content_html != ''
                 AND (images_json IS NULL OR images_json = '[]')"""
        )
        repaired = 0
        for row in cursor.fetchall():
            item_id = row[0]
            content_html = row[1]
            images = _extract_images_from_html(content_html)
            if images:
                conn.execute(
                    "UPDATE feed_items SET images_json = ? WHERE item_id = ?",
                    (_serialize_images(images), item_id),
                )
                repaired += 1
        conn.commit()
        return repaired
    finally:
        conn.close()


def update_embeddings(config: dict, items: list[FeedItem]) -> int:
    """Update the embedding column for items that have been embedded.

    Args:
        config: The OffScroll config dict.
        items: FeedItems with embedding field populated.

    Returns:
        The count of items updated.
    """
    conn = _get_connection(config)
    try:
        count = 0
        for item in items:
            if item.embedding is not None:
                conn.execute(
                    "UPDATE feed_items SET embedding = ? WHERE item_id = ?",
                    (_serialize_embedding(item.embedding), item.item_id),
                )
                count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def get_feed_name_map(config: dict) -> dict[str, str]:
    """Return a mapping of feed URL to feed name.

    Only includes feeds that have a non-NULL name.

    Args:
        config: The OffScroll config dict.

    Returns:
        Dict mapping feed_url -> feed_name for all named sources.
    """
    conn = _get_connection(config)
    try:
        rows = conn.execute("SELECT url, name FROM feed_sources WHERE name IS NOT NULL").fetchall()
        return {row[0]: row[1] for row in rows}
    finally:
        conn.close()


def get_feed_stats(config: dict) -> list[dict]:
    """Return statistics for each configured feed source.

    Returns a list of dicts with keys: url, name, source_type, item_count
    """
    conn = _get_connection(config)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT fs.url, fs.name, fs.source_type,
                      COUNT(fi.item_id) as item_count
               FROM feed_sources fs
               LEFT JOIN feed_items fi ON fs.url = fi.feed_url
               GROUP BY fs.url"""
        ).fetchall()
        return [
            {
                "url": row["url"],
                "name": row["name"],
                "source_type": row["source_type"],
                "item_count": row["item_count"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_db_stats(config: dict) -> dict:
    """Return overall database statistics.

    Returns a dict with keys: total_items, total_feeds, total_editions
    """
    conn = _get_connection(config)
    try:
        total_items = conn.execute("SELECT COUNT(*) FROM feed_items").fetchone()[0]
        total_feeds = conn.execute("SELECT COUNT(*) FROM feed_sources").fetchone()[0]
        total_editions = conn.execute("SELECT COUNT(*) FROM editions").fetchone()[0]
        return {
            "total_items": total_items,
            "total_feeds": total_feeds,
            "total_editions": total_editions,
        }
    finally:
        conn.close()


def get_edition_count(config: dict) -> int:
    """Return the total number of editions in the DB."""
    conn = _get_connection(config)
    try:
        row = conn.execute("SELECT COUNT(*) FROM editions").fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def get_items_for_embedding(config: dict) -> list[FeedItem]:
    """Return all FeedItems that do not yet have embeddings.

    Query: SELECT * FROM feed_items WHERE embedding IS NULL
    """
    conn = _get_connection(config)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM feed_items WHERE embedding IS NULL").fetchall()
        return [_row_to_feed_item(row) for row in rows]
    finally:
        conn.close()


def get_items_for_clustering(config: dict) -> list[FeedItem]:
    """Return all FeedItems that have embeddings but no cluster_id.

    Query: SELECT * FROM feed_items WHERE embedding IS NOT NULL
           AND cluster_id IS NULL
    """
    conn = _get_connection(config)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM feed_items WHERE embedding IS NOT NULL AND cluster_id IS NULL"
        ).fetchall()
        return [_row_to_feed_item(row) for row in rows]
    finally:
        conn.close()


def update_cluster_ids(config: dict, items: list[FeedItem]) -> int:
    """Update the cluster_id column for items that have been clustered.

    Args:
        config: The OffScroll config dict.
        items: FeedItems with cluster_id field populated.

    Returns:
        The count of items updated.
    """
    conn = _get_connection(config)
    try:
        count = 0
        for item in items:
            if item.cluster_id is not None:
                conn.execute(
                    "UPDATE feed_items SET cluster_id = ? WHERE item_id = ?",
                    (item.cluster_id, item.item_id),
                )
                count += 1
        conn.commit()
        return count
    finally:
        conn.close()
