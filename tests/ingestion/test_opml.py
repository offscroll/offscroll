"""Tests for OPML import.

comprehensive tests for OPML parsing and feed registration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_DATA = Path(__file__).parent.parent / "sample_data"


# ============================================================================
# import_opml() tests
# ============================================================================


def test_import_opml_basic():
    """Parses sample OPML file and returns feed entries."""
    from offscroll.ingestion.opml import import_opml

    opml_path = SAMPLE_DATA / "feeds" / "sample_opml.xml"
    feeds = import_opml(opml_path)

    # Sample OPML has 10 feeds across 3 folders
    assert len(feeds) == 10
    assert all(feed["source_type"] == "rss" for feed in feeds)


def test_import_opml_extracts_urls():
    """All xmlUrl values are extracted."""
    from offscroll.ingestion.opml import import_opml

    opml_path = SAMPLE_DATA / "feeds" / "sample_opml.xml"
    feeds = import_opml(opml_path)

    urls = [feed["url"] for feed in feeds]
    assert "https://example.com/tech-feed.xml" in urls
    assert "https://news.example.com/feed.xml" in urls
    assert "https://alice.example.com/feed.xml" in urls


def test_import_opml_extracts_names():
    """Title/text attributes are used as names."""
    from offscroll.ingestion.opml import import_opml

    opml_path = SAMPLE_DATA / "feeds" / "sample_opml.xml"
    feeds = import_opml(opml_path)

    # Check that names are extracted (should be title or text)
    names = [feed["name"] for feed in feeds]
    assert "Example Tech Blog" in names
    assert "Alice's Blog" in names


def test_import_opml_no_duplicates():
    """Duplicate xmlUrl entries are deduplicated."""
    import tempfile
    from pathlib import Path

    from offscroll.ingestion.opml import import_opml

    # Create a temporary OPML with duplicates
    opml_content = """<?xml version="1.0"?>
<opml version="2.0">
  <head><title>Test</title></head>
  <body>
    <outline type="rss" title="Feed 1" xmlUrl="https://example.com/feed.xml" />
    <outline type="rss" title="Feed 2" xmlUrl="https://example.com/feed.xml" />
    <outline type="rss" title="Feed 3" xmlUrl="https://other.com/feed.xml" />
  </body>
</opml>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(opml_content)
        temp_path = Path(f.name)

    try:
        feeds = import_opml(temp_path)
        assert len(feeds) == 2
        urls = [feed["url"] for feed in feeds]
        assert urls.count("https://example.com/feed.xml") == 1
    finally:
        temp_path.unlink()


def test_import_opml_file_not_found():
    """FileNotFoundError for missing file."""
    from offscroll.ingestion.opml import import_opml

    with pytest.raises(FileNotFoundError):
        import_opml(Path("/nonexistent/file.opml"))


def test_import_opml_invalid_xml():
    """ValueError for malformed XML."""
    import tempfile
    from pathlib import Path

    from offscroll.ingestion.opml import import_opml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write("This is not valid XML <unclosed>")
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Failed to parse OPML XML"):
            import_opml(temp_path)
    finally:
        temp_path.unlink()


def test_import_opml_no_feeds():
    """ValueError for OPML with no outline elements with xmlUrl."""
    import tempfile
    from pathlib import Path

    from offscroll.ingestion.opml import import_opml

    opml_content = """<?xml version="1.0"?>
<opml version="2.0">
  <head><title>Empty</title></head>
  <body>
    <outline text="Folder with no feeds" />
  </body>
</opml>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(opml_content)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="contains no feed outlines"):
            import_opml(temp_path)
    finally:
        temp_path.unlink()


def test_import_opml_flattens_nested():
    """Nested outlines are flattened correctly."""
    from offscroll.ingestion.opml import import_opml

    opml_path = SAMPLE_DATA / "feeds" / "sample_opml.xml"
    feeds = import_opml(opml_path)

    # All feeds should be at top level (flattened)
    # The sample OPML has feeds nested in folder outlines
    assert len(feeds) == 10
    # Each feed should have URL and name
    for feed in feeds:
        assert "url" in feed
        assert "name" in feed
        assert feed["source_type"] == "rss"


# ============================================================================
# register_opml_feeds() tests
# ============================================================================


def test_register_opml_feeds_basic(sample_config, tmp_path):
    """Feeds are registered in DB."""
    from offscroll.ingestion.opml import register_opml_feeds
    from offscroll.ingestion.store import get_feed_stats, init_db

    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
    }

    opml_path = SAMPLE_DATA / "feeds" / "sample_opml.xml"
    init_db(config)
    count = register_opml_feeds(config, opml_path)

    assert count == 10

    # Verify feeds are in DB
    stats = get_feed_stats(config)
    assert len(stats) == 10  # 10 feeds from OPML


def test_register_opml_feeds_skip_existing(sample_config, tmp_path):
    """Existing feeds are not duplicated."""
    from offscroll.ingestion.opml import register_opml_feeds
    from offscroll.ingestion.store import get_feed_stats, init_db, register_feed_source

    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
    }

    init_db(config)

    # Pre-register one feed from the OPML
    register_feed_source(
        config,
        "https://example.com/tech-feed.xml",
        "rss",
        "Example Tech Blog",
    )

    opml_path = SAMPLE_DATA / "feeds" / "sample_opml.xml"
    count = register_opml_feeds(config, opml_path)

    # Only 9 new feeds (1 already exists)
    assert count == 9

    # Verify correct count in DB
    stats = get_feed_stats(config)
    # 1 pre-registered + 9 new from OPML = 10 total
    assert len(stats) == 10


def test_register_opml_feeds_file_not_found(sample_config):
    """FileNotFoundError for missing OPML file."""
    from offscroll.ingestion.opml import register_opml_feeds

    with pytest.raises(FileNotFoundError):
        register_opml_feeds(sample_config, Path("/nonexistent.opml"))


def test_register_opml_feeds_with_names(sample_config, tmp_path):
    """Feed names from OPML are preserved in DB."""
    from offscroll.ingestion.opml import register_opml_feeds
    from offscroll.ingestion.store import get_feed_stats, init_db

    config = {
        **sample_config,
        "output": {"data_dir": str(tmp_path)},
    }

    opml_path = SAMPLE_DATA / "feeds" / "sample_opml.xml"
    init_db(config)
    register_opml_feeds(config, opml_path)

    stats = get_feed_stats(config)
    # Find a feed with known name from sample OPML
    alice_feed = next(
        (s for s in stats if "alice.example.com/feed.xml" in s["url"]),
        None,
    )
    assert alice_feed is not None
    assert alice_feed["name"] == "Alice's Blog"
