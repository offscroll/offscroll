"""Tests for RSS/Atom feed parsing.

parse_feed() tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from offscroll.ingestion.feeds import (
    _discover_feed_url,
    _extract_images_from_html,
    _looks_like_html,
    _probe_common_feed_paths,
    _strip_boilerplate,
    ingest_all_feeds,
    parse_feed,
)
from offscroll.models import SourceType

SAMPLE_DATA = Path(__file__).parent.parent / "sample_data"


@pytest.fixture
def rss_xml() -> str:
    """Load sample RSS XML."""
    return (SAMPLE_DATA / "feeds" / "sample_rss.xml").read_text()


@pytest.fixture
def atom_xml() -> str:
    """Load sample Atom XML."""
    return (SAMPLE_DATA / "feeds" / "sample_atom.xml").read_text()


@pytest.fixture
def atom_extended_xml() -> str:
    """Load extended Atom XML with edge cases."""
    return (SAMPLE_DATA / "feeds" / "sample_atom_extended.xml").read_text()


# ---------------------------------------------------------------------------
# RSS parsing
# ---------------------------------------------------------------------------


def test_parse_rss_basic(rss_xml):
    """Parse a simple RSS 2.0 feed into FeedItems."""
    result = parse_feed(rss_xml, feed_url="https://example.com/feed.xml")
    items = result.items
    assert len(items) == 3


def test_parse_rss_fields(rss_xml):
    """All expected fields are populated on parsed RSS items."""
    items = parse_feed(rss_xml, feed_url="https://example.com/feed.xml").items
    first = items[0]
    assert first.title == "First Test Post"
    assert first.item_url == "https://example.com/post-1"
    assert first.content_text != ""
    assert first.feed_url == "https://example.com/feed.xml"


def test_parse_rss_author_extraction(rss_xml):
    """Author name is extracted from 'email (Name)' format."""
    items = parse_feed(rss_xml, feed_url="https://example.com/feed.xml").items
    first = items[0]
    assert first.author == "Alice"


def test_parse_rss_image_from_enclosure(rss_xml):
    """Enclosure with image type creates ImageContent."""
    items = parse_feed(rss_xml, feed_url="https://example.com/feed.xml").items
    # Second item has an enclosure
    second = items[1]
    assert len(second.images) == 1
    assert "image.jpg" in second.images[0].url


def test_parse_rss_word_count(rss_xml):
    """Word count is auto-computed from content_text."""
    items = parse_feed(rss_xml, feed_url="https://example.com/feed.xml").items
    for item in items:
        assert item.word_count > 0


def test_parse_rss_publication_order(rss_xml):
    """Items are sorted oldest first by published_at."""
    items = parse_feed(rss_xml, feed_url="https://example.com/feed.xml").items
    for i in range(len(items) - 1):
        assert items[i].published_at <= items[i + 1].published_at


# ---------------------------------------------------------------------------
# Atom parsing
# ---------------------------------------------------------------------------


def test_parse_atom_basic(atom_xml):
    """Parse a simple Atom feed into FeedItems."""
    items = parse_feed(atom_xml, feed_url="https://example.com/atom.xml").items
    assert len(items) == 1


def test_parse_atom_content(atom_xml):
    """Atom HTML content is converted to plain text."""
    items = parse_feed(atom_xml, feed_url="https://example.com/atom.xml").items
    first = items[0]
    assert first.content_text != ""
    # Should not contain HTML tags
    assert "<p>" not in first.content_text
    assert "</" not in first.content_text


def test_parse_atom_no_title(atom_extended_xml):
    """Atom entry with no title has title=None."""
    items = parse_feed(atom_extended_xml, feed_url="https://example.com/atom.xml").items
    notitle_items = [item for item in items if item.author == "NoTitle"]
    assert len(notitle_items) == 1
    assert notitle_items[0].title is None


# ---------------------------------------------------------------------------
# Auto-detection and overrides
# ---------------------------------------------------------------------------


def test_parse_feed_auto_detect_rss(rss_xml):
    """Source type is auto-detected as RSS for RSS feeds."""
    items = parse_feed(rss_xml, feed_url="https://example.com/feed.xml").items
    for item in items:
        assert item.source_type == SourceType.RSS


def test_parse_feed_auto_detect_atom(atom_xml):
    """Source type is auto-detected as ATOM for Atom feeds."""
    items = parse_feed(atom_xml, feed_url="https://example.com/atom.xml").items
    for item in items:
        assert item.source_type == SourceType.ATOM


def test_parse_feed_explicit_type(rss_xml):
    """Explicit source_type parameter overrides auto-detection."""
    items = parse_feed(
        rss_xml,
        feed_url="https://example.com/feed.xml",
        source_type=SourceType.ATOM,
    ).items
    for item in items:
        assert item.source_type == SourceType.ATOM


def test_parse_feed_empty():
    """Empty or invalid XML raises ValueError."""
    with pytest.raises(ValueError):
        parse_feed("", feed_url="https://example.com/feed.xml")


# ---------------------------------------------------------------------------
# ingest_all_feeds 
# ---------------------------------------------------------------------------


def test_ingest_all_feeds_basic(rss_xml, tmp_path):
    """Mocked HTTP returns sample XML, items stored."""
    config = {
        "feeds": {"rss": [{"url": "https://example.com/feed.xml"}]},
        "output": {"data_dir": str(tmp_path)},
    }
    mock_response = MagicMock()
    mock_response.text = rss_xml
    mock_response.raise_for_status = MagicMock()
    with patch("offscroll.ingestion.feeds.httpx.get", return_value=mock_response):
        count = ingest_all_feeds(config)
    assert count == 3  # sample_rss.xml has 3 items


def test_ingest_all_feeds_skip_duplicates(rss_xml, tmp_path):
    """Second call returns 0 new items."""
    config = {
        "feeds": {"rss": [{"url": "https://example.com/feed.xml"}]},
        "output": {"data_dir": str(tmp_path)},
    }
    mock_response = MagicMock()
    mock_response.text = rss_xml
    mock_response.raise_for_status = MagicMock()
    with patch("offscroll.ingestion.feeds.httpx.get", return_value=mock_response):
        first = ingest_all_feeds(config)
        second = ingest_all_feeds(config)
    assert first == 3
    assert second == 0


def test_ingest_all_feeds_bad_feed(rss_xml, tmp_path):
    """One bad URL logged, other feeds still processed."""
    config = {
        "feeds": {
            "rss": [
                {"url": "https://bad.example.com/feed.xml"},
                {"url": "https://good.example.com/feed.xml"},
            ]
        },
        "output": {"data_dir": str(tmp_path)},
    }
    mock_response = MagicMock()
    mock_response.text = rss_xml
    mock_response.raise_for_status = MagicMock()

    def side_effect(url, **kwargs):
        if "bad" in url:
            raise httpx.HTTPError("Connection refused")
        return mock_response

    import httpx

    with patch("offscroll.ingestion.feeds.httpx.get", side_effect=side_effect):
        count = ingest_all_feeds(config)
    assert count == 3  # Only the good feed's items


def test_ingest_all_feeds_empty_config(tmp_path):
    """No feeds configured, returns 0."""
    config = {
        "feeds": {"rss": []},
        "output": {"data_dir": str(tmp_path)},
    }
    count = ingest_all_feeds(config)
    assert count == 0


def test_ingest_all_feeds_mapping_proxy_config(rss_xml, tmp_path):
    """Config wrapped in MappingProxyType (as load_config returns) works.

    Regression test: isinstance(feed_conf, dict) returned False for
    MappingProxyType objects, causing a crash when the wizard wrote
    feeds as [{"url": "..."}] and load_config() wrapped them.
    """
    from offscroll.config import _recursive_proxy

    raw_config = {
        "feeds": {"rss": [{"url": "https://example.com/feed.xml"}]},
        "output": {"data_dir": str(tmp_path)},
    }
    proxied_config = _recursive_proxy(raw_config)

    mock_response = MagicMock()
    mock_response.text = rss_xml
    mock_response.raise_for_status = MagicMock()
    with patch("offscroll.ingestion.feeds.httpx.get", return_value=mock_response):
        count = ingest_all_feeds(proxied_config)
    assert count == 3


# ---------------------------------------------------------------------------
# HTML detection and RSS autodiscovery
# ---------------------------------------------------------------------------

HTML_WITH_RSS_LINK = """\
<!DOCTYPE html>
<html>
<head>
    <title>My Blog</title>
    <link rel="alternate" type="application/rss+xml"
          title="RSS Feed" href="https://myblog.com/feed.xml" />
</head>
<body><h1>Welcome</h1></body>
</html>
"""

HTML_WITH_ATOM_LINK = """\
<!DOCTYPE html>
<html>
<head>
    <title>My Blog</title>
    <link rel="alternate" type="application/atom+xml"
          title="Atom Feed" href="/atom.xml" />
</head>
<body><h1>Welcome</h1></body>
</html>
"""

HTML_NO_FEED_LINK = """\
<!DOCTYPE html>
<html>
<head><title>My Blog</title></head>
<body><h1>Welcome</h1></body>
</html>
"""


def test_looks_like_html_content_type():
    """HTML detected by content-type header."""
    assert _looks_like_html("anything", "text/html; charset=utf-8") is True


def test_looks_like_html_doctype():
    """HTML detected by <!DOCTYPE tag."""
    assert _looks_like_html("<!DOCTYPE html><html>") is True


def test_looks_like_html_tag():
    """HTML detected by <html tag."""
    assert _looks_like_html("<html lang='en'>") is True


def test_looks_like_html_xml():
    """XML content is not detected as HTML."""
    assert _looks_like_html('<?xml version="1.0"?><rss>') is False


def test_discover_feed_url_rss():
    """RSS autodiscovery link is extracted from HTML."""
    url = _discover_feed_url(HTML_WITH_RSS_LINK, "https://myblog.com/")
    assert url == "https://myblog.com/feed.xml"


def test_discover_feed_url_atom_relative():
    """Atom autodiscovery link with relative URL is resolved."""
    url = _discover_feed_url(HTML_WITH_ATOM_LINK, "https://myblog.com/")
    assert url == "https://myblog.com/atom.xml"


def test_discover_feed_url_none():
    """HTML without feed links returns None."""
    url = _discover_feed_url(HTML_NO_FEED_LINK, "https://myblog.com/")
    assert url is None


def test_ingest_html_with_autodiscovery(rss_xml, tmp_path):
    """Blog URL triggers autodiscovery and fetches the real feed."""
    config = {
        "feeds": {"rss": [{"url": "https://myblog.com/"}]},
        "output": {"data_dir": str(tmp_path)},
    }

    html_response = MagicMock()
    html_response.text = HTML_WITH_RSS_LINK
    html_response.headers = {"content-type": "text/html; charset=utf-8"}
    html_response.raise_for_status = MagicMock()

    feed_response = MagicMock()
    feed_response.text = rss_xml
    feed_response.headers = {"content-type": "application/rss+xml"}
    feed_response.raise_for_status = MagicMock()

    def side_effect(url, **kwargs):
        if url == "https://myblog.com/":
            return html_response
        if url == "https://myblog.com/feed.xml":
            return feed_response
        raise httpx.HTTPError(f"Unexpected URL: {url}")

    import httpx

    with patch("offscroll.ingestion.feeds.httpx.get", side_effect=side_effect):
        count = ingest_all_feeds(config)
    assert count == 3


def test_ingest_html_no_autodiscovery(tmp_path, caplog):
    """Blog URL without feed link logs a clear error message."""
    config = {
        "feeds": {"rss": [{"url": "https://myblog.com/"}]},
        "output": {"data_dir": str(tmp_path)},
    }

    html_response = MagicMock()
    html_response.text = HTML_NO_FEED_LINK
    html_response.headers = {"content-type": "text/html; charset=utf-8"}
    html_response.raise_for_status = MagicMock()

    with patch("offscroll.ingestion.feeds.httpx.get", return_value=html_response):
        count = ingest_all_feeds(config)

    assert count == 0
    assert "appears to be a web page, not an RSS feed" in caplog.text


def test_ingest_real_rss_still_works(rss_xml, tmp_path):
    """Actual RSS XML content still works normally (no false HTML detection)."""
    config = {
        "feeds": {"rss": [{"url": "https://example.com/feed.xml"}]},
        "output": {"data_dir": str(tmp_path)},
    }
    mock_response = MagicMock()
    mock_response.text = rss_xml
    mock_response.headers = {"content-type": "application/rss+xml"}
    mock_response.raise_for_status = MagicMock()
    with patch("offscroll.ingestion.feeds.httpx.get", return_value=mock_response):
        count = ingest_all_feeds(config)
    assert count == 3


# ---------------------------------------------------------------------------
# Item 1: _extract_images_from_html
# ---------------------------------------------------------------------------


def test_extract_images_from_html_basic():
    """Extracts img tags with src and alt attributes."""
    html = '<p>Text</p><img src="https://example.com/comic.png" alt="A comic">'
    images = _extract_images_from_html(html)
    assert len(images) == 1
    assert images[0].url == "https://example.com/comic.png"
    assert images[0].alt_text == "A comic"


def test_extract_images_from_html_no_images():
    """Returns empty list when no img tags present."""
    html = "<p>Just some text with no images</p>"
    images = _extract_images_from_html(html)
    assert images == []


def test_extract_images_from_html_skips_data_uris():
    """Data URIs are skipped."""
    html = '<img src="data:image/png;base64,abc123" alt="inline">'
    images = _extract_images_from_html(html)
    assert images == []


def test_extract_images_from_html_skips_tracking_pixels():
    """1x1 tracking pixels are skipped."""
    html = '<img src="https://tracker.com/pixel.gif" width="1" height="1">'
    images = _extract_images_from_html(html)
    assert images == []


def test_extract_images_from_html_multiple():
    """Multiple images are extracted."""
    html = (
        '<img src="https://example.com/a.png" alt="first">'
        '<img src="https://example.com/b.png" alt="second">'
    )
    images = _extract_images_from_html(html)
    assert len(images) == 2


def test_parse_feed_html_image_fallback():
    """parse_feed extracts images from HTML content when no enclosures exist."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
    <channel><title>Comics</title>
    <item>
        <title>Episode 1</title>
        <link>https://example.com/1</link>
        <description>&lt;p&gt;Text&lt;/p&gt;&lt;img src="https://example.com/comic.png"
 alt="Comic"&gt;</description>
        <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
    </item>
    </channel></rss>"""
    items = parse_feed(xml, feed_url="https://example.com/rss.xml").items
    assert len(items) == 1
    assert len(items[0].images) == 1
    assert "comic.png" in items[0].images[0].url


def test_parse_feed_enclosure_preferred_over_html_img():
    """Enclosure images take priority; HTML images are only a fallback."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
    <channel><title>Blog</title>
    <item>
        <title>Post</title>
        <link>https://example.com/1</link>
        <description>&lt;img src="https://example.com/inline.png"&gt;</description>
        <enclosure url="https://example.com/enclosure.jpg" type="image/jpeg" length="12345" />
        <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
    </item>
    </channel></rss>"""
    items = parse_feed(xml, feed_url="https://example.com/rss.xml").items
    assert len(items) == 1
    # Should have only the enclosure image, not the inline one
    assert len(items[0].images) == 1
    assert "enclosure.jpg" in items[0].images[0].url


# ---------------------------------------------------------------------------
# Item 2: _probe_common_feed_paths
# ---------------------------------------------------------------------------


def test_probe_common_feed_paths_finds_feed():
    """Finds a feed at /feed path."""

    def head_side_effect(url, **kwargs):
        resp = MagicMock()
        if url.endswith("/feed"):
            resp.status_code = 200
            resp.headers = {"content-type": "application/rss+xml"}
        else:
            resp.status_code = 404
            resp.headers = {"content-type": "text/html"}
        return resp

    with patch("offscroll.ingestion.feeds.httpx.head", side_effect=head_side_effect):
        result = _probe_common_feed_paths("https://example.com")
    assert result == "https://example.com/feed"


def test_probe_common_feed_paths_finds_api_feed():
    """Finds a feed at api.{domain}/feed/ path."""

    def head_side_effect(url, **kwargs):
        resp = MagicMock()
        if "api.example.com/feed/" in url:
            resp.status_code = 200
            resp.headers = {"content-type": "application/atom+xml"}
        else:
            resp.status_code = 404
            resp.headers = {"content-type": "text/html"}
        return resp

    with patch("offscroll.ingestion.feeds.httpx.head", side_effect=head_side_effect):
        result = _probe_common_feed_paths("https://example.com")
    assert result == "https://api.example.com/feed/"


def test_probe_common_feed_paths_returns_none():
    """Returns None when no common paths respond with feed content-type."""

    def head_side_effect(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 404
        resp.headers = {"content-type": "text/html"}
        return resp

    with patch("offscroll.ingestion.feeds.httpx.head", side_effect=head_side_effect):
        result = _probe_common_feed_paths("https://example.com")
    assert result is None


def test_ingest_html_with_probe_fallback(rss_xml, tmp_path):
    """Blog URL without autodiscovery link falls back to path probing."""
    config = {
        "feeds": {"rss": [{"url": "https://myblog.com/"}]},
        "output": {"data_dir": str(tmp_path)},
    }

    html_response = MagicMock()
    html_response.text = HTML_NO_FEED_LINK
    html_response.headers = {"content-type": "text/html; charset=utf-8"}
    html_response.raise_for_status = MagicMock()

    feed_response = MagicMock()
    feed_response.text = rss_xml
    feed_response.headers = {"content-type": "application/rss+xml"}
    feed_response.raise_for_status = MagicMock()

    def get_side_effect(url, **kwargs):
        if url == "https://myblog.com/":
            return html_response
        if url == "https://myblog.com/feed":
            return feed_response
        raise httpx.HTTPError(f"Unexpected URL: {url}")

    def head_side_effect(url, **kwargs):
        resp = MagicMock()
        if url == "https://myblog.com/feed":
            resp.status_code = 200
            resp.headers = {"content-type": "application/rss+xml"}
        else:
            resp.status_code = 404
            resp.headers = {"content-type": "text/html"}
        return resp

    import httpx

    with (
        patch("offscroll.ingestion.feeds.httpx.get", side_effect=get_side_effect),
        patch("offscroll.ingestion.feeds.httpx.head", side_effect=head_side_effect),
    ):
        count = ingest_all_feeds(config)
    assert count == 3


# ---------------------------------------------------------------------------
# Item 6: _strip_boilerplate
# ---------------------------------------------------------------------------


def test_strip_boilerplate_removes_donation():
    """Removes paragraphs with donation text."""
    text = "Good article content.\n\nDonating = loving. Please support us."
    result = _strip_boilerplate(text)
    assert "article content" in result
    assert "Donating" not in result


def test_strip_boilerplate_removes_newsletter():
    """Removes paragraphs with newsletter signup text."""
    text = "Real content here.\n\nSign up for our free weekly newsletter!"
    result = _strip_boilerplate(text)
    assert "Real content" in result
    assert "newsletter" not in result


def test_strip_boilerplate_removes_share_block():
    """Removes 'share this on' paragraphs."""
    text = "Article text.\n\nShare this on Twitter and Facebook."
    result = _strip_boilerplate(text)
    assert "Article text" in result
    assert "Share this" not in result


def test_strip_boilerplate_preserves_content():
    """Normal article content is preserved."""
    text = "This is a normal article.\n\nWith multiple paragraphs.\n\nAbout interesting topics."
    result = _strip_boilerplate(text)
    assert result == text


def test_strip_boilerplate_handles_empty():
    """Empty string returns empty string."""
    assert _strip_boilerplate("") == ""


def test_strip_boilerplate_removes_filed_under():
    """Removes 'filed under:' paragraphs."""
    text = "Content.\n\nFiled under: politics, technology"
    result = _strip_boilerplate(text)
    assert "Content" in result
    assert "Filed under" not in result


def test_extract_plain_text_decodes_numeric_entities():
    """_extract_plain_text decodes numeric HTML entities."""
    from offscroll.ingestion.feeds import _extract_plain_text

    result = _extract_plain_text("Smart quotes: &#8220;hello&#8221;")
    assert "\u201c" in result  # left double quote
    assert "\u201d" in result  # right double quote


def test_extract_plain_text_decodes_named_entities():
    """_extract_plain_text decodes named HTML entities."""
    from offscroll.ingestion.feeds import _extract_plain_text

    result = _extract_plain_text("&amp; &lt; &gt; &mdash;")
    assert result == "& < > \u2014"


def test_parse_feed_decodes_entities():
    """Integration: parse_feed decodes HTML entities in content text."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
    <channel><title>Blog</title>
    <item>
        <title>Post</title>
        <link>https://example.com/1</link>
        <description>He said &amp;ldquo;hello&amp;rdquo; &amp;mdash; and left.</description>
        <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
    </item>
    </channel></rss>"""
    items = parse_feed(xml, feed_url="https://example.com/rss.xml").items
    assert len(items) == 1
    # Double-encoded entities should be fully decoded
    assert "&ldquo;" not in items[0].content_text
    assert "&rdquo;" not in items[0].content_text
    assert "&mdash;" not in items[0].content_text


def test_parse_feed_strips_boilerplate():
    """Integration: parse_feed strips boilerplate from content text."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
    <channel><title>Blog</title>
    <item>
        <title>Post</title>
        <link>https://example.com/1</link>
        <description>Real article content.

Donating = loving. Please support us.</description>
        <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
    </item>
    </channel></rss>"""
    items = parse_feed(xml, feed_url="https://example.com/rss.xml").items
    assert len(items) == 1
    assert "Real article content" in items[0].content_text
    assert "Donating" not in items[0].content_text


# ---------------------------------------------------------------------------
# Feed title extraction (source attribution pipeline)
# ---------------------------------------------------------------------------


def test_parse_feed_extracts_rss_title(rss_xml):
    """parse_feed returns feed-level title from RSS channel."""
    result = parse_feed(rss_xml, feed_url="https://example.com/feed.xml")
    assert result.feed_title == "Test Blog"


def test_parse_feed_extracts_atom_title(atom_xml):
    """parse_feed returns feed-level title from Atom feed."""
    result = parse_feed(atom_xml, feed_url="https://example.com/atom.xml")
    assert result.feed_title == "Test Atom Feed"


def test_parse_feed_empty_title():
    """parse_feed returns empty string when feed has no title."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
    <channel>
    <item>
        <title>Post</title>
        <link>https://example.com/1</link>
        <description>Content.</description>
        <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
    </item>
    </channel></rss>"""
    result = parse_feed(xml, feed_url="https://example.com/rss.xml")
    assert result.feed_title == ""


def test_ingest_stores_feed_title(rss_xml, tmp_path):
    """ingest_all_feeds stores the feed title in feed_sources."""
    import sqlite3

    config = {
        "feeds": {"rss": [{"url": "https://example.com/feed.xml"}]},
        "output": {"data_dir": str(tmp_path)},
    }
    mock_response = MagicMock()
    mock_response.text = rss_xml
    mock_response.raise_for_status = MagicMock()
    with patch("offscroll.ingestion.feeds.httpx.get", return_value=mock_response):
        ingest_all_feeds(config)

    db_path = tmp_path / "offscroll.db"
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT name FROM feed_sources WHERE url = ?",
        ("https://example.com/feed.xml",),
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "Test Blog"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_strip_boilerplate_marginalian_donation():
    """12.3: Marginalian donation text is stripped."""
    from offscroll.ingestion.feeds import _strip_boilerplate

    article = (
        "Annie Dillard wrote about the importance of paying attention.\n\n"
        "She observed nature with a poet's eye.\n\n"
        "donating = loving\n\n"
        "Share this to spread the word."
    )
    result = _strip_boilerplate(article)
    assert "donating" not in result
    assert "Annie Dillard" in result
    assert "poet's eye" in result


def test_strip_boilerplate_truncation():
    """12.3: Truncation at first boilerplate match removes all trailing text."""
    from offscroll.ingestion.feeds import _strip_boilerplate

    article = (
        "This is the real article content.\n\n"
        "More important ideas here.\n\n"
        "If this labor makes something worthwhile, consider donating.\n\n"
        "You can also join by email."
    )
    result = _strip_boilerplate(article)
    assert "real article content" in result
    assert "More important ideas" in result
    # Everything from "If this labor" onward should be gone
    assert "labor makes" not in result
    assert "join by email" not in result


def test_strip_boilerplate_marginalian_complement():
    """12.3: 'complement ... labors' pattern is stripped."""
    from offscroll.ingestion.feeds import _strip_boilerplate

    article = (
        "A thoughtful essay on living.\n\n"
        "complement these labors of love by becoming a patron"
    )
    result = _strip_boilerplate(article)
    assert "thoughtful essay" in result
    assert "complement" not in result


def test_strip_boilerplate_preserves_content_sprint12():
    """12.3: Legitimate article content is not removed."""
    from offscroll.ingestion.feeds import _strip_boilerplate

    article = (
        "The Framework laptop is a revolution in repairability.\n\n"
        "Users can swap out components in minutes.\n\n"
        "The 16-inch model features a powerful GPU."
    )
    result = _strip_boilerplate(article)
    assert result == article


def test_strip_boilerplate_new_patterns():
    """12.3: New boilerplate patterns are detected."""
    from offscroll.ingestion.feeds import _strip_boilerplate

    # "the post X first appeared on Y"
    text = "Great content.\n\nThe post My Article first appeared on Some Blog."
    assert "first appeared" not in _strip_boilerplate(text)

    # "for more ... subscribe"
    text2 = "Great content.\n\nFor more articles like this, subscribe to our newsletter."
    assert "subscribe" not in _strip_boilerplate(text2)


def test_extract_images_from_html_srcset():
    """12.7: _extract_images_from_html picks best srcset candidate."""
    from offscroll.ingestion.feeds import _extract_images_from_html

    html = '''
    <img src="small.jpg" srcset="medium.jpg 800w, large.jpg 1200w" alt="Test">
    '''
    images = _extract_images_from_html(html)
    assert len(images) == 1
    # Should pick the largest srcset candidate (1200w)
    assert images[0].url == "large.jpg"
    assert images[0].alt_text == "Test"


def test_extract_images_from_html_deduplicates():
    """12.7: _extract_images_from_html de-duplicates by URL."""
    from offscroll.ingestion.feeds import _extract_images_from_html

    html = '''
    <img src="same.jpg" alt="First">
    <img src="same.jpg" alt="Duplicate">
    <img src="different.jpg" alt="Other">
    '''
    images = _extract_images_from_html(html)
    assert len(images) == 2
    urls = [img.url for img in images]
    assert "same.jpg" in urls
    assert "different.jpg" in urls
