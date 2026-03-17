"""RSS/Atom feed parsing and polling.

parse_feed() converts raw XML into FeedItem objects.
ingest_all_feeds() polls configured feeds via HTTP.
"""

from __future__ import annotations

import calendar
import collections.abc
import hashlib
import html as html_module
import logging
import re
from datetime import UTC, datetime
from typing import NamedTuple
from urllib.parse import urlparse

import feedparser
import httpx

from offscroll.ingestion.store import init_db, store_item
from offscroll.models import FeedItem, ImageContent, SourceType

logger = logging.getLogger(__name__)


class ParsedFeed(NamedTuple):
    """Result of parsing a feed: items plus feed-level metadata."""

    items: list[FeedItem]
    feed_title: str  # Feed-level <title> element (empty string if absent)


# --- Boilerplate patterns (compiled once at module level) ---
_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"donating\s*=\s*loving",
        r"has a free weekly newsletter",
        r"share this on\b",
        r"^subscribe\b",
        r"join the newsletter",
        r"filed under:",
        r"sign up for .*newsletter",
        r"follow us on\b",
        r"if you enjoyed this",
        r"support .* on patreon",
        r"become a patron",
        r"click here to\b",
        r"unsubscribe\b.*\bhere\b",
        r"if this labor makes",
        r"complement .* labors",
        r"published \w+ \d+",
        r"^the post .+ first appeared on\b",
        r"^this essay .+ first appeared",
        r"you can also .+ email",
        r"for more .+ subscribe",
    ]
]

#  Truncation patterns. When one of these is found *anywhere*
# in the text (not just at paragraph boundaries), everything from that
# match onward is discarded. Covers boilerplate that does not fall on
# paragraph boundaries.
_TRUNCATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"donating\s*=\s*loving",
        r"if this labor makes",
        r"complement .* labors",
        r"has a free weekly newsletter",
        r"you can also .+ email",
        r"for seventeen years,? I have been spending",
        r"I have been spending hundreds of hours",
        r"please consider a one-time donation",
        r"you are welcome to .+ on Patreon",
        r"Published \w+ \d{1,2},? \d{4}\b",
    ]
]


def _strip_boilerplate(text: str) -> str:
    """Remove common boilerplate paragraphs from article text.

     Two-pass approach.
    Pass 1: Split on double-newlines, discard paragraphs matching
    known boilerplate patterns.
    Pass 2: Truncate at the first truncation pattern match (catches
    boilerplate that does not fall on paragraph boundaries, e.g.
    The Marginalian's donation text).
    """
    if not text:
        return text

    # Pass 1: Per-paragraph filtering
    paragraphs = text.split("\n\n")
    kept = []
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        if any(pat.search(stripped) for pat in _BOILERPLATE_PATTERNS):
            continue
        kept.append(para)
    result = "\n\n".join(kept)

    # Pass 2: Truncation at first match
    earliest_pos = len(result)
    for pat in _TRUNCATION_PATTERNS:
        match = pat.search(result)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()

    if earliest_pos < len(result):
        result = result[:earliest_pos].rstrip()

    return result


def _extract_plain_text(html_str: str) -> str:
    """Convert HTML to plain text preserving structural whitespace.

     Block-level tags (<h1>-<h6>, <p>, <br>, <div>, <li>,
    <blockquote>) are converted to double-newline paragraph breaks
    before tag stripping. This prevents subheadings from being
    concatenated into body text .
    """
    text = html_str
    # Insert paragraph breaks before block-level opening tags
    text = re.sub(
        r"<(?:h[1-6]|p|div|blockquote|li|article|section|header|footer)[\s>]",
        r"\n\n",
        text,
        flags=re.IGNORECASE,
    )
    #  Insert paragraph breaks before closing block-level tags
    # too. Without this, text inside <h2>...</h2> concatenates with
    # the following text when tags are stripped (
    # "new.Streamlined schedulingAdding").
    text = re.sub(
        r"</(?:h[1-6]|p|div|blockquote|li|article|section|header|footer)>",
        r"\n\n",
        text,
        flags=re.IGNORECASE,
    )
    # Convert <br> tags to newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # : Insert a space when stripping remaining
    # HTML tags.  Without this, inline tags like <strong>, <em>, <a>
    # that wrap subheading-like content cause word concatenation when
    # stripped (e.g. "new:<strong>Include comments</strong>" becomes
    # "new:Include comments" instead of "new: Include comments").
    text = re.sub(r"<[^>]+>", " ", text).strip()
    # Collapse multiple spaces to single space (preserving newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse excessive newlines (3+ -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return html_module.unescape(text)


def _extract_author_name(raw_author: str) -> str:
    """Extract author name from various formats.

    RSS often uses 'email (Name)' format. Atom uses plain names.
    """
    # Match "email (Name)" pattern
    match = re.match(r"[^(]+\(([^)]+)\)", raw_author)
    if match:
        return match.group(1).strip()
    return raw_author.strip()


def _parse_published(entry) -> datetime:
    """Parse publication date from a feed entry.

    Returns datetime.now(UTC) if no date is available.
    """
    parsed = getattr(entry, "published_parsed", None)
    if parsed is None:
        parsed = getattr(entry, "updated_parsed", None)
    if parsed is None:
        return datetime.now(UTC)
    timestamp = calendar.timegm(parsed)
    return datetime.fromtimestamp(timestamp, tz=UTC)


def _generate_item_id(feed_url: str, entry) -> str:
    """Generate a stable item ID when the feed doesn't provide one."""
    title = getattr(entry, "title", "") or ""
    published = getattr(entry, "published", "") or ""
    raw = f"{feed_url}|{title}|{published}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _extract_images(entry) -> list[ImageContent]:
    """Extract image references from feed entry enclosures."""
    images = []
    for enclosure in getattr(entry, "enclosures", []):
        enc_type = getattr(enclosure, "type", "") or ""
        if enc_type.startswith("image/"):
            images.append(
                ImageContent(
                    url=enclosure.get("href", enclosure.get("url", "")),
                    alt_text=None,
                )
            )
    # Also check media_content
    for media in getattr(entry, "media_content", []):
        media_type = media.get("type", "") or ""
        if media_type.startswith("image/"):
            images.append(
                ImageContent(
                    url=media.get("url", ""),
                    alt_text=None,
                )
            )
    return images


def _extract_images_from_html(html_str: str) -> list[ImageContent]:
    """Extract image references from HTML content via <img> tags.

    Fallback for feeds that embed images in HTML content rather than
    using <enclosure> or <media:content> elements. Skips data URIs
    and tracking pixels (1x1 images, common tracker domains).

    (C1 part 2): Also extracts srcset first-candidate images
    and de-duplicates by URL to avoid redundant downloads.
    """
    if not html_str:
        return []

    img_pattern = re.compile(
        r'<img\b[^>]*\bsrc=["\']([^"\']+)["\'][^>]*>',
        re.IGNORECASE,
    )
    alt_pattern = re.compile(
        r'\balt=["\']([^"\']*)["\']',
        re.IGNORECASE,
    )
    srcset_pattern = re.compile(
        r'\bsrcset=["\']([^"\']+)["\']',
        re.IGNORECASE,
    )

    seen_urls: set[str] = set()
    images = []
    for match in img_pattern.finditer(html_str):
        src = match.group(1)
        tag = match.group(0)

        # Skip data URIs
        if src.startswith("data:"):
            continue

        # Skip common tracking pixels
        is_pixel = (
            'width="1"' in tag or "width='1'" in tag or 'height="1"' in tag or "height='1'" in tag
        )
        if is_pixel:
            continue

        # Try to get a higher-quality URL from srcset
        srcset_match = srcset_pattern.search(tag)
        if srcset_match:
            # srcset format: "url1 800w, url2 1200w, ..."
            # Pick the largest candidate
            candidates = srcset_match.group(1).split(",")
            best_url = src
            best_width = 0
            for candidate in candidates:
                parts = candidate.strip().split()
                if len(parts) >= 2:
                    c_url = parts[0]
                    descriptor = parts[1]
                    if descriptor.endswith("w"):
                        try:
                            w = int(descriptor[:-1])
                            if w > best_width:
                                best_width = w
                                best_url = c_url
                        except ValueError:
                            pass
            if best_width > 0:
                src = best_url

        # De-duplicate by URL
        if src in seen_urls:
            continue
        seen_urls.add(src)

        # Extract alt text
        alt_match = alt_pattern.search(tag)
        alt_text = alt_match.group(1) if alt_match else None

        images.append(ImageContent(url=src, alt_text=alt_text))

    return images


def _detect_source_type(feed) -> SourceType:
    """Auto-detect source type from parsed feed version."""
    version = getattr(feed, "version", "") or ""
    if version.startswith("atom"):
        return SourceType.ATOM
    return SourceType.RSS


def parse_feed(
    raw_xml: str,
    feed_url: str,
    source_type: SourceType | None = None,
) -> ParsedFeed:
    """Parse RSS/Atom XML into FeedItem objects.

    Args:
        raw_xml: The raw XML string (RSS 2.0 or Atom 1.0).
        feed_url: The URL this feed was fetched from.
        source_type: If None, auto-detect from the parsed feed
            (RSS -> SourceType.RSS, Atom -> SourceType.ATOM).

    Returns:
        A ParsedFeed namedtuple with ``items`` (list of FeedItem
        objects in publication order, oldest first) and
        ``feed_title`` (the feed-level title, empty string if absent).

    Raises:
        ValueError: If the XML cannot be parsed by feedparser or
            contains no entries.
    """
    feed = feedparser.parse(raw_xml)

    if feed.bozo and not feed.entries:
        raise ValueError(f"Failed to parse feed XML: {feed.bozo_exception}")

    if not feed.entries:
        raise ValueError("Feed contains no entries")

    detected_type = source_type or _detect_source_type(feed)

    # Extract feed-level title
    feed_title = (feed.feed.get("title", "") or "").strip()

    items = []
    for entry in feed.entries:
        # Extract item ID
        item_id = getattr(entry, "id", None) or getattr(entry, "link", None)
        if not item_id:
            item_id = _generate_item_id(feed_url, entry)

        # Extract title (may be None for some Atom entries)
        title = getattr(entry, "title", None)

        # Extract content text and HTML
        content_html = None
        content_text = ""

        # For Atom, prefer entry.content[0].value
        entry_content = getattr(entry, "content", None)
        if entry_content and len(entry_content) > 0:
            content_html = entry_content[0].get("value", "")
            content_text = _strip_boilerplate(_extract_plain_text(content_html))
        elif hasattr(entry, "summary"):
            raw_summary = entry.summary
            # Check if summary contains HTML
            if "<" in raw_summary:
                content_html = raw_summary
                content_text = _strip_boilerplate(_extract_plain_text(raw_summary))
            else:
                content_text = _strip_boilerplate(html_module.unescape(raw_summary.strip()))
        elif hasattr(entry, "description"):
            raw_desc = entry.description
            if "<" in raw_desc:
                content_html = raw_desc
                content_text = _strip_boilerplate(_extract_plain_text(raw_desc))
            else:
                content_text = _strip_boilerplate(html_module.unescape(raw_desc.strip()))

        # Extract author
        author = None
        author_detail = getattr(entry, "author_detail", None)
        if author_detail and hasattr(author_detail, "name"):
            author = author_detail.name
        elif hasattr(entry, "author") and entry.author:
            author = _extract_author_name(entry.author)

        # Extract item URL
        item_url = getattr(entry, "link", None)

        # Extract publication date
        published_at = _parse_published(entry)

        # Extract images from enclosures
        images = _extract_images(entry)

        # Fallback: extract images from HTML content if no enclosures found
        if not images and content_html:
            images = _extract_images_from_html(content_html)

        items.append(
            FeedItem(
                item_id=item_id,
                source_type=detected_type,
                feed_url=feed_url,
                item_url=item_url,
                author=author,
                title=title,
                content_text=content_text,
                content_html=content_html,
                published_at=published_at,
                images=images,
            )
        )

    # Sort by publication date (oldest first)
    items.sort(key=lambda item: item.published_at or datetime.min.replace(tzinfo=UTC))

    return ParsedFeed(items=items, feed_title=feed_title)


def _looks_like_html(content: str, content_type: str = "") -> bool:
    """Check if response content appears to be HTML rather than RSS/Atom XML."""
    if "text/html" in content_type:
        return True
    stripped = content.lstrip()
    if stripped[:15].lower().startswith("<!doctype"):
        return True
    return bool(stripped[:10].lower().startswith("<html"))


def _discover_feed_url(html: str, base_url: str) -> str | None:
    """Attempt RSS/Atom autodiscovery from HTML link tags.

    Looks for <link rel="alternate" type="application/rss+xml" ...> or
    <link rel="alternate" type="application/atom+xml" ...> in the HTML.

    Returns the discovered feed URL or None.
    """
    pattern = re.compile(
        r"<link\b[^>]*"
        r'type=["\']application/(?:rss|atom)\+xml["\']'
        r'[^>]*href=["\']([^"\']+)["\']'
        r"[^>]*/?>",
        re.IGNORECASE,
    )
    match = pattern.search(html)
    if not match:
        # Try the alternate order: href before type
        pattern2 = re.compile(
            r"<link\b[^>]*"
            r'href=["\']([^"\']+)["\']'
            r'[^>]*type=["\']application/(?:rss|atom)\+xml["\']'
            r"[^>]*/?>",
            re.IGNORECASE,
        )
        match = pattern2.search(html)
    if not match:
        return None

    feed_url = match.group(1)
    # Resolve relative URLs
    if feed_url.startswith("/"):
        # Extract scheme + host from base_url
        parts = base_url.split("/", 3)
        if len(parts) >= 3:
            feed_url = f"{parts[0]}//{parts[2]}{feed_url}"
    elif not feed_url.startswith("http"):
        feed_url = base_url.rstrip("/") + "/" + feed_url

    return feed_url


_FEED_CONTENT_TYPES = frozenset(
    [
        "application/rss+xml",
        "application/atom+xml",
        "application/xml",
        "text/xml",
    ]
)


def _probe_common_feed_paths(base_url: str) -> str | None:
    """Try common feed URL patterns when autodiscovery fails.

    Sends HEAD requests to common feed paths and returns the first
    one that responds with a feed-like Content-Type.
    """
    parsed = urlparse(base_url)
    bare_domain = parsed.hostname or ""
    scheme = parsed.scheme or "https"
    base = base_url.rstrip("/")

    candidates = [
        f"{scheme}://api.{bare_domain}/feed/",
        f"{base}/feed",
        f"{base}/feed/",
        f"{base}/rss",
        f"{base}/rss.xml",
        f"{base}/atom.xml",
        f"{base}/index.xml",
    ]

    for candidate in candidates:
        try:
            resp = httpx.head(candidate, timeout=10.0, follow_redirects=True)
            if resp.status_code < 400:
                ct = resp.headers.get("content-type", "")
                if any(feed_ct in ct for feed_ct in _FEED_CONTENT_TYPES):
                    logger.info("Probed feed URL: %s", candidate)
                    return candidate
        except (httpx.HTTPError, httpx.TimeoutException):
            continue

    return None


def ingest_all_feeds(config: dict) -> int:
    """Poll all configured RSS/Atom feeds and store new items.

    Reads feed URLs from config["feeds"]["rss"]. For each feed,
    fetches the XML via httpx, parses it with parse_feed(), and
    stores each item with store_item(). Skips items that already
    exist (store_item returns False for duplicates).

    Args:
        config: The OffScroll config dict.

    Returns:
        The count of new items ingested across all feeds.
    """
    init_db(config)

    feeds = config.get("feeds", {}).get("rss", [])
    if not feeds:
        logger.info("No RSS feeds configured")
        return 0

    new_count = 0
    for feed_conf in feeds:
        url = feed_conf["url"] if isinstance(feed_conf, collections.abc.Mapping) else feed_conf
        try:
            response = httpx.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.error("Failed to fetch feed %s: %s", url, exc)
            continue

        # Detect HTML pages and attempt RSS autodiscovery
        content_type = response.headers.get("content-type", "")
        if _looks_like_html(response.text, content_type):
            discovered = _discover_feed_url(response.text, url)
            if discovered:
                logger.info(
                    "URL %s is a web page; discovered feed URL: %s",
                    url,
                    discovered,
                )
                try:
                    response = httpx.get(discovered, timeout=30.0, follow_redirects=True)
                    response.raise_for_status()
                    url = discovered
                except (httpx.HTTPError, httpx.TimeoutException) as exc:
                    logger.error(
                        "Failed to fetch discovered feed %s: %s",
                        discovered,
                        exc,
                    )
                    continue
            else:
                # Try common feed paths as a last resort
                probed = _probe_common_feed_paths(url)
                if probed:
                    logger.info(
                        "URL %s is a web page; probed feed URL: %s",
                        url,
                        probed,
                    )
                    try:
                        response = httpx.get(probed, timeout=30.0, follow_redirects=True)
                        response.raise_for_status()
                        url = probed
                    except (httpx.HTTPError, httpx.TimeoutException) as exc:
                        logger.error(
                            "Failed to fetch probed feed %s: %s",
                            probed,
                            exc,
                        )
                        continue
                else:
                    logger.error(
                        "URL %s appears to be a web page, not an RSS feed. "
                        "Try looking for the site's RSS feed URL "
                        "(often /feed, /rss, or /atom.xml)",
                        url,
                    )
                    continue

        try:
            parsed = parse_feed(response.text, feed_url=url)
        except ValueError as exc:
            logger.error("Failed to parse feed %s: %s", url, exc)
            continue

        feed_new = 0
        new_items = []
        for item in parsed.items:
            if store_item(config, item, feed_name=parsed.feed_title or None):
                feed_new += 1
                new_items.append(item)
        new_count += feed_new
        logger.info("Feed %s: %d new items", url, feed_new)

        # Download images for new items if enabled
        if new_items and config.get("ingestion", {}).get("download_images", False):
            try:
                from offscroll.ingestion.images import download_images

                img_count = download_images(config, new_items)
                if img_count > 0:
                    from offscroll.ingestion.store import update_image_paths

                    for item in new_items:
                        if any(img.local_path for img in item.images):
                            update_image_paths(config, item.item_id, item.images)
                    logger.info("Feed %s: %d images downloaded", url, img_count)
            except Exception:
                logger.exception("Image download failed for feed %s", url)

    return new_count
