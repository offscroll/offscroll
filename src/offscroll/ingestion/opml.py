"""OPML import/export.

Imports feeds from OPML files.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from offscroll.ingestion.store import register_feed_source

logger = logging.getLogger(__name__)


def import_opml(opml_path: Path) -> list[dict]:
    """Parse an OPML file and return feed entries.

    Args:
        opml_path: Path to the OPML XML file.

    Returns:
        List of dicts with keys: url, name, source_type.
        source_type is always "rss" (OPML is RSS-only).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the XML cannot be parsed or
            contains no feed outlines.
    """
    if not opml_path.exists():
        raise FileNotFoundError(f"OPML file not found: {opml_path}")

    try:
        tree = ET.parse(str(opml_path))
        root = tree.getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse OPML XML: {exc}") from exc

    # Find all outline elements with xmlUrl attribute
    feeds = []
    seen_urls = set()

    for outline in root.iter("outline"):
        xml_url = outline.get("xmlUrl")
        if xml_url:
            # Skip duplicates
            if xml_url in seen_urls:
                continue
            seen_urls.add(xml_url)

            # Extract name (prefer title, fall back to text, then URL)
            name = outline.get("title") or outline.get("text") or xml_url

            feeds.append(
                {
                    "url": xml_url,
                    "name": name,
                    "source_type": "rss",
                }
            )

    if not feeds:
        raise ValueError("OPML file contains no feed outlines with xmlUrl")

    return feeds


def register_opml_feeds(
    config: dict,
    opml_path: Path,
) -> int:
    """Import feeds from OPML and register them.

    Parses the OPML file, then for each feed URL:
    1. Check if it already exists in feed_sources.
    2. If not, insert it into feed_sources with source_type="rss".

    Args:
        config: The OffScroll config dict.
        opml_path: Path to the OPML file.

    Returns:
        Count of new feeds registered.
    """
    feeds = import_opml(opml_path)

    count = 0
    for feed in feeds:
        if register_feed_source(
            config,
            url=feed["url"],
            source_type=feed["source_type"],
            name=feed.get("name"),
        ):
            count += 1
            logger.info("Registered feed: %s (%s)", feed.get("name", feed["url"]), feed["url"])
        else:
            logger.debug("Feed already exists: %s", feed["url"])

    logger.info("Imported %d new feeds from OPML", count)
    return count
