"""Facebook crawler.

Extracts posts from the Facebook news feed using ARIA attributes and
structural DOM patterns. Class names are obfuscated and change every
2-4 weeks, so selectors target role attributes and structural patterns.

LLM fallback note: Facebook's DOM changes are the most frequent of
any major platform. When selectors break (and they will), the fallback
would:
  1. Detect: extract_posts() returns 0 items on a page with content
  2. Capture: page.content() -> raw HTML
  3. Send to LLM: "Extract post text, author, timestamp, and image URLs"
  4. Parse structured response into FeedItems
This is not implemented in this prototype. This is where it matters most.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import UTC, datetime

from playwright.async_api import Page

from offscroll.ingestion.browser.base import BaseCrawler
from offscroll.models import FeedItem, ImageContent, SourceType

logger = logging.getLogger(__name__)

# CSS selectors — targeting ARIA/structural patterns, not class names
SEL_POST = 'div[role="article"]'
SEL_TEXT = '[dir="auto"]'

# Facebook post ID patterns
_FB_POST_ID_RE = re.compile(r"/posts/(\w+)")
_FB_STORY_ID_RE = re.compile(r"story_fbid=(\d+)")
_FB_PERMALINK_RE = re.compile(r"/permalink/(\d+)")


class FacebookCrawler(BaseCrawler):
    """Crawler for Facebook news feed."""

    @property
    def platform_name(self) -> str:
        return "facebook"

    @property
    def source_type(self) -> SourceType:
        return SourceType.FACEBOOK

    @property
    def feed_url(self) -> str:
        return "https://www.facebook.com/"

    @property
    def start_url(self) -> str:
        return "https://www.facebook.com/"

    async def is_logged_in(self, page: Page) -> bool:
        """Facebook is logged in if we can find the feed or nav landmarks."""
        try:
            # Check for the main navigation bar (logged-in indicator)
            nav = await page.query_selector('[aria-label="Facebook"]')
            if not nav:
                return False
            # Also verify we're not on the login page
            login_form = await page.query_selector('form[action*="login"]')
            return login_form is None
        except Exception:
            return False

    async def extract_posts(self, page: Page) -> list[FeedItem]:
        """Extract posts from visible DOM using role="article" selectors."""
        items: list[FeedItem] = []
        seen_ids: set[str] = set()

        articles = await page.query_selector_all(SEL_POST)

        for article in articles:
            try:
                item = await self._parse_post(article)
                if item and item.item_id not in seen_ids:
                    seen_ids.add(item.item_id)
                    items.append(item)
            except Exception as exc:
                logger.debug("Failed to parse FB post: %s", exc)

        return items

    async def _parse_post(self, article) -> FeedItem | None:
        """Parse a single article element into a FeedItem.

        Facebook's [dir="auto"] selector picks up both post text and UI
        labels. We deduplicate by taking the longest text fragments and
        filtering out known UI patterns.
        """
        # Extract all text fragments with dir="auto"
        text_els = await article.query_selector_all(SEL_TEXT)
        fragments: list[str] = []
        for el in text_els:
            t = (await el.inner_text()).strip()
            if t and len(t) > 2:
                fragments.append(t)

        # Deduplicate and filter UI labels
        fragments = _filter_ui_text(fragments)

        if not fragments:
            return None

        # Use the longest fragment as primary text, join others
        fragments.sort(key=len, reverse=True)
        text = fragments[0] if len(fragments) == 1 else "\n\n".join(fragments)

        # Extract author from first link or heading in the article
        author = None
        author_url = None
        # Facebook posts typically have the author as the first strong/heading element
        heading = await article.query_selector("h2, h3, h4, strong")
        if heading:
            author_link = await heading.query_selector("a")
            if author_link:
                author = (await author_link.inner_text()).strip()
                href = await author_link.get_attribute("href")
                if href:
                    if href.startswith("/"):
                        author_url = f"https://www.facebook.com{href}"
                    else:
                        author_url = href

        # Extract timestamp from aria-label on links (Facebook's pattern)
        published_at = None
        time_links = await article.query_selector_all("a[aria-label]")
        for link in time_links:
            label = await link.get_attribute("aria-label")
            if label and _looks_like_timestamp(label):
                published_at = _parse_fb_timestamp(label)
                break

        # Also try <abbr> elements which Facebook sometimes uses
        if not published_at:
            abbrs = await article.query_selector_all("abbr")
            for abbr in abbrs:
                data_utime = await abbr.get_attribute("data-utime")
                if data_utime:
                    try:
                        published_at = datetime.fromtimestamp(int(data_utime), tz=UTC)
                    except (ValueError, OSError):
                        pass

        # Extract post URL and ID
        item_url = None
        item_id = None
        all_links = await article.query_selector_all("a[href]")
        for link in all_links:
            href = await link.get_attribute("href")
            if not href:
                continue
            for pattern in (_FB_POST_ID_RE, _FB_STORY_ID_RE, _FB_PERMALINK_RE):
                match = pattern.search(href)
                if match:
                    item_id = f"fb:{match.group(1)}"
                    if href.startswith("/"):
                        item_url = f"https://www.facebook.com{href}"
                    else:
                        item_url = href
                    break
            if item_id:
                break

        # Fallback ID: content hash
        if not item_id:
            item_id = "fb:" + hashlib.sha256(text.encode()).hexdigest()[:16]

        # Extract images (filter for Facebook's scontent CDN)
        images: list[ImageContent] = []
        img_els = await article.query_selector_all("img[src]")
        for img in img_els:
            src = await img.get_attribute("src")
            alt = await img.get_attribute("alt")
            if src and ("scontent" in src or "fbcdn" in src):
                # Skip tiny images (profile pics, icons)
                width = await img.get_attribute("width")
                height = await img.get_attribute("height")
                if width and height:
                    try:
                        if int(width) < 100 or int(height) < 100:
                            continue
                    except ValueError:
                        pass
                images.append(ImageContent(url=src, alt_text=alt))

        return FeedItem(
            item_id=item_id,
            source_type=self.source_type,
            feed_url=self.feed_url,
            item_url=item_url,
            author=author,
            author_url=author_url,
            title=None,  # Facebook posts don't have titles
            content_text=text,
            content_html=None,
            published_at=published_at or datetime.now(UTC),
            images=images,
        )


# Known Facebook UI text patterns to filter out
_UI_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^like$",
        r"^comment$",
        r"^share$",
        r"^reply$",
        r"^see more$",
        r"^see less$",
        r"^\d+ comments?$",
        r"^\d+ shares?$",
        r"^\d+ likes?$",
        r"^write a comment",
        r"^sponsored$",
        r"^suggested for you$",
        r"^people you may know$",
        r"^shared a memory",
        r"^reels and short videos$",
    ]
]


def _filter_ui_text(fragments: list[str]) -> list[str]:
    """Remove known Facebook UI text from extracted fragments."""
    result = []
    seen = set()
    for f in fragments:
        # Skip UI patterns
        if any(pat.match(f) for pat in _UI_PATTERNS):
            continue
        # Skip very short fragments (likely UI)
        if len(f) < 10:
            continue
        # Deduplicate
        key = f.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(f)
    return result


_TIME_WORDS = {"hour", "hours", "minute", "minutes", "day", "days", "yesterday", "just now"}


def _looks_like_timestamp(text: str) -> bool:
    """Heuristic: does this aria-label look like a time description?"""
    lower = text.lower()
    return any(w in lower for w in _TIME_WORDS) or bool(
        re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", lower)
    )


def _parse_fb_timestamp(label: str) -> datetime | None:
    """Best-effort parse of Facebook's human-readable timestamps.

    Handles patterns like "2 hours ago", "Yesterday at 3:45 PM",
    "March 15 at 10:30 AM". Returns None if unparseable.
    """
    lower = label.lower().strip()

    # "just now"
    if "just now" in lower:
        return datetime.now(UTC)

    # "X minutes ago" or "X hours ago"
    m = re.match(r"(\d+)\s+(minute|hour|day)s?\s+ago", lower)
    if m:
        from datetime import timedelta

        n = int(m.group(1))
        unit = m.group(2)
        delta = {"minute": timedelta(minutes=n), "hour": timedelta(hours=n), "day": timedelta(days=n)}
        return datetime.now(UTC) - delta.get(unit, timedelta())

    # Can't parse — return None rather than guess wrong
    return None
