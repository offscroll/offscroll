"""X (Twitter) crawler.

Extracts tweets from the home timeline using data-testid CSS selectors.
These selectors target React testing attributes that X leaves in
production and have been stable for 2+ years.

LLM fallback note: if extract_posts() returns 0 items on a page with
substantial HTML content, the fallback would:
  1. Capture page.content()
  2. Send to LLM: "Extract tweet text, author, timestamp, and image URLs"
  3. Parse structured response into FeedItems
This is not implemented in this prototype.
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

# CSS selectors — X uses data-testid attributes (React testing IDs)
SEL_TWEET = 'article[data-testid="tweet"]'
SEL_TEXT = '[data-testid="tweetText"]'
SEL_USER = '[data-testid="User-Name"]'
SEL_PHOTO = '[data-testid="tweetPhoto"] img'
SEL_PRIMARY_COLUMN = '[data-testid="primaryColumn"]'

# Regex to extract tweet ID from permalink
_TWEET_ID_RE = re.compile(r"/status/(\d+)")


class XCrawler(BaseCrawler):
    """Crawler for X (Twitter) home timeline."""

    @property
    def platform_name(self) -> str:
        return "x"

    @property
    def source_type(self) -> SourceType:
        return SourceType.TWITTER

    @property
    def feed_url(self) -> str:
        return "https://x.com/home"

    @property
    def start_url(self) -> str:
        return "https://x.com/home"

    async def is_logged_in(self, page: Page) -> bool:
        """X is logged in if the primary column is present."""
        try:
            el = await page.query_selector(SEL_PRIMARY_COLUMN)
            return el is not None
        except Exception:
            return False

    async def extract_posts(self, page: Page) -> list[FeedItem]:
        """Extract tweets from visible DOM using data-testid selectors."""
        items: list[FeedItem] = []
        seen_ids: set[str] = set()

        tweets = await page.query_selector_all(SEL_TWEET)

        for tweet in tweets:
            try:
                item = await self._parse_tweet(tweet)
                if item and item.item_id not in seen_ids:
                    seen_ids.add(item.item_id)
                    items.append(item)
            except Exception as exc:
                logger.debug("Failed to parse tweet: %s", exc)

        return items

    async def _parse_tweet(self, tweet) -> FeedItem | None:
        """Parse a single tweet article element into a FeedItem."""
        # Extract text content
        text_el = await tweet.query_selector(SEL_TEXT)
        text = ""
        if text_el:
            text = (await text_el.inner_text()).strip()

        if not text:
            return None

        # Extract author
        author = None
        author_url = None
        user_el = await tweet.query_selector(SEL_USER)
        if user_el:
            # The User-Name element contains display name and @handle
            links = await user_el.query_selector_all("a")
            if links:
                href = await links[0].get_attribute("href")
                if href:
                    author_url = f"https://x.com{href}" if href.startswith("/") else href
                    # Handle is the path without /
                    author = href.lstrip("/")
                # Try to get display name from first link text
                display_name = (await links[0].inner_text()).strip()
                if display_name:
                    author = display_name

        # Extract timestamp
        published_at = None
        time_el = await tweet.query_selector("time")
        if time_el:
            dt_str = await time_el.get_attribute("datetime")
            if dt_str:
                try:
                    published_at = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

        # Extract tweet URL and ID
        item_url = None
        item_id = None
        # Look for the timestamp link which contains the permalink
        time_links = await tweet.query_selector_all("a[href*='/status/']")
        for link in time_links:
            href = await link.get_attribute("href")
            if href and "/status/" in href:
                item_url = f"https://x.com{href}" if href.startswith("/") else href
                match = _TWEET_ID_RE.search(href)
                if match:
                    item_id = f"x:{match.group(1)}"
                break

        # Fallback ID: content hash
        if not item_id:
            item_id = "x:" + hashlib.sha256(text.encode()).hexdigest()[:16]

        # Extract images
        images: list[ImageContent] = []
        photo_els = await tweet.query_selector_all(SEL_PHOTO)
        for img in photo_els:
            src = await img.get_attribute("src")
            alt = await img.get_attribute("alt")
            if src and "pbs.twimg.com" in src:
                images.append(ImageContent(url=src, alt_text=alt))

        return FeedItem(
            item_id=item_id,
            source_type=self.source_type,
            feed_url=self.feed_url,
            item_url=item_url,
            author=author,
            author_url=author_url,
            title=None,  # Tweets don't have titles
            content_text=text,
            content_html=None,
            published_at=published_at or datetime.now(UTC),
            images=images,
        )
