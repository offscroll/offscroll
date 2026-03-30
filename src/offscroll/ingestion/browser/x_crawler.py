"""X (Twitter) crawler.

Extracts posts from the X home feed using data-testid CSS selectors.
X leaves React testing IDs in production; these have been stable for
2+ years and are the most reliable extraction strategy.

Selector table:
    Tweet container:  article[data-testid="tweet"]
    Tweet text:       [data-testid="tweetText"]
    Author info:      [data-testid="User-Name"]
    Timestamp:        time[datetime]
    Photos:           [data-testid="tweetPhoto"] img
    Login check:      [data-testid="primaryColumn"]

Expected selector lifetime: months. data-testid attributes are part
of X's testing infrastructure and change rarely.

LLM FALLBACK: If extract_posts() returns 0 items on a page that has
visible tweets, that's the signal that selectors have broken. The
fallback would:
1. Capture page.content() → raw HTML
2. Send to LLM: "Extract tweet text, author, timestamp, and URLs"
3. Parse structured response into FeedItem objects
Not built in this prototype. See extract_posts() docstring.
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

# Regex to extract tweet ID from permalink URL
_TWEET_ID_RE = re.compile(r"/status/(\d+)")


class XCrawler(BaseCrawler):
    """Crawler for X (Twitter) home feed."""

    @property
    def platform_name(self) -> str:
        return "x"

    @property
    def source_type(self) -> SourceType:
        return SourceType.TWITTER

    @property
    def start_url(self) -> str:
        return "https://x.com/home"

    async def is_logged_in(self, page: Page) -> bool:
        """Check for the primary column (only visible when logged in)."""
        try:
            el = await page.query_selector('[data-testid="primaryColumn"]')
            return el is not None
        except Exception:
            return False

    async def extract_posts(self, page: Page) -> list[FeedItem]:
        """Extract tweets from the current DOM state.

        Uses data-testid selectors. Returns FeedItem objects with
        item_id derived from the tweet's permalink URL (/status/ID).
        Falls back to SHA-256 content hash when permalink is unavailable.

        LLM FALLBACK HOOK: If this method returns an empty list, the
        caller should check whether the page actually has content
        (e.g., page.query_selector('article') returns results). If so,
        selectors have broken and LLM extraction should be triggered:
            html = await page.content()
            items = llm_extract_tweets(html)  # Not yet implemented
        """
        items: list[FeedItem] = []

        tweets = await page.query_selector_all('article[data-testid="tweet"]')
        for tweet in tweets:
            try:
                item = await self._parse_tweet(tweet, page)
                if item:
                    items.append(item)
            except Exception:
                logger.debug("Failed to parse a tweet element", exc_info=True)
                continue

        return items

    async def _parse_tweet(self, tweet, page: Page) -> FeedItem | None:
        """Parse a single tweet article element into a FeedItem."""

        # --- Text content ---
        text_el = await tweet.query_selector('[data-testid="tweetText"]')
        content_text = ""
        content_html = None
        if text_el:
            content_text = (await text_el.inner_text()).strip()
            content_html = await text_el.inner_html()

        if not content_text:
            # Skip empty tweets (e.g., promoted content with no text)
            return None

        # --- Author ---
        author = "Unknown"
        author_url = None
        username_el = await tweet.query_selector('[data-testid="User-Name"]')
        if username_el:
            # The User-Name container has display name + @handle
            spans = await username_el.query_selector_all("span")
            if spans:
                author = (await spans[0].inner_text()).strip()
            # Extract @handle link
            links = await username_el.query_selector_all("a")
            for link in links:
                href = await link.get_attribute("href")
                if href and href.startswith("/") and "/status/" not in href:
                    author_url = f"https://x.com{href}"
                    break

        # --- Timestamp ---
        published_at = None
        time_el = await tweet.query_selector("time[datetime]")
        if time_el:
            dt_str = await time_el.get_attribute("datetime")
            if dt_str:
                try:
                    published_at = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

        # --- Permalink / Item ID ---
        item_url = None
        item_id = None
        # Look for the timestamp's parent <a> which links to the tweet
        if time_el:
            parent_link = await time_el.evaluate("el => el.closest('a')?.href")
            if parent_link:
                item_url = parent_link
                match = _TWEET_ID_RE.search(parent_link)
                if match:
                    item_id = f"x:{match.group(1)}"

        # Fallback ID: content hash
        if not item_id:
            content_hash = hashlib.sha256(content_text.encode()).hexdigest()[:16]
            item_id = f"x:hash:{content_hash}"

        # --- Images ---
        images: list[ImageContent] = []
        photo_els = await tweet.query_selector_all('[data-testid="tweetPhoto"] img')
        for img in photo_els:
            src = await img.get_attribute("src")
            alt = await img.get_attribute("alt")
            if src and "pbs.twimg.com" in src:
                images.append(ImageContent(url=src, alt_text=alt))

        return FeedItem(
            item_id=item_id,
            source_type=SourceType.TWITTER,
            feed_url="https://x.com/home",
            item_url=item_url,
            author=author,
            author_url=author_url,
            title=None,  # Tweets don't have titles
            content_text=content_text,
            content_html=content_html,
            published_at=published_at,
            images=images,
        )
