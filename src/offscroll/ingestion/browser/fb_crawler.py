"""Facebook crawler.

Extracts posts from the Facebook news feed using structural/ARIA
selectors. Facebook has no data-testid attributes in production;
class names are obfuscated and change every 2-4 weeks. The selectors
here target ARIA roles and structural patterns that are more stable.

Selector table:
    Post container:  div[role="article"]
    Post text:       [dir="auto"] (filtered — picks up UI labels too)
    Images:          img[src] (filtered for scontent CDN)
    Timestamp:       a[aria-label] with time heuristic
    Login check:     [aria-label="Facebook"] + feed structure

Expected selector lifetime: 2-4 weeks. Facebook's DOM is the most
volatile of any major platform. The text extraction via [dir="auto"]
will pick up UI labels alongside post text — deduplication logic
filters known boilerplate but is imperfect.

LLM FALLBACK: This is where it matters most. When selectors break
(and they will every few weeks), the fallback would:
1. Detect: extract_posts() returns 0 items on a page with content
2. Capture: page.content() → raw HTML
3. Send to LLM: "Extract post text, author, timestamp, and URLs
   from this Facebook feed HTML"
4. Parse structured response into FeedItem objects
Not built in this prototype. The extraction function documents
exactly where the fallback plugs in.
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

# Known UI labels that [dir="auto"] picks up alongside post text.
# These get filtered out during extraction.
_UI_LABEL_PATTERNS = {
    "Like",
    "Comment",
    "Share",
    "Send",
    "Reply",
    "View more comments",
    "Most relevant",
    "All comments",
    "Write a comment",
    "Write a public comment",
    "Newest",
    "See more",
    "See less",
}

# Regex to extract post ID from Facebook permalink URLs
_POST_ID_RE = re.compile(r"/posts/(\w+)")
_STORY_ID_RE = re.compile(r"story_fbid=(\d+)")

# Time-like patterns in aria-label for timestamp detection
_TIME_LABEL_RE = re.compile(
    r"\b(\d+)\s*(hour|minute|second|day|week|month|year)s?\s*ago\b"
    r"|"
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}"
    r"|"
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|"
    r"\bYesterday\b"
    r"|"
    r"\b\d{1,2}:\d{2}\s*(AM|PM)\b",
    re.IGNORECASE,
)


class FacebookCrawler(BaseCrawler):
    """Crawler for Facebook news feed."""

    @property
    def platform_name(self) -> str:
        return "facebook"

    @property
    def source_type(self) -> SourceType:
        return SourceType.FACEBOOK

    @property
    def start_url(self) -> str:
        return "https://www.facebook.com/"

    async def is_logged_in(self, page: Page) -> bool:
        """Check for logged-in state via navigation bar + feed presence."""
        try:
            # Facebook shows an aria-label="Facebook" nav element when logged in,
            # plus feed content (role="feed" or role="main" with articles).
            nav = await page.query_selector('[aria-label="Facebook"]')
            if not nav:
                return False
            # Also check for at least one article (feed content)
            article = await page.query_selector('div[role="article"]')
            return article is not None
        except Exception:
            return False

    async def extract_posts(self, page: Page) -> list[FeedItem]:
        """Extract posts from visible Facebook feed articles.

        Uses role="article" containers and structural heuristics.
        Text extraction via [dir="auto"] is noisy — UI labels get
        mixed in and are filtered by known-pattern matching.

        LLM FALLBACK HOOK: If this method returns an empty list,
        check whether the page has articles:
            articles = await page.query_selector_all('div[role="article"]')
            if articles:
                # Selectors broke — trigger LLM extraction
                html = await page.content()
                items = llm_extract_fb_posts(html)  # Not yet implemented
        """
        items: list[FeedItem] = []

        articles = await page.query_selector_all('div[role="article"]')
        for article in articles:
            try:
                item = await self._parse_article(article, page)
                if item:
                    items.append(item)
            except Exception:
                logger.debug("Failed to parse a Facebook article", exc_info=True)
                continue

        return items

    async def _parse_article(self, article, page: Page) -> FeedItem | None:
        """Parse a single Facebook article element into a FeedItem."""

        # --- Text content ---
        # [dir="auto"] captures all auto-direction text blocks, which
        # includes post content but also UI labels, comments, etc.
        text_els = await article.query_selector_all('[dir="auto"]')
        text_fragments: list[str] = []
        for el in text_els:
            text = (await el.inner_text()).strip()
            if not text:
                continue
            # Filter out known UI labels
            if text in _UI_LABEL_PATTERNS:
                continue
            # Filter very short fragments likely to be UI (1-2 words, no punctuation)
            if len(text.split()) <= 2 and not any(c in text for c in ".!?,:;"):
                continue
            text_fragments.append(text)

        # Deduplicate: adjacent duplicates from nested [dir="auto"] elements
        deduped: list[str] = []
        for frag in text_fragments:
            if not deduped or frag != deduped[-1]:
                deduped.append(frag)

        content_text = "\n\n".join(deduped)
        if not content_text or len(content_text) < 10:
            return None

        # --- Author ---
        # First link with strong text in the article is typically the author.
        # Facebook's author name is usually the first <a> with a profile link.
        author = "Unknown"
        author_url = None
        heading = await article.query_selector("h2, h3, h4, [role='heading']")
        if heading:
            links = await heading.query_selector_all("a")
            if links:
                author = (await links[0].inner_text()).strip()
                href = await links[0].get_attribute("href")
                if href:
                    if href.startswith("/"):
                        author_url = f"https://www.facebook.com{href}"
                    elif href.startswith("http"):
                        author_url = href
        if author == "Unknown":
            # Fallback: first <a> in the article with substantial text
            links = await article.query_selector_all("a")
            for link in links:
                text = (await link.inner_text()).strip()
                if text and len(text) > 2 and text not in _UI_LABEL_PATTERNS:
                    author = text
                    href = await link.get_attribute("href")
                    if href and href.startswith("/"):
                        author_url = f"https://www.facebook.com{href}"
                    elif href and href.startswith("http"):
                        author_url = href
                    break

        # --- Timestamp ---
        # Facebook timestamps are in aria-label attributes of <a> elements.
        # We look for links whose aria-label matches time patterns.
        published_at = None
        timestamp_text = None
        links = await article.query_selector_all("a[aria-label]")
        for link in links:
            label = await link.get_attribute("aria-label")
            if label and _TIME_LABEL_RE.search(label):
                timestamp_text = label
                break

        # We store the raw label but don't parse it into a datetime here —
        # Facebook's relative timestamps ("2 hours ago") need the current
        # time, and their absolute formats vary by locale. For the prototype,
        # we use ingestion time as published_at and note the label.
        if timestamp_text:
            # Attempt basic "X hours/minutes ago" parsing
            published_at = _parse_relative_time(timestamp_text)

        # --- Permalink / Item ID ---
        item_url = None
        item_id = None
        # Look for permalink-like links (containing /posts/ or story_fbid)
        all_links = await article.query_selector_all("a[href]")
        for link in all_links:
            href = await link.get_attribute("href")
            if not href:
                continue
            post_match = _POST_ID_RE.search(href)
            story_match = _STORY_ID_RE.search(href)
            if post_match:
                item_id = f"fb:{post_match.group(1)}"
                item_url = href if href.startswith("http") else f"https://www.facebook.com{href}"
                break
            if story_match:
                item_id = f"fb:{story_match.group(1)}"
                item_url = href if href.startswith("http") else f"https://www.facebook.com{href}"
                break

        # Fallback ID: content hash
        if not item_id:
            content_hash = hashlib.sha256(content_text.encode()).hexdigest()[:16]
            item_id = f"fb:hash:{content_hash}"

        # --- Images ---
        images: list[ImageContent] = []
        img_els = await article.query_selector_all("img[src]")
        for img in img_els:
            src = await img.get_attribute("src")
            alt = await img.get_attribute("alt")
            if src and "scontent" in src:
                # Facebook CDN images (scontent-*.xx.fbcdn.net)
                images.append(ImageContent(url=src, alt_text=alt))

        return FeedItem(
            item_id=item_id,
            source_type=SourceType.FACEBOOK,
            feed_url="https://www.facebook.com/",
            item_url=item_url,
            author=author,
            author_url=author_url,
            title=None,  # Facebook posts don't have titles
            content_text=content_text,
            content_html=None,  # HTML extraction would require innerHTML of post body
            published_at=published_at,
            images=images,
        )


def _parse_relative_time(label: str) -> datetime | None:
    """Best-effort parsing of Facebook relative timestamps.

    Handles "X hours/minutes/days ago" patterns. Returns None for
    anything it can't parse — the ingestion timestamp (set by FeedItem
    default) serves as fallback.
    """
    match = re.search(r"(\d+)\s*(hour|minute|second|day|week|month|year)s?\s*ago", label, re.I)
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2).lower()
    now = datetime.now(UTC)

    from datetime import timedelta

    deltas = {
        "second": timedelta(seconds=value),
        "minute": timedelta(minutes=value),
        "hour": timedelta(hours=value),
        "day": timedelta(days=value),
        "week": timedelta(weeks=value),
        "month": timedelta(days=value * 30),  # Approximate
        "year": timedelta(days=value * 365),  # Approximate
    }
    delta = deltas.get(unit)
    if delta:
        return now - delta
    return None
