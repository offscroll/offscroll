"""Abstract base class for walled-garden browser crawlers.

Handles browser lifecycle, persistent profile management, stealth
patches, and the login-wait flow. Platform-specific crawlers
subclass this and implement extract_posts() and login detection.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

from playwright.async_api import BrowserContext, Page, async_playwright

from offscroll.ingestion.browser import behavior
from offscroll.ingestion.store import init_db, store_item
from offscroll.models import FeedItem, SourceType

logger = logging.getLogger(__name__)

# JavaScript to patch navigator.webdriver and remove Playwright globals
_STEALTH_SCRIPT = """
() => {
    // Remove navigator.webdriver flag
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
    });

    // Remove Playwright-injected globals
    for (const key of Object.getOwnPropertyNames(window)) {
        if (key.startsWith('__playwright') || key.startsWith('__pw_')) {
            delete window[key];
        }
    }
}
"""

DEFAULT_PROFILE_BASE = Path.home() / ".offscroll" / "data" / "browser_profiles"


class BaseCrawler(ABC):
    """Base class for browser-based walled-garden crawlers.

    Subclasses must implement:
        platform_name: str property
        source_type: SourceType property
        feed_url: str property
        start_url: str property
        is_logged_in(page) -> bool
        extract_posts(page) -> list[FeedItem]
    """

    def __init__(self, config: dict, profile_dir: Path | None = None):
        self.config = config
        self.profile_dir = profile_dir or (DEFAULT_PROFILE_BASE / self.platform_name)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Short name for logging and profile directory."""
        ...

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """SourceType enum value for FeedItems."""
        ...

    @property
    @abstractmethod
    def feed_url(self) -> str:
        """Canonical feed URL for FeedItem.feed_url."""
        ...

    @property
    @abstractmethod
    def start_url(self) -> str:
        """URL to navigate to at session start."""
        ...

    @abstractmethod
    async def is_logged_in(self, page: Page) -> bool:
        """Check if the page shows a logged-in state."""
        ...

    @abstractmethod
    async def extract_posts(self, page: Page) -> list[FeedItem]:
        """Extract feed items from the current page state.

        LLM fallback insertion point: if this returns an empty list
        but the page clearly has content (e.g., page.content() length
        is substantial), a future LLM-based extractor could be called
        here as a fallback.
        """
        ...

    async def run(self) -> int:
        """Run a complete crawl session. Returns count of new items stored."""
        init_db(self.config)

        session_mins = behavior.session_duration_minutes()
        deadline = time.monotonic() + session_mins * 60
        logger.info(
            "%s: starting session (%.0f min)",
            self.platform_name,
            session_mins,
        )

        new_count = 0
        async with async_playwright() as pw:
            context = await pw.firefox.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=False,
                viewport={"width": 1280, "height": 900},
                locale="en-US",
                timezone_id="America/New_York",
                args=["--disable-blink-features=AutomationControlled"],
            )

            page = context.pages[0] if context.pages else await context.new_page()

            # Apply stealth patches
            await context.add_init_script(_STEALTH_SCRIPT)
            await page.add_init_script(_STEALTH_SCRIPT)

            try:
                # Navigate to platform
                await page.goto(self.start_url, wait_until="domcontentloaded", timeout=30000)
                await behavior.pause(2.0, 4.0)

                # Check login state; wait for manual login if needed
                if not await self.is_logged_in(page):
                    logged_in = await self._wait_for_login(page)
                    if not logged_in:
                        logger.error("%s: login timeout — aborting session", self.platform_name)
                        return 0

                logger.info("%s: logged in, beginning feed scroll", self.platform_name)

                # Scroll-and-extract loop
                scroll_count = 0
                while time.monotonic() < deadline:
                    # Extract visible posts
                    posts = await self.extract_posts(page)
                    for item in posts:
                        if store_item(self.config, item):
                            new_count += 1

                    if posts:
                        logger.info(
                            "%s: extracted %d posts (%d new so far)",
                            self.platform_name,
                            len(posts),
                            new_count,
                        )

                    # Dwell on content
                    await behavior.dwell()

                    # Scroll down
                    await behavior.human_scroll(page, "down")
                    scroll_count += 1

                    # Wait for new content to load
                    await behavior.pause(1.5, 3.5)

                    # Periodic check: are we still logged in?
                    if scroll_count % 10 == 0:
                        if not await self.is_logged_in(page):
                            logger.warning(
                                "%s: lost login state after %d scrolls",
                                self.platform_name,
                                scroll_count,
                            )
                            break

            finally:
                await context.close()

        logger.info(
            "%s: session complete — %d new items stored",
            self.platform_name,
            new_count,
        )
        return new_count

    async def _wait_for_login(self, page: Page, timeout_s: float = 300) -> bool:
        """Wait up to timeout_s for the user to manually log in.

        Checks is_logged_in() every 3 seconds.
        """
        logger.info(
            "%s: not logged in — please log in manually in the browser window "
            "(waiting up to %.0f minutes)",
            self.platform_name,
            timeout_s / 60,
        )
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            await asyncio.sleep(3.0)
            try:
                if await self.is_logged_in(page):
                    return True
            except Exception:
                pass  # page may be navigating
        return False
