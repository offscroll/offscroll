"""Abstract base class for browser-based walled-garden crawlers.

Handles browser lifecycle (launch, persistent profile, stealth patches),
session timing, and the crawl→extract→store loop. Platform-specific
subclasses implement login detection, feed scrolling, and post extraction.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

from playwright.async_api import BrowserContext, Page, async_playwright

from offscroll.ingestion.browser.behavior import (
    BehaviorConfig,
    dwell,
    human_scroll,
    inter_action_pause,
    session_duration_minutes,
)
from offscroll.ingestion.store import init_db, store_item
from offscroll.models import FeedItem, SourceType

logger = logging.getLogger(__name__)

# JavaScript to remove Playwright automation fingerprints.
# Firefox uses the Juggler protocol (not CDP), so we only need to:
# 1. Unset navigator.webdriver
# 2. Remove __playwright* globals
_STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {
    get: () => undefined,
});
// Remove Playwright-injected globals
for (const key of Object.getOwnPropertyNames(window)) {
    if (key.startsWith('__playwright') || key.startsWith('__pw_')) {
        delete window[key];
    }
}
"""

# Default profile base directory
_DEFAULT_PROFILE_BASE = Path.home() / ".offscroll" / "data" / "browser_profiles"


class BaseCrawler(ABC):
    """Base class for headed-Firefox browser crawlers.

    Subclasses must implement:
    - platform_name: str property
    - source_type: SourceType property
    - start_url: str property
    - is_logged_in(page) -> bool
    - extract_posts(page) -> list[FeedItem]
    """

    def __init__(
        self,
        config: dict,
        behavior_cfg: BehaviorConfig | None = None,
        profile_dir: Path | None = None,
        headless: bool = False,
    ) -> None:
        self.config = config
        self.behavior_cfg = behavior_cfg or BehaviorConfig()
        self.profile_dir = profile_dir or (_DEFAULT_PROFILE_BASE / self.platform_name)
        self.headless = headless

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Short name for this platform (used in profile dir, logs)."""

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """SourceType enum value for items from this platform."""

    @property
    @abstractmethod
    def start_url(self) -> str:
        """URL to navigate to at session start (the feed page)."""

    @abstractmethod
    async def is_logged_in(self, page: Page) -> bool:
        """Check if the page shows a logged-in state.

        Called after navigating to start_url. Should check for
        platform-specific indicators (e.g., feed column presence).
        """

    @abstractmethod
    async def extract_posts(self, page: Page) -> list[FeedItem]:
        """Extract visible posts from the current page state.

        Returns FeedItem objects ready for store_item(). Called
        after each scroll-and-dwell cycle. Must handle deduplication
        within a single session (same post may appear in DOM across
        multiple extraction passes).

        LLM FALLBACK HOOK: If this returns an empty list but the page
        clearly has content, that's the signal to invoke LLM-based
        extraction. The fallback is not built in this prototype —
        subclasses document where it would plug in via docstrings.
        """

    async def _launch_context(self, pw) -> BrowserContext:
        """Launch a persistent Firefox context with stealth patches."""
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        context = await pw.firefox.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
            # No extra args needed for Firefox — no CDP to patch
        )
        # Apply stealth patches to all new pages
        await context.add_init_script(_STEALTH_SCRIPT)
        return context

    async def _wait_for_login(self, page: Page, timeout_s: int = 300) -> bool:
        """Wait for the user to log in manually.

        Opens the platform URL and polls for login indicators.
        Returns True if login detected, False if timeout.
        """
        logger.info(
            "%s: Waiting for manual login (timeout %ds)...",
            self.platform_name,
            timeout_s,
        )
        start = time.monotonic()
        check_interval = 3.0  # seconds between login checks
        while time.monotonic() - start < timeout_s:
            if await self.is_logged_in(page):
                logger.info("%s: Login detected.", self.platform_name)
                return True
            await asyncio.sleep(check_interval)
        logger.warning(
            "%s: Login timeout after %ds. Aborting session.",
            self.platform_name,
            timeout_s,
        )
        return False

    async def crawl(self) -> int:
        """Run a single crawl session. Returns count of new items stored.

        Flow:
        1. Launch persistent Firefox context
        2. Navigate to start_url
        3. If not logged in, wait for manual login (5 min timeout)
        4. Scroll feed with behavioral simulation
        5. Extract posts after each dwell
        6. Store via store_item() (dedup via INSERT OR IGNORE)
        7. Stop when session duration expires
        """
        init_db(self.config)
        new_items = 0
        seen_ids: set[str] = set()
        session_minutes = session_duration_minutes(self.behavior_cfg)
        session_seconds = session_minutes * 60

        logger.info(
            "%s: Starting crawl session (%.0f min)",
            self.platform_name,
            session_minutes,
        )

        async with async_playwright() as pw:
            context = await self._launch_context(pw)
            page = context.pages[0] if context.pages else await context.new_page()

            try:
                await page.goto(self.start_url, wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(3)  # Let JS hydrate

                if not await self.is_logged_in(page):
                    if not await self._wait_for_login(page):
                        return 0

                session_start = time.monotonic()

                while time.monotonic() - session_start < session_seconds:
                    # Dwell (read current viewport)
                    await dwell(self.behavior_cfg)

                    # Extract posts
                    posts = await self.extract_posts(page)
                    for item in posts:
                        if item.item_id in seen_ids:
                            continue
                        seen_ids.add(item.item_id)
                        if store_item(self.config, item):
                            new_items += 1
                            logger.debug(
                                "%s: Stored item %s",
                                self.platform_name,
                                item.item_id,
                            )

                    # Scroll down
                    await human_scroll(page, "down", self.behavior_cfg)
                    await inter_action_pause(self.behavior_cfg)

                    elapsed = time.monotonic() - session_start
                    logger.debug(
                        "%s: %.0f/%.0fs elapsed, %d new items",
                        self.platform_name,
                        elapsed,
                        session_seconds,
                        new_items,
                    )

            except Exception:
                logger.exception("%s: Error during crawl session", self.platform_name)
            finally:
                await context.close()

        logger.info(
            "%s: Session complete. %d new items stored.",
            self.platform_name,
            new_items,
        )
        return new_items
