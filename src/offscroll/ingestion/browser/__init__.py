"""Browser-based crawlers for walled-garden platforms.

Requires the 'browser' optional dependency group:
    pip install -e ".[browser]"
    playwright install firefox
"""

from offscroll.ingestion.browser.fb_crawler import FacebookCrawler
from offscroll.ingestion.browser.x_crawler import XCrawler

__all__ = ["XCrawler", "FacebookCrawler"]
