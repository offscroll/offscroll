"""Mastodon and Bluesky ingestion.

Mastodon ingestion uses mastodon.py.
Bluesky ingestion uses atproto SDK.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from datetime import datetime

from offscroll.ingestion.store import init_db, store_item
from offscroll.models import FeedItem, ImageContent, SourceType

logger = logging.getLogger(__name__)


def _extract_plain_text(html: str) -> str:
    """Convert HTML to plain text preserving structural whitespace.

     Matches feeds.py behaviour -- block-level tags are
    converted to paragraph breaks before stripping .
    """
    text = html
    # Insert paragraph breaks before block-level opening tags
    text = re.sub(
        r"<(?:h[1-6]|p|div|blockquote|li|article|section|header|footer)[\s>]",
        r"\n\n",
        text,
        flags=re.IGNORECASE,
    )
    # Insert paragraph breaks before closing block-level tags
    text = re.sub(
        r"</(?:h[1-6]|p|div|blockquote|li|article|section|header|footer)>",
        r"\n\n",
        text,
        flags=re.IGNORECASE,
    )
    # Convert <br> tags to newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text).strip()
    # Collapse excessive newlines (3+ -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _status_to_feed_item(
    status: dict,
    instance: str,
) -> FeedItem:
    """Convert a Mastodon status dict to FeedItem.

    Handles boosts (uses reblogged status), HTML stripping,
    thread detection, and image attachments.

    Args:
        status: A Mastodon status dict from the API.
        instance: The Mastodon instance URL (base URL).

    Returns:
        A FeedItem instance.
    """
    # Handle boosts: use reblogged status for content
    if status.get("reblog") is not None:
        reblog = status["reblog"]
        original_author = reblog["account"]["display_name"] or f"@{reblog['account']['acct']}"
        # Optionally prefix with boost context
        content_html = reblog["content"]
        content_text = _extract_plain_text(content_html)
        # Add boost context prefix (optional, per spec)
        booster = status["account"]["display_name"] or f"@{status['account']['acct']}"
        content_text = f"[Boosted by {booster}]\n\n{content_text}"
        author_url = reblog["account"]["url"]
        original_status = reblog
    else:
        original_author = status["account"]["display_name"] or f"@{status['account']['acct']}"
        content_html = status["content"]
        content_text = _extract_plain_text(content_html)
        author_url = status["account"]["url"]
        original_status = status

    # Extract images from media_attachments
    images = []
    for media in original_status.get("media_attachments", []):
        if media.get("type") == "image":
            images.append(
                ImageContent(
                    url=media.get("url", ""),
                    alt_text=media.get("description"),
                )
            )

    # Thread detection
    is_thread = original_status.get("in_reply_to_id") is not None
    thread_id = str(original_status["in_reply_to_id"]) if is_thread else None

    return FeedItem(
        item_id=str(original_status["id"]),
        source_type=SourceType.MASTODON,
        feed_url=instance,
        item_url=original_status.get("url"),
        author=original_author,
        author_url=author_url,
        title=None,  # Mastodon posts don't have titles
        content_text=content_text,
        content_html=content_html,
        published_at=original_status.get("created_at"),
        images=images,
        is_thread=is_thread,
        thread_id=thread_id,
    )


def ingest_mastodon(config: dict) -> int:
    """Poll configured Mastodon timelines and store new items.

    Reads Mastodon config from config["feeds"]["mastodon"]. Each entry has:
      - instance: str (e.g. "https://mastodon.social")
      - access_token_env: str (env var name)
      - timeline: str ("home", "public", or "list:<list_id>")

    Auth: OAuth access token read from the environment variable
    named in access_token_env.

    Args:
        config: The OffScroll config dict.

    Returns:
        Count of new items ingested.
    """
    mastodon_feeds = config.get("feeds", {}).get("mastodon", [])
    if not mastodon_feeds:
        logger.info("No Mastodon feeds configured")
        return 0

    try:
        from mastodon import Mastodon
    except ImportError as exc:
        raise ImportError(
            "Mastodon.py not installed. Install: pip install 'offscroll[fediverse]'"
        ) from exc

    init_db(config)

    total_new = 0

    for feed_conf in mastodon_feeds:
        instance = feed_conf["instance"]
        access_token_env = feed_conf["access_token_env"]
        timeline_type = feed_conf.get("timeline", "home")

        # Read token from environment
        try:
            access_token = os.environ[access_token_env]
        except KeyError:
            logger.error(
                "Environment variable %s not set for Mastodon instance %s",
                access_token_env,
                instance,
            )
            continue

        # Create API client
        try:
            api = Mastodon(
                access_token=access_token,
                api_base_url=instance,
            )
        except Exception as exc:
            logger.error("Failed to authenticate to Mastodon %s: %s", instance, exc)
            continue

        # Fetch timeline
        try:
            if timeline_type == "home":
                statuses = api.timeline_home(limit=40)
            elif timeline_type == "public":
                statuses = api.timeline_public(limit=40)
            elif timeline_type.startswith("list:"):
                list_id = timeline_type.split(":", 1)[1]
                statuses = api.timeline_list(list_id, limit=40)
            else:
                logger.warning("Unknown timeline type: %s", timeline_type)
                continue
        except Exception as exc:
            logger.error("Failed to fetch timeline from %s: %s", instance, exc)
            continue

        # Convert and store statuses
        feed_new = 0
        for status in statuses:
            # Skip deleted or incomplete statuses
            if not status.get("content"):
                continue

            try:
                item = _status_to_feed_item(status, instance)
                if store_item(config, item):
                    feed_new += 1
            except Exception as exc:
                logger.warning("Failed to convert status %s: %s", status.get("id"), exc)
                continue

        total_new += feed_new
        logger.info("Mastodon %s: %d new items", instance, feed_new)

    return total_new


def _bsky_post_to_feed_item(
    feed_view: dict,
) -> FeedItem:
    """Convert a Bluesky feed view to FeedItem.

    Handles quote posts, thread detection, and image attachments.

    Args:
        feed_view: A feed view dict from the atproto SDK.

    Returns:
        A FeedItem instance.
    """
    post = feed_view.get("post", {})
    record = post.get("record", {})
    author = post.get("author", {})

    # Extract author display name or handle
    author_name = author.get("display_name") or author.get("handle", "unknown")

    # Extract content
    content_text = record.get("text", "")

    # Handle quote posts: append quoted content
    embed = post.get("embed") or {}
    if embed.get("$type") == "app.bsky.embed.record#view":
        quoted_post = embed.get("record", {})
        if quoted_post:
            quoted_author = quoted_post.get("author", {}).get("handle", "unknown")
            quoted_text = quoted_post.get("record", {}).get("text", "")
            if quoted_text:
                content_text += f"\n\n[Quoting @{quoted_author}]: {quoted_text}"

    # Extract images
    images = []
    if embed.get("$type") == "app.bsky.embed.images#view":
        for image in embed.get("images", []):
            images.append(
                ImageContent(
                    url=image.get("thumb", ""),
                    alt_text=image.get("alt"),
                )
            )

    # Extract published date (ISO format string)
    published_at = None
    if record.get("created_at"):
        with contextlib.suppress(ValueError, AttributeError):
            published_at = datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))

    # Construct item URL from URI (AT URI format)
    post_id = post.get("uri", "").split("/")[-1]
    item_url = f"https://bsky.app/profile/{author.get('handle')}/post/{post_id}"

    # Thread detection
    is_thread = record.get("reply") is not None
    thread_id = None
    if is_thread and record.get("reply", {}).get("root"):
        thread_id = record["reply"]["root"].get("uri")

    return FeedItem(
        item_id=post.get("uri", ""),
        source_type=SourceType.BLUESKY,
        feed_url="https://bsky.app",
        item_url=item_url,
        author=author_name,
        author_url=f"https://bsky.app/profile/{author.get('handle')}",
        title=None,  # Bluesky posts don't have titles
        content_text=content_text,
        content_html=None,  # Bluesky doesn't provide HTML
        published_at=published_at,
        images=images,
        is_thread=is_thread,
        thread_id=thread_id,
    )


def ingest_bluesky(config: dict) -> int:
    """Poll configured Bluesky timelines and store new items.

    Reads Bluesky config from config["feeds"]["bluesky"]. Each entry has:
      - handle: str (e.g. "user.bsky.social")
      - app_password_env: str (env var name)
      - feed: str ("timeline" or "author:<did>")

    Auth: App password read from the environment variable
    named in app_password_env.

    Args:
        config: The OffScroll config dict.

    Returns:
        Count of new items ingested.
    """
    bluesky_feeds = config.get("feeds", {}).get("bluesky", [])
    if not bluesky_feeds:
        logger.info("No Bluesky feeds configured")
        return 0

    try:
        from atproto import Client as BskyClient
    except ImportError as exc:
        raise ImportError(
            "atproto not installed. Install: pip install 'offscroll[fediverse]'"
        ) from exc

    init_db(config)

    total_new = 0

    for feed_conf in bluesky_feeds:
        handle = feed_conf["handle"]
        app_password_env = feed_conf["app_password_env"]
        feed_type = feed_conf.get("feed", "timeline")

        # Read password from environment
        try:
            app_password = os.environ[app_password_env]
        except KeyError:
            logger.error(
                "Environment variable %s not set for Bluesky user %s",
                app_password_env,
                handle,
            )
            continue

        # Create client and login
        try:
            client = BskyClient()
            client.login(handle, app_password)
        except Exception as exc:
            logger.error("Failed to authenticate to Bluesky as %s: %s", handle, exc)
            continue

        # Fetch timeline
        try:
            if feed_type == "timeline":
                response = client.get_timeline(limit=50)
            elif feed_type.startswith("author:"):
                did = feed_type.split(":", 1)[1]
                response = client.get_author_feed(actor=did, limit=50)
            else:
                logger.warning("Unknown Bluesky feed type: %s", feed_type)
                continue
        except Exception as exc:
            logger.error("Failed to fetch Bluesky timeline for %s: %s", handle, exc)
            continue

        # Convert and store posts
        feed_new = 0
        feed_views = response.feed if hasattr(response, "feed") else response.get("feed", [])
        for feed_view in feed_views:
            post = feed_view.get("post", {})
            record = post.get("record", {})

            # Skip if no content
            if not record.get("text"):
                continue

            try:
                item = _bsky_post_to_feed_item(feed_view)
                if store_item(config, item):
                    feed_new += 1
            except Exception as exc:
                logger.warning("Failed to convert Bluesky post %s: %s", post.get("uri"), exc)
                continue

        total_new += feed_new
        logger.info("Bluesky %s: %d new items", handle, feed_new)

    return total_new
