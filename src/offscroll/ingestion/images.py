"""Image download for feed items.

Downloads images referenced in feed items to the local data directory
and updates their local_path fields.

image download implementation.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from offscroll.models import ImageContent

if TYPE_CHECKING:
    from offscroll.models import FeedItem

logger = logging.getLogger(__name__)


def _image_extension(url: str, content_type: str | None = None) -> str:
    """Determine file extension from URL or content type."""
    if content_type:
        ct = content_type.lower()
        if "png" in ct:
            return "png"
        if "gif" in ct:
            return "gif"
        if "webp" in ct:
            return "webp"
        if "svg" in ct:
            return "svg"
        if "jpeg" in ct or "jpg" in ct:
            return "jpg"

    # Fall back to URL extension
    path = url.split("?")[0].split("#")[0]
    if "." in path:
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in ("jpg", "jpeg", "png", "gif", "webp", "svg"):
            return ext

    return "jpg"  # default


def _image_hash(url: str) -> str:
    """Generate a short hash from the image URL for filenames."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


def download_images(
    config: dict,
    items: list[FeedItem],
) -> int:
    """Download images for feed items.

    For each item with images that have a URL but no local_path,
    download the image to the data directory. Skip images below the
    min_image_dimension threshold (using Content-Length as a rough
    proxy). Update the ImageContent.local_path field.

    Args:
        config: The OffScroll config dict.
        items: FeedItems with images to download.

    Returns:
        Count of images downloaded.
    """
    ingestion_config = config.get("ingestion", {})
    if not ingestion_config.get("download_images", True):
        logger.info("Image download disabled in config")
        return 0

    min_dimension = ingestion_config.get("min_image_dimension", 200)
    # Use a rough heuristic: images below ~10KB are likely too small
    # (icons, spacers, etc.). min_dimension^2 * 3 bytes / 10 gives
    # a conservative lower bound for content length.
    min_content_length = max(1000, min_dimension * min_dimension * 3 // 10)

    data_dir = Path(config.get("output", {}).get("data_dir", "~/.offscroll/data"))
    data_dir = data_dir.expanduser()
    images_dir = data_dir / "images"

    count = 0
    for item in items:
        for image in item.images:
            if image.local_path is not None:
                continue
            if not image.url:
                continue

            try:
                downloaded = _download_single_image(
                    image=image,
                    item_id=item.item_id,
                    images_dir=images_dir,
                    min_content_length=min_content_length,
                )
                if downloaded:
                    count += 1
            except Exception:
                logger.exception(
                    "Failed to download image %s for item %s",
                    image.url,
                    item.item_id,
                )

    return count


def _download_single_image(
    image: ImageContent,
    item_id: str,
    images_dir: Path,
    min_content_length: int,
) -> bool:
    """Download a single image. Returns True if downloaded.

    Args:
        image: The ImageContent to download. Updates local_path in place.
        item_id: The parent item's ID (used for directory structure).
        images_dir: Base directory for image storage.
        min_content_length: Skip images with Content-Length below this.

    Returns:
        True if the image was downloaded, False if skipped.
    """
    url = image.url
    try:
        response = httpx.get(
            url,
            timeout=30.0,
            follow_redirects=True,
        )
        response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.warning("HTTP error downloading %s: %s", url, exc)
        raise

    # Check content length
    content_length = len(response.content)
    if content_length < min_content_length:
        logger.debug(
            "Skipping image %s: size %d below threshold %d",
            url,
            content_length,
            min_content_length,
        )
        return False

    # Determine extension and filename
    content_type = response.headers.get("content-type")
    ext = _image_extension(url, content_type)
    filename = f"{_image_hash(url)}.{ext}"

    # Create item directory
    item_dir = images_dir / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    # Write the image
    image_path = item_dir / filename
    image_path.write_bytes(response.content)

    # Update the ImageContent with relative path from data_dir
    image.local_path = str(Path("images") / item_id / filename)

    # Measure image dimensions using PIL (Pillow, already a
    # dependency via WeasyPrint)
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            image.width, image.height = img.size
    except Exception:
        logger.debug("Could not measure dimensions for %s", image_path)

    logger.info("Downloaded image %s -> %s", url, image.local_path)
    return True
