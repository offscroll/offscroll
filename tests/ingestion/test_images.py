"""Tests for image download functionality.

image download implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from offscroll.ingestion.images import (
    _image_extension,
    _image_hash,
    download_images,
)
from offscroll.models import FeedItem, ImageContent, SourceType


def _make_item(
    item_id: str = "test-001",
    images: list[ImageContent] | None = None,
) -> FeedItem:
    """Helper to create a FeedItem with images."""
    return FeedItem(
        item_id=item_id,
        source_type=SourceType.RSS,
        feed_url="https://example.com/feed.xml",
        content_text="Test content for image download testing.",
        images=images or [],
    )


def _make_response(
    content: bytes = b"\xff\xd8\xff" + b"\x00" * 50000,
    status_code: int = 200,
    content_type: str = "image/jpeg",
) -> httpx.Response:
    """Create a mock httpx Response."""
    response = MagicMock(spec=httpx.Response)
    response.content = content
    response.status_code = status_code
    response.headers = {"content-type": content_type}
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
    return response


class TestImageExtension:
    """Tests for _image_extension helper."""

    def test_extension_from_content_type_jpeg(self):
        assert _image_extension("http://x.com/img", "image/jpeg") == "jpg"

    def test_extension_from_content_type_png(self):
        assert _image_extension("http://x.com/img", "image/png") == "png"

    def test_extension_from_url(self):
        assert _image_extension("http://x.com/photo.png") == "png"

    def test_extension_from_url_with_query(self):
        assert _image_extension("http://x.com/photo.webp?w=200") == "webp"

    def test_extension_default_jpg(self):
        assert _image_extension("http://x.com/img") == "jpg"


class TestImageHash:
    """Tests for _image_hash helper."""

    def test_hash_is_deterministic(self):
        h1 = _image_hash("http://example.com/img.jpg")
        h2 = _image_hash("http://example.com/img.jpg")
        assert h1 == h2

    def test_hash_length(self):
        h = _image_hash("http://example.com/img.jpg")
        assert len(h) == 12


class TestDownloadImages:
    """Tests for download_images function."""

    def test_successful_download(self, tmp_path):
        """Image downloads to the correct path and local_path is set."""
        config = {
            "ingestion": {"download_images": True, "min_image_dimension": 50},
            "output": {"data_dir": str(tmp_path)},
        }
        image = ImageContent(url="https://example.com/photo.jpg")
        item = _make_item(images=[image])

        with patch("offscroll.ingestion.images.httpx.get") as mock_get:
            mock_get.return_value = _make_response()
            count = download_images(config, [item])

        assert count == 1
        assert image.local_path is not None
        assert image.local_path.startswith("images/")
        # Verify file was actually written
        full_path = tmp_path / image.local_path
        assert full_path.exists()

    def test_skip_when_disabled(self, tmp_path):
        """No downloads when download_images is False."""
        config = {
            "ingestion": {"download_images": False},
            "output": {"data_dir": str(tmp_path)},
        }
        image = ImageContent(url="https://example.com/photo.jpg")
        item = _make_item(images=[image])

        count = download_images(config, [item])

        assert count == 0
        assert image.local_path is None

    def test_skip_below_size_threshold(self, tmp_path):
        """Images below min_content_length are skipped."""
        config = {
            "ingestion": {"download_images": True, "min_image_dimension": 200},
            "output": {"data_dir": str(tmp_path)},
        }
        image = ImageContent(url="https://example.com/tiny.jpg")
        item = _make_item(images=[image])

        # Return a very small image (100 bytes)
        with patch("offscroll.ingestion.images.httpx.get") as mock_get:
            mock_get.return_value = _make_response(content=b"\x00" * 100)
            count = download_images(config, [item])

        assert count == 0
        assert image.local_path is None

    def test_http_error_does_not_crash(self, tmp_path):
        """HTTP errors are logged but pipeline continues."""
        config = {
            "ingestion": {"download_images": True, "min_image_dimension": 50},
            "output": {"data_dir": str(tmp_path)},
        }
        image = ImageContent(url="https://example.com/missing.jpg")
        item = _make_item(images=[image])

        with patch("offscroll.ingestion.images.httpx.get") as mock_get:
            mock_get.side_effect = httpx.HTTPError("404 Not Found")
            count = download_images(config, [item])

        assert count == 0
        assert image.local_path is None

    def test_skip_already_downloaded(self, tmp_path):
        """Images with local_path already set are skipped."""
        config = {
            "ingestion": {"download_images": True, "min_image_dimension": 50},
            "output": {"data_dir": str(tmp_path)},
        }
        image = ImageContent(
            url="https://example.com/photo.jpg",
            local_path="images/existing/abc.jpg",
        )
        item = _make_item(images=[image])

        with patch("offscroll.ingestion.images.httpx.get") as mock_get:
            count = download_images(config, [item])

        assert count == 0
        mock_get.assert_not_called()

    def test_multiple_images_per_item(self, tmp_path):
        """Multiple images on a single item are all downloaded."""
        config = {
            "ingestion": {"download_images": True, "min_image_dimension": 50},
            "output": {"data_dir": str(tmp_path)},
        }
        images = [
            ImageContent(url="https://example.com/img1.jpg"),
            ImageContent(url="https://example.com/img2.png"),
            ImageContent(url="https://example.com/img3.webp"),
        ]
        item = _make_item(images=images)

        with patch("offscroll.ingestion.images.httpx.get") as mock_get:
            mock_get.return_value = _make_response()
            count = download_images(config, [item])

        assert count == 3
        assert all(img.local_path is not None for img in images)

    def test_multiple_items(self, tmp_path):
        """Images from multiple items are all downloaded."""
        config = {
            "ingestion": {"download_images": True, "min_image_dimension": 50},
            "output": {"data_dir": str(tmp_path)},
        }
        item1 = _make_item(
            item_id="item-001",
            images=[ImageContent(url="https://example.com/a.jpg")],
        )
        item2 = _make_item(
            item_id="item-002",
            images=[ImageContent(url="https://example.com/b.jpg")],
        )

        with patch("offscroll.ingestion.images.httpx.get") as mock_get:
            mock_get.return_value = _make_response()
            count = download_images(config, [item1, item2])

        assert count == 2

    def test_partial_failure(self, tmp_path):
        """One image failure does not prevent others from downloading."""
        config = {
            "ingestion": {"download_images": True, "min_image_dimension": 50},
            "output": {"data_dir": str(tmp_path)},
        }
        images = [
            ImageContent(url="https://example.com/good.jpg"),
            ImageContent(url="https://example.com/bad.jpg"),
        ]
        item = _make_item(images=images)

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise httpx.HTTPError("Server error")
            return _make_response()

        with patch("offscroll.ingestion.images.httpx.get") as mock_get:
            mock_get.side_effect = side_effect
            count = download_images(config, [item])

        assert count == 1
        assert images[0].local_path is not None
        assert images[1].local_path is None


# ---------------------------------------------------------------------------
#  Image Dimension Measurement Tests
# ---------------------------------------------------------------------------


class TestImageDimensionMeasurement:
    """(11.7): Image dimensions measured on download."""

    def test_dimensions_set_after_download(self, tmp_path):
        """Downloaded images have width and height populated."""
        import io

        from PIL import Image as PILImage

        config = {
            "ingestion": {"download_images": True, "min_image_dimension": 50},
            "output": {"data_dir": str(tmp_path)},
        }
        image = ImageContent(url="https://example.com/photo.jpg")
        item = _make_item(images=[image])

        # Create a real image large enough to pass size threshold
        # (400x300 produces ~2.5KB JPEG, above the 1000-byte minimum)
        img = PILImage.new("RGB", (400, 300), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        real_image_bytes = buf.getvalue()

        with patch("offscroll.ingestion.images.httpx.get") as mock_get:
            mock_get.return_value = _make_response(content=real_image_bytes)
            count = download_images(config, [item])

        assert count == 1
        assert image.width == 400
        assert image.height == 300

    def test_dimensions_for_png(self, tmp_path):
        """PNG images also get dimensions measured."""
        import io

        from PIL import Image as PILImage

        config = {
            "ingestion": {"download_images": True, "min_image_dimension": 50},
            "output": {"data_dir": str(tmp_path)},
        }
        image = ImageContent(url="https://example.com/photo.png")
        item = _make_item(images=[image])

        # Create a PNG image large enough (400x300 produces ~1.2KB PNG)
        img = PILImage.new("RGBA", (400, 300), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        with patch("offscroll.ingestion.images.httpx.get") as mock_get:
            mock_get.return_value = _make_response(
                content=png_bytes, content_type="image/png"
            )
            count = download_images(config, [item])

        assert count == 1
        assert image.width == 400
        assert image.height == 300

    def test_dimension_failure_does_not_crash(self, tmp_path):
        """If PIL cannot read the image, download still succeeds."""
        config = {
            "ingestion": {"download_images": True, "min_image_dimension": 50},
            "output": {"data_dir": str(tmp_path)},
        }
        image = ImageContent(url="https://example.com/corrupt.jpg")
        item = _make_item(images=[image])

        # Send non-image bytes that are large enough to pass size check
        corrupt_bytes = b"\x00" * 50000

        with patch("offscroll.ingestion.images.httpx.get") as mock_get:
            mock_get.return_value = _make_response(content=corrupt_bytes)
            count = download_images(config, [item])

        assert count == 1
        assert image.local_path is not None
        # Dimensions should remain None (PIL couldn't read it)
        assert image.width is None
        assert image.height is None
