"""End-to-end integration tests for the full OffScroll pipeline.

Validates: mock HTTP -> ingest -> curate -> render PDF + email.
Uses local mock HTTP server, stub embeddings, and sample data.
"""

from __future__ import annotations

import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from offscroll.curation.digest import render_digest
from offscroll.curation.selection import curate_edition
from offscroll.ingestion.feeds import ingest_all_feeds
from offscroll.ingestion.store import init_db
from offscroll.layout.renderer import render_newspaper_pdf
from offscroll.models import CuratedEdition

SAMPLE_DATA = Path(__file__).parent.parent / "sample_data"


def test_end_to_end_pipeline(tmp_path):
    """Full pipeline: mock HTTP -> ingest -> curate -> render PDF + email -> verify outputs.

    Uses a local mock HTTP server serving RSS XML, stub embeddings, and stub editorial
    (no Ollama).
    """
    # 1. Start mock HTTP server
    rss_content = (SAMPLE_DATA / "feeds" / "sample_rss.xml").read_text()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.end_headers()
            self.wfile.write(rss_content.encode())

        def log_message(self, *args):
            pass  # suppress output

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    try:
        # 2. Build config pointing at mock server
        config = {
            "feeds": {
                "rss": [
                    {
                        "url": f"http://127.0.0.1:{port}/feed.xml",
                        "name": "Test",
                    }
                ],
                "mastodon": [],
                "bluesky": [],
                "opml_files": [],
            },
            "ingestion": {
                "poll_interval_minutes": 60,
                "max_items_per_feed": 100,
                "download_images": False,
                "min_image_dimension": 200,
            },
            "embedding": {
                "provider": "stub",
                "ollama_url": "http://localhost:11434",
            },
            "curation": {
                "model": "ollama",
                "ollama_model": "llama3.2:3b",
                "ollama_url": "http://localhost:11434",
                "min_word_count": 10,
                "weights": {
                    "coverage": 1.0,
                    "redundancy": 1.0,
                    "quality": 1.0,
                    "diversity": 1.0,
                    "fit": 1.0,
                },
                "optimizer_iterations": 10,
            },
            "clustering": {
                "min_cluster_size": 2,
            },
            "newspaper": {
                "title": "Integration Test",
                "subtitle_pattern": "Vol. {volume}, No. {issue}",
                "page_target": 4,
                "columns": 3,
                "page_size": "letter",
                "margin_top": 0.5,
                "margin_bottom": 0.5,
                "margin_left": 0.5,
                "margin_right": 0.5,
                "column_gap": 0.2,
            },
            "email": {"enabled": False},
            "output": {
                "data_dir": str(tmp_path / "data"),
            },
            "logging": {
                "level": "WARNING",
                "file": None,
            },
        }

        # 3. Ingest
        init_db(config)
        count = ingest_all_feeds(config)
        assert count > 0, "No items ingested from mock server"

        # 4. Curate (uses stub embeddings, skips editorial -- Ollama not running)
        edition_path = curate_edition(config)
        assert edition_path.exists()
        assert edition_path.suffix == ".json"

        # 5. Render PDF
        pdf_path = render_newspaper_pdf(config, edition_path=edition_path)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000

        # 6. Render email digest
        digest_path = render_digest(config, edition_path=edition_path, send=False)
        assert digest_path.exists()
        html = digest_path.read_text()
        assert "<html" in html.lower()
        assert "Integration Test" in html

    finally:
        server.shutdown()


def test_end_to_end_with_load_config(tmp_path):
    """Full pipeline using load_config() to reproduce the MappingProxyType path.

    Regression test for the MappingProxyType crash: the setup wizard
    writes feeds as [{"url": "..."}] dicts. load_config() wraps them in
    _recursive_proxy(), which converts dicts to MappingProxyType and
    lists to tuples. Code that uses isinstance(..., dict) breaks.

    This test writes a config.yaml to disk, loads it through load_config()
    (getting MappingProxyType wrapping), and runs the full pipeline:
    ingest -> embed -> cluster -> curate -> render PDF.
    """
    import yaml

    from offscroll.config import load_config

    # 1. Start mock HTTP server
    rss_content = (SAMPLE_DATA / "feeds" / "sample_rss.xml").read_text()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.end_headers()
            self.wfile.write(rss_content.encode())

        def log_message(self, *args):
            pass  # suppress output

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    try:
        # 2. Write config.yaml exactly as the wizard would
        config_yaml = {
            "feeds": {
                "rss": [
                    {
                        "url": f"http://127.0.0.1:{port}/feed.xml",
                        "name": "Test Feed",
                    }
                ],
                "mastodon": [],
                "bluesky": [],
                "opml_files": [],
            },
            "ingestion": {
                "poll_interval_minutes": 60,
                "download_images": False,
            },
            "embedding": {
                "provider": "stub",
            },
            "curation": {
                "model": "ollama",
                "optimizer_iterations": 10,
                "min_word_count": 10,
            },
            "clustering": {
                "min_cluster_size": 2,
            },
            "newspaper": {
                "title": "Proxy Test Gazette",
                "page_target": 4,
                "page_size": "letter",
            },
            "email": {"enabled": False},
            "output": {
                "data_dir": str(tmp_path / "data"),
            },
            "logging": {
                "level": "WARNING",
                "file": str(tmp_path / "test.log"),
            },
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_yaml, f, default_flow_style=False)

        # 3. Load config through load_config() -- produces MappingProxyType
        config = load_config(config_path)

        # Verify we actually got a MappingProxyType, not a plain dict
        import types

        assert isinstance(config, types.MappingProxyType), (
            "load_config should return MappingProxyType"
        )

        # 4. Ingest (this is where the original crash happened)
        init_db(config)
        count = ingest_all_feeds(config)
        assert count > 0, "No items ingested from mock server"

        # 5. Curate (embed + cluster + select + editorial)
        edition_path = curate_edition(config)
        assert edition_path.exists()
        assert edition_path.suffix == ".json"

        # 6. Render PDF
        pdf_path = render_newspaper_pdf(config, edition_path=edition_path)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000

        # 7. Verify edition JSON is valid
        edition = CuratedEdition.from_json(edition_path)
        assert edition.edition.title == "Proxy Test Gazette"
        assert len(edition.sections) > 0

        # 8. Render email digest
        digest_path = render_digest(config, edition_path=edition_path, send=False)
        assert digest_path.exists()
        assert "Proxy Test Gazette" in digest_path.read_text()

    finally:
        server.shutdown()


def test_end_to_end_produces_valid_json(tmp_path):
    """The curated JSON contains required fields."""
    # Use the sample full edition
    sample = SAMPLE_DATA / "editions" / "sample_edition_full.json"
    edition = CuratedEdition.from_json(sample)
    assert edition.edition.title
    assert len(edition.sections) > 0

    # Round-trip through to_json / from_json
    out_path = tmp_path / "roundtrip.json"
    edition.to_json(out_path)
    reloaded = CuratedEdition.from_json(out_path)
    assert len(reloaded.sections) == len(edition.sections)
