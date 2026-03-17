"""End-to-end integration tests through the full pipeline using load_config().

Two test paths:
1. test_e2e_stub_embeddings: Uses stub embeddings (no Ollama needed).
   Validates that the full pipeline works through load_config()
   (MappingProxyType wrapping) end-to-end: ingest -> curate -> render.

2. test_e2e_real_ollama: Uses real Ollama for embeddings and curation.
   Marked @pytest.mark.slow. Skipped when Ollama is not reachable.
   Validates the production code path with a real LLM backend.

Both tests use a local mock HTTP server to serve RSS XML, so no
external feed dependencies.
"""

from __future__ import annotations

import threading
import types
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import httpx
import pytest
import yaml

from offscroll.config import load_config
from offscroll.curation.selection import curate_edition
from offscroll.ingestion.feeds import ingest_all_feeds
from offscroll.ingestion.store import init_db
from offscroll.layout.renderer import render_newspaper_html, render_newspaper_pdf
from offscroll.models import CuratedEdition, RankedEdition, load_edition

SAMPLE_DATA = Path(__file__).parent / "sample_data"


def _ollama_is_reachable(url: str = "http://localhost:11434") -> bool:
    """Check whether Ollama is running and reachable."""
    try:
        resp = httpx.get(f"{url}/api/tags", timeout=3.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _ollama_has_model(model: str, url: str = "http://localhost:11434") -> bool:
    """Check whether a specific model is available in Ollama."""
    try:
        resp = httpx.get(f"{url}/api/tags", timeout=3.0)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        # Match with or without tag suffix
        return any(model in m or m.startswith(model) for m in models)
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _start_mock_feed_server() -> tuple[HTTPServer, int]:
    """Start a local HTTP server that serves sample RSS XML."""
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
    return server, port


def _write_config(tmp_path: Path, port: int, embedding_provider: str = "stub") -> Path:
    """Write a config.yaml file and return its path.

    When embedding_provider is "ollama", uses real Ollama settings.
    When "stub", uses deterministic stub embeddings (no Ollama needed).
    """
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
            "provider": embedding_provider,
            "ollama_model": "nomic-embed-text",
            "ollama_url": "http://localhost:11434",
        },
        "curation": {
            "model": "ollama",
            "ollama_model": "llama3.2:3b",
            "ollama_url": "http://localhost:11434",
            "optimizer_iterations": 10,
            "min_word_count": 10,
        },
        "clustering": {
            "min_cluster_size": 2,
        },
        "newspaper": {
            "title": "Integration Test Gazette",
            "subtitle_pattern": "Vol. {volume}, No. {issue}",
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
    return config_path


def test_e2e_stub_embeddings(tmp_path):
    """Full pipeline with load_config() and stub embeddings.

    Validates: load_config() -> MappingProxyType -> ingest -> curate
    -> render HTML + PDF. Uses stub embeddings so no Ollama is needed.

    This is the core regression test for the MappingProxyType crash:
    every config value is wrapped in MappingProxyType (dicts) or tuples
    (lists), and the full pipeline must handle that correctly.
    """
    server, port = _start_mock_feed_server()

    try:
        config_path = _write_config(tmp_path, port, embedding_provider="stub")
        config = load_config(config_path)

        # Verify we got MappingProxyType, not a plain dict
        assert isinstance(config, types.MappingProxyType), (
            "load_config should return MappingProxyType"
        )

        # Verify nested values are also proxied
        assert isinstance(config["feeds"], types.MappingProxyType)
        assert isinstance(config["feeds"]["rss"], tuple)  # lists become tuples

        # Step 1: Ingest
        init_db(config)
        count = ingest_all_feeds(config)
        assert count > 0, "No items ingested from mock server"

        # Step 2: Curate (stub embeddings, no LLM editorial)
        edition_path = curate_edition(config)
        assert edition_path.exists()
        assert edition_path.suffix == ".json"

        # Step 3: Load and verify edition
        edition = load_edition(edition_path)
        assert isinstance(edition, (CuratedEdition, RankedEdition))
        assert edition.edition.title == "Integration Test Gazette"

        # Step 4: Render HTML
        html_path = render_newspaper_html(config, edition_path=edition_path)
        assert html_path.exists()
        html_content = html_path.read_text()
        assert "<html" in html_content.lower()
        assert "Integration Test Gazette" in html_content

        # Step 5: Render PDF
        pdf_path = render_newspaper_pdf(config, edition_path=edition_path)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000  # Not an empty file

    finally:
        server.shutdown()


@pytest.mark.slow
def test_e2e_real_ollama(tmp_path):
    """Full pipeline with load_config() and real Ollama.

    Validates the production code path: real Ollama embeddings
    (nomic-embed-text) and real LLM curation (llama3.2:3b).

    Requires:
    - Ollama running at localhost:11434
    - nomic-embed-text model pulled
    - llama3.2:3b model pulled

    Skips gracefully if Ollama is not available.
    """
    if not _ollama_is_reachable():
        pytest.skip("Ollama is not running at localhost:11434")

    if not _ollama_has_model("nomic-embed-text"):
        pytest.skip("nomic-embed-text model not available in Ollama")

    if not _ollama_has_model("llama3.2:3b"):
        pytest.skip("llama3.2:3b model not available in Ollama")

    server, port = _start_mock_feed_server()

    try:
        config_path = _write_config(tmp_path, port, embedding_provider="ollama")
        config = load_config(config_path)

        # Verify MappingProxyType wrapping
        assert isinstance(config, types.MappingProxyType)

        # Step 1: Ingest
        init_db(config)
        count = ingest_all_feeds(config)
        assert count > 0, "No items ingested from mock server"

        # Step 2: Curate with real Ollama (embeddings + LLM editorial)
        edition_path = curate_edition(config)
        assert edition_path.exists()
        assert edition_path.suffix == ".json"

        # Step 3: Load and verify edition
        edition = load_edition(edition_path)
        assert isinstance(edition, (CuratedEdition, RankedEdition))
        assert edition.edition.title == "Integration Test Gazette"

        # Step 4: Render HTML
        html_path = render_newspaper_html(config, edition_path=edition_path)
        assert html_path.exists()
        html_content = html_path.read_text()
        assert "<html" in html_content.lower()
        assert "Integration Test Gazette" in html_content

        # Step 5: Render PDF
        pdf_path = render_newspaper_pdf(config, edition_path=edition_path)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000

    finally:
        server.shutdown()
