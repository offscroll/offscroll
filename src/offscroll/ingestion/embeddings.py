"""Embedding generation (cloud + local).

embedding provider interface with stub implementation.
Real provider implementations (OpenAI, Ollama, sentence_transformers)
are a task.
"""

from __future__ import annotations

import hashlib
import logging
import math

from offscroll.models import FeedItem

logger = logging.getLogger(__name__)


def _embed_stub(texts: list[str]) -> list[list[float]]:
    """Stub embedding provider for testing.

    Returns deterministic 8-dimensional embeddings based on a
    hash of each text. This allows tests to run without any
    model or API.
    """
    embeddings = []
    for text in texts:
        digest = hashlib.md5(text.encode()).digest()  # noqa: S324
        raw = [digest[i] / 255.0 for i in range(8)]
        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in raw))
        if norm > 0.0:
            raw = [x / norm for x in raw]
        embeddings.append(raw)
    return embeddings


def _embed_ollama(
    texts: list[str],
    config: dict,
) -> list[list[float]]:
    """Embed texts using Ollama's embedding API.

    Args:
        texts: List of text strings to embed.
        config: Config dict. Uses config["embedding"]["ollama_model"]
            (default: "nomic-embed-text") and
            config["embedding"]["ollama_url"]
            (default: "http://localhost:11434").

    Returns:
        List of embedding vectors (one per text).

    Raises:
        ConnectionError: If Ollama is not reachable.
    """
    try:
        import ollama as ollama_client
    except ImportError as e:
        raise ImportError("ollama package not installed. Install with: pip install ollama") from e

    embedding_config = config.get("embedding", {})
    model_name = embedding_config.get("ollama_model", "nomic-embed-text")
    ollama_url = embedding_config.get("ollama_url", "http://localhost:11434")

    try:
        client = ollama_client.Client(host=ollama_url)
    except Exception as e:
        raise ConnectionError(
            f"Cannot connect to Ollama at {ollama_url}. "
            "Is Ollama running? Start it with: ollama serve"
        ) from e

    # nomic-embed-text has an 8192-token context window. Using a
    # conservative character limit to stay safely under that ceiling.
    max_chars = 6000

    embeddings = []
    for text in texts:
        if len(text) > max_chars:
            logger.warning(
                "Text truncated from %d to %d characters for Ollama embedding",
                len(text),
                max_chars,
            )
            text = text[:max_chars]
        try:
            response = client.embed(model=model_name, input=text)
            embeddings.append(response["embeddings"][0])
        except Exception as e:
            raise ConnectionError(f"Ollama embedding request failed: {e}") from e

    return embeddings


def _embed_texts(texts: list[str], config: dict) -> list[list[float]]:
    """Dispatch to the configured embedding provider.

    Args:
        texts: List of text strings to embed.
        config: Config dict with provider settings.

    Returns:
        List of embedding vectors (one per text).

    Raises:
        NotImplementedError: If the provider requires a running
            service (openai, ollama) and is not available.
    """
    provider = config.get("embedding", {}).get("provider", "stub")

    if provider == "stub":
        return _embed_stub(texts)

    if provider == "ollama":
        return _embed_ollama(texts, config)

    if provider in ("openai", "sentence_transformers"):
        raise NotImplementedError(f"Provider '{provider}' not yet implemented")

    raise NotImplementedError(f"Unknown embedding provider: '{provider}'")


def embed_items(
    items: list[FeedItem],
    config: dict,
) -> list[FeedItem]:
    """Generate embeddings for a batch of FeedItems.

    Dispatches to the configured provider (openai, ollama, or
    sentence_transformers). Updates each item's embedding field
    in-place and returns the list.

    Args:
        items: FeedItems to embed. Items that already have
            embeddings are skipped.
        config: The OffScroll config dict. Uses
            config["embedding"]["provider"] to select the backend.

    Returns:
        The same list of items with embeddings populated.
    """
    if not items:
        return items

    # Filter to items needing embeddings
    to_embed = [item for item in items if item.embedding is None]

    if not to_embed:
        return items

    # Build text input: prepend title if present
    texts = []
    for item in to_embed:
        if item.title:
            texts.append(f"{item.title}: {item.content_text}")
        else:
            texts.append(item.content_text)

    vectors = _embed_texts(texts, config)

    for item, vector in zip(to_embed, vectors, strict=True):
        item.embedding = vector

    return items
