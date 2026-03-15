"""OffScroll configuration loading and validation.

Usage in any component:

    from offscroll.config import load_config
    config = load_config()         # Loads from default path
    config = load_config(path)     # Loads from explicit path

The returned object is a read-only mapping. Access with bracket notation:

    api_key = os.environ[config["curation"]["claude_api_key_env"]]
    page_target = config["newspaper"]["page_target"]

We do NOT wrap the config in a class or dataclass. A dict is simple,
debuggable, and serializable. If config access becomes painful, we
revisit.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path.home() / ".offscroll" / "config.yaml"

# Defaults are applied when a key is missing from the user's config.
# This is the single source of truth for default values.
DEFAULTS: dict[str, Any] = {
    "feeds": {"rss": [], "mastodon": [], "bluesky": [], "opml_files": []},
    "ingestion": {
        "poll_interval_minutes": 60,
        "max_items_per_feed": 100,
        "download_images": True,
        "min_image_dimension": 200,
    },
    "embedding": {
        "provider": "ollama",
        "openai_model": "text-embedding-3-small",
        "openai_api_key_env": "OPENAI_API_KEY",
        "ollama_model": "nomic-embed-text",
        "ollama_url": "http://localhost:11434",
        "st_model": "all-MiniLM-L6-v2",
    },
    "curation": {
        "model": "ollama",
        "claude_model": "claude-sonnet-4-20250514",
        "claude_api_key_env": "ANTHROPIC_API_KEY",
        "openai_model": "gpt-4o",
        "openai_api_key_env": "OPENAI_API_KEY",
        "ollama_model": "llama3.1:8b",
        "ollama_url": "http://localhost:11434",
        "weights": {
            "coverage": 1.0,
            "redundancy": 1.0,
            "quality": 1.0,
            "diversity": 1.0,
            "fit": 1.0,
        },
        "optimizer_iterations": 500,
    },
    "newspaper": {
        "title": "The Morning Dispatch",
        "subtitle_pattern": "Vol. {volume}, No. {issue}",
        "page_target": 10,
        "columns": 3,
        "page_size": "letter",
        "margin_top": 0.5,
        "margin_bottom": 0.5,
        "margin_left": 0.5,
        "margin_right": 0.5,
        "column_gap": 0.2,
    },
    "email": {
        "enabled": False,
        "smtp_host": "",
        "smtp_port": 587,
        "smtp_user_env": "OFFSCROLL_SMTP_USER",
        "smtp_password_env": "OFFSCROLL_SMTP_PASSWORD",
        "from_address": "",
        "to_addresses": [],
    },
    "output": {
        "data_dir": "~/.offscroll/data",
        "retention_weeks": 0,
    },
    "schedule": {
        "compile_day": "saturday",
        "compile_time": "21:00",
        "timezone": "America/New_York",
    },
    "logging": {
        "level": "INFO",
        "file": "~/.offscroll/offscroll.log",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_paths(config: dict) -> dict:
    """Expand ~ in path-like config values."""
    for key in ("data_dir", "file"):
        for section in config.values():
            if isinstance(section, dict) and key in section:
                section[key] = str(Path(section[key]).expanduser())
    return config


def _recursive_proxy(obj: Any) -> Any:
    """Recursively wrap dicts in MappingProxyType for read-only access."""
    if isinstance(obj, dict):
        return types.MappingProxyType({k: _recursive_proxy(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return tuple(_recursive_proxy(item) for item in obj)
    return obj


def _validate(config: dict) -> list[str]:
    """Return a list of validation errors. Empty list means valid."""
    errors = []

    # Must have at least one feed source
    feeds = config.get("feeds", {})
    has_feeds = (
        bool(feeds.get("rss"))
        or bool(feeds.get("mastodon"))
        or bool(feeds.get("bluesky"))
        or bool(feeds.get("opml_files"))
    )
    if not has_feeds:
        msg = "No feed sources configured. Add at least one RSS, Mastodon, or Bluesky feed."
        errors.append(msg)

    # Validate embedding provider
    provider = config.get("embedding", {}).get("provider", "")
    if provider not in ("openai", "ollama", "sentence_transformers", "stub"):
        errors.append(f"Unknown embedding provider: {provider}")

    # Validate curation model
    model = config.get("curation", {}).get("model", "")
    if model not in ("claude", "openai", "ollama"):
        errors.append(f"Unknown curation model: {model}")

    # Validate page size
    page_size = config.get("newspaper", {}).get("page_size", "")
    if page_size not in ("letter", "a4"):
        errors.append(f"Unknown page size: {page_size}")

    # Validate poll interval
    poll = config.get("ingestion", {}).get("poll_interval_minutes", 60)
    if poll < 15:
        errors.append(f"Poll interval too aggressive: {poll} minutes (minimum 15)")

    return errors


def load_config(path: Path | None = None) -> types.MappingProxyType:
    """Load, validate, and return the config dict (as read-only MappingProxyType).

    Raises SystemExit with a clear message on errors.
    """
    config_path = path or DEFAULT_CONFIG_PATH

    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        msg = "Run 'offscroll init' to create a config file, or copy config.example.yaml."
        print(msg, file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        user_config = yaml.safe_load(f) or {}

    config = _deep_merge(DEFAULTS, user_config)
    config = _expand_paths(config)

    # Allow OLLAMA_HOST environment variable to override config
    ollama_host = os.environ.get("OLLAMA_HOST")
    if ollama_host:
        config["embedding"]["ollama_url"] = ollama_host
        config["curation"]["ollama_url"] = ollama_host

    errors = _validate(config)
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    # Wrap the entire config (including nested dicts) in MappingProxyType for read-only access
    return _recursive_proxy(config)  # type: ignore
