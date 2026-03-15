"""Tests for the CLI setup and init commands.

Docker, setup CLI, and integration tests.
"""

from __future__ import annotations

import yaml
from click.testing import CliRunner

from offscroll.cli import cli


def test_setup_creates_config(tmp_path, monkeypatch):
    """setup command writes config.yaml to ~/.offscroll/."""
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    runner.invoke(
        cli,
        ["setup"],
        input=(
            "Test Paper\n"  # newspaper name
            "10\n"  # pages
            "https://example.com/feed\n"  # RSS
            "Test Feed\n"  # name
            "\n"  # empty = done with RSS
            "n\n"  # no OPML
            "n\n"  # no Mastodon
            "n\n"  # no Bluesky
            "http://localhost:11434\n"  # Ollama
            "n\n"  # no email
        ),
    )
    config_path = tmp_path / ".offscroll" / "config.yaml"
    assert config_path.exists()
    config = yaml.safe_load(config_path.read_text())
    assert config["newspaper"]["title"] == "Test Paper"
    assert config["newspaper"]["page_target"] == 10
    assert len(config["feeds"]["rss"]) == 1
    assert config["feeds"]["rss"][0]["url"] == "https://example.com/feed"
    assert config["feeds"]["rss"][0]["name"] == "Test Feed"


def test_setup_overwrites_with_confirm(tmp_path, monkeypatch):
    """setup prompts before overwriting existing config."""
    monkeypatch.setenv("HOME", str(tmp_path))
    offscroll_dir = tmp_path / ".offscroll"
    offscroll_dir.mkdir()
    config_path = offscroll_dir / "config.yaml"
    config_path.write_text("feeds:\n  rss:\n    - url: https://old.com/feed\n")

    runner = CliRunner()
    runner.invoke(
        cli,
        ["setup"],
        input=(
            "y\n"  # confirm overwrite
            "New Paper\n"
            "5\n"
            "https://new.com/feed\n"
            "New Feed\n"
            "\n"
            "n\n"
            "n\n"
            "n\n"
            "http://localhost:11434\n"
            "n\n"
        ),
    )
    config = yaml.safe_load(config_path.read_text())
    assert config["newspaper"]["title"] == "New Paper"


def test_setup_no_overwrite_on_decline(tmp_path, monkeypatch):
    """setup respects declining to overwrite existing config."""
    monkeypatch.setenv("HOME", str(tmp_path))
    offscroll_dir = tmp_path / ".offscroll"
    offscroll_dir.mkdir()
    config_path = offscroll_dir / "config.yaml"
    config_path.write_text("feeds:\n  rss:\n    - url: https://old.com/feed\n")

    runner = CliRunner()
    runner.invoke(
        cli,
        ["setup"],
        input="n\n",  # decline overwrite
    )
    # Should still have the old content
    assert "https://old.com/feed" in config_path.read_text()


def test_setup_minimal_config(tmp_path, monkeypatch):
    """setup creates valid config with minimal inputs."""
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    runner.invoke(
        cli,
        ["setup"],
        input=(
            "Minimal\n"
            "1\n"
            "\n"  # no RSS feeds
            "n\n"  # no OPML
            "n\n"  # no Mastodon
            "n\n"  # no Bluesky
            "http://localhost:11434\n"
            "n\n"  # no email
        ),
    )
    config_path = tmp_path / ".offscroll" / "config.yaml"
    assert config_path.exists()
    config = yaml.safe_load(config_path.read_text())
    assert config["newspaper"]["title"] == "Minimal"
    assert len(config["feeds"]["rss"]) == 0


def test_setup_ollama_check_offline(tmp_path, monkeypatch):
    """setup warns when Ollama is unreachable."""
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["setup"],
        input=(
            "Offline Test\n"
            "5\n"
            "\n"
            "n\n"
            "n\n"
            "n\n"
            "http://localhost:99999\n"  # invalid port
            "n\n"
        ),
    )
    # Should still complete despite offline Ollama
    config_path = tmp_path / ".offscroll" / "config.yaml"
    assert config_path.exists()
    assert "Warning" in result.output or "Could not connect" in result.output


# ---------------------------------------------------------------------------
#  feeds add command
# ---------------------------------------------------------------------------


def test_feeds_add_basic(tmp_path, monkeypatch):
    """feeds add appends a feed URL to config.yaml."""
    monkeypatch.setenv("HOME", str(tmp_path))
    offscroll_dir = tmp_path / ".offscroll"
    offscroll_dir.mkdir()
    config_path = offscroll_dir / "config.yaml"
    config_path.write_text("feeds:\n  rss:\n    - url: https://existing.com/feed\n")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["feeds", "add", "https://new.example.com/feed.xml"],
    )
    assert result.exit_code == 0
    assert "Added feed" in result.output

    config = yaml.safe_load(config_path.read_text())
    urls = [e["url"] for e in config["feeds"]["rss"]]
    assert "https://new.example.com/feed.xml" in urls
    assert "https://existing.com/feed" in urls


def test_feeds_add_with_name(tmp_path, monkeypatch):
    """feeds add --name stores the name alongside the URL."""
    monkeypatch.setenv("HOME", str(tmp_path))
    offscroll_dir = tmp_path / ".offscroll"
    offscroll_dir.mkdir()
    config_path = offscroll_dir / "config.yaml"
    config_path.write_text("feeds:\n  rss: []\n")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["feeds", "add", "--name", "My Blog", "https://blog.example.com/rss"],
    )
    assert result.exit_code == 0
    assert "My Blog" in result.output

    config = yaml.safe_load(config_path.read_text())
    assert config["feeds"]["rss"][0]["url"] == "https://blog.example.com/rss"
    assert config["feeds"]["rss"][0]["name"] == "My Blog"


def test_feeds_add_duplicate(tmp_path, monkeypatch):
    """feeds add detects duplicate URLs."""
    monkeypatch.setenv("HOME", str(tmp_path))
    offscroll_dir = tmp_path / ".offscroll"
    offscroll_dir.mkdir()
    config_path = offscroll_dir / "config.yaml"
    config_path.write_text("feeds:\n  rss:\n    - url: https://dup.example.com/feed\n")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["feeds", "add", "https://dup.example.com/feed"],
    )
    assert "already exists" in result.output

    config = yaml.safe_load(config_path.read_text())
    # Should still be just one entry
    assert len(config["feeds"]["rss"]) == 1


def test_feeds_add_creates_rss_section(tmp_path, monkeypatch):
    """feeds add creates feeds.rss section if missing."""
    monkeypatch.setenv("HOME", str(tmp_path))
    offscroll_dir = tmp_path / ".offscroll"
    offscroll_dir.mkdir()
    config_path = offscroll_dir / "config.yaml"
    config_path.write_text("newspaper:\n  title: Test\n")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["feeds", "add", "https://new.example.com/feed.xml"],
    )
    assert result.exit_code == 0

    config = yaml.safe_load(config_path.read_text())
    assert len(config["feeds"]["rss"]) == 1


def test_feeds_add_valid_yaml(tmp_path, monkeypatch):
    """Config file remains valid YAML after feeds add."""
    monkeypatch.setenv("HOME", str(tmp_path))
    offscroll_dir = tmp_path / ".offscroll"
    offscroll_dir.mkdir()
    config_path = offscroll_dir / "config.yaml"
    config_path.write_text("feeds:\n  rss: []\nnewspaper:\n  title: Test\n")

    runner = CliRunner()
    runner.invoke(
        cli,
        ["feeds", "add", "https://example.com/feed.xml"],
    )
    runner.invoke(
        cli,
        ["feeds", "add", "https://another.example.com/rss"],
    )

    # Should still be valid YAML
    config = yaml.safe_load(config_path.read_text())
    assert len(config["feeds"]["rss"]) == 2
    assert config["newspaper"]["title"] == "Test"


def test_init_calls_setup(tmp_path, monkeypatch):
    """init command delegates to setup."""
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    runner.invoke(
        cli,
        ["init"],
        input=(
            "Test Init\n"
            "8\n"
            "https://example.com/init-feed\n"
            "Init Feed\n"
            "\n"
            "n\n"
            "n\n"
            "n\n"
            "http://localhost:11434\n"
            "n\n"
        ),
    )
    config_path = tmp_path / ".offscroll" / "config.yaml"
    assert config_path.exists()
    config = yaml.safe_load(config_path.read_text())
    assert config["newspaper"]["title"] == "Test Init"


# ---------------------------------------------------------------------------
#  compile command, db export
# ---------------------------------------------------------------------------


def test_compile_command_exists():
    """The compile command is registered in the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli, ["compile", "--help"])
    assert result.exit_code == 0
    assert "curate" in result.output.lower() or "compile" in result.output.lower()


def test_db_export_no_editions(tmp_path, monkeypatch):
    """db export reports error when no editions exist."""
    import copy

    from offscroll.config import DEFAULTS

    monkeypatch.setenv("HOME", str(tmp_path))
    offscroll_dir = tmp_path / ".offscroll"
    offscroll_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = offscroll_dir / "config.yaml"
    config = copy.deepcopy(DEFAULTS)
    config["feeds"] = {
        "rss": [{"url": "https://example.com/feed"}],
        "mastodon": [],
        "bluesky": [],
        "opml_files": [],
    }
    config["output"]["data_dir"] = str(data_dir)

    import yaml as yaml_mod

    with open(config_path, "w") as f:
        yaml_mod.dump(config, f)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(config_path), "db", "export", str(tmp_path / "out.json")],
    )
    assert result.exit_code != 0 or "No editions found" in result.output


def test_db_export_copies_file(tmp_path, monkeypatch):
    """db export copies latest edition JSON to the specified path."""
    import copy
    import json

    from offscroll.config import DEFAULTS

    monkeypatch.setenv("HOME", str(tmp_path))
    offscroll_dir = tmp_path / ".offscroll"
    offscroll_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = offscroll_dir / "config.yaml"
    config = copy.deepcopy(DEFAULTS)
    config["feeds"] = {
        "rss": [{"url": "https://example.com/feed"}],
        "mastodon": [],
        "bluesky": [],
        "opml_files": [],
    }
    config["output"]["data_dir"] = str(data_dir)

    import yaml as yaml_mod

    with open(config_path, "w") as f:
        yaml_mod.dump(config, f)

    # Create a fake edition file
    edition_data = {
        "edition": {
            "date": "2026-03-05",
            "title": "Export Test",
            "subtitle": "Vol. 1",
        },
        "sections": [],
        "pull_quotes": [],
    }
    edition_file = data_dir / "edition-2026-03-05.json"
    with open(edition_file, "w") as f:
        json.dump(edition_data, f)

    out_path = tmp_path / "exported.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(config_path), "db", "export", str(out_path)],
    )
    assert result.exit_code == 0
    assert out_path.exists()
    exported = json.loads(out_path.read_text())
    assert exported["edition"]["title"] == "Export Test"
