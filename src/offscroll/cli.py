"""OffScroll CLI entry point.

Usage:
    offscroll ingest          Poll feeds and store new items
    offscroll curate          Run full curation pipeline
    offscroll render pdf      Generate newspaper PDF
    offscroll run             Full pipeline: ingest -> curate -> render
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from offscroll.config import load_config


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config.yaml",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, verbose: bool) -> None:
    """OffScroll -- your personal newspaper from the open web."""
    ctx.ensure_object(dict)

    # Set up logging before anything else
    from offscroll.logging import setup_logging

    level = logging.DEBUG if verbose else None  # None = use config level
    default_path = Path.home() / ".offscroll" / "config.yaml"
    effective_path = config_path or (default_path if default_path.exists() else None)
    if effective_path:
        try:
            config = load_config(effective_path)
            if level is None:
                level = getattr(logging, config.get("logging", {}).get("level", "INFO"))
            log_file = config.get("logging", {}).get("file")
            setup_logging(level=level, log_file=log_file)
            ctx.obj["config"] = config
        except SystemExit:
            # Config exists but failed validation -- allow commands that
            # don't need a valid config (setup, init, feeds add) to proceed
            setup_logging(level=level or logging.INFO)
            ctx.obj["config"] = None
    else:
        # No config yet -- only 'init' should work
        setup_logging(level=level or logging.INFO)
        ctx.obj["config"] = None


@cli.command()
@click.pass_context
def setup(ctx: click.Context) -> None:
    """Interactive setup wizard for OffScroll.

    Walks through:
    1. Create ~/.offscroll directory
    2. Newspaper name and settings
    3. RSS feed URLs (manual entry)
    4. OPML import (optional)
    5. Mastodon credentials (optional)
    6. Bluesky credentials (optional)
    7. Ollama connection check
    8. Email settings (optional)
    9. Write config.yaml
    """
    import yaml

    offscroll_dir = Path.home() / ".offscroll"
    offscroll_dir.mkdir(exist_ok=True)
    config_path = offscroll_dir / "config.yaml"

    if config_path.exists() and not click.confirm(f"Config exists at {config_path}. Overwrite?"):
        return

    config = {}

    # 1. Newspaper settings
    click.echo("\n--- Newspaper Settings ---")
    title = click.prompt(
        "Newspaper name",
        default="The Morning Dispatch",
    )
    page_target = click.prompt(
        "Pages per issue",
        default=10,
        type=int,
    )
    config["newspaper"] = {
        "title": title,
        "page_target": page_target,
    }

    # 2. RSS feeds
    click.echo("\n--- RSS Feeds ---")
    click.echo("Enter feed URLs one per line. Empty line to finish.")
    rss_feeds = []
    while True:
        url = click.prompt(
            "Feed URL",
            default="",
            show_default=False,
        )
        if not url:
            break
        name = click.prompt(
            "  Name (optional)",
            default="",
            show_default=False,
        )
        entry = {"url": url}
        if name:
            entry["name"] = name
        rss_feeds.append(entry)

    # 3. OPML import
    opml_files = []
    if click.confirm(
        "Import feeds from an OPML file?",
        default=False,
    ):
        opml_path = click.prompt("Path to OPML file")
        opml_files.append(opml_path)

    # 4. Mastodon
    mastodon = []
    if click.confirm(
        "Add a Mastodon account?",
        default=False,
    ):
        instance = click.prompt(
            "Instance URL",
            default="https://mastodon.social",
        )
        token_env = click.prompt(
            "Env var for access token",
            default="OFFSCROLL_MASTODON_TOKEN",
        )
        timeline = click.prompt(
            "Timeline (home/public)",
            default="home",
        )
        mastodon.append(
            {
                "instance": instance,
                "access_token_env": token_env,
                "timeline": timeline,
            }
        )

    # 5. Bluesky
    bluesky = []
    if click.confirm(
        "Add a Bluesky account?",
        default=False,
    ):
        handle = click.prompt("Handle")
        pw_env = click.prompt(
            "Env var for app password",
            default="OFFSCROLL_BLUESKY_PASSWORD",
        )
        bluesky.append(
            {
                "handle": handle,
                "app_password_env": pw_env,
                "feed": "timeline",
            }
        )

    config["feeds"] = {
        "rss": rss_feeds,
        "mastodon": mastodon,
        "bluesky": bluesky,
        "opml_files": opml_files,
    }

    # 6. Ollama check
    click.echo("\n--- Ollama ---")
    ollama_url = click.prompt(
        "Ollama URL",
        default="http://localhost:11434",
    )
    config["embedding"] = {
        "provider": "ollama",
        "ollama_url": ollama_url,
    }
    config["curation"] = {
        "model": "ollama",
        "ollama_url": ollama_url,
    }

    # Test Ollama connection
    try:
        import httpx

        r = httpx.get(
            f"{ollama_url}/api/tags",
            timeout=5.0,
        )
        if r.status_code == 200:
            models = r.json().get("models", [])
            names = [m["name"] for m in models]
            click.echo(f"Ollama connected. Models: {', '.join(names) or 'none'}")
            if not any("nomic" in n for n in names):
                click.echo("Note: nomic-embed-text not found. Run: ollama pull nomic-embed-text")
        else:
            click.echo(f"Warning: Ollama responded but returned status {r.status_code}")
    except Exception:
        click.echo(
            f"Warning: Could not connect to Ollama at {ollama_url}. Make sure Ollama is running."
        )

    # 7. Email (optional)
    if click.confirm(
        "\nEnable email digest?",
        default=False,
    ):
        smtp_host = click.prompt("SMTP host")
        smtp_port = click.prompt("SMTP port", default=587, type=int)
        from_addr = click.prompt("From address")
        to_addr = click.prompt("To address")
        config["email"] = {
            "enabled": True,
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
            "from_address": from_addr,
            "to_addresses": [to_addr],
        }
    else:
        config["email"] = {"enabled": False}

    # 8. Write config
    with open(config_path, "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    click.echo(f"\nConfig written to {config_path}")
    click.echo("Next steps:")
    click.echo("  offscroll ingest   # poll your feeds")
    click.echo("  offscroll run      # full pipeline")


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Create config (alias for 'setup')."""
    ctx.invoke(setup)


@cli.command()
@click.pass_context
def ingest(ctx: click.Context) -> None:
    """Poll all configured feeds and store new items."""
    config = _require_config(ctx)
    from offscroll.ingestion.feeds import ingest_all_feeds

    total = 0

    # RSS/Atom
    total += ingest_all_feeds(config)

    # Mastodon
    if config.get("feeds", {}).get("mastodon"):
        try:
            from offscroll.ingestion.fediverse import ingest_mastodon

            total += ingest_mastodon(config)
        except ImportError:
            click.echo("Mastodon.py not installed. Skipping Mastodon feeds.")

    # Bluesky
    if config.get("feeds", {}).get("bluesky"):
        try:
            from offscroll.ingestion.fediverse import ingest_bluesky

            total += ingest_bluesky(config)
        except ImportError:
            click.echo("atproto not installed. Skipping Bluesky feeds.")

    click.echo(f"Ingested {total} new items.")


@cli.command()
@click.pass_context
def embed(ctx: click.Context) -> None:
    """Generate embeddings for un-embedded items."""
    config = _require_config(ctx)
    from offscroll.ingestion.embeddings import embed_items
    from offscroll.ingestion.store import (
        get_items_for_embedding,
        update_embeddings,
    )

    items = get_items_for_embedding(config)
    if not items:
        click.echo("No un-embedded items found.")
        return
    embed_items(items, config)
    count = update_embeddings(config, items)
    click.echo(f"Embedded {count} items.")


@cli.command()
@click.pass_context
def cluster(ctx: click.Context) -> None:
    """Cluster embedded items."""
    config = _require_config(ctx)
    from offscroll.ingestion.clustering import cluster_items
    from offscroll.ingestion.store import (
        get_items_for_clustering,
        update_cluster_ids,
    )

    items = get_items_for_clustering(config)
    if not items:
        click.echo("No embedded items found for clustering.")
        return
    cluster_items(items, config)
    count = update_cluster_ids(config, items)
    click.echo(f"Clustered {count} items.")


@cli.command()
@click.option(
    "--fresh",
    is_flag=True,
    default=False,
    help="Ignore edition history and skip recording. For dev/test only.",
)
@click.pass_context
def curate(ctx: click.Context, fresh: bool) -> None:
    """Run the full curation pipeline: embed, cluster, select, editorial."""
    config = _require_config(ctx)
    from offscroll.curation.selection import curate_edition

    if fresh:
        click.echo("FRESH MODE: full item pool, edition will NOT be recorded.")
    edition_path = curate_edition(config, fresh=fresh)
    click.echo(f"Curated edition written to {edition_path}")


@cli.group()
def render() -> None:
    """Render a curated edition to an output format."""
    pass


@render.command("pdf")
@click.option(
    "--edition",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to curated edition JSON (default: latest)",
)
@click.pass_context
def render_pdf(ctx: click.Context, edition: Path | None) -> None:
    """Render curated edition to newspaper PDF (via styled HTML + WeasyPrint)."""
    config = _require_config(ctx)
    from offscroll.layout.renderer import render_newspaper_pdf as do_render

    pdf_path = do_render(config, edition_path=edition)
    click.echo(f"PDF written to {pdf_path}")


@render.command("newspaper-html")
@click.option(
    "--edition",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to curated edition JSON (default: latest)",
)
@click.pass_context
def render_newspaper_html(ctx: click.Context, edition: Path | None) -> None:
    """Render curated edition to styled newspaper HTML (browser-viewable, skip PDF)."""
    config = _require_config(ctx)
    from offscroll.layout.renderer import render_newspaper_html as do_render

    html_path = do_render(config, edition_path=edition)
    click.echo(f"Newspaper HTML written to {html_path}")


@render.command("email")
@click.option(
    "--edition",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to curated edition JSON (default: latest)",
)
@click.option("--send/--no-send", default=False, help="Actually send the email")
@click.pass_context
def render_email(ctx: click.Context, edition: Path | None, send: bool) -> None:
    """Render curated edition to email digest."""
    config = _require_config(ctx)
    from offscroll.curation.digest import render_digest

    html_path = render_digest(config, edition_path=edition, send=send)
    click.echo(f"Email digest written to {html_path}")
    if send:
        click.echo("Email sent.")


@cli.command("compile")
@click.pass_context
def compile_edition(ctx: click.Context) -> None:
    """Compile the weekly edition: curate, render, and deliver.

    Runs: curate -> render PDF -> send email (if enabled).
    Does NOT ingest. Use 'offscroll run' for the full pipeline
    including ingestion, or run 'offscroll ingest' separately
    (e.g. hourly via cron).
    """
    config = _require_config(ctx)
    ctx.invoke(curate)
    ctx.invoke(render_pdf)
    if config.get("email", {}).get("enabled"):
        ctx.invoke(render_email, send=True)
    click.echo("Compilation complete.")


@cli.command()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Full pipeline: ingest -> curate -> render all enabled outputs."""
    config = _require_config(ctx)
    ctx.invoke(ingest)
    ctx.invoke(curate)
    ctx.invoke(render_pdf)
    if config.get("email", {}).get("enabled"):
        ctx.invoke(render_email, send=True)
    click.echo("Pipeline complete.")


@cli.group()
def feeds() -> None:
    """Manage feed sources."""
    pass


@feeds.command("list")
@click.pass_context
def feeds_list(ctx: click.Context) -> None:
    """List configured feeds and their status."""
    config = _require_config(ctx)
    from offscroll.ingestion.store import get_feed_stats

    stats = get_feed_stats(config)
    for s in stats:
        click.echo(f"  {s['name'] or s['url']}  ({s['source_type']})  {s['item_count']} items")


@feeds.command("add")
@click.argument("url")
@click.option("--name", default=None, help="Human-readable feed name")
def feeds_add(url: str, name: str | None) -> None:
    """Add an RSS/Atom feed URL to the config."""
    import yaml

    # Find the config file (operate directly on YAML, not the loaded config)
    config_path = Path.home() / ".offscroll" / "config.yaml"

    if not config_path.exists():
        click.echo(f"Config file not found at {config_path}. Run 'offscroll setup' first.", err=True)
        sys.exit(1)

    # Load the raw config
    with open(config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    # Ensure feeds.rss list exists
    if "feeds" not in raw_config:
        raw_config["feeds"] = {}
    if "rss" not in raw_config["feeds"] or raw_config["feeds"]["rss"] is None:
        raw_config["feeds"]["rss"] = []

    # Check for duplicates
    existing_urls = set()
    for entry in raw_config["feeds"]["rss"]:
        if isinstance(entry, dict):
            existing_urls.add(entry.get("url", ""))
        else:
            existing_urls.add(entry)

    if url in existing_urls:
        click.echo(f"Feed already exists: {url}")
        return

    # Add the new feed
    new_entry: dict[str, str] = {"url": url}
    if name:
        new_entry["name"] = name
    raw_config["feeds"]["rss"].append(new_entry)

    # Write back
    with open(config_path, "w") as f:
        yaml.dump(raw_config, f, default_flow_style=False, sort_keys=False)

    display_name = f"{name} ({url})" if name else url
    click.echo(f"Added feed: {display_name}")


@feeds.command("import")
@click.argument("opml", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def feeds_import(ctx: click.Context, opml: Path) -> None:
    """Import feeds from an OPML file."""
    config = _require_config(ctx)
    from offscroll.ingestion.opml import register_opml_feeds

    count = register_opml_feeds(config, opml)
    click.echo(f"Imported {count} new feeds.")


@cli.group()
def db() -> None:
    """Database operations."""
    pass


@db.command("stats")
@click.pass_context
def db_stats(ctx: click.Context) -> None:
    """Show database statistics."""
    config = _require_config(ctx)
    from offscroll.ingestion.store import get_db_stats

    stats = get_db_stats(config)
    click.echo(f"Total items:    {stats['total_items']}")
    click.echo(f"Total feeds:    {stats['total_feeds']}")
    click.echo(f"Total editions: {stats['total_editions']}")


@db.command("export")
@click.argument("path", type=click.Path(path_type=Path))
@click.pass_context
def db_export(ctx: click.Context, path: Path) -> None:
    """Export latest curated edition JSON to a file."""
    import shutil

    config = _require_config(ctx)
    data_dir = Path(config["output"]["data_dir"])
    editions = sorted(data_dir.glob("edition-*.json"), reverse=True)
    if not editions:
        click.echo("No editions found. Run 'offscroll curate' first.", err=True)
        sys.exit(1)
    shutil.copy2(editions[0], path)
    click.echo(f"Exported {editions[0].name} to {path}")


def _require_config(ctx: click.Context) -> dict:
    """Get config from context or fail with a helpful message."""
    config = ctx.obj.get("config")
    if config is None:
        click.echo("No config file found. Run 'offscroll init' first.", err=True)
        sys.exit(1)
    return config


def main() -> None:
    """Entry point for the 'offscroll' command."""
    cli(obj={})


if __name__ == "__main__":
    main()
