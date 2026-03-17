"""OffScroll CLI entry point.

Usage:
    offscroll run             Full pipeline: ingest -> curate -> render
    offscroll status          Show current state at a glance
    offscroll open            Open latest newspaper in default viewer
    offscroll ingest          Poll feeds and store new items
    offscroll curate          Run full curation pipeline
    offscroll render pdf      Generate newspaper PDF
    offscroll feeds list      Show feeds with health indicators
    offscroll feeds add-starters   Add curated starter feeds
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
@click.option("--quiet", "-q", is_flag=True, help="Suppress logs; show only summary and errors")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, verbose: bool, quiet: bool) -> None:
    """OffScroll -- your personal newspaper from the open web."""
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet

    # Set up logging before anything else
    from offscroll.logging import setup_logging

    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = None  # None = use config level
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

    # Render HTML (reliable fallback)
    try:
        ctx.invoke(render_newspaper_html)
    except Exception as exc:
        logger.warning("HTML render failed: %s", exc)

    # Attempt PDF (graceful fallback)
    try:
        ctx.invoke(render_pdf)
    except Exception as exc:
        logger.warning("PDF render failed (HTML is available): %s", exc)

    if config.get("email", {}).get("enabled"):
        ctx.invoke(render_email, send=True)
    click.echo("Compilation complete.")


@cli.command()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Full pipeline: ingest -> curate -> render all enabled outputs."""
    config = _require_config(ctx)
    quiet = ctx.obj.get("quiet", False)
    pdf_ok = False
    html_ok = False
    email_ok = False
    pdf_path = None
    html_path = None

    try:
        ctx.invoke(ingest)
        ctx.invoke(curate)
    except Exception:
        _pipeline_fail("Curation failed. See ~/.offscroll/offscroll.log", quiet)
        sys.exit(1)

    # Render HTML first (reliable)
    try:
        html_path = ctx.invoke(render_newspaper_html)
        html_ok = True
    except Exception as exc:
        logger.warning("HTML render failed: %s", exc)

    # Attempt PDF (may fail due to WeasyPrint bugs)
    try:
        pdf_path = ctx.invoke(render_pdf)
        pdf_ok = True
    except Exception as exc:
        logger.warning("PDF render failed (HTML is available): %s", exc)

    # Email if enabled
    if config.get("email", {}).get("enabled"):
        try:
            ctx.invoke(render_email, send=True)
            email_ok = True
        except Exception as exc:
            logger.warning("Email send failed: %s", exc)

    # Print summary
    _print_pipeline_summary(
        config,
        pdf_path,
        html_path,
        pdf_ok,
        html_ok,
        email_ok,
        quiet,
    )

    if not pdf_ok and not html_ok:
        sys.exit(1)


@cli.group()
def feeds() -> None:
    """Manage feed sources."""
    pass


@feeds.command("list")
@click.pass_context
def feeds_list(ctx: click.Context) -> None:
    """List configured feeds with health indicators."""
    config = _require_config(ctx)
    from offscroll.ingestion.store import get_feed_health

    feeds = get_feed_health(config)
    if not feeds:
        click.echo("No feeds configured. Run 'offscroll feeds add <url>' to add one.")
        return
    for f in feeds:
        icon = {"ok": "\u2713", "stale": "!", "empty": "\u2717"}.get(f["status"], "?")
        name = f["name"] or f["url"]
        last = ""
        if f["last_ingested"]:
            last = f" (last: {_relative_time(f['last_ingested'])})"
        click.echo(f"  {icon} {name:<40s} {f['item_count']:>4d} items{last}")


@feeds.command("add")
@click.argument("url")
@click.option("--name", default=None, help="Human-readable feed name")
def feeds_add(url: str, name: str | None) -> None:
    """Add an RSS/Atom feed URL to the config."""
    import yaml

    # Find the config file (operate directly on YAML, not the loaded config)
    config_path = Path.home() / ".offscroll" / "config.yaml"

    if not config_path.exists():
        click.echo(
            f"Config file not found at {config_path}. Run 'offscroll setup' first.",
            err=True,
        )
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


# ---------------------------------------------------------------------------
#  Starter feed pack (task #74)
# ---------------------------------------------------------------------------

STARTER_FEEDS = [
    {"url": "https://www.quantamagazine.org/feed", "name": "Quanta Magazine"},
    {"url": "https://www.themarginalian.org/feed/", "name": "The Marginalian"},
    {"url": "https://hnrss.org/best", "name": "Hacker News Best"},
    {"url": "https://arstechnica.com/feed/", "name": "Ars Technica"},
    {"url": "https://nautil.us/feed/", "name": "Nautilus"},
    {"url": "https://www.theverge.com/rss/index.xml", "name": "The Verge"},
    {"url": "https://longreads.com/feed/", "name": "Longreads"},
    {"url": "https://feeds.aeonmedia.co/aeon/feed", "name": "Aeon"},
    {"url": "https://kottke.org/feed/json", "name": "Kottke.org"},
    {"url": "https://www.sciencedaily.com/rss/all.xml", "name": "ScienceDaily"},
    {"url": "https://www.newyorker.com/feed/everything", "name": "The New Yorker"},
    {"url": "https://feeds.feedburner.com/brainpickings/rss", "name": "Brain Pickings"},
    {"url": "https://pluralistic.net/feed/", "name": "Pluralistic (Cory Doctorow)"},
    {"url": "https://www.openculture.com/feed", "name": "Open Culture"},
    {"url": "https://lwn.net/headlines/rss", "name": "LWN.net"},
]


@feeds.command("add-starters")
def feeds_add_starters() -> None:
    """Add a curated pack of ~15 popular feeds across domains.

    Great for first-run: get a newspaper immediately without
    hunting for RSS URLs. Remove later with 'feeds remove-starters'.
    """
    import yaml

    config_path = Path.home() / ".offscroll" / "config.yaml"
    if not config_path.exists():
        click.echo(
            f"Config file not found at {config_path}. Run 'offscroll setup' first.",
            err=True,
        )
        sys.exit(1)

    with open(config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    if "feeds" not in raw_config:
        raw_config["feeds"] = {}
    if "rss" not in raw_config["feeds"] or raw_config["feeds"]["rss"] is None:
        raw_config["feeds"]["rss"] = []

    existing_urls = set()
    for entry in raw_config["feeds"]["rss"]:
        if isinstance(entry, dict):
            existing_urls.add(entry.get("url", ""))
        else:
            existing_urls.add(entry)

    added = 0
    for starter in STARTER_FEEDS:
        if starter["url"] not in existing_urls:
            raw_config["feeds"]["rss"].append(dict(starter))
            added += 1

    with open(config_path, "w") as f:
        yaml.dump(raw_config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Added {added} starter feeds ({len(STARTER_FEEDS) - added} already present).")
    if added:
        click.echo("Run 'offscroll ingest' then 'offscroll run' to generate your first newspaper.")


@feeds.command("remove-starters")
def feeds_remove_starters() -> None:
    """Remove the starter feed pack, keeping any feeds you added yourself."""
    import yaml

    config_path = Path.home() / ".offscroll" / "config.yaml"
    if not config_path.exists():
        click.echo("Config file not found.", err=True)
        sys.exit(1)

    with open(config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    rss = raw_config.get("feeds", {}).get("rss", [])
    starter_urls = {s["url"] for s in STARTER_FEEDS}

    before = len(rss)
    rss = [
        entry
        for entry in rss
        if (isinstance(entry, dict) and entry.get("url", "") not in starter_urls)
        or (isinstance(entry, str) and entry not in starter_urls)
    ]
    raw_config["feeds"]["rss"] = rss
    removed = before - len(rss)

    with open(config_path, "w") as f:
        yaml.dump(raw_config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Removed {removed} starter feeds. {len(rss)} custom feeds remain.")


# ---------------------------------------------------------------------------
#  offscroll open
# ---------------------------------------------------------------------------


@cli.command("open")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pdf", "html"]),
    default=None,
    help="Preferred format (default: PDF if available, else HTML).",
)
@click.pass_context
def open_newspaper(ctx: click.Context, fmt: str | None) -> None:
    """Open the latest newspaper in your default viewer."""
    config = _require_config(ctx)
    import webbrowser

    data_dir = Path(config["output"]["data_dir"])

    # Find latest newspaper files
    pdfs = sorted(data_dir.glob("newspaper-*.pdf"), reverse=True)
    htmls = sorted(data_dir.glob("newspaper-*.html"), reverse=True)

    target = None
    if fmt == "pdf":
        target = pdfs[0] if pdfs else None
    elif fmt == "html":
        target = htmls[0] if htmls else None
    else:
        # Prefer PDF, fall back to HTML
        target = (pdfs[0] if pdfs else None) or (htmls[0] if htmls else None)

    if target is None:
        click.echo("No newspaper found. Run 'offscroll run' first.", err=True)
        sys.exit(1)

    # Try to open; on headless systems, print path instead
    try:
        opened = webbrowser.open(target.as_uri())
        if not opened:
            raise OSError("No browser available")
    except Exception:
        click.echo("Could not open viewer. Copy this file to your device:")
        click.echo(f"  {target}")
        return

    click.echo(f"Opened: {target}")


# ---------------------------------------------------------------------------
#  offscroll status
# ---------------------------------------------------------------------------


@cli.command("status")
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current OffScroll state at a glance."""
    config = _require_config(ctx)
    from offscroll.ingestion.store import get_db_stats, get_feed_health, get_latest_edition_info

    stats = get_db_stats(config)
    health = get_feed_health(config)
    latest = get_latest_edition_info(config)

    # Feeds summary
    active = sum(1 for f in health if f["status"] == "ok")
    stale = sum(1 for f in health if f["status"] == "stale")
    empty = sum(1 for f in health if f["status"] == "empty")
    click.echo(
        f"Feeds:      {active} active"
        + (f", {stale} stale" if stale else "")
        + (f", {empty} empty" if empty else "")
        + f" ({stats['total_items']} items stored)"
    )

    # Latest edition
    if latest:
        data_dir = Path(config["output"]["data_dir"])
        # Try to load the edition JSON for metadata
        json_path = latest.get("json_path")
        edition_title = None
        edition_subtitle = None
        edition_date = None
        article_count = latest["item_count"]
        if json_path:
            import json

            try:
                jp = Path(json_path)
                if not jp.is_absolute():
                    jp = data_dir / jp
                with open(jp) as f:
                    ed_data = json.load(f)
                edition_meta = ed_data.get("edition", {})
                edition_title = edition_meta.get("title")
                edition_subtitle = edition_meta.get("subtitle")
                edition_date = edition_meta.get("date")
            except Exception:
                pass

        if edition_title:
            click.echo(f"Latest:     {edition_title} {edition_subtitle or ''}")
        else:
            click.echo(f"Latest:     Edition {latest['edition_id']}")

        if edition_date:
            click.echo(f"            {article_count} articles | {edition_date}")

        # Show file paths
        if edition_date:
            pdf_path = data_dir / f"newspaper-{edition_date}.pdf"
            html_path = data_dir / f"newspaper-{edition_date}.html"
            if pdf_path.exists():
                click.echo(f"            PDF:  {pdf_path}")
            if html_path.exists():
                click.echo(f"            HTML: {html_path}")
    else:
        click.echo("Latest:     No editions yet. Run 'offscroll run' to create one.")

    click.echo(f"Editions:   {stats['total_editions']} total")


# ---------------------------------------------------------------------------
#  Pipeline summary helper
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _print_pipeline_summary(
    config: dict,
    pdf_path: Path | None,
    html_path: Path | None,
    pdf_ok: bool,
    html_ok: bool,
    email_ok: bool,
    quiet: bool,
) -> None:
    """Print the post-pipeline summary block."""
    from offscroll.ingestion.store import get_latest_edition_info

    latest = get_latest_edition_info(config)
    title = config.get("newspaper", {}).get("title", "OffScroll")

    # Build one-line summary for quiet/cron mode
    article_count = latest["item_count"] if latest else 0
    subtitle = ""
    if latest and latest.get("json_path"):
        import json

        try:
            jp = Path(latest["json_path"])
            data_dir = Path(config["output"]["data_dir"])
            if not jp.is_absolute():
                jp = data_dir / jp
            with open(jp) as f:
                ed_data = json.load(f)
            subtitle = ed_data.get("edition", {}).get("subtitle", "")
        except Exception:
            pass

    formats = []
    if pdf_ok:
        formats.append("PDF")
    if html_ok:
        formats.append("HTML")
    if email_ok:
        formats.append("email")
    fmt_str = " + ".join(formats) if formats else "none"

    one_line = f"OK: {title} {subtitle} ({article_count} articles, {fmt_str})"
    if not pdf_ok and html_ok:
        one_line = f"WARN: {title} {subtitle} ({article_count} articles, PDF failed, HTML OK)"
    elif not pdf_ok and not html_ok:
        one_line = "FAIL: No output generated. See ~/.offscroll/offscroll.log"

    if quiet:
        click.echo(one_line)
        return

    # Full summary block
    click.echo("")
    click.echo("\u2501" * 50)
    click.echo(f"  {title} \u2014 {subtitle}" if subtitle else f"  {title}")
    click.echo(f"  {article_count} articles")
    if pdf_ok and pdf_path:
        click.echo(f"  PDF:  {pdf_path}")
    elif not pdf_ok:
        click.echo("  PDF:  failed (WeasyPrint error)")
    if html_ok and html_path:
        click.echo(f"  HTML: {html_path}")
    if email_ok:
        click.echo("  Email sent")
    click.echo("\u2501" * 50)


def _pipeline_fail(msg: str, quiet: bool) -> None:
    """Print a pipeline failure message."""
    if quiet:
        click.echo(f"FAIL: {msg}", err=True)
    else:
        click.echo(f"\nPipeline failed: {msg}", err=True)


def _relative_time(iso_str: str) -> str:
    """Convert an ISO datetime string to a human-readable relative time."""
    from datetime import UTC, datetime, timedelta

    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - dt
        if delta < timedelta(hours=1):
            mins = max(1, int(delta.total_seconds() / 60))
            return f"{mins}m ago"
        if delta < timedelta(days=1):
            hours = int(delta.total_seconds() / 3600)
            return f"{hours}h ago"
        days = delta.days
        return f"{days}d ago"
    except Exception:
        return iso_str


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
