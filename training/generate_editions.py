#!/usr/bin/env python3
"""Training set generation pipeline for OffScroll layout optimization.

Generates ~100 newspaper editions from diverse feed configs, renders
them via Typst, extracts per-page PNGs, and organizes spreads for
Neville's grading protocol.

Pipeline steps per config:
  1. Fetch current RSS content (lightweight, no DB/Ollama)
  2. Build CuratedEdition with randomized item ordering
  3. Render via Typst to PDF
  4. Extract per-page PNGs via PyMuPDF
  5. Save page metadata and edition context

Usage:
    cd offscroll/
    .venv/bin/python training/generate_editions.py [--max-configs N] [--workers N]

IRAS Task #172 — Belle
"""

from __future__ import annotations

import hashlib
import html as html_module
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import fitz  # PyMuPDF
import httpx
import yaml

# Add src to path so we can import offscroll
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from offscroll.ingestion.feeds import parse_feed
from offscroll.layout.typst_renderer import build_typst_markup, render_typst_pdf
from offscroll.models import (
    CuratedEdition,
    CuratedImage,
    CuratedItem,
    CuratedThread,
    EditionMeta,
    LayoutHint,
    PullQuote,
    Section,
)

logger = logging.getLogger("training.generate")

CONFIGS_DIR = Path(__file__).parent / "configs"
EDITIONS_DIR = Path(__file__).parent / "editions"
SPREADS_DIR = EDITIONS_DIR / "spreads"

# Typst assets
TYPST_DIR = Path(__file__).parent.parent / "src" / "offscroll" / "layout" / "typst"
FONTS_DIR = Path(__file__).parent.parent / "src" / "offscroll" / "layout" / "fonts"

# Feed fetch settings
FETCH_TIMEOUT = 15.0
FETCH_WORKERS = 10

# Layout hints assigned by word count thresholds
FEATURE_MIN_WORDS = 500
BRIEF_MAX_WORDS = 100

# PNG extraction DPI
PAGE_DPI = 150

# Regex to strip residual HTML tags from content
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    """Aggressively strip HTML residue from feed content text."""
    if not text:
        return text
    # Strip any remaining HTML tags
    text = _HTML_TAG_RE.sub(" ", text)
    # Collapse whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    # Strip HTML entity leftovers
    text = html_module.unescape(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 1: Fetch RSS content
# ---------------------------------------------------------------------------

def fetch_feed(url: str, name: str) -> list[dict]:
    """Fetch a single RSS/Atom feed and return parsed items as dicts."""
    try:
        resp = httpx.get(url, timeout=FETCH_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        parsed = parse_feed(resp.text, url)
        items = []
        for fi in parsed.items:
            clean_content = _clean_text(fi.content_text)
            items.append({
                "item_id": fi.item_id,
                "title": fi.title,
                "author": fi.author or name,
                "content_text": clean_content,
                "word_count": len(clean_content.split()) if clean_content else 0,
                "images": [
                    {"url": img.url, "alt_text": img.alt_text}
                    for img in fi.images
                ],
                "feed_name": name,
                "feed_url": url,
                "published_at": fi.published_at.isoformat() if fi.published_at else None,
            })
        return items
    except Exception as e:
        logger.warning("Failed to fetch %s (%s): %s", name, url, e)
        return []


def fetch_all_feeds(config: dict) -> list[dict]:
    """Fetch all feeds from a config concurrently."""
    feeds = config.get("feeds", {}).get("rss", [])
    all_items = []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {
            pool.submit(fetch_feed, f["url"], f.get("name", f["url"])): f
            for f in feeds
        }
        for future in as_completed(futures):
            all_items.extend(future.result())
    return all_items


# ---------------------------------------------------------------------------
# Step 2: Build CuratedEdition with randomized ordering
# ---------------------------------------------------------------------------

def select_pull_quote(item_text: str, item_id: str, author: str) -> PullQuote | None:
    """Extract a heuristic pull quote from article text."""
    if not item_text or len(item_text) < 200:
        return None
    sentences = [
        s.strip() for s in item_text.replace("\n", " ").split(".")
        if 40 < len(s.strip()) < 200
    ]
    if not sentences:
        return None
    # Pick a sentence from the middle third (more likely to be interesting)
    mid_start = len(sentences) // 3
    mid_end = max(mid_start + 1, 2 * len(sentences) // 3)
    chosen = random.choice(sentences[mid_start:mid_end])
    return PullQuote(
        text=chosen.strip() + ".",
        attribution=author,
        source_item_id=item_id,
    )


def build_edition(
    config: dict,
    items: list[dict],
    config_id: str,
    rng: random.Random,
) -> CuratedEdition | None:
    """Build a CuratedEdition from fetched items with random ordering."""
    if len(items) < 5:
        logger.warning("Config %s: only %d items, skipping", config_id, len(items))
        return None

    newspaper = config.get("newspaper", {})
    page_target = newspaper.get("page_target", 10)

    # Shuffle items for variety
    rng.shuffle(items)

    # Assign layout hints by word count
    curated_items = []
    feature_assigned = False
    for item in items:
        wc = item["word_count"]
        if wc < 20:
            continue  # Too short to render

        if wc >= FEATURE_MIN_WORDS and not feature_assigned:
            hint = LayoutHint.FEATURE
            feature_assigned = True
        elif wc <= BRIEF_MAX_WORDS:
            hint = LayoutHint.BRIEF
        else:
            hint = LayoutHint.STANDARD

        ci = CuratedItem(
            item_id=item["item_id"],
            display_text=item["content_text"],
            author=item["author"],
            source_name=item["feed_name"],
            title=item.get("title"),
            images=[],  # No downloaded images for training
            layout_hint=hint,
            word_count=wc,
        )
        curated_items.append(ci)

    if len(curated_items) < 5:
        logger.warning("Config %s: only %d valid items after filter", config_id, len(curated_items))
        return None

    # Budget: ~3-5 items per page for standards, 1 feature = ~2 pages
    # Rough item budget based on page target
    items_budget = page_target * 3
    curated_items = curated_items[:items_budget]

    # If no feature was assigned, promote the longest standard
    if not any(ci.layout_hint == LayoutHint.FEATURE for ci in curated_items):
        longest = max(
            (ci for ci in curated_items if ci.layout_hint == LayoutHint.STANDARD),
            key=lambda ci: ci.word_count,
            default=None,
        )
        if longest:
            longest.layout_hint = LayoutHint.FEATURE

    # Group into sections by feed source (2-4 sections)
    feeds_used = list({ci.source_name for ci in curated_items})
    rng.shuffle(feeds_used)

    # Create 2-4 sections
    n_sections = min(max(2, len(feeds_used) // 3), 4)
    section_names = ["Front Page", "Features", "Analysis", "Briefs"][:n_sections]
    sections = [Section(heading=name) for name in section_names]

    # Distribute items across sections
    for i, ci in enumerate(curated_items):
        if ci.layout_hint == LayoutHint.FEATURE:
            sections[0].items.append(ci)
        elif ci.layout_hint == LayoutHint.BRIEF:
            sections[-1].items.append(ci)
        else:
            # Distribute standards across middle sections
            sec_idx = (i % max(1, n_sections - 1))
            if sec_idx == 0 and len(sections[0].items) > 0:
                sec_idx = 1 if n_sections > 1 else 0
            sections[min(sec_idx, n_sections - 1)].items.append(ci)

    # Remove empty sections
    sections = [s for s in sections if s.items]

    # Generate pull quotes (1 per 4 items, from longer articles)
    pull_quotes = []
    long_items = [ci for ci in curated_items if ci.word_count > 200]
    n_pqs = max(1, len(long_items) // 4)
    for ci in rng.sample(long_items, min(n_pqs, len(long_items))):
        pq = select_pull_quote(ci.display_text, ci.item_id, ci.author)
        if pq:
            pull_quotes.append(pq)

    title = newspaper.get("title", "The Daily")
    subtitle = newspaper.get("subtitle_pattern", "Training Edition {issue}").format(
        volume=1, issue=config_id.replace("training-", ""),
    )

    edition = CuratedEdition(
        edition=EditionMeta(
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            title=title,
            subtitle=subtitle,
        ),
        sections=sections,
        pull_quotes=pull_quotes,
        page_target=page_target,
    )
    return edition


# ---------------------------------------------------------------------------
# Step 3: Render via Typst
# ---------------------------------------------------------------------------

def render_edition_typst(
    edition: CuratedEdition,
    config: dict,
    output_dir: Path,
    config_id: str,
) -> tuple[Path | None, float]:
    """Render edition to PDF via Typst. Returns (pdf_path, render_seconds)."""
    typst_bin = shutil.which("typst")
    if typst_bin is None:
        raise FileNotFoundError("Typst CLI not found")

    output_dir.mkdir(parents=True, exist_ok=True)

    typ_path = output_dir / f"{config_id}.typ"
    pdf_path = output_dir / f"{config_id}.pdf"

    # Generate markup
    markup = build_typst_markup(edition, config)
    typ_path.write_text(markup)

    # Copy template files
    templates_dest = output_dir / "templates.typ"
    shutil.copy2(TYPST_DIR / "templates.typ", templates_dest)

    # Compile
    t0 = time.monotonic()
    result = subprocess.run(
        [typst_bin, "compile", "--font-path", str(FONTS_DIR), str(typ_path), str(pdf_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    render_time = time.monotonic() - t0

    # Clean up template copy
    templates_dest.unlink(missing_ok=True)

    if result.returncode != 0:
        logger.error("Typst failed for %s:\n%s", config_id, result.stderr[:500])
        return None, render_time

    return pdf_path, render_time


# ---------------------------------------------------------------------------
# Step 4: Extract per-page PNGs
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: Path, output_dir: Path) -> list[dict]:
    """Extract each page as PNG and return page metadata."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        # Render at target DPI
        mat = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        png_path = output_dir / f"page-{i + 1:03d}.png"
        pix.save(str(png_path))

        pages.append({
            "page_number": i + 1,
            "png_path": str(png_path.relative_to(EDITIONS_DIR)),
            "width_pt": page.rect.width,
            "height_pt": page.rect.height,
        })
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Step 5: Organize spreads for grading
# ---------------------------------------------------------------------------

def build_spreads(
    pages: list[dict],
    config_id: str,
    edition_idx: int,
) -> list[dict]:
    """Build spread units from page list.

    Spread rules:
    - Page 1 is always a solo spread
    - Interior pages pair as (2,3), (4,5), (6,7), ...
    - Final odd page is a solo spread
    """
    spreads = []
    n = len(pages)
    if n == 0:
        return spreads

    # Page 1: solo
    spread_id = f"s-{edition_idx:03d}-001"
    spreads.append({
        "spread_id": spread_id,
        "edition_id": config_id,
        "type": "solo",
        "pages": [pages[0]],
    })

    # Interior pairs
    i = 1
    spread_num = 2
    while i < n:
        if i + 1 < n:
            # Facing pair
            spread_id = f"s-{edition_idx:03d}-{spread_num:03d}"
            spreads.append({
                "spread_id": spread_id,
                "edition_id": config_id,
                "type": "spread",
                "pages": [pages[i], pages[i + 1]],
            })
            i += 2
        else:
            # Final odd page: solo
            spread_id = f"s-{edition_idx:03d}-{spread_num:03d}"
            spreads.append({
                "spread_id": spread_id,
                "edition_id": config_id,
                "type": "solo",
                "pages": [pages[i]],
            })
            i += 1
        spread_num += 1

    return spreads


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_one_config(
    config_path: Path,
    config_idx: int,
) -> dict:
    """Process a single training config end-to-end. Returns result dict."""
    config_id = config_path.stem  # e.g. "training-001"
    result = {
        "config_id": config_id,
        "config_path": str(config_path),
        "status": "failed",
        "error": None,
        "num_feeds": 0,
        "num_items_fetched": 0,
        "num_items_used": 0,
        "num_pages": 0,
        "num_spreads": 0,
        "render_time_s": 0.0,
        "page_target": 0,
        "archetype": "",
        "feeds_contributed": [],
    }

    try:
        with open(config_path) as f:
            raw = f.read()
            config = yaml.safe_load(raw)

        # Extract archetype from comment
        for line in raw.split("\n"):
            if "Archetype:" in line:
                result["archetype"] = line.split("Archetype:")[-1].strip()
                break

        feeds = config.get("feeds", {}).get("rss", [])
        result["num_feeds"] = len(feeds)
        result["page_target"] = config.get("newspaper", {}).get("page_target", 10)

        # Fetch feeds
        logger.info("[%s] Fetching %d feeds...", config_id, len(feeds))
        items = fetch_all_feeds(config)
        result["num_items_fetched"] = len(items)

        if len(items) < 5:
            result["error"] = f"Insufficient items: {len(items)}"
            return result

        # Track which feeds contributed
        result["feeds_contributed"] = list({it["feed_name"] for it in items})

        # Build edition with deterministic seed per config for reproducibility
        seed = int(hashlib.md5(config_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        edition = build_edition(config, items, config_id, rng)
        if edition is None:
            result["error"] = "Failed to build edition"
            return result

        total_items = sum(len(s.items) for s in edition.sections)
        result["num_items_used"] = total_items

        # Render
        edition_dir = EDITIONS_DIR / config_id
        edition_dir.mkdir(parents=True, exist_ok=True)

        # Override output data_dir to our training directory
        render_config = dict(config)
        render_config["output"] = {"data_dir": str(edition_dir)}

        pdf_path, render_time = render_edition_typst(
            edition, render_config, edition_dir, config_id,
        )
        result["render_time_s"] = round(render_time, 2)

        if pdf_path is None:
            result["error"] = "Typst compilation failed"
            return result

        # Extract pages
        pages_dir = edition_dir / "pages"
        pages_dir.mkdir(exist_ok=True)
        pages = extract_pages(pdf_path, pages_dir)
        result["num_pages"] = len(pages)

        # Build spreads
        spreads = build_spreads(pages, config_id, config_idx)
        result["num_spreads"] = len(spreads)

        # Save edition metadata
        edition_meta = {
            "config_id": config_id,
            "archetype": result["archetype"],
            "date_generated": datetime.now(UTC).isoformat(),
            "page_target": result["page_target"],
            "num_pages": len(pages),
            "num_items": total_items,
            "feeds_contributed": result["feeds_contributed"],
            "render_time_s": result["render_time_s"],
            "pages": pages,
            "spreads": spreads,
        }
        with open(edition_dir / "metadata.json", "w") as f:
            json.dump(edition_meta, f, indent=2)

        # Save the curated edition JSON for reference
        edition.to_json(edition_dir / "edition.json")

        result["status"] = "success"
        logger.info(
            "[%s] OK: %d pages, %d spreads, %.1fs render",
            config_id, len(pages), len(spreads), render_time,
        )

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error("[%s] Error: %s", config_id, traceback.format_exc())

    return result


def build_grading_manifest(results: list[dict]) -> dict:
    """Build the master grading manifest with randomized spreads."""
    all_spreads = []

    for r in results:
        if r["status"] != "success":
            continue
        meta_path = EDITIONS_DIR / r["config_id"] / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        for spread in meta["spreads"]:
            all_spreads.append({
                "spread_id": spread["spread_id"],
                "type": spread["type"],
                "page_pngs": [p["png_path"] for p in spread["pages"]],
                # Edition context stored separately for post-grading analysis
                "_edition_id": spread["edition_id"],
            })

    # Randomize spread order for blind grading
    rng = random.Random(42)
    rng.shuffle(all_spreads)

    # Split into grading batches and non-revealing manifest
    grading_units = []
    edition_map = {}  # spread_id -> edition_id (separate file)
    for i, sp in enumerate(all_spreads):
        grading_units.append({
            "grading_index": i + 1,
            "spread_id": sp["spread_id"],
            "type": sp["type"],
            "page_pngs": sp["page_pngs"],
        })
        edition_map[sp["spread_id"]] = sp["_edition_id"]

    return {
        "grading_units": grading_units,
        "edition_map": edition_map,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate OffScroll training editions")
    parser.add_argument("--max-configs", type=int, default=100,
                        help="Max configs to process (default: all 100)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel edition workers (default: 1, sequential)")
    parser.add_argument("--start", type=int, default=1,
                        help="Start from config N (1-indexed)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Verify typst is available
    if shutil.which("typst") is None:
        logger.error("Typst CLI not found. Install: https://github.com/typst/typst")
        sys.exit(1)

    # Find configs
    configs = sorted(CONFIGS_DIR.glob("training-*.yaml"))
    if args.start > 1:
        configs = configs[args.start - 1:]
    configs = configs[:args.max_configs]
    logger.info("Processing %d configs (start=%d)", len(configs), args.start)

    # Ensure output directories
    EDITIONS_DIR.mkdir(parents=True, exist_ok=True)
    SPREADS_DIR.mkdir(parents=True, exist_ok=True)

    # Process configs
    t_total = time.monotonic()
    results = []

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_one_config, cfg, i + 1): cfg
                for i, cfg in enumerate(configs)
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for i, cfg in enumerate(configs):
            results.append(process_one_config(cfg, i + 1))

    total_time = time.monotonic() - t_total

    # Sort results by config_id
    results.sort(key=lambda r: r["config_id"])

    # Build grading manifest
    manifest = build_grading_manifest(results)

    # Save grading units (no edition context — blind)
    grading_path = EDITIONS_DIR / "grading-manifest.json"
    with open(grading_path, "w") as f:
        json.dump(manifest["grading_units"], f, indent=2)

    # Save edition map separately (for post-grading analysis only)
    map_path = EDITIONS_DIR / "edition-map.json"
    with open(map_path, "w") as f:
        json.dump(manifest["edition_map"], f, indent=2)

    # Save full results
    results_path = EDITIONS_DIR / "generation-results.json"
    with open(results_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(UTC).isoformat(),
            "total_time_s": round(total_time, 1),
            "configs_processed": len(results),
            "results": results,
        }, f, indent=2)

    # Print summary
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    total_pages = sum(r["num_pages"] for r in success)
    total_spreads = sum(r["num_spreads"] for r in success)
    avg_render = (
        sum(r["render_time_s"] for r in success) / len(success)
        if success else 0
    )

    print(f"\n{'=' * 60}")
    print(f"Training Set Generation Complete")
    print(f"{'=' * 60}")
    print(f"Editions rendered:  {len(success)} / {len(results)}")
    print(f"Total pages:        {total_pages}")
    print(f"Total spreads:      {total_spreads}")
    print(f"Avg render time:    {avg_render:.1f}s")
    print(f"Total time:         {total_time:.0f}s")
    print(f"Failed:             {len(failed)}")
    if failed:
        for r in failed[:10]:
            print(f"  {r['config_id']}: {r['error']}")
    print(f"\nOutputs:")
    print(f"  Editions:         {EDITIONS_DIR}/training-*/")
    print(f"  Grading manifest: {grading_path}")
    print(f"  Edition map:      {map_path}")
    print(f"  Full results:     {results_path}")


if __name__ == "__main__":
    main()
