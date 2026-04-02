#!/usr/bin/env python3
"""Re-render existing training editions with the fixed Typst renderer.

This script fixes two rendering bugs in previously generated editions:
1. `}{` delimiter leakage — caused by old renderer wrapping single-column
   rows in `{ }` code blocks and omitting `#` before function calls in
   multi-column content blocks.  Current renderer is already fixed.
2. HTML attribute leakage — `id="..."` fragments leaked from incomplete
   HTML tag stripping.  Fixed by `_strip_html_attr_prefixes` in renderer.

Usage:
    cd offscroll/
    .venv/bin/python training/rerender_editions.py [--editions N] [--start M]

IRAS Task #243 — Belle
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from offscroll.layout.typst_renderer import build_typst_markup
from offscroll.models import CuratedEdition

logger = logging.getLogger("rerender")

EDITIONS_DIR = Path(__file__).parent / "editions"
TYPST_DIR = Path(__file__).parent.parent / "src" / "offscroll" / "layout" / "typst"
FONTS_DIR = Path(__file__).parent.parent / "src" / "offscroll" / "layout" / "fonts"
PAGE_DPI = 150


def rerender_one(edition_dir: Path) -> dict:
    """Re-render a single training edition. Returns status dict."""
    config_id = edition_dir.name  # e.g. "training-001"
    edition_json = edition_dir / "edition.json"

    result = {
        "edition_id": config_id,
        "status": "failed",
        "error": None,
        "num_pages": 0,
        "render_seconds": 0.0,
    }

    if not edition_json.exists():
        result["error"] = "edition.json not found"
        return result

    try:
        edition = CuratedEdition.from_json(edition_json)
    except Exception as e:
        result["error"] = f"Failed to load edition.json: {e}"
        return result

    # Minimal config: data_dir is the edition directory itself
    config = {
        "output": {"data_dir": str(edition_dir)},
        "newspaper": {"debug_mode": False},
    }

    try:
        markup = build_typst_markup(edition, config)
    except Exception as e:
        result["error"] = f"build_typst_markup failed: {e}"
        return result

    typ_path = edition_dir / f"{config_id}.typ"
    pdf_path = edition_dir / f"{config_id}.pdf"

    typ_path.write_text(markup, encoding="utf-8")
    result["typ_generated"] = True

    # Verify the generated .typ is clean before attempting compilation
    issues = verify_typ_file(typ_path)
    result["typ_issues"] = issues

    typst_bin = shutil.which("typst")
    if typst_bin is None:
        # Can't compile, but .typ generation and verification succeeded
        result["status"] = "typ_only"
        result["note"] = "typst CLI not found; .typ generated and verified only"
        return result

    # Copy templates alongside so the import works
    templates_dest = edition_dir / "templates.typ"
    shutil.copy2(TYPST_DIR / "templates.typ", templates_dest)

    t0 = time.monotonic()
    proc = subprocess.run(
        [typst_bin, "compile", "--font-path", str(FONTS_DIR), str(typ_path), str(pdf_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    result["render_seconds"] = time.monotonic() - t0

    templates_dest.unlink(missing_ok=True)

    if proc.returncode != 0:
        result["error"] = f"Typst compile failed:\n{proc.stderr[:500]}"
        return result

    # Extract pages
    pages_dir = edition_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    for old_png in pages_dir.glob("page-*.png"):
        old_png.unlink()

    try:
        doc = fitz.open(str(pdf_path))
        for i in range(len(doc)):
            page = doc[i]
            mat = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
            pix = page.get_pixmap(matrix=mat)
            pix.save(str(pages_dir / f"page-{i + 1:03d}.png"))
        result["num_pages"] = len(doc)
        doc.close()
    except Exception as e:
        result["error"] = f"Page extraction failed: {e}"
        return result

    result["status"] = "ok"
    return result


def verify_typ_file(typ_path: Path) -> list[str]:
    """Check a .typ file for known bad patterns. Returns list of issues found."""
    issues = []
    try:
        content = typ_path.read_text(encoding="utf-8")
    except Exception:
        return ["Could not read .typ file"]

    lines = content.splitlines()
    # Track whether we're inside the page-setup block (footer context)
    in_page_setup = False
    brace_depth = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track page setup block to skip its braces
        if stripped.startswith("#set page("):
            in_page_setup = True
        if in_page_setup:
            brace_depth += stripped.count("{") - stripped.count("}")
            if stripped == ")" and brace_depth <= 0:
                in_page_setup = False
                brace_depth = 0
            continue

        # Old bug: bare { or } as only content on a line,
        # NOT inside a content paragraph (i.e. not indented inside [])
        # Real article content has braces escaped as \{ and \}
        if stripped in ("{", "}") and not line.startswith("  "):
            issues.append(f"Line {i}: unescaped bare '{stripped}' at top level (old code-block wrapper?)")

        # HTML attribute leakage in content: unescaped id=" inside brackets
        # After fix, these should appear as id= only if they survived stripping
        if 'id="' in line and "[" in line and not stripped.startswith("//"):
            # Distinguish: class/id attributes vs normal content
            if re.search(r'\bid\s*=\s*"[^"]*".*>', line):
                issues.append(f"Line {i}: possible id= attribute leak: {line[:120]}")

        # function call without # inside content block (old multi-col bug)
        # Pattern: line starts with spaces then a known function name without #
        if re.match(r'^\s{4,}(standard|feature|thread|brief)-article\(', line):
            issues.append(f"Line {i}: function call missing # in content block: {line[:80]}")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-render training editions with fixed renderer")
    parser.add_argument("--editions", type=int, default=10, help="Number of editions to re-render")
    parser.add_argument("--start", type=int, default=1, help="Starting edition number (1-based)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    edition_dirs = sorted(EDITIONS_DIR.glob("training-*"))
    selected = edition_dirs[args.start - 1 : args.start - 1 + args.editions]

    if not selected:
        logger.error("No edition directories found in %s", EDITIONS_DIR)
        sys.exit(1)

    logger.info("Re-rendering %d editions (starting from training-%03d)", len(selected), args.start)

    ok = 0
    typ_only = 0
    failed = 0
    total_issues = []

    for ed_dir in selected:
        logger.info("  %s ...", ed_dir.name)
        res = rerender_one(ed_dir)
        issues = res.get("typ_issues", [])
        if issues:
            logger.warning("    .typ ISSUES:")
            for iss in issues:
                logger.warning("      %s", iss)
            total_issues.extend(issues)
        if res["status"] == "ok":
            ok += 1
            logger.info("    OK — %d pages in %.1fs", res["num_pages"], res["render_seconds"])
            if not issues:
                logger.info("    .typ file clean (no known bad patterns)")
        elif res["status"] == "typ_only":
            typ_only += 1
            logger.info("    .typ generated — %s", res.get("note", ""))
            if not issues:
                logger.info("    .typ file clean (no known bad patterns)")
        else:
            failed += 1
            logger.error("    FAILED: %s", res["error"])

    logger.info("")
    logger.info(
        "Results: %d compiled+ok, %d typ-generated, %d failed out of %d",
        ok, typ_only, failed, len(selected),
    )
    if total_issues:
        logger.warning("%d residual issues found in .typ files", len(total_issues))
    else:
        logger.info("No residual bad patterns found in any .typ file")


if __name__ == "__main__":
    main()
