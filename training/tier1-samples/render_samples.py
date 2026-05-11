#!/usr/bin/env python3
"""Render three sample editions (tight / standard / open) from the
test fixture to verify Tier 1 typographic diversity.

Used to confirm acceptance criteria for brief #414:
- Scale parameter applies consistently across an edition
- Lead items differentiated on each section
- Three weights with small-caps kickers
- Existing layouts still render at standard scale (no regression)

Run:
    cd offscroll
    uv run python training/tier1-samples/render_samples.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from offscroll.layout.typst_renderer import build_typst_markup  # noqa: E402
from offscroll.models import CuratedEdition  # noqa: E402

FIXTURE = REPO / "tests" / "sample_data" / "editions" / "sample_edition_full.json"
TYPST_DIR = REPO / "src" / "offscroll" / "layout" / "typst"
FONTS_DIR = REPO / "src" / "offscroll" / "layout" / "fonts"
OUT_DIR = Path(__file__).parent

PRESETS = [
    {
        "name": "tight",
        "config": {
            "scale": "tight",
            "weights": "three",
            "lead_amplifications": ["scale-bump"],
        },
    },
    {
        "name": "standard",
        "config": {
            "scale": "standard",
            "weights": "three",
            "lead_amplifications": ["deck", "scale-bump"],
        },
    },
    {
        "name": "open",
        "config": {
            "scale": "open",
            "weights": "three",
            "lead_amplifications": ["deck", "scale-bump"],
        },
    },
    # Regression check — two-weight strategy with no lead amplification
    # mirrors the prior (pre-#414) visual treatment of the standard scale.
    {
        "name": "legacy",
        "config": {
            "scale": "standard",
            "weights": "two",
            "lead_amplifications": [],
        },
    },
]


def render(preset: dict) -> Path:
    name = preset["name"]
    edition = CuratedEdition.from_json(str(FIXTURE))
    config = {
        "output": {"data_dir": str(OUT_DIR)},
        "newspaper": {
            "debug_mode": False,
            "typography": preset["config"],
        },
    }
    markup = build_typst_markup(edition, config)
    typ_path = OUT_DIR / f"sample-{name}.typ"
    pdf_path = OUT_DIR / f"sample-{name}.pdf"
    typ_path.write_text(markup)

    # Copy templates alongside generated .typ so #import resolves
    templates_dst = OUT_DIR / "templates.typ"
    shutil.copy2(TYPST_DIR / "templates.typ", templates_dst)

    try:
        result = subprocess.run(
            [
                "typst",
                "compile",
                "--font-path",
                str(FONTS_DIR),
                str(typ_path),
                str(pdf_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        templates_dst.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"[{name}] FAILED:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"[{name}] OK -> {pdf_path.relative_to(REPO)}")
    return pdf_path


def main():
    for preset in PRESETS:
        render(preset)


if __name__ == "__main__":
    main()
