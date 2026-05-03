#!/usr/bin/env python3
"""
Compute layout features for graded spreads — batch 002 (Task #329).

Fixes five broken features from the original compute_features.py:
  - d2_orphans, d2_widows, d4_col_balance, d6_dead_space: were NA
    (required rendered output analysis, now extracted from PDFs via PyMuPDF)
  - d5_fill_fraction: was a miscomputed word-density proxy (r=1.0 with
    est_words_per_page). Now computed as actual spatial fill from PDF
    text block bounding boxes.

All other features are preserved unchanged from the original script.

Seed: 42. Split: 80% train / 20% val.
"""

import csv
import json
import math
import os
import random

import fitz  # PyMuPDF
import numpy as np

BASE = "/home/modus/repos/belle/offscroll/training"
GRADES_PATH = f"{BASE}/grades/batch-002.csv"
EDITIONS_PATH = f"{BASE}/editions"
OUTPUT_PATH = f"{BASE}/features/features-002.csv"

# Page geometry (US letter, 0.5in margins)
PAGE_WIDTH_PT = 612.0
PAGE_HEIGHT_PT = 792.0
MARGIN_PT = 36.0  # 0.5in

# Printable area
PRINT_LEFT = MARGIN_PT
PRINT_RIGHT = PAGE_WIDTH_PT - MARGIN_PT
PRINT_TOP = MARGIN_PT
PRINT_BOTTOM = PAGE_HEIGHT_PT - MARGIN_PT
PRINT_WIDTH = PRINT_RIGHT - PRINT_LEFT   # 540pt
PRINT_HEIGHT = PRINT_BOTTOM - PRINT_TOP   # 720pt
PRINTABLE_AREA = PRINT_WIDTH * PRINT_HEIGHT  # per page

# Footer occupies bottom ~15pt of printable area
FOOTER_HEIGHT_PT = 18.0
CONTENT_BOTTOM = PRINT_BOTTOM - FOOTER_HEIGHT_PT  # effective content bottom
CONTENT_HEIGHT = CONTENT_BOTTOM - PRINT_TOP

# Column detection: if blocks cluster into two x-groups separated by > this, it's 2-col
COL_GAP_THRESHOLD_PT = 20.0

# Dead space: gaps within a column larger than this count as trapped whitespace
DEAD_SPACE_GAP_PT = 36.0  # 0.5in

# Body text identification
BODY_FONT = "SourceSerif4-Regular"
BODY_SIZE = 10.0
HEADLINE_FONTS = {"SourceSans3-Bold", "SourceSans3-Black"}


# ─────────────────────────────────────────────────────────────
# Helpers (unchanged from original)
# ─────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def template_entropy(items):
    if not items:
        return 0.0
    counts = {}
    for item in items:
        hint = item.get("layout_hint") or "unknown"
        counts[hint] = counts.get(hint, 0) + 1
    n = len(items)
    ent = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            ent -= p * math.log2(p)
    return round(ent, 6)


def distinct_sources(items):
    return len({i.get("source_name") or "" for i in items if i.get("source_name")})


def total_images(items):
    return sum(len(i.get("images") or []) for i in items)


def word_count_cv(items):
    wcs = [i.get("word_count") or 0 for i in items]
    if len(wcs) < 2:
        return 0.0
    mean = sum(wcs) / len(wcs)
    if mean == 0:
        return 0.0
    variance = sum((w - mean) ** 2 for w in wcs) / len(wcs)
    return round(math.sqrt(variance) / mean, 6)


def anchor_strength_ratio(items):
    wcs = [i.get("word_count") or 0 for i in items]
    if not wcs:
        return 1.0
    mean = sum(wcs) / len(wcs)
    if mean == 0:
        return 1.0
    return round(max(wcs) / mean, 6)


def estimate_spread_words(all_items, left_page, right_page, total_pages):
    if not all_items or total_pages == 0:
        return 0.0, []
    total_words = sum(i.get("word_count") or 0 for i in all_items)
    if total_words == 0:
        n = len(all_items)
        p_start = (left_page - 1) / total_pages
        p_end = right_page / total_pages
        overlapping = [i for idx, i in enumerate(all_items)
                       if idx / n < p_end and (idx + 1) / n > p_start]
        return 0.0, overlapping
    words_per_page = total_words / total_pages
    page_start_word = (left_page - 1) * words_per_page
    page_end_word = right_page * words_per_page
    est_words = 0.0
    overlapping = []
    cum = 0.0
    for item in all_items:
        w = item.get("word_count") or 0
        item_start = cum
        item_end = cum + w
        overlap_start = max(item_start, page_start_word)
        overlap_end = min(item_end, page_end_word)
        if overlap_end > overlap_start:
            if w > 0:
                est_words += (overlap_end - overlap_start)
            overlapping.append(item)
        cum = item_end
    return round(est_words, 2), overlapping


def page_role(left_page, right_page, total_pages, spread_type):
    if left_page == 1 and spread_type == "solo":
        return "front"
    if spread_type == "solo" and left_page == total_pages:
        return "solo_terminal"
    if left_page == total_pages or right_page == total_pages:
        return "terminal"
    return "interior"


WORDS_PER_PAGE_CAPACITY = 450  # kept for compatibility but not used for d5


# ─────────────────────────────────────────────────────────────
# PDF-based rendered feature extraction
# ─────────────────────────────────────────────────────────────

def _is_footer_block(block, page_height):
    """Check if a block is in the footer region."""
    if block["type"] != 0:
        return True
    y_bottom = block["bbox"][3]
    return y_bottom > page_height - MARGIN_PT - 5


def _is_header_element(block):
    """Check if a block is a masthead/section-label element (not body content)."""
    if block["type"] != 0:
        return False
    lines = block.get("lines", [])
    if not lines:
        return False
    for line in lines:
        for span in line.get("spans", []):
            font = span.get("font", "")
            size = span.get("size", 0)
            # Masthead title is very large
            if size > 20:
                return True
    return False


def _block_is_body_text(block):
    """Check if a text block is body text (not headline/byline)."""
    if block["type"] != 0:
        return False
    lines = block.get("lines", [])
    if not lines:
        return False
    # Check first span
    for line in lines[:1]:
        for span in line.get("spans", []):
            font = span.get("font", "")
            size = span.get("size", 0)
            if font == BODY_FONT and abs(size - BODY_SIZE) < 1:
                return True
    return False


def _block_is_headline(block):
    """Check if a text block is a headline."""
    if block["type"] != 0:
        return False
    lines = block.get("lines", [])
    if not lines:
        return False
    for line in lines[:1]:
        for span in line.get("spans", []):
            font = span.get("font", "")
            if font in HEADLINE_FONTS:
                return True
    return False


def _detect_columns(blocks):
    """Detect column layout from block x-coordinates.

    Returns list of (x_left, x_right) column boundaries.
    For single-column: returns [(PRINT_LEFT, PRINT_RIGHT)].
    For two-column: returns two ranges separated by gutter.
    """
    if not blocks:
        return [(PRINT_LEFT, PRINT_RIGHT)]

    # Collect x-centers of blocks
    x_centers = []
    for b in blocks:
        if b["type"] != 0:
            continue
        x0, _, x1, _ = b["bbox"]
        x_centers.append((x0 + x1) / 2)

    if not x_centers:
        return [(PRINT_LEFT, PRINT_RIGHT)]

    x_centers.sort()

    # Check for a gap in x-centers that indicates column separation
    # Look for the largest gap in the middle third of the x-range
    page_center = (PRINT_LEFT + PRINT_RIGHT) / 2
    best_gap = 0
    best_gap_pos = None
    for i in range(len(x_centers) - 1):
        gap = x_centers[i + 1] - x_centers[i]
        # Gap must be near page center
        gap_center = (x_centers[i] + x_centers[i + 1]) / 2
        if abs(gap_center - page_center) < PRINT_WIDTH * 0.3 and gap > best_gap:
            best_gap = gap
            best_gap_pos = (x_centers[i], x_centers[i + 1])

    if best_gap > COL_GAP_THRESHOLD_PT and best_gap_pos:
        # Two columns
        gutter_center = (best_gap_pos[0] + best_gap_pos[1]) / 2
        return [
            (PRINT_LEFT, gutter_center - best_gap / 2),
            (gutter_center + best_gap / 2, PRINT_RIGHT),
        ]

    return [(PRINT_LEFT, PRINT_RIGHT)]


def _assign_block_to_column(block, columns):
    """Return index of the column a block belongs to."""
    x_center = (block["bbox"][0] + block["bbox"][2]) / 2
    best_col = 0
    best_dist = float("inf")
    for i, (cl, cr) in enumerate(columns):
        col_center = (cl + cr) / 2
        dist = abs(x_center - col_center)
        if dist < best_dist:
            best_dist = dist
            best_col = i
    return best_col


def analyze_page_pdf(doc, page_num_0indexed):
    """Extract layout metrics from a single PDF page.

    Returns dict with:
      fill_area: total area of content blocks (pt²)
      col_bottoms: list of column bottom y-positions
      dead_space_area: area of trapped whitespace (pt²)
      orphans: count of orphaned elements
      widows: count of widow lines
    """
    page = doc[page_num_0indexed]
    all_blocks = page.get_text("dict")["blocks"]

    # Filter to content blocks (not footer, within printable area)
    blocks = []
    for b in all_blocks:
        if b["type"] != 0:
            continue
        if _is_footer_block(b, PAGE_HEIGHT_PT):
            continue
        x0, y0, x1, y1 = b["bbox"]
        # Must overlap with printable area
        if x1 < PRINT_LEFT or x0 > PRINT_RIGHT:
            continue
        if y1 < PRINT_TOP or y0 > CONTENT_BOTTOM:
            continue
        blocks.append(b)

    if not blocks:
        return {
            "fill_area": 0.0,
            "col_bottoms": [],
            "dead_space_area": 0.0,
            "orphans": 0,
            "widows": 0,
        }

    # Detect columns
    columns = _detect_columns(blocks)

    # --- Fill area ---
    fill_area = 0.0
    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        # Clip to printable area
        x0 = max(x0, PRINT_LEFT)
        x1 = min(x1, PRINT_RIGHT)
        y0 = max(y0, PRINT_TOP)
        y1 = min(y1, CONTENT_BOTTOM)
        if x1 > x0 and y1 > y0:
            fill_area += (x1 - x0) * (y1 - y0)

    # --- Per-column analysis ---
    # Group blocks by column
    col_blocks = {i: [] for i in range(len(columns))}
    for b in blocks:
        ci = _assign_block_to_column(b, columns)
        col_blocks[ci].append(b)

    col_bottoms = []
    dead_space_area = 0.0
    orphans = 0
    widows = 0

    for ci in range(len(columns)):
        cblocks = col_blocks[ci]
        if not cblocks:
            col_bottoms.append(PRINT_TOP)
            continue

        # Sort by y-position
        cblocks.sort(key=lambda b: b["bbox"][1])

        # Column bottom = bottom of the lowest block
        col_bot = max(b["bbox"][3] for b in cblocks)
        col_bottoms.append(col_bot)

        # --- Dead space: gaps between consecutive blocks ---
        for i in range(len(cblocks) - 1):
            gap_top = cblocks[i]["bbox"][3]
            gap_bot = cblocks[i + 1]["bbox"][1]
            gap = gap_bot - gap_top
            if gap > DEAD_SPACE_GAP_PT:
                col_width = columns[ci][1] - columns[ci][0]
                dead_space_area += gap * col_width

        # --- Widow detection ---
        # First block in column: if it's 1 line of body text and
        # there's a significant gap after it, it's likely a widow
        first_block = cblocks[0]
        if _block_is_body_text(first_block):
            first_lines = first_block.get("lines", [])
            first_height = first_block["bbox"][3] - first_block["bbox"][1]
            if len(first_lines) == 1 and first_height < 20:
                # Check gap after
                if len(cblocks) > 1:
                    gap_after = cblocks[1]["bbox"][1] - first_block["bbox"][3]
                    if gap_after > 15:  # significant gap after single line
                        widows += 1
                else:
                    # Only block in column = isolated single line
                    widows += 1

        # --- Orphan detection ---
        # Last block in column: if it's a headline without body text
        # following (nothing after it in this column), it's orphaned
        last_block = cblocks[-1]
        if _block_is_headline(last_block):
            last_height = last_block["bbox"][3] - last_block["bbox"][1]
            # Check it's near the bottom of the column content area
            remaining_space = CONTENT_BOTTOM - last_block["bbox"][3]
            if remaining_space > 100:  # significant empty space below
                # Headline at bottom with empty space = orphaned headline
                orphans += 1
        # Also check for isolated short body text blocks at bottom
        elif _block_is_body_text(last_block):
            last_lines = last_block.get("lines", [])
            last_height = last_block["bbox"][3] - last_block["bbox"][1]
            if len(last_lines) == 1 and last_height < 20:
                # Single body line at bottom — check gap before it
                if len(cblocks) > 1:
                    gap_before = last_block["bbox"][1] - cblocks[-2]["bbox"][3]
                    if gap_before > 20:
                        orphans += 1

    return {
        "fill_area": fill_area,
        "col_bottoms": col_bottoms,
        "dead_space_area": dead_space_area,
        "orphans": orphans,
        "widows": widows,
    }


def compute_rendered_features(edition_id, page_numbers):
    """Compute rendered-output features for a spread's pages.

    Args:
        edition_id: e.g. 'training-009'
        page_numbers: list of 1-indexed page numbers in the spread

    Returns dict with d2_orphans, d2_widows, d4_col_balance,
    d5_fill_fraction, d6_dead_space.
    """
    pdf_path = f"{EDITIONS_PATH}/{edition_id}/{edition_id}.pdf"
    if not os.path.exists(pdf_path):
        return None

    doc = fitz.open(pdf_path)
    n_pages = len(page_numbers)

    total_fill_area = 0.0
    total_dead_space = 0.0
    total_orphans = 0
    total_widows = 0
    max_col_balance_pt = 0.0
    total_printable = n_pages * PRINT_WIDTH * CONTENT_HEIGHT

    for pnum in page_numbers:
        page_idx = pnum - 1  # 0-indexed
        if page_idx >= len(doc):
            continue

        result = analyze_page_pdf(doc, page_idx)
        total_fill_area += result["fill_area"]
        total_dead_space += result["dead_space_area"]
        total_orphans += result["orphans"]
        total_widows += result["widows"]

        # Column balance for this page
        bottoms = result["col_bottoms"]
        if len(bottoms) >= 2:
            balance = max(bottoms) - min(bottoms)
            if balance > max_col_balance_pt:
                max_col_balance_pt = balance

    doc.close()

    fill_fraction = total_fill_area / total_printable if total_printable > 0 else 0.0
    dead_space_ratio = total_dead_space / total_printable if total_printable > 0 else 0.0

    return {
        "d5_fill_fraction": round(fill_fraction, 6),
        "d4_col_balance": round(max_col_balance_pt, 2),
        "d6_dead_space": round(dead_space_ratio, 6),
        "d2_orphans": total_orphans,
        "d2_widows": total_widows,
    }


# ─────────────────────────────────────────────────────────────
# Load grades
# ─────────────────────────────────────────────────────────────

grades = {}
with open(GRADES_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        grades[row["spread_id"]] = {
            "technical": int(row["technical"]),
            "style": int(row["style"]),
        }

spread_ids = list(grades.keys())
print(f"Loaded {len(spread_ids)} graded spreads")

# ─────────────────────────────────────────────────────────────
# Train/val split — preserved from original features-002.csv
# (The original split was generated from batch-001 spread ordering
# with seed 42. We preserve it exactly rather than recomputing,
# since recomputing from batch-002 ordering would give a different
# split due to different row order in the CSV.)
# ─────────────────────────────────────────────────────────────

_ORIGINAL_VAL = {
    "s-004-001", "s-006-007", "s-010-002", "s-014-004", "s-036-002",
    "s-044-002", "s-051-005", "s-065-009", "s-074-002", "s-084-006",
}
train_set = set(spread_ids) - _ORIGINAL_VAL
val_set = _ORIGINAL_VAL & set(spread_ids)

print(f"Split: {len(train_set)} train, {len(val_set)} val")

# ─────────────────────────────────────────────────────────────
# Per-edition cache
# ─────────────────────────────────────────────────────────────

edition_map = load_json(f"{EDITIONS_PATH}/edition-map.json")
_meta_cache = {}
_ed_cache = {}


def get_meta(edition_id):
    if edition_id not in _meta_cache:
        _meta_cache[edition_id] = load_json(f"{EDITIONS_PATH}/{edition_id}/metadata.json")
    return _meta_cache[edition_id]


def get_edition(edition_id):
    if edition_id not in _ed_cache:
        _ed_cache[edition_id] = load_json(f"{EDITIONS_PATH}/{edition_id}/edition.json")
    return _ed_cache[edition_id]


def all_items_flat(edition_data):
    items = []
    for section in edition_data.get("sections") or []:
        for item in section.get("items") or []:
            items.append(item)
    return items


# ─────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────

rows = []

for spread_id in spread_ids:
    edition_id = edition_map.get(spread_id)
    if not edition_id:
        print(f"WARNING: no mapping for {spread_id}")
        continue

    try:
        meta = get_meta(edition_id)
        edition = get_edition(edition_id)
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        continue

    # Find this spread in metadata
    spread_meta = next(
        (s for s in (meta.get("spreads") or []) if s["spread_id"] == spread_id),
        None,
    )
    if spread_meta is None:
        print(f"WARNING: {spread_id} not in {edition_id} metadata")
        continue

    pages = spread_meta["pages"]
    spread_type = spread_meta["type"]
    left_page = pages[0]["page_number"]
    right_page = pages[-1]["page_number"]
    n_pages_in_spread = len(pages)
    total_pages = meta.get("num_pages") or len(meta.get("pages") or [])
    page_numbers = [p["page_number"] for p in pages]

    # ── Edition-level aggregates (unchanged) ────────────────
    items = all_items_flat(edition)
    n_items = len(items)
    ed_word_total = sum(i.get("word_count") or 0 for i in items)
    ed_word_mean = round(ed_word_total / n_items, 2) if n_items > 0 else 0.0
    hints = [i.get("layout_hint") or "unknown" for i in items]
    ed_brief_frac = round(hints.count("brief") / len(hints), 6) if hints else 0.0
    ed_standard_frac = round(hints.count("standard") / len(hints), 6) if hints else 0.0
    ed_template_entropy = template_entropy(items)
    ed_image_count = total_images(items)
    ed_source_count = distinct_sources(items)
    ed_section_count = len(edition.get("sections") or [])

    # ── Spread-estimated features (unchanged) ────────────────
    est_words, spread_items = estimate_spread_words(
        items, left_page, right_page, total_pages
    )
    est_n = len(spread_items)
    sp_hints = [i.get("layout_hint") or "unknown" for i in spread_items]
    est_brief_count = sp_hints.count("brief")
    est_standard_count = sp_hints.count("standard")
    est_image_count = total_images(spread_items)
    est_source_count = distinct_sources(spread_items)
    est_words_per_page = round(est_words / n_pages_in_spread, 2)
    est_items_per_page = round(est_n / n_pages_in_spread, 4)
    est_word_mean = round(est_words / est_n, 2) if est_n > 0 else 0.0

    # D7, D8, anchor, D3 (unchanged metadata proxies)
    d7_template_entropy = template_entropy(spread_items)
    d8_word_count_cv = word_count_cv(spread_items)
    anchor_str = anchor_strength_ratio(spread_items)
    d3_image_fraction = round(est_image_count / est_n, 6) if est_n > 0 else 0.0

    # ── Spread position features (unchanged) ─────────────────
    role = page_role(left_page, right_page, total_pages, spread_type)
    is_front = int(role == "front")
    is_terminal = int(role in ("terminal", "solo_terminal"))
    is_solo = int(spread_type == "solo")
    page_pos_frac = round(
        (left_page - 1) / (total_pages - 1) if total_pages > 1 else 0.0, 6
    )

    # ── Rendered-output features (NEW — from PDF analysis) ───
    rendered = compute_rendered_features(edition_id, page_numbers)
    if rendered is None:
        print(f"WARNING: no PDF for {edition_id}, using NA for rendered features")
        d5_fill = "NA"
        d4_col = "NA"
        d6_dead = "NA"
        d2_orph = "NA"
        d2_wid = "NA"
    else:
        d5_fill = rendered["d5_fill_fraction"]
        d4_col = rendered["d4_col_balance"]
        d6_dead = rendered["d6_dead_space"]
        d2_orph = rendered["d2_orphans"]
        d2_wid = rendered["d2_widows"]

    # ── Assemble row ────────────────────────────────────────
    g = grades[spread_id]
    rows.append({
        "spread_id": spread_id,
        "split": "train" if spread_id in train_set else "val",
        "technical_grade": g["technical"],
        "style_grade": g["style"],
        "spread_type": spread_type,
        "is_solo": is_solo,
        "is_front": is_front,
        "is_terminal": is_terminal,
        "page_role": role,
        "left_page": left_page,
        "right_page": right_page,
        "n_pages_in_spread": n_pages_in_spread,
        "edition_page_count": total_pages,
        "page_position_frac": page_pos_frac,
        "edition_item_count": n_items,
        "edition_word_count_total": ed_word_total,
        "edition_word_count_mean": ed_word_mean,
        "edition_brief_frac": ed_brief_frac,
        "edition_standard_frac": ed_standard_frac,
        "edition_template_entropy": ed_template_entropy,
        "edition_image_count_total": ed_image_count,
        "edition_source_count": ed_source_count,
        "edition_section_count": ed_section_count,
        "est_item_count": est_n,
        "est_items_per_page": est_items_per_page,
        "est_word_count": est_words,
        "est_words_per_page": est_words_per_page,
        "est_word_count_mean": est_word_mean,
        "est_brief_count": est_brief_count,
        "est_standard_count": est_standard_count,
        "est_image_count": est_image_count,
        "est_source_count": est_source_count,
        "d3_image_fraction": d3_image_fraction,
        "d5_fill_fraction": d5_fill,
        "d7_template_entropy": d7_template_entropy,
        "d8_word_count_cv": d8_word_count_cv,
        "anchor_strength": anchor_str,
        "d2_orphans": d2_orph,
        "d2_widows": d2_wid,
        "d4_col_balance": d4_col,
        "d6_dead_space": d6_dead,
    })

print(f"Computed features for {len(rows)} spreads")

# ─────────────────────────────────────────────────────────────
# Write features-002.csv
# ─────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fieldnames = list(rows[0].keys())

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {OUTPUT_PATH}")

# ─────────────────────────────────────────────────────────────
# Summary stats
# ─────────────────────────────────────────────────────────────

def stats(vals):
    if not vals:
        return "empty"
    return f"min={min(vals):.4f} max={max(vals):.4f} mean={sum(vals)/len(vals):.4f}"


train_count = sum(1 for r in rows if r["split"] == "train")
val_count = sum(1 for r in rows if r["split"] == "val")
print(f"\nTrain: {train_count}, Val: {val_count}")

for col in [
    "d5_fill_fraction", "d4_col_balance", "d6_dead_space",
    "d2_orphans", "d2_widows",
    "d7_template_entropy", "d8_word_count_cv", "anchor_strength",
    "est_items_per_page", "est_words_per_page", "page_position_frac",
]:
    vals = [r[col] for r in rows if isinstance(r[col], (int, float))]
    print(f"  {col}: {stats(vals)}")

# Check correlation between d5_fill_fraction and est_words_per_page
fill_vals = [r["d5_fill_fraction"] for r in rows if isinstance(r["d5_fill_fraction"], float)]
word_vals = [r["est_words_per_page"] for r in rows if isinstance(r["d5_fill_fraction"], float)]
if fill_vals and word_vals:
    fill_arr = np.array(fill_vals)
    word_arr = np.array(word_vals)
    corr = np.corrcoef(fill_arr, word_arr)[0, 1]
    print(f"\n  Correlation d5_fill_fraction vs est_words_per_page: {corr:.4f}")
    print(f"  (Was r=1.000 — should now be much lower)")

# Check direction: do high-grade spreads have higher fill?
for grade_col in ["technical_grade"]:
    high_grade = [r["d5_fill_fraction"] for r in rows
                  if isinstance(r["d5_fill_fraction"], float) and r[grade_col] >= 6]
    low_grade = [r["d5_fill_fraction"] for r in rows
                 if isinstance(r["d5_fill_fraction"], float) and r[grade_col] <= 3]
    if high_grade and low_grade:
        print(f"  d5_fill high-grade (T>=6) mean: {sum(high_grade)/len(high_grade):.4f}")
        print(f"  d5_fill low-grade  (T<=3) mean: {sum(low_grade)/len(low_grade):.4f}")
        print(f"  (Direction should be: high-grade > low-grade for true spatial fill)")
