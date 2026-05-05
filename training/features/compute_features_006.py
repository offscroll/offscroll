#!/usr/bin/env python3
"""
Compute layout features for graded spreads — n=200, sixth iteration.

This revision adds **visual-hierarchy features** to the schema in
features-005.csv. Ada's #344 report identified the technical-feature
ceiling at R² ≈ 0.50 and the style floor at R² ≈ 0.28. The 18
structural features measure ``health'' (fill, orphans, balance) but
not visual hierarchy. New features here are designed to lift both
ceilings by capturing what the eye actually sees: type-size variation,
weight diversity, anchor strength of rendered blocks, white-space
rhythm, and pull-quote/headline presence.

New features (h-prefix, all computed from the rendered PDF):

  Type-size variation:
    h_distinct_font_sizes      Count of distinct font sizes on the spread
    h_size_std_chars           Character-weighted std of font sizes
    h_max_size_to_body         Max non-nameplate size / 10pt body baseline

  Weight / family diversity:
    h_distinct_weights         Distinct (family, bold, italic) combinations
    h_bold_char_frac           Char-fraction in Bold/Black weights
    h_italic_char_frac         Char-fraction in italic styles
    h_sans_char_frac           Char-fraction in sans-serif (headline material)

  Element-scale (anchor strength on rendered blocks):
    h_block_area_max_to_median Largest block area / median block area
    h_block_area_cv            CV of block areas (visual mass dispersion)

  White-space rhythm:
    h_gap_cv                   CV of inter-block gaps within columns
    h_max_gap_to_median        Max gap / median gap

  Headline / pull-quote presence:
    h_headline_count           Count of headline blocks (sans + size>=12)
    h_headline_area_frac       Headline block area / total block area
    h_pull_quote_count         Count of italic-serif blocks at size>=12

The corpus uses a fixed font system (SourceSerif4 for body / serif
italics; SourceSans3 for headlines; both have Bold and Italic cuts).
The 48pt size is used only for the page-1 nameplate; the
``h_max_size_to_body'' feature ignores it so that a non-nameplate
spread isn't mis-rated.

Schema: superset of features-005.csv (all 18 prior features + 14
visual-hierarchy features = 32 numeric + 7 categorical/binary).
Same 80/20 train/val split with seed 42 over the lex-sorted spread
IDs (must match features-005 so n=200 results compare cleanly).

Outputs features-006.csv to the same directory.
"""

import csv
import json
import math
import os
import random
import re
from collections import Counter

import fitz  # PyMuPDF
import numpy as np

BASE = "/home/modus/offscroll/training"
GRADES_PATHS = [
    f"{BASE}/grades/batch-002.csv",
    f"{BASE}/grades/batch-003.csv",
    f"{BASE}/grades/batch-004.csv",
    f"{BASE}/grades/batch-005.csv",
]
EDITIONS_PATH = f"{BASE}/editions"
OUTPUT_PATH = f"{BASE}/features/features-006.csv"

# Page geometry (US letter, 0.5in margins)
PAGE_WIDTH_PT = 612.0
PAGE_HEIGHT_PT = 792.0
MARGIN_PT = 36.0

PRINT_LEFT = MARGIN_PT
PRINT_RIGHT = PAGE_WIDTH_PT - MARGIN_PT
PRINT_TOP = MARGIN_PT
PRINT_BOTTOM = PAGE_HEIGHT_PT - MARGIN_PT
PRINT_WIDTH = PRINT_RIGHT - PRINT_LEFT
PRINT_HEIGHT = PRINT_BOTTOM - PRINT_TOP
PRINTABLE_AREA = PRINT_WIDTH * PRINT_HEIGHT

FOOTER_HEIGHT_PT = 18.0
CONTENT_BOTTOM = PRINT_BOTTOM - FOOTER_HEIGHT_PT
CONTENT_HEIGHT = CONTENT_BOTTOM - PRINT_TOP

COL_GAP_THRESHOLD_PT = 20.0
DEAD_SPACE_GAP_PT = 36.0

BODY_FONT = "SourceSerif4-Regular"
BODY_SIZE = 10.0
HEADLINE_FONTS = {"SourceSans3-Bold", "SourceSans3-Black"}

# Visual-hierarchy thresholds
HEADLINE_MIN_SIZE = 12.0      # blocks at >=12pt with sans-serif are headlines
PULL_QUOTE_MIN_SIZE = 12.0    # blocks at >=12pt italic-serif are pull quotes
NAMEPLATE_SIZE_THRESHOLD = 36.0  # ignore for "non-nameplate" max-size feature

# Widow detection guards (unchanged from compute_features_005.py)
WIDOW_MIN_TEXT_LEN = 5
WIDOW_MIN_PARA_RATIO = 1.5


# ─────────────────────────────────────────────────────────────
# Helpers (unchanged from compute_features_005.py)
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


# ─────────────────────────────────────────────────────────────
# PDF block helpers
# ─────────────────────────────────────────────────────────────

def _is_footer_block(block, page_height):
    if block["type"] != 0:
        return True
    y_bottom = block["bbox"][3]
    return y_bottom > page_height - MARGIN_PT - 5


def _block_is_body_text(block):
    if block["type"] != 0:
        return False
    lines = block.get("lines", [])
    if not lines:
        return False
    for line in lines[:1]:
        for span in line.get("spans", []):
            font = span.get("font", "")
            size = span.get("size", 0)
            if font == BODY_FONT and abs(size - BODY_SIZE) < 1:
                return True
    return False


def _block_is_headline(block):
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


def _block_text(block):
    parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            parts.append(span.get("text", ""))
    return "".join(parts).strip()


def _norm_text(s):
    return re.sub(r'\s+', ' ', s).strip().lower()


def _detect_columns(blocks):
    if not blocks:
        return [(PRINT_LEFT, PRINT_RIGHT)]

    x_centers = []
    for b in blocks:
        if b["type"] != 0:
            continue
        x0, _, x1, _ = b["bbox"]
        x_centers.append((x0 + x1) / 2)

    if not x_centers:
        return [(PRINT_LEFT, PRINT_RIGHT)]

    x_centers.sort()
    page_center = (PRINT_LEFT + PRINT_RIGHT) / 2
    best_gap = 0
    best_gap_pos = None
    for i in range(len(x_centers) - 1):
        gap = x_centers[i + 1] - x_centers[i]
        gap_center = (x_centers[i] + x_centers[i + 1]) / 2
        if abs(gap_center - page_center) < PRINT_WIDTH * 0.3 and gap > best_gap:
            best_gap = gap
            best_gap_pos = (x_centers[i], x_centers[i + 1])

    if best_gap > COL_GAP_THRESHOLD_PT and best_gap_pos:
        gutter_center = (best_gap_pos[0] + best_gap_pos[1]) / 2
        return [
            (PRINT_LEFT, gutter_center - best_gap / 2),
            (gutter_center + best_gap / 2, PRINT_RIGHT),
        ]

    return [(PRINT_LEFT, PRINT_RIGHT)]


def _assign_block_to_column(block, columns):
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


# ─────────────────────────────────────────────────────────────
# Source paragraphs (for widow tail-match)
# ─────────────────────────────────────────────────────────────

_HTML_ATTR_PREFIX = re.compile(
    r'^[a-z]+="[^"]*"\s*(?:[a-z]+="[^"]*"\s*)*>\s*',
    flags=re.IGNORECASE,
)


def _strip_html_attr_prefix(p):
    return _HTML_ATTR_PREFIX.sub('', p)


def collect_source_paragraphs(edition_data):
    paragraphs = []
    for section in edition_data.get("sections") or []:
        for item in section.get("items") or []:
            txt = item.get("display_text") or ""
            for para in re.split(r'\n\n+', txt):
                p = _strip_html_attr_prefix(para.strip())
                if p:
                    paragraphs.append(_norm_text(p))
    return paragraphs


def is_widow_tail(block_text, source_paragraphs):
    ntext = _norm_text(block_text)
    if len(ntext) < WIDOW_MIN_TEXT_LEN:
        return False
    for np_ in source_paragraphs:
        if (
            len(np_) > len(ntext) * WIDOW_MIN_PARA_RATIO
            and np_.endswith(ntext)
        ):
            return True
    return False


# ─────────────────────────────────────────────────────────────
# Visual-hierarchy classification
# ─────────────────────────────────────────────────────────────

def _font_class(font_name: str):
    """Classify a font name into (family, is_bold, is_italic).

    family ∈ {"sans", "serif", "other"}; case insensitive substring
    matching keeps this robust to libertinus/sourcesans3/etc.
    """
    f = font_name or ""
    family = "other"
    fl = f.lower()
    if "sans" in fl:
        family = "sans"
    elif "serif" in fl:
        family = "serif"
    is_bold = "bold" in fl or "black" in fl
    is_italic = ("-it" in fl or "italic" in fl)
    return family, is_bold, is_italic


def analyze_block_typography(block):
    """Pull typography signals out of a single text block.

    Returns dict with character-weighted counts and the dominant
    classification of the block (used for headline / pull-quote
    detection on the spread).
    """
    sizes = []  # one entry per character for character-weighted stats
    bold_chars = 0
    italic_chars = 0
    sans_chars = 0
    total_chars = 0
    block_max_size = 0.0
    has_italic_serif_large = False
    has_sans_large = False
    family_weight_keys = set()

    for line in block.get("lines", []):
        for span in line.get("spans", []):
            f = span.get("font", "")
            sz = span.get("size", 0)
            text_len = len(span.get("text", ""))
            if text_len <= 0:
                continue
            sizes.extend([sz] * text_len)
            family, is_bold, is_italic = _font_class(f)
            family_weight_keys.add((family, is_bold, is_italic))
            if is_bold:
                bold_chars += text_len
            if is_italic:
                italic_chars += text_len
            if family == "sans":
                sans_chars += text_len
            total_chars += text_len
            if sz > block_max_size:
                block_max_size = sz
            if family == "serif" and is_italic and sz >= PULL_QUOTE_MIN_SIZE:
                has_italic_serif_large = True
            if family == "sans" and sz >= HEADLINE_MIN_SIZE:
                has_sans_large = True

    return {
        "sizes_chars": sizes,
        "bold_chars": bold_chars,
        "italic_chars": italic_chars,
        "sans_chars": sans_chars,
        "total_chars": total_chars,
        "max_size": block_max_size,
        "is_pull_quote": has_italic_serif_large,
        "is_headline": has_sans_large and not has_italic_serif_large,
        "family_weight_keys": family_weight_keys,
    }


# ─────────────────────────────────────────────────────────────
# Per-page rendered analysis
# ─────────────────────────────────────────────────────────────

def analyze_page_pdf(doc, page_num_0indexed, source_paragraphs):
    page = doc[page_num_0indexed]
    all_blocks = page.get_text("dict")["blocks"]

    blocks = []
    for b in all_blocks:
        if b["type"] != 0:
            continue
        if _is_footer_block(b, PAGE_HEIGHT_PT):
            continue
        x0, y0, x1, y1 = b["bbox"]
        if x1 < PRINT_LEFT or x0 > PRINT_RIGHT:
            continue
        if y1 < PRINT_TOP or y0 > CONTENT_BOTTOM:
            continue
        blocks.append(b)

    page_result = {
        "fill_area": 0.0,
        "col_bottoms": [],
        "dead_space_area": 0.0,
        "orphans": 0,
        "widows": 0,
        # New visual-hierarchy aggregates (per-page; combine across spread):
        "block_areas": [],
        "block_typography": [],
        "col_gaps": [],     # all inter-block gaps within columns on this page
        "headline_count": 0,
        "pull_quote_count": 0,
        "headline_area": 0.0,
        "page_fill_fraction": 0.0,
    }

    if not blocks:
        return page_result

    columns = _detect_columns(blocks)

    fill_area = 0.0
    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        x0 = max(x0, PRINT_LEFT)
        x1 = min(x1, PRINT_RIGHT)
        y0 = max(y0, PRINT_TOP)
        y1 = min(y1, CONTENT_BOTTOM)
        if x1 > x0 and y1 > y0:
            fill_area += (x1 - x0) * (y1 - y0)

    col_blocks = {i: [] for i in range(len(columns))}
    for b in blocks:
        ci = _assign_block_to_column(b, columns)
        col_blocks[ci].append(b)

    col_bottoms = []
    dead_space_area = 0.0
    orphans = 0
    widows = 0
    col_gaps = []

    for ci in range(len(columns)):
        cblocks = col_blocks[ci]
        if not cblocks:
            col_bottoms.append(PRINT_TOP)
            continue

        cblocks.sort(key=lambda b: b["bbox"][1])

        col_bot = max(b["bbox"][3] for b in cblocks)
        col_bottoms.append(col_bot)

        for i in range(len(cblocks) - 1):
            gap_top = cblocks[i]["bbox"][3]
            gap_bot = cblocks[i + 1]["bbox"][1]
            gap = gap_bot - gap_top
            if gap > 0:
                col_gaps.append(gap)
            if gap > DEAD_SPACE_GAP_PT:
                col_width = columns[ci][1] - columns[ci][0]
                dead_space_area += gap * col_width

        first_block = cblocks[0]
        if _block_is_body_text(first_block):
            first_lines = first_block.get("lines", [])
            first_height = first_block["bbox"][3] - first_block["bbox"][1]
            if len(first_lines) == 1 and first_height < 20:
                btext = _block_text(first_block)
                if is_widow_tail(btext, source_paragraphs):
                    widows += 1

        last_block = cblocks[-1]
        if _block_is_headline(last_block):
            remaining_space = CONTENT_BOTTOM - last_block["bbox"][3]
            if remaining_space > 100:
                orphans += 1
        elif _block_is_body_text(last_block):
            last_lines = last_block.get("lines", [])
            last_height = last_block["bbox"][3] - last_block["bbox"][1]
            if len(last_lines) == 1 and last_height < 20:
                if len(cblocks) > 1:
                    gap_before = last_block["bbox"][1] - cblocks[-2]["bbox"][3]
                    if gap_before > 20:
                        orphans += 1

    # Visual-hierarchy per-page aggregates
    block_areas = []
    block_typography = []
    headline_count = 0
    pull_quote_count = 0
    headline_area = 0.0

    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        x0c = max(x0, PRINT_LEFT)
        x1c = min(x1, PRINT_RIGHT)
        y0c = max(y0, PRINT_TOP)
        y1c = min(y1, CONTENT_BOTTOM)
        if x1c <= x0c or y1c <= y0c:
            continue
        area = (x1c - x0c) * (y1c - y0c)
        block_areas.append(area)
        typ = analyze_block_typography(b)
        block_typography.append(typ)
        if typ["is_headline"]:
            headline_count += 1
            headline_area += area
        if typ["is_pull_quote"]:
            pull_quote_count += 1

    page_printable = PRINT_WIDTH * CONTENT_HEIGHT
    page_result.update({
        "fill_area": fill_area,
        "col_bottoms": col_bottoms,
        "dead_space_area": dead_space_area,
        "orphans": orphans,
        "widows": widows,
        "block_areas": block_areas,
        "block_typography": block_typography,
        "col_gaps": col_gaps,
        "headline_count": headline_count,
        "pull_quote_count": pull_quote_count,
        "headline_area": headline_area,
        "page_fill_fraction": fill_area / page_printable if page_printable > 0 else 0.0,
    })
    return page_result


def _cv(values):
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    m = float(arr.mean())
    if m == 0:
        return 0.0
    return float(arr.std() / m)


def compute_rendered_features(edition_id, page_numbers, source_paragraphs):
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

    spread_block_areas = []
    spread_block_typography = []
    spread_col_gaps = []
    spread_headline_count = 0
    spread_pull_quote_count = 0
    spread_headline_area = 0.0
    page_fills = []

    for pnum in page_numbers:
        page_idx = pnum - 1
        if page_idx >= len(doc):
            continue

        result = analyze_page_pdf(doc, page_idx, source_paragraphs)
        total_fill_area += result["fill_area"]
        total_dead_space += result["dead_space_area"]
        total_orphans += result["orphans"]
        total_widows += result["widows"]

        bottoms = result["col_bottoms"]
        if len(bottoms) >= 2:
            balance = max(bottoms) - min(bottoms)
            if balance > max_col_balance_pt:
                max_col_balance_pt = balance

        spread_block_areas.extend(result["block_areas"])
        spread_block_typography.extend(result["block_typography"])
        spread_col_gaps.extend(result["col_gaps"])
        spread_headline_count += result["headline_count"]
        spread_pull_quote_count += result["pull_quote_count"]
        spread_headline_area += result["headline_area"]
        page_fills.append(result["page_fill_fraction"])

    doc.close()

    fill_fraction = total_fill_area / total_printable if total_printable > 0 else 0.0
    dead_space_ratio = total_dead_space / total_printable if total_printable > 0 else 0.0

    # ─── Visual-hierarchy aggregations across the spread ───
    sizes_all = []
    bold_chars = 0
    italic_chars = 0
    sans_chars = 0
    total_chars = 0
    distinct_sizes = set()
    family_weight_keys = set()
    non_nameplate_max = 0.0

    for typ in spread_block_typography:
        sizes_all.extend(typ["sizes_chars"])
        bold_chars += typ["bold_chars"]
        italic_chars += typ["italic_chars"]
        sans_chars += typ["sans_chars"]
        total_chars += typ["total_chars"]
        family_weight_keys.update(typ["family_weight_keys"])
        for s in typ["sizes_chars"]:
            distinct_sizes.add(round(s, 1))
        # non-nameplate block max
        block_size = typ["max_size"]
        if block_size < NAMEPLATE_SIZE_THRESHOLD and block_size > non_nameplate_max:
            non_nameplate_max = block_size

    h_distinct_font_sizes = len(distinct_sizes)
    if sizes_all:
        h_size_std_chars = float(np.std(np.asarray(sizes_all, dtype=float)))
    else:
        h_size_std_chars = 0.0
    h_max_size_to_body = non_nameplate_max / BODY_SIZE if non_nameplate_max > 0 else 0.0
    h_distinct_weights = len(family_weight_keys)
    h_bold_char_frac = bold_chars / total_chars if total_chars > 0 else 0.0
    h_italic_char_frac = italic_chars / total_chars if total_chars > 0 else 0.0
    h_sans_char_frac = sans_chars / total_chars if total_chars > 0 else 0.0

    if spread_block_areas:
        median_area = float(np.median(spread_block_areas))
        max_area = float(max(spread_block_areas))
        h_block_area_max_to_median = max_area / median_area if median_area > 0 else 0.0
        h_block_area_cv = _cv(spread_block_areas)
        total_block_area = sum(spread_block_areas)
    else:
        h_block_area_max_to_median = 0.0
        h_block_area_cv = 0.0
        total_block_area = 0.0

    if spread_col_gaps:
        h_gap_cv = _cv(spread_col_gaps)
        median_gap = float(np.median(spread_col_gaps))
        h_max_gap_to_median = max(spread_col_gaps) / median_gap if median_gap > 0 else 0.0
    else:
        h_gap_cv = 0.0
        h_max_gap_to_median = 0.0

    h_headline_count = spread_headline_count
    h_pull_quote_count = spread_pull_quote_count
    h_headline_area_frac = (
        spread_headline_area / total_block_area if total_block_area > 0 else 0.0
    )

    return {
        "d5_fill_fraction": round(fill_fraction, 6),
        "d4_col_balance": round(max_col_balance_pt, 2),
        "d6_dead_space": round(dead_space_ratio, 6),
        "d2_orphans": total_orphans,
        "d2_widows": total_widows,
        # New visual-hierarchy features:
        "h_distinct_font_sizes": int(h_distinct_font_sizes),
        "h_size_std_chars": round(h_size_std_chars, 6),
        "h_max_size_to_body": round(h_max_size_to_body, 6),
        "h_distinct_weights": int(h_distinct_weights),
        "h_bold_char_frac": round(h_bold_char_frac, 6),
        "h_italic_char_frac": round(h_italic_char_frac, 6),
        "h_sans_char_frac": round(h_sans_char_frac, 6),
        "h_block_area_max_to_median": round(h_block_area_max_to_median, 6),
        "h_block_area_cv": round(h_block_area_cv, 6),
        "h_gap_cv": round(h_gap_cv, 6),
        "h_max_gap_to_median": round(h_max_gap_to_median, 6),
        "h_headline_count": int(h_headline_count),
        "h_headline_area_frac": round(h_headline_area_frac, 6),
        "h_pull_quote_count": int(h_pull_quote_count),
    }


# ─────────────────────────────────────────────────────────────
# Load and merge grades (identical to compute_features_005.py)
# ─────────────────────────────────────────────────────────────

grades = {}
batch_of = {}
filtered_rows = []
for path in GRADES_PATHS:
    batch_label = os.path.basename(path).replace(".csv", "")
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("spread_id") or "").strip()
            tech_raw = (row.get("technical") or "").strip()
            style_raw = (row.get("style") or "").strip()
            if not sid.startswith("s-") or not tech_raw or not style_raw:
                filtered_rows.append((path, sid))
                continue
            if sid in grades:
                raise ValueError(
                    f"Duplicate spread_id {sid} in {path} "
                    f"(also in {batch_of[sid]})"
                )
            grades[sid] = {
                "technical": int(tech_raw),
                "style": int(style_raw),
            }
            batch_of[sid] = batch_label

spread_ids = list(grades.keys())
print(f"Loaded {len(spread_ids)} graded spreads from {len(GRADES_PATHS)} batches")
if filtered_rows:
    print(f"Filtered {len(filtered_rows)} non-spread rows: {filtered_rows}")

# ─────────────────────────────────────────────────────────────
# Train/val split — identical procedure to features-005
# ─────────────────────────────────────────────────────────────

sorted_ids = sorted(spread_ids)
rng = random.Random(42)
shuffled = sorted_ids[:]
rng.shuffle(shuffled)
n_val = int(round(len(shuffled) * 0.20))
val_set = set(shuffled[:n_val])
train_set = set(shuffled[n_val:])
assert len(val_set) + len(train_set) == len(spread_ids)
print(f"Split: {len(train_set)} train, {len(val_set)} val")

# ─────────────────────────────────────────────────────────────
# Per-edition cache
# ─────────────────────────────────────────────────────────────

edition_map = load_json(f"{EDITIONS_PATH}/edition-map.json")
_meta_cache = {}
_ed_cache = {}
_para_cache = {}


def get_meta(edition_id):
    if edition_id not in _meta_cache:
        _meta_cache[edition_id] = load_json(f"{EDITIONS_PATH}/{edition_id}/metadata.json")
    return _meta_cache[edition_id]


def get_edition(edition_id):
    if edition_id not in _ed_cache:
        _ed_cache[edition_id] = load_json(f"{EDITIONS_PATH}/{edition_id}/edition.json")
    return _ed_cache[edition_id]


def get_paragraphs(edition_id):
    if edition_id not in _para_cache:
        _para_cache[edition_id] = collect_source_paragraphs(get_edition(edition_id))
    return _para_cache[edition_id]


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
skipped = []

for spread_id in spread_ids:
    edition_id = edition_map.get(spread_id)
    if not edition_id:
        parts = spread_id.split("-")
        if len(parts) == 3 and parts[0] == "s":
            candidate = f"training-{parts[1]}"
            cand_meta_path = f"{EDITIONS_PATH}/{candidate}/metadata.json"
            if os.path.exists(cand_meta_path):
                edition_id = candidate
                print(
                    f"NOTE: {spread_id} not in edition-map; "
                    f"derived edition_id={edition_id} from ID format"
                )
        if not edition_id:
            print(f"WARNING: no mapping for {spread_id}")
            skipped.append((spread_id, "no edition mapping"))
            continue

    try:
        meta = get_meta(edition_id)
        edition = get_edition(edition_id)
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        skipped.append((spread_id, str(e)))
        continue

    spread_meta = next(
        (s for s in (meta.get("spreads") or []) if s["spread_id"] == spread_id),
        None,
    )
    if spread_meta is None:
        print(f"WARNING: {spread_id} not in {edition_id} metadata")
        skipped.append((spread_id, "not in edition metadata"))
        continue

    pages = spread_meta["pages"]
    spread_type = spread_meta["type"]
    left_page = pages[0]["page_number"]
    right_page = pages[-1]["page_number"]
    n_pages_in_spread = len(pages)
    total_pages = meta.get("num_pages") or len(meta.get("pages") or [])
    page_numbers = [p["page_number"] for p in pages]

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

    d7_template_entropy = template_entropy(spread_items)
    d8_word_count_cv = word_count_cv(spread_items)
    anchor_str = anchor_strength_ratio(spread_items)
    d3_image_fraction = round(est_image_count / est_n, 6) if est_n > 0 else 0.0

    role = page_role(left_page, right_page, total_pages, spread_type)
    is_front = int(role == "front")
    is_terminal = int(role in ("terminal", "solo_terminal"))
    is_solo = int(spread_type == "solo")
    page_pos_frac = round(
        (left_page - 1) / (total_pages - 1) if total_pages > 1 else 0.0, 6
    )

    source_paragraphs = get_paragraphs(edition_id)
    rendered = compute_rendered_features(edition_id, page_numbers, source_paragraphs)
    if rendered is None:
        print(f"WARNING: no PDF for {edition_id}, using NA for rendered features")
        d5_fill = "NA"
        d4_col = "NA"
        d6_dead = "NA"
        d2_orph = "NA"
        d2_wid = "NA"
        h_features = {k: "NA" for k in [
            "h_distinct_font_sizes", "h_size_std_chars", "h_max_size_to_body",
            "h_distinct_weights", "h_bold_char_frac", "h_italic_char_frac",
            "h_sans_char_frac", "h_block_area_max_to_median", "h_block_area_cv",
            "h_gap_cv", "h_max_gap_to_median", "h_headline_count",
            "h_headline_area_frac", "h_pull_quote_count",
        ]}
    else:
        d5_fill = rendered["d5_fill_fraction"]
        d4_col = rendered["d4_col_balance"]
        d6_dead = rendered["d6_dead_space"]
        d2_orph = rendered["d2_orphans"]
        d2_wid = rendered["d2_widows"]
        h_features = {k: rendered[k] for k in [
            "h_distinct_font_sizes", "h_size_std_chars", "h_max_size_to_body",
            "h_distinct_weights", "h_bold_char_frac", "h_italic_char_frac",
            "h_sans_char_frac", "h_block_area_max_to_median", "h_block_area_cv",
            "h_gap_cv", "h_max_gap_to_median", "h_headline_count",
            "h_headline_area_frac", "h_pull_quote_count",
        ]}

    g = grades[spread_id]
    row = {
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
    }
    row.update(h_features)
    rows.append(row)

print(f"Computed features for {len(rows)} spreads (skipped {len(skipped)})")
if skipped:
    for sid, reason in skipped:
        print(f"  SKIPPED {sid}: {reason}")

# ─────────────────────────────────────────────────────────────
# Write features-006.csv
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

batch_counts = Counter(batch_of[r["spread_id"]] for r in rows)
print(f"\nPer-batch row counts: {dict(batch_counts)}")

new_features = [
    "h_distinct_font_sizes", "h_size_std_chars", "h_max_size_to_body",
    "h_distinct_weights", "h_bold_char_frac", "h_italic_char_frac",
    "h_sans_char_frac", "h_block_area_max_to_median", "h_block_area_cv",
    "h_gap_cv", "h_max_gap_to_median", "h_headline_count",
    "h_headline_area_frac", "h_pull_quote_count",
]
print("\n=== New visual-hierarchy feature stats ===")
for col in new_features:
    vals = [r[col] for r in rows if isinstance(r[col], (int, float))]
    print(f"  {col}: {stats(vals)}")

print("\nCorrelations with technical_grade:")
tg = np.array([r["technical_grade"] for r in rows], dtype=float)
sg = np.array([r["style_grade"] for r in rows], dtype=float)
for col in new_features:
    vals = [r[col] for r in rows if isinstance(r[col], (int, float))]
    if len(vals) == len(rows) and np.std(vals) > 0:
        arr = np.array(vals, dtype=float)
        corr_t = np.corrcoef(arr, tg)[0, 1]
        corr_s = np.corrcoef(arr, sg)[0, 1]
        print(f"  {col}: r(tech)={corr_t:+.4f}  r(style)={corr_s:+.4f}")
    else:
        print(f"  {col}: zero variance or NA")
