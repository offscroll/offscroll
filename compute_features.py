#!/usr/bin/env python3
"""
Compute layout features for graded spreads (Task #238).

For each graded spread in batch-001.csv, extract computable features
from edition metadata.json and edition.json. Features are written to
training/features/features.csv along with features-metadata.md.

Features requiring image analysis are noted but not computed (NA).
Spread-level item distributions are approximated via linear word-count
allocation (exact placement requires a render).

Seed: 42. Split: 80% train / 20% val.
"""

import csv
import json
import math
import os
import random

BASE = "/home/modus/repos/belle/offscroll/training"
GRADES_PATH = f"{BASE}/grades/batch-001.csv"
EDITIONS_PATH = f"{BASE}/editions"
OUTPUT_PATH = f"{BASE}/features/features.csv"
META_PATH = f"{BASE}/features/features-metadata.md"

WORDS_PER_PAGE_CAPACITY = 450  # rough typographic estimate for 2–3 col newspaper page


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def template_entropy(items):
    """Shannon entropy (bits) of layout_hint distribution."""
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
    """Coefficient of variation of word counts across items."""
    wcs = [i.get("word_count") or 0 for i in items]
    if len(wcs) < 2:
        return 0.0
    mean = sum(wcs) / len(wcs)
    if mean == 0:
        return 0.0
    variance = sum((w - mean) ** 2 for w in wcs) / len(wcs)
    return round(math.sqrt(variance) / mean, 6)


def anchor_strength_ratio(items):
    """max word count / mean word count across items."""
    wcs = [i.get("word_count") or 0 for i in items]
    if not wcs:
        return 1.0
    mean = sum(wcs) / len(wcs)
    if mean == 0:
        return 1.0
    return round(max(wcs) / mean, 6)


def estimate_spread_words(all_items, left_page, right_page, total_pages):
    """
    Estimate word count on pages [left_page, right_page] using linear
    cumulative word-count allocation. Pro-rates partial item overlaps
    so the result approximates actual words per spread rather than
    summing entire items that straddle boundaries.

    Returns: (estimated_words, list_of_overlapping_items)
    'overlapping_items' includes any item with any overlap, used for
    counting discrete quantities (brief count, source count, etc.).
    """
    if not all_items or total_pages == 0:
        return 0.0, []

    total_words = sum(i.get("word_count") or 0 for i in all_items)
    if total_words == 0:
        # Uniform by item count
        n = len(all_items)
        item_frac = 1.0 / n
        p_start = (left_page - 1) / total_pages
        p_end = right_page / total_pages
        est = total_words  # 0 anyway
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
        # Overlap with spread window
        overlap_start = max(item_start, page_start_word)
        overlap_end = min(item_end, page_end_word)
        if overlap_end > overlap_start:
            # Pro-rate word count by overlap fraction
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
# 80/20 train/val split (seeded for reproducibility)
# ─────────────────────────────────────────────────────────────

rng = random.Random(42)
shuffled = list(spread_ids)
rng.shuffle(shuffled)
n_train = round(len(shuffled) * 0.8)
train_set = set(shuffled[:n_train])
val_set = set(shuffled[n_train:])

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
    spread_meta = next((s for s in (meta.get("spreads") or []) if s["spread_id"] == spread_id), None)
    if spread_meta is None:
        print(f"WARNING: {spread_id} not in {edition_id} metadata")
        continue

    pages = spread_meta["pages"]
    spread_type = spread_meta["type"]
    left_page = pages[0]["page_number"]
    right_page = pages[-1]["page_number"]
    n_pages_in_spread = len(pages)
    total_pages = meta.get("num_pages") or len(meta.get("pages") or [])

    # ── Edition-level aggregates ────────────────────────────
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

    # ── Spread-estimated features ───────────────────────────
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

    # D7: template entropy for estimated spread items
    d7_template_entropy = template_entropy(spread_items)

    # D8 proxy: word count coefficient of variation for spread items
    d8_word_count_cv = word_count_cv(spread_items)

    # Anchor strength: max / mean word count for spread items
    anchor_str = anchor_strength_ratio(spread_items)

    # D3 proxy: image fraction (images per item on spread)
    d3_image_fraction = round(est_image_count / est_n, 6) if est_n > 0 else 0.0

    # D5 proxy: fill fraction vs. fixed typographic capacity
    # capacity = WORDS_PER_PAGE_CAPACITY * pages_in_spread
    d5_fill_fraction = round(
        est_words / (WORDS_PER_PAGE_CAPACITY * n_pages_in_spread), 6
    )

    # ── Spread position features ────────────────────────────
    role = page_role(left_page, right_page, total_pages, spread_type)
    is_front = int(role == "front")
    is_terminal = int(role in ("terminal", "solo_terminal"))
    is_solo = int(spread_type == "solo")
    page_pos_frac = round(
        (left_page - 1) / (total_pages - 1) if total_pages > 1 else 0.0, 6
    )

    # ── Assemble row ────────────────────────────────────────
    g = grades[spread_id]
    rows.append({
        "spread_id": spread_id,
        "split": "train" if spread_id in train_set else "val",
        "technical_grade": g["technical"],
        "style_grade": g["style"],
        # Spread structure
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
        # Edition-level features
        "edition_item_count": n_items,
        "edition_word_count_total": ed_word_total,
        "edition_word_count_mean": ed_word_mean,
        "edition_brief_frac": ed_brief_frac,
        "edition_standard_frac": ed_standard_frac,
        "edition_template_entropy": ed_template_entropy,
        "edition_image_count_total": ed_image_count,
        "edition_source_count": ed_source_count,
        "edition_section_count": ed_section_count,
        # Spread-estimated features
        "est_item_count": est_n,
        "est_items_per_page": est_items_per_page,
        "est_word_count": est_words,
        "est_words_per_page": est_words_per_page,
        "est_word_count_mean": est_word_mean,
        "est_brief_count": est_brief_count,
        "est_standard_count": est_standard_count,
        "est_image_count": est_image_count,
        "est_source_count": est_source_count,
        # D-feature proxies (metadata-derivable)
        "d3_image_fraction": d3_image_fraction,
        "d5_fill_fraction": d5_fill_fraction,
        "d7_template_entropy": d7_template_entropy,
        "d8_word_count_cv": d8_word_count_cv,
        "anchor_strength": anchor_str,
        # Features requiring rendered output (not computable from metadata)
        "d2_orphans": "NA",
        "d2_widows": "NA",
        "d4_col_balance": "NA",
        "d6_dead_space": "NA",
    })

print(f"Computed features for {len(rows)} spreads")

# ─────────────────────────────────────────────────────────────
# Write features.csv
# ─────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fieldnames = list(rows[0].keys())

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {OUTPUT_PATH}")

# ─────────────────────────────────────────────────────────────
# Summary stats for sanity check
# ─────────────────────────────────────────────────────────────

def stats(vals):
    if not vals:
        return "empty"
    return f"min={min(vals):.3f} max={max(vals):.3f} mean={sum(vals)/len(vals):.3f}"

train_count = sum(1 for r in rows if r["split"] == "train")
val_count = sum(1 for r in rows if r["split"] == "val")
print(f"Train: {train_count}, Val: {val_count}")

for col in ["d5_fill_fraction", "d7_template_entropy", "d8_word_count_cv", "anchor_strength",
            "est_items_per_page", "est_words_per_page", "page_position_frac"]:
    vals = [r[col] for r in rows if isinstance(r[col], (int, float))]
    print(f"  {col}: {stats(vals)}")
