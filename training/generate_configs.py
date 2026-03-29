#!/usr/bin/env python3
"""
Brief #171: Generate 100 random subset configs for OffScroll training.

Steps:
1. Validate all feed URLs from the corpus (parallel HTTP HEAD/GET)
2. Generate 100 diverse subsets with plausible newspaper profiles
3. Write OffScroll-compatible YAML configs
"""

import json
import random
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml

CORPUS_PATH = os.path.expanduser(
    "~/repos/belle/iras/docs/offscroll/feed-corpus.json"
)
OUTPUT_DIR = os.path.expanduser("~/offscroll/training/configs")
VALIDATED_CORPUS_PATH = os.path.expanduser(
    "~/repos/belle/iras/docs/offscroll/feed-corpus-validated.json"
)

# Newspaper archetypes — each defines a content strategy
ARCHETYPES = [
    {
        "name": "Tech Daily",
        "category_weights": {
            "technology": 5, "programming": 3, "ai-ml": 3,
            "engineering-blogs": 2, "web-development": 2,
            "ios-apple": 1, "android": 1, "startups": 1,
        },
        "size_range": (15, 25),
        "frequency_bias": "daily",
    },
    {
        "name": "Deep Tech Weekly",
        "category_weights": {
            "engineering-blogs": 5, "programming": 4,
            "ai-ml": 3, "science": 2, "technology": 1,
        },
        "size_range": (12, 20),
        "frequency_bias": "weekly",
    },
    {
        "name": "News & Politics",
        "category_weights": {
            "journalism": 5, "politics": 5, "business": 3,
            "environment": 2, "education": 1,
        },
        "size_range": (15, 25),
        "frequency_bias": "daily",
    },
    {
        "name": "Culture & Arts",
        "category_weights": {
            "books": 4, "movies-tv": 4, "music": 3,
            "photography": 3, "architecture-design": 3,
            "fashion-beauty": 2, "personal-essays": 2,
            "humor": 2,
        },
        "size_range": (15, 25),
        "frequency_bias": None,
    },
    {
        "name": "Lifestyle",
        "category_weights": {
            "food": 4, "travel": 4, "diy-home": 3,
            "health": 3, "fashion-beauty": 2,
            "personal-finance": 2, "niche-hobbies": 2,
        },
        "size_range": (15, 25),
        "frequency_bias": None,
    },
    {
        "name": "Science & Discovery",
        "category_weights": {
            "science": 5, "science-space": 5, "environment": 3,
            "history": 3, "education": 2, "ai-ml": 1,
        },
        "size_range": (12, 20),
        "frequency_bias": None,
    },
    {
        "name": "Indie Dev",
        "category_weights": {
            "programming": 4, "engineering-blogs": 4,
            "web-development": 3, "startups": 3,
            "personal-essays": 2, "niche-hobbies": 1,
        },
        "size_range": (10, 18),
        "frequency_bias": "irregular",
    },
    {
        "name": "Visual & Design",
        "category_weights": {
            "photography": 5, "architecture-design": 5,
            "fashion-beauty": 3, "movies-tv": 2,
            "gaming": 1,
        },
        "size_range": (12, 20),
        "content_profile_bias": "image-heavy",
        "frequency_bias": None,
    },
    {
        "name": "Long Reads",
        "category_weights": {
            "personal-essays": 5, "journalism": 4,
            "books": 3, "history": 3, "science": 2,
            "programming": 1,
        },
        "size_range": (10, 15),
        "length_bias": "long",
        "frequency_bias": None,
    },
    {
        "name": "Eclectic",
        "category_weights": {},  # Equal weight to all categories
        "size_range": (15, 28),
        "frequency_bias": None,
    },
    {
        "name": "Gaming & Entertainment",
        "category_weights": {
            "gaming": 5, "movies-tv": 4, "music": 3,
            "humor": 3, "technology": 1,
        },
        "size_range": (12, 20),
        "frequency_bias": None,
    },
    {
        "name": "Finance & Business",
        "category_weights": {
            "personal-finance": 5, "business": 5,
            "startups": 3, "journalism": 2,
            "technology": 1,
        },
        "size_range": (12, 20),
        "frequency_bias": None,
    },
    {
        "name": "Apple Ecosystem",
        "category_weights": {
            "ios-apple": 5, "technology": 3,
            "programming": 2, "engineering-blogs": 1,
        },
        "size_range": (10, 18),
        "frequency_bias": None,
    },
    {
        "name": "Mobile Dev",
        "category_weights": {
            "ios-apple": 4, "android": 4,
            "programming": 3, "engineering-blogs": 2,
            "web-development": 2,
        },
        "size_range": (12, 20),
        "frequency_bias": None,
    },
    {
        "name": "Food & Travel",
        "category_weights": {
            "food": 5, "travel": 5, "photography": 2,
            "personal-essays": 1,
        },
        "size_range": (12, 20),
        "frequency_bias": None,
    },
]

# Newspaper name templates
NEWSPAPER_NAMES = [
    "The {adj} {noun}",
    "The {adj} {noun}",
    "{adj} {noun}",
    "The {noun}",
]

ADJECTIVES = [
    "Morning", "Evening", "Daily", "Weekly", "Sunday", "Digital",
    "Modern", "New", "Open", "Free", "Independent", "Global",
    "Local", "Electric", "Midnight", "Golden", "Silver", "Iron",
    "Pacific", "Atlantic", "Northern", "Southern", "Eastern", "Western",
    "Coastal", "Mountain", "Prairie", "Desert", "Urban", "Metro",
]

NOUNS = [
    "Dispatch", "Herald", "Tribune", "Chronicle", "Gazette",
    "Observer", "Review", "Standard", "Post", "Record",
    "Beacon", "Sentinel", "Signal", "Telegraph", "Bulletin",
    "Register", "Courier", "Times", "Journal", "Press",
    "Wire", "Index", "Ledger", "Banner", "Mirror",
]


def validate_feed(feed, timeout=10):
    """Validate a single feed URL. Returns (feed, is_valid)."""
    url = feed["url"]
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "OffScroll/1.0 (feed validator)"},
            allow_redirects=True,
        )
        # Accept any 2xx response; also check for XML-ish content
        if resp.status_code < 200 or resp.status_code >= 400:
            return (feed, False, f"HTTP {resp.status_code}")
        content_type = resp.headers.get("Content-Type", "").lower()
        text_start = resp.text[:500].lower()
        is_feed = any(
            marker in content_type or marker in text_start
            for marker in ["xml", "rss", "atom", "feed", "<channel", "<entry"]
        )
        if not is_feed:
            return (feed, False, "Not a feed (no XML/RSS markers)")
        return (feed, True, "OK")
    except requests.exceptions.Timeout:
        return (feed, False, "Timeout")
    except requests.exceptions.ConnectionError:
        return (feed, False, "Connection error")
    except Exception as e:
        return (feed, False, str(e)[:80])


def validate_corpus(feeds):
    """Validate all feeds in parallel. Returns list of valid feeds."""
    valid = []
    dead = []
    print(f"Validating {len(feeds)} feeds (parallel, 10s timeout)...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(validate_feed, f): f for f in feeds}
        done = 0
        for future in as_completed(futures):
            done += 1
            feed, is_valid, reason = future.result()
            if is_valid:
                valid.append(feed)
            else:
                dead.append((feed, reason))
            if done % 50 == 0:
                print(f"  {done}/{len(feeds)} checked...")

    print(f"\nValidation complete:")
    print(f"  Valid: {len(valid)}")
    print(f"  Dead:  {len(dead)} ({100*len(dead)/len(feeds):.1f}%)")
    if dead:
        print(f"\nDead feeds:")
        for feed, reason in sorted(dead, key=lambda x: x[0]["name"]):
            print(f"  - {feed['name']}: {reason}")
    return valid, dead


def generate_newspaper_name(used_names):
    """Generate a unique newspaper name."""
    for _ in range(100):
        template = random.choice(NEWSPAPER_NAMES)
        adj = random.choice(ADJECTIVES)
        noun = random.choice(NOUNS)
        name = template.format(adj=adj, noun=noun)
        if name not in used_names:
            used_names.add(name)
            return name
    # Fallback with number
    i = len(used_names)
    return f"Edition {i}"


def select_feeds_for_archetype(archetype, feeds_by_category, all_feeds, target_size):
    """Select feeds matching an archetype's profile."""
    selected = []
    selected_urls = set()
    weights = archetype["category_weights"]

    # If eclectic (empty weights), use all categories equally
    if not weights:
        all_cats = list(feeds_by_category.keys())
        weights = {cat: 1 for cat in all_cats}

    # Build weighted category pool
    cat_pool = []
    for cat, weight in weights.items():
        cat_pool.extend([cat] * weight)

    # Optionally filter by content profile or length
    content_bias = archetype.get("content_profile_bias")
    length_bias = archetype.get("length_bias")
    freq_bias = archetype.get("frequency_bias")

    attempts = 0
    while len(selected) < target_size and attempts < target_size * 10:
        attempts += 1
        cat = random.choice(cat_pool)
        candidates = feeds_by_category.get(cat, [])
        if not candidates:
            continue

        # Apply biases as soft preferences (70% chance of matching)
        if content_bias and random.random() < 0.7:
            biased = [f for f in candidates if f["content_profile"] == content_bias]
            if biased:
                candidates = biased
        if length_bias and random.random() < 0.7:
            biased = [f for f in candidates if f["typical_length"] == length_bias]
            if biased:
                candidates = biased
        if freq_bias and random.random() < 0.5:
            biased = [f for f in candidates if f["post_frequency"] == freq_bias]
            if biased:
                candidates = biased

        feed = random.choice(candidates)
        if feed["url"] not in selected_urls:
            selected.append(feed)
            selected_urls.add(feed["url"])

    # If we're still short, fill from random feeds
    if len(selected) < target_size:
        remaining = [f for f in all_feeds if f["url"] not in selected_urls]
        random.shuffle(remaining)
        for feed in remaining[: target_size - len(selected)]:
            selected.append(feed)

    return selected


def generate_config(subset_id, feeds, newspaper_name, archetype_name):
    """Generate an OffScroll-compatible config dict."""
    # Vary some settings per config
    page_target = random.choice([6, 8, 10, 12])
    columns = random.choice([2, 3, 4])
    page_size = random.choice(["letter", "a4"])

    days = ["monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"]

    config = {
        "feeds": {
            "rss": [
                {"url": f["url"], "name": f["name"]}
                for f in feeds
            ],
        },
        "newspaper": {
            "title": newspaper_name,
            "subtitle_pattern": "Vol. {volume}, No. {issue}",
            "page_target": page_target,
            "columns": columns,
            "page_size": page_size,
            "margin_top": 0.5,
            "margin_bottom": 0.5,
            "margin_left": 0.5,
            "margin_right": 0.5,
            "column_gap": 0.2,
        },
        "schedule": {
            "compile_day": random.choice(days),
            "compile_time": f"{random.randint(5,22):02d}:00",
            "timezone": "America/New_York",
        },
        "output": {
            "data_dir": f"~/.offscroll/training/{subset_id}/data",
            "retention_weeks": 0,
        },
        "_meta": {
            "training_id": subset_id,
            "archetype": archetype_name,
            "feed_count": len(feeds),
            "categories": sorted(set(f["category"] for f in feeds)),
            "content_profiles": {
                p: sum(1 for f in feeds if f["content_profile"] == p)
                for p in ["text-heavy", "mixed", "image-heavy"]
                if any(f["content_profile"] == p for f in feeds)
            },
            "frequency_mix": {
                p: sum(1 for f in feeds if f["post_frequency"] == p)
                for p in ["daily", "weekly", "irregular", "monthly"]
                if any(f["post_frequency"] == p for f in feeds)
            },
        },
    }
    return config


def main():
    # Load corpus
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} feeds from corpus")

    # Validate
    valid_feeds, dead_feeds = validate_corpus(corpus)

    # Save validated corpus
    with open(VALIDATED_CORPUS_PATH, "w") as f:
        json.dump(valid_feeds, f, indent=2)
    print(f"\nSaved validated corpus ({len(valid_feeds)} feeds) to {VALIDATED_CORPUS_PATH}")

    # Index by category
    feeds_by_category = {}
    for feed in valid_feeds:
        cat = feed["category"]
        feeds_by_category.setdefault(cat, []).append(feed)

    print(f"\nCategories in validated corpus:")
    for cat, feeds in sorted(feeds_by_category.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(feeds)}")

    # Generate 100 subsets
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    used_names = set()
    random.seed(171)  # Reproducible

    # Distribute archetypes across 100 configs
    # Each archetype gets ~6-7 configs, with variation
    archetype_assignments = []
    for i in range(100):
        archetype_assignments.append(ARCHETYPES[i % len(ARCHETYPES)])
    random.shuffle(archetype_assignments)

    stats = {
        "total_configs": 100,
        "archetype_distribution": {},
        "size_distribution": {"10-15": 0, "16-20": 0, "21-25": 0, "26-30": 0},
        "category_coverage": set(),
        "feed_usage_count": {},
    }

    for i in range(100):
        subset_id = f"training-{i+1:03d}"
        archetype = archetype_assignments[i]
        arch_name = archetype["name"]

        lo, hi = archetype["size_range"]
        target_size = random.randint(lo, hi)

        feeds = select_feeds_for_archetype(
            archetype, feeds_by_category, valid_feeds, target_size
        )
        name = generate_newspaper_name(used_names)
        config = generate_config(subset_id, feeds, name, arch_name)

        # Write YAML
        config_path = os.path.join(OUTPUT_DIR, f"{subset_id}.yaml")
        with open(config_path, "w") as f:
            f.write(f"# {subset_id}: {name}\n")
            f.write(f"# Archetype: {arch_name}\n")
            f.write(f"# Feeds: {len(feeds)}\n\n")
            yaml.dump(config, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)

        # Track stats
        stats["archetype_distribution"][arch_name] = (
            stats["archetype_distribution"].get(arch_name, 0) + 1
        )
        if target_size <= 15:
            stats["size_distribution"]["10-15"] += 1
        elif target_size <= 20:
            stats["size_distribution"]["16-20"] += 1
        elif target_size <= 25:
            stats["size_distribution"]["21-25"] += 1
        else:
            stats["size_distribution"]["26-30"] += 1

        for feed in feeds:
            stats["category_coverage"].add(feed["category"])
            stats["feed_usage_count"][feed["url"]] = (
                stats["feed_usage_count"].get(feed["url"], 0) + 1
            )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Generated 100 configs in {OUTPUT_DIR}")
    print(f"\nArchetype distribution:")
    for arch, count in sorted(stats["archetype_distribution"].items()):
        print(f"  {arch}: {count}")
    print(f"\nSize distribution:")
    for bucket, count in stats["size_distribution"].items():
        print(f"  {bucket} feeds: {count}")
    print(f"\nCategory coverage: {len(stats['category_coverage'])} of "
          f"{len(feeds_by_category)} categories")
    usage = list(stats["feed_usage_count"].values())
    print(f"\nFeed usage: {len(usage)} unique feeds used")
    print(f"  Min uses: {min(usage)}, Max uses: {max(usage)}, "
          f"Mean: {sum(usage)/len(usage):.1f}")

    # Write summary doc
    summary_path = os.path.join(os.path.dirname(OUTPUT_DIR), "GENERATION-SUMMARY.md")
    stats["category_coverage"] = sorted(stats["category_coverage"])
    with open(summary_path, "w") as f:
        f.write("# Training Config Generation Summary\n\n")
        f.write(f"Generated: 2026-03-29 by Belle (Brief #171)\n\n")
        f.write(f"## Corpus\n\n")
        f.write(f"- Source: `docs/offscroll/feed-corpus.json` ({len(corpus)} feeds)\n")
        f.write(f"- After validation: {len(valid_feeds)} live feeds "
                f"({len(dead_feeds)} dead, {100*len(dead_feeds)/len(corpus):.1f}%)\n")
        f.write(f"- Validated corpus: `docs/offscroll/feed-corpus-validated.json`\n\n")
        f.write(f"## Generation Strategy\n\n")
        f.write("Each of the 100 configs represents a plausible newspaper with a\n")
        f.write("distinct content profile. Configs are generated from 15 archetypes:\n\n")
        for arch in ARCHETYPES:
            cats = ", ".join(arch["category_weights"].keys()) if arch["category_weights"] else "all"
            f.write(f"- **{arch['name']}** ({arch['size_range'][0]}-{arch['size_range'][1]} feeds): {cats}\n")
        f.write(f"\nEach archetype is assigned ~{100//len(ARCHETYPES)}-{100//len(ARCHETYPES)+1} configs. "
                f"Within each archetype,\nfeed selection uses weighted random sampling with soft biases for\n"
                f"content profile, post frequency, and article length.\n\n")
        f.write(f"## Distribution\n\n")
        f.write(f"### Archetype Counts\n\n")
        f.write("| Archetype | Count |\n|---|---|\n")
        for arch, count in sorted(stats["archetype_distribution"].items()):
            f.write(f"| {arch} | {count} |\n")
        f.write(f"\n### Size Distribution\n\n")
        f.write("| Feed Count | Configs |\n|---|---|\n")
        for bucket, count in stats["size_distribution"].items():
            f.write(f"| {bucket} | {count} |\n")
        f.write(f"\n### Category Coverage\n\n")
        f.write(f"{len(stats['category_coverage'])} of {len(feeds_by_category)} categories represented:\n")
        f.write(", ".join(stats["category_coverage"]) + "\n\n")
        f.write(f"### Feed Usage\n\n")
        f.write(f"- Unique feeds used: {len(usage)}\n")
        f.write(f"- Usage range: {min(usage)}-{max(usage)} (mean {sum(usage)/len(usage):.1f})\n")

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
