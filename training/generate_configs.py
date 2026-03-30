#!/usr/bin/env python3
"""
Generate 100 random OffScroll training configs from the feed corpus.

Strategy:
- Define newspaper archetypes with category weight profiles
- Each archetype produces a plausible newspaper with a coherent content mix
- Vary size (10-30 feeds), content profile, post frequency mix
- Ensure diversity: no two configs are the same archetype+seed combo

Archetypes:
  1. Tech Daily (tech + programming heavy)
  2. Hacker News (programming + engineering blogs + startups)
  3. Science Journal (science + environment + health)
  4. Culture Mag (movies, music, books, humor)
  5. Lifestyle (food, travel, fashion, diy)
  6. News Desk (journalism + politics + business)
  7. Design Studio (architecture, photography, fashion)
  8. Gaming Hub (gaming + tech + humor)
  9. Finance Focus (personal-finance + business + startups)
 10. Eclectic Mix (uniform random across all categories)
 11. Long Reads (text-heavy, long articles only)
 12. Visual Feed (image-heavy feeds prioritized)
 13. Apple Insider (ios-apple + tech + programming)
 14. Android World (android + tech + programming)
 15. AI/ML Focus (ai-ml + programming + engineering-blogs)
 16. Green Planet (environment + food + science)
 17. History & Essays (history + personal-essays + books)
 18. Sports & Cars (sports + sports-cars + niche-hobbies)
 19. Education (education + science + books)
 20. Web Dev (web-development + programming + engineering-blogs)
"""

import json
import random
import yaml
import sys
import os
from pathlib import Path
from collections import Counter

CORPUS_PATH = os.path.expanduser(
    "~/repos/belle/iras/docs/offscroll/feed-corpus.json"
)
OUTPUT_DIR = os.path.expanduser(
    "~/repos/belle/offscroll/training/configs"
)

# Newspaper name templates
NEWSPAPER_NAMES = [
    "The {adj} {noun}",
    "The {adj} {noun}",
    "{adj} {noun}",
    "The {noun}",
]

NAME_ADJS = [
    "Morning", "Evening", "Daily", "Weekly", "Sunday", "Digital",
    "Modern", "Global", "Local", "Independent", "Free", "Open",
    "Connected", "Curious", "Informed", "Thoughtful", "Critical",
    "Electric", "Silicon", "Pacific", "Atlantic", "Northern",
    "Southern", "Coastal", "Urban", "Metropolitan", "Civic",
    "Public", "Pioneer", "Frontier", "New", "Old", "Bold",
    "Quiet", "Bright", "Sharp", "Clear", "Deep", "Wide", "Swift",
]

NAME_NOUNS = [
    "Dispatch", "Herald", "Tribune", "Gazette", "Observer",
    "Chronicle", "Times", "Post", "Register", "Beacon",
    "Standard", "Record", "Review", "Examiner", "Monitor",
    "Sentinel", "Journal", "Bulletin", "Courier", "Signal",
    "Wire", "Digest", "Report", "Edition", "Press",
    "Ledger", "Mirror", "Star", "Sun", "Globe",
]

# Archetype definitions: category weights and constraints
ARCHETYPES = {
    "tech-daily": {
        "weights": {
            "technology": 5, "programming": 3, "engineering-blogs": 3,
            "ai-ml": 2, "web-development": 1, "startups": 1,
        },
        "size_range": (15, 25),
        "freq_pref": "daily",
    },
    "hacker-news": {
        "weights": {
            "programming": 5, "engineering-blogs": 5, "startups": 3,
            "technology": 2, "ai-ml": 2, "web-development": 2,
        },
        "size_range": (18, 28),
        "freq_pref": None,
    },
    "science-journal": {
        "weights": {
            "science": 5, "science-space": 4, "environment": 3,
            "health": 3, "education": 2, "technology": 1,
        },
        "size_range": (12, 20),
        "freq_pref": None,
    },
    "culture-mag": {
        "weights": {
            "movies-tv": 4, "music": 4, "books": 4, "humor": 3,
            "personal-essays": 3, "gaming": 1,
        },
        "size_range": (12, 22),
        "freq_pref": None,
    },
    "lifestyle": {
        "weights": {
            "food": 5, "travel": 5, "fashion-beauty": 4,
            "diy-home": 3, "photography": 2, "health": 2,
        },
        "size_range": (12, 22),
        "freq_pref": None,
    },
    "news-desk": {
        "weights": {
            "journalism": 5, "politics": 5, "business": 3,
            "technology": 1, "environment": 1,
        },
        "size_range": (15, 25),
        "freq_pref": "daily",
    },
    "design-studio": {
        "weights": {
            "architecture-design": 5, "photography": 4,
            "fashion-beauty": 3, "diy-home": 2, "niche-hobbies": 2,
        },
        "size_range": (10, 18),
        "freq_pref": None,
        "profile_pref": "image-heavy",
    },
    "gaming-hub": {
        "weights": {
            "gaming": 5, "technology": 3, "humor": 2,
            "movies-tv": 1, "niche-hobbies": 1,
        },
        "size_range": (12, 20),
        "freq_pref": None,
    },
    "finance-focus": {
        "weights": {
            "personal-finance": 5, "business": 5, "startups": 3,
            "technology": 1, "journalism": 1,
        },
        "size_range": (12, 20),
        "freq_pref": None,
    },
    "eclectic": {
        "weights": {},  # uniform across all
        "size_range": (15, 30),
        "freq_pref": None,
    },
    "long-reads": {
        "weights": {
            "personal-essays": 5, "journalism": 4, "books": 3,
            "history": 3, "programming": 2, "science": 2,
        },
        "size_range": (10, 18),
        "freq_pref": None,
        "length_pref": "long",
        "profile_pref": "text-heavy",
    },
    "visual-feed": {
        "weights": {
            "photography": 5, "architecture-design": 4,
            "fashion-beauty": 4, "food": 3, "travel": 3,
            "diy-home": 2,
        },
        "size_range": (12, 20),
        "freq_pref": None,
        "profile_pref": "image-heavy",
    },
    "apple-insider": {
        "weights": {
            "ios-apple": 5, "technology": 3, "programming": 2,
            "engineering-blogs": 1,
        },
        "size_range": (12, 20),
        "freq_pref": None,
    },
    "android-world": {
        "weights": {
            "android": 5, "technology": 3, "programming": 2,
            "engineering-blogs": 1,
        },
        "size_range": (12, 20),
        "freq_pref": None,
    },
    "ai-ml-focus": {
        "weights": {
            "ai-ml": 5, "programming": 3, "engineering-blogs": 3,
            "science": 2, "technology": 1,
        },
        "size_range": (12, 22),
        "freq_pref": None,
    },
    "green-planet": {
        "weights": {
            "environment": 5, "food": 3, "science": 3,
            "health": 2, "travel": 1,
        },
        "size_range": (10, 18),
        "freq_pref": None,
    },
    "history-essays": {
        "weights": {
            "history": 5, "personal-essays": 5, "books": 4,
            "education": 2, "journalism": 1,
        },
        "size_range": (10, 18),
        "freq_pref": None,
        "profile_pref": "text-heavy",
    },
    "sports-autos": {
        "weights": {
            "sports": 5, "sports-cars": 5, "niche-hobbies": 3,
            "humor": 1,
        },
        "size_range": (10, 18),
        "freq_pref": None,
    },
    "education": {
        "weights": {
            "education": 5, "science": 3, "books": 3,
            "history": 2, "programming": 1,
        },
        "size_range": (10, 18),
        "freq_pref": None,
    },
    "web-dev": {
        "weights": {
            "web-development": 5, "programming": 4,
            "engineering-blogs": 3, "technology": 1,
        },
        "size_range": (12, 22),
        "freq_pref": None,
    },
}

# Page sizes and column counts for variety
PAGE_SIZES = ["letter", "a4"]
COLUMN_OPTIONS = [2, 3, 4]
PAGE_TARGETS = [6, 8, 10, 12, 14, 16]


def load_corpus():
    with open(CORPUS_PATH) as f:
        return json.load(f)


def generate_newspaper_name(rng):
    template = rng.choice(NEWSPAPER_NAMES)
    adj = rng.choice(NAME_ADJS)
    noun = rng.choice(NAME_NOUNS)
    return template.format(adj=adj, noun=noun)


def select_feeds(corpus, archetype_name, rng):
    """Select feeds for a config based on archetype weights."""
    arch = ARCHETYPES[archetype_name]
    weights = arch["weights"]
    lo, hi = arch["size_range"]
    target_size = rng.randint(lo, hi)
    freq_pref = arch.get("freq_pref")
    length_pref = arch.get("length_pref")
    profile_pref = arch.get("profile_pref")

    # Build per-feed scores
    scored = []
    for feed in corpus:
        cat = feed["category"]
        if weights:
            base = weights.get(cat, 0)
            # Small chance to include off-category feeds for variety
            if base == 0:
                base = 0.1
        else:
            # Eclectic: uniform
            base = 1.0

        # Boost for frequency preference
        if freq_pref and feed.get("post_frequency") == freq_pref:
            base *= 1.5

        # Boost for length preference
        if length_pref and feed.get("typical_length") == length_pref:
            base *= 2.0

        # Boost for content profile preference
        if profile_pref and feed.get("content_profile") == profile_pref:
            base *= 2.0

        scored.append((feed, base))

    # Weighted sample without replacement
    selected = []
    remaining = list(scored)
    for _ in range(min(target_size, len(remaining))):
        if not remaining:
            break
        total = sum(w for _, w in remaining)
        if total <= 0:
            break
        r = rng.random() * total
        cumulative = 0
        for i, (feed, w) in enumerate(remaining):
            cumulative += w
            if cumulative >= r:
                selected.append(feed)
                remaining.pop(i)
                break

    return selected


def make_config(feeds, name, page_target, columns, page_size, rng):
    """Build an OffScroll-compatible config dict."""
    rss_entries = []
    for feed in feeds:
        entry = {"url": feed["url"]}
        if feed.get("name"):
            entry["name"] = feed["name"]
        rss_entries.append(entry)

    config = {
        "feeds": {
            "rss": rss_entries,
        },
        "ingestion": {
            "poll_interval_minutes": 60,
            "max_items_per_feed": 100,
            "download_images": True,
            "min_image_dimension": 200,
        },
        "embedding": {
            "provider": "ollama",
            "ollama_model": "nomic-embed-text",
            "ollama_url": "http://localhost:11434",
        },
        "curation": {
            "model": "ollama",
            "ollama_model": "llama3.1:8b",
            "ollama_url": "http://localhost:11434",
            "weights": {
                "coverage": round(0.5 + rng.random(), 2),
                "redundancy": round(0.5 + rng.random(), 2),
                "quality": round(0.5 + rng.random(), 2),
                "diversity": round(0.5 + rng.random(), 2),
                "fit": round(0.5 + rng.random(), 2),
            },
            "optimizer_iterations": 500,
        },
        "newspaper": {
            "title": name,
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
        "output": {
            "data_dir": "~/.offscroll/data",
            "retention_weeks": 0,
        },
        "schedule": {
            "compile_day": rng.choice(
                ["monday", "tuesday", "wednesday", "thursday",
                 "friday", "saturday", "sunday"]
            ),
            "compile_time": f"{rng.randint(6, 22):02d}:00",
            "timezone": "America/New_York",
        },
        "logging": {
            "level": "INFO",
            "file": "~/.offscroll/offscroll.log",
        },
    }
    return config


def generate_all():
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} feeds from corpus")

    # Index by category
    by_category = {}
    for feed in corpus:
        cat = feed["category"]
        by_category.setdefault(cat, []).append(feed)

    # Distribute 100 configs across archetypes
    archetype_names = list(ARCHETYPES.keys())
    # 5 of each archetype = 100 configs
    assignments = []
    for arch in archetype_names:
        assignments.extend([arch] * 5)

    assert len(assignments) == 100

    # Shuffle to randomize numbering
    master_rng = random.Random(42)
    master_rng.shuffle(assignments)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    used_names = set()
    stats = {
        "archetype_counts": Counter(),
        "size_distribution": [],
        "category_coverage": Counter(),
        "profile_distribution": Counter(),
        "freq_distribution": Counter(),
    }

    for i, arch_name in enumerate(assignments, 1):
        seed = 1000 + i
        rng = random.Random(seed)

        feeds = select_feeds(corpus, arch_name, rng)

        # Generate unique newspaper name
        for _ in range(50):
            name = generate_newspaper_name(rng)
            if name not in used_names:
                used_names.add(name)
                break

        page_target = rng.choice(PAGE_TARGETS)
        columns = rng.choice(COLUMN_OPTIONS)
        page_size = rng.choice(PAGE_SIZES)

        config = make_config(feeds, name, page_target, columns, page_size, rng)

        filename = f"training-{i:03d}.yaml"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w") as f:
            f.write(f"# OffScroll Training Config {i:03d}\n")
            f.write(f"# Archetype: {arch_name}\n")
            f.write(f"# Feeds: {len(feeds)}\n")
            f.write(f"# Generated by Belle (IRAS #171)\n\n")
            yaml.dump(config, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)

        # Stats
        stats["archetype_counts"][arch_name] += 1
        stats["size_distribution"].append(len(feeds))
        for feed in feeds:
            stats["category_coverage"][feed["category"]] += 1
            stats["profile_distribution"][feed.get("content_profile", "unknown")] += 1
            stats["freq_distribution"][feed.get("post_frequency", "unknown")] += 1

    # Print summary
    print(f"\nGenerated 100 configs in {OUTPUT_DIR}")
    sizes = stats["size_distribution"]
    print(f"Feed count per config: min={min(sizes)}, max={max(sizes)}, "
          f"avg={sum(sizes)/len(sizes):.1f}")
    print(f"\nArchetype distribution:")
    for arch, count in sorted(stats["archetype_counts"].items()):
        print(f"  {arch}: {count}")
    print(f"\nCategory coverage (total feed slots across all configs):")
    for cat, count in stats["category_coverage"].most_common():
        print(f"  {cat}: {count}")
    print(f"\nContent profile distribution:")
    for prof, count in sorted(stats["profile_distribution"].items()):
        print(f"  {prof}: {count}")
    print(f"\nPost frequency distribution:")
    for freq, count in sorted(stats["freq_distribution"].items()):
        print(f"  {freq}: {count}")

    return stats


if __name__ == "__main__":
    generate_all()
