#!/usr/bin/env python3
"""Validate feed corpus URLs. Remove dead feeds. Regenerate configs."""

import json
import os
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

CORPUS_PATH = os.path.expanduser(
    "~/repos/belle/iras/docs/offscroll/feed-corpus.json"
)

TIMEOUT = 10  # seconds per feed


def check_feed(feed):
    """Return (feed, alive, status_info)."""
    url = feed["url"]
    ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "OffScroll/1.0 (feed validator)"},
        )
        resp = urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx)
        ct = resp.headers.get("content-type", "").lower()
        # Check for XML/RSS/Atom content
        is_feed = any(x in ct for x in ["xml", "rss", "atom", "json", "text"])
        if not is_feed:
            # Some feeds return text/html but are actually valid
            # Read a small chunk to check for XML markers
            chunk = resp.read(500).decode("utf-8", errors="ignore")
            is_feed = any(x in chunk.lower() for x in [
                "<?xml", "<rss", "<feed", "<atom", '"items"'
            ])
        return (feed, is_feed, f"status={resp.status} ct={ct[:40]}")
    except Exception as e:
        return (feed, False, str(e)[:80])


def main():
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    print(f"Validating {len(corpus)} feeds (timeout={TIMEOUT}s)...")

    alive = []
    dead = []

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(check_feed, feed): feed for feed in corpus}
        done = 0
        for future in as_completed(futures):
            done += 1
            feed, is_alive, info = future.result()
            if is_alive:
                alive.append(feed)
            else:
                dead.append((feed, info))
            if done % 50 == 0:
                print(f"  {done}/{len(corpus)} checked...")

    print(f"\nResults:")
    print(f"  Alive: {len(alive)}")
    print(f"  Dead:  {len(dead)} ({100*len(dead)/len(corpus):.1f}%)")

    if dead:
        print(f"\nDead feeds:")
        for feed, info in sorted(dead, key=lambda x: x[0]["name"]):
            print(f"  - {feed['name']}: {info}")

    # Write validated corpus
    with open(CORPUS_PATH, "w") as f:
        json.dump(alive, f, indent=2, ensure_ascii=False)
    print(f"\nUpdated corpus: {len(alive)} feeds")

    return len(alive), len(dead)


if __name__ == "__main__":
    main()
