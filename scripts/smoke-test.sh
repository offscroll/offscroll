#!/usr/bin/env bash
# smoke-test.sh — Quick sanity check for the OffScroll pipeline.
#
# Tests:
#   1. CLI is installed and responds to --help
#   2. Setup creates config directory and file from a test config
#   3. Ingest runs without crashing against the test config
#
# Uses a temp directory so it doesn't touch ~/.offscroll.

set -euo pipefail

PASS=0
FAIL=0
TMPDIR=""

cleanup() {
    if [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ]; then
        rm -rf "$TMPDIR"
    fi
}
trap cleanup EXIT

report() {
    local status="$1"
    local label="$2"
    if [ "$status" -eq 0 ]; then
        echo "  PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $label"
        FAIL=$((FAIL + 1))
    fi
}

echo "OffScroll Smoke Test"
echo "==================="
echo

# --- Test 1: CLI is installed ---
if offscroll --help > /dev/null 2>&1; then
    report 0 "CLI is installed (offscroll --help)"
else
    report 1 "CLI is installed (offscroll --help)"
    echo "  offscroll is not on PATH. Did you run 'pip install -e .'?"
    echo
    echo "Results: $PASS passed, $FAIL failed"
    exit 1
fi

# --- Set up temp directory with test config ---
TMPDIR=$(mktemp -d)
DATADIR="$TMPDIR/data"
CONFIGFILE="$TMPDIR/config.yaml"
mkdir -p "$DATADIR"

cat > "$CONFIGFILE" <<EOF
feeds:
  rss: []
  mastodon: []
  bluesky: []

embedding:
  provider: "ollama"
  ollama_model: "nomic-embed-text"
  ollama_url: "http://localhost:11434"

curation:
  model: "ollama"
  ollama_model: "llama3.2:3b"
  ollama_url: "http://localhost:11434"

newspaper:
  title: "Smoke Test Edition"
  page_target: 2
  columns: 3
  page_size: "letter"

output:
  data_dir: "$DATADIR"

logging:
  level: "WARNING"
EOF

# --- Test 2: Config file is valid (offscroll loads it) ---
if offscroll --config "$CONFIGFILE" ingest > /dev/null 2>&1; then
    report 0 "Ingest runs with empty feed list (no crash)"
else
    report 1 "Ingest runs with empty feed list (no crash)"
fi

# --- Test 3: Ingest with a feed URL (uses a public test feed) ---
# Add a known stable RSS feed to the config
cat > "$CONFIGFILE" <<EOF
feeds:
  rss:
    - url: "https://www.rssboard.org/files/sample-rss-2.xml"

embedding:
  provider: "ollama"
  ollama_model: "nomic-embed-text"
  ollama_url: "http://localhost:11434"

curation:
  model: "ollama"
  ollama_model: "llama3.2:3b"
  ollama_url: "http://localhost:11434"

newspaper:
  title: "Smoke Test Edition"
  page_target: 2
  columns: 3
  page_size: "letter"

output:
  data_dir: "$DATADIR"

logging:
  level: "INFO"
EOF

if offscroll --config "$CONFIGFILE" ingest 2>&1; then
    report 0 "Ingest with live RSS feed"
else
    report 1 "Ingest with live RSS feed"
fi

# --- Summary ---
echo
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
