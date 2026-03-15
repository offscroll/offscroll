# OffScroll

Your personal newspaper from the open web.

OffScroll ingests content from RSS feeds, Mastodon, and Bluesky,
curates it with AI into a cohesive selection, and renders a
newspaper-quality PDF you can print and read offline. Wake up to a
personalized newspaper with your morning coffee -- no algorithms, no
ads, no tracking.

## Why

Social media feeds are designed to keep you scrolling. OffScroll
inverts that: it pulls content from your chosen sources, curates it
into a finite, readable edition, and gets you off the screen. You
control what you read. Your data stays on your machine.

## Features

- **Multi-source ingestion** -- RSS/Atom feeds, Mastodon, Bluesky
- **AI-powered curation** -- clusters similar content, selects the
  best, generates editorial context
- **Newspaper-quality PDF output** -- multi-column layout,
  typography, pull quotes, images
- **Email digest** -- optional HTML email delivery
- **Fully local by default** -- uses Ollama for embeddings and
  curation. No cloud API keys required.
- **Cloud LLM option** -- optionally use Claude or OpenAI for
  higher-quality curation
- **Self-hosted** -- runs on your hardware, from a Raspberry Pi to
  a full desktop

## Quick Start

```bash
# Clone
git clone https://github.com/offscroll/offscroll.git
cd offscroll

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# System dependencies (WeasyPrint needs these)
# Debian/Ubuntu:
sudo apt-get install libcairo2-dev libpango1.0-dev libgdk-pixbuf2.0-dev libffi-dev
# Arch Linux:
sudo pacman -S cairo pango gdk-pixbuf2 libffi
# macOS:
brew install cairo pango gdk-pixbuf libffi
# Other: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html

# Install and start Ollama (https://ollama.com)
ollama pull nomic-embed-text   # ~274 MB (embeddings)
ollama pull llama3.1:8b        # ~4.9 GB (curation)

# Interactive setup
offscroll setup
```

The setup wizard walks you through newspaper name, feed URLs,
optional Mastodon/Bluesky accounts, Ollama connection, and email
settings. It writes `~/.offscroll/config.yaml`.

For non-interactive environments (SSH, containers), copy
`config.example.yaml` to `~/.offscroll/config.yaml` and edit
directly.

## Usage

```bash
# Individual pipeline steps
offscroll ingest          # Poll feeds, store new items
offscroll curate          # Embed, cluster, select, editorial
offscroll render pdf      # Generate newspaper PDF

# Full pipeline
offscroll run             # ingest -> curate -> render

# Manage feeds
offscroll feeds add URL   # Add an RSS feed
offscroll feeds import FILE.opml  # Import from OPML
offscroll feeds list      # Show configured feeds

# Database
offscroll db stats        # Show item/feed/edition counts
```

## Docker

Docker Compose includes an Ollama companion container -- no host
Ollama install needed.

```bash
docker compose build
docker compose up -d
docker compose run offscroll ingest
```

## Configuration

Edit `~/.offscroll/config.yaml`. See `config.example.yaml` for all
options including:

- Feed sources (RSS, Mastodon, Bluesky, OPML import)
- Embedding provider (Ollama, OpenAI, sentence-transformers)
- Curation model (Ollama, Claude, OpenAI)
- Newspaper settings (title, page count, columns, page size)
- Email digest (SMTP settings)
- Output directory and retention
- Scheduling (compile day/time)

### Cloud Providers (Optional)

```bash
pip install -e ".[cloud]"
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

## Hardware Requirements

| Setup | RAM | Disk | Notes |
|-------|-----|------|-------|
| Desktop with `llama3.1:8b` | 8 GB+ | ~6 GB | Recommended |
| Raspberry Pi 5 (8 GB) with `llama3.2:3b` | 8 GB | ~3 GB | Works, curation slower |
| Raspberry Pi 4 (4 GB) with `llama3.2:3b` | 4 GB | ~3 GB | Marginal, may swap |
| Low-RAM devices | -- | -- | Use cloud provider option |

For low-RAM devices, use the smaller model:

```yaml
curation:
  ollama_model: "llama3.2:3b"
```

## Automation

```bash
# Daily newspaper at 6am (cron):
0 6 * * * cd /path/to/offscroll && .venv/bin/offscroll run >> ~/.offscroll/cron.log 2>&1

# Docker:
0 6 * * * cd /path/to/offscroll && docker compose run --rm offscroll run >> ~/.offscroll/cron.log 2>&1
```

## Architecture

```
src/offscroll/
  cli.py               # CLI entry point (click)
  config.py            # Configuration loading and validation
  models.py            # Shared data models
  logging.py           # Logging setup
  ingestion/           # RSS/Atom, Mastodon, Bluesky, embeddings, clustering, storage
  curation/            # Loss-function selection, LLM editorial, email digest
  layout/              # Jinja2 templates, CSS, WeasyPrint PDF rendering
    templates/         # HTML templates (newspaper components)
    styles/            # CSS (newspaper layout, typography)
    fonts/             # Source Sans/Serif/Code Pro (bundled)
```

## Development

```bash
pip install -e ".[dev]"

# Run tests
pytest
pytest -m "not slow"              # Skip API tests
pytest --cov=offscroll             # With coverage

# Lint and type check
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/offscroll/
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
