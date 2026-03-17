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
- **Cloud LLM option** *(planned)* -- Claude and OpenAI curation
  support is planned but not yet implemented
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

Docker is the recommended deployment method. The Compose file
includes an Ollama companion container so no host Ollama install
is needed.

```bash
# Clone and build
git clone https://github.com/offscroll/offscroll.git
cd offscroll
docker compose build

# Create host directories for persistent data
mkdir -p config data

# Copy example config and edit
cp config.example.yaml config/config.yaml
# Edit config/config.yaml with your feeds, newspaper name, etc.

# Pull Ollama models (first run only)
docker compose up -d ollama
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.1:8b

# Run the full pipeline
docker compose run --rm offscroll run

# Or run individual steps
docker compose run --rm offscroll ingest
docker compose run --rm offscroll curate
docker compose run --rm offscroll render pdf
docker compose run --rm offscroll status
```

Set your timezone in `.env` or export it:

```bash
echo "TZ=America/New_York" > .env
```

For Raspberry Pi or low-RAM devices, use the smaller curation
model in your `config/config.yaml`:

```yaml
curation:
  ollama_model: "llama3.2:3b"
```

### Automation with Docker

```bash
# Daily newspaper at 6am (add to crontab):
0 6 * * * cd /path/to/offscroll && docker compose run --rm offscroll -q run >> data/cron.log 2>&1
```


## Configuration

Edit `~/.offscroll/config.yaml`. See `config.example.yaml` for all
options including:

- Feed sources (RSS, Mastodon, Bluesky, OPML import)
- Embedding provider (Ollama; OpenAI and sentence-transformers planned)
- Curation model (Ollama; Claude and OpenAI planned)
- Newspaper settings (title, page count, columns, page size)
- Email digest (SMTP settings)
- Output directory and retention
- Scheduling (compile day/time)

### Cloud Providers *(Planned — not yet implemented)*

Cloud LLM curation (Claude, OpenAI) is on the roadmap but not yet
functional. The `[cloud]` extras install the client libraries, but the
curation pipeline does not use them yet. Only Ollama works today.

```bash
# Not yet functional — planned for a future release:
# pip install -e ".[cloud]"
# export ANTHROPIC_API_KEY="sk-ant-..."
# export OPENAI_API_KEY="sk-..."
```

## Hardware Requirements

| Setup | RAM | Disk | Notes |
|-------|-----|------|-------|
| Desktop with `llama3.1:8b` | 8 GB+ | ~6 GB | Recommended |
| Raspberry Pi 5 (8 GB) with `llama3.2:3b` | 8 GB | ~3 GB | Works, curation slower |
| Raspberry Pi 4 (4 GB) with `llama3.2:3b` | 4 GB | ~3 GB | Marginal, may swap |
| Low-RAM devices | -- | -- | Cloud provider option planned; use smallest Ollama model for now |

For low-RAM devices, use the smaller model:

```yaml
curation:
  ollama_model: "llama3.2:3b"
```

## Automation (bare-metal)

```bash
# Daily newspaper at 6am (cron):
0 6 * * * cd /path/to/offscroll && .venv/bin/offscroll -q run >> ~/.offscroll/cron.log 2>&1
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

## Known Issues / Platform Status

| Platform | Status | Notes |
|----------|--------|-------|
| x86_64 Linux | **Tested** | Primary development and CI platform |
| arm64 Linux (Raspberry Pi 5, etc.) | Untested | Dockerfile builds for arm64 via buildx; WeasyPrint deps available in Debian arm64 but not yet verified at runtime |
| Apple Silicon (Docker Desktop) | Untested | Should work via arm64 image; font rendering may differ from Linux |
| Windows (Docker Desktop / WSL2) | Untested | Expected to work via x86_64 image |

**WeasyPrint font rendering** may vary across architectures and
host font configurations. OffScroll bundles Source Sans/Serif/Code
Pro and runs `fc-cache` at image build time, but system-level
font fallback behavior can differ between amd64 and arm64 base
images.

**PDF generation on arm64:** WeasyPrint's native dependencies
(`libcairo`, `libpango`, `libgdk-pixbuf`) are available in
Debian arm64 repositories. If PDF rendering fails, the CLI falls
back to HTML-only output gracefully.

**Multi-arch Docker images** are built with `docker buildx` for
`linux/amd64` and `linux/arm64`. Tagged releases are published
automatically to GHCR. To pull a specific architecture:

```bash
docker pull --platform linux/arm64 ghcr.io/offscroll/offscroll:latest
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
