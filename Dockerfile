# OffScroll Dockerfile
# Multi-arch: amd64 + arm64 (Raspberry Pi 4/5)

# --- Build stage ---
FROM python:3.14-slim AS builder

WORKDIR /build

# System deps for building (WeasyPrint needs development headers)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf-2.0-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

# Install all dependency groups
RUN pip install --no-cache-dir \
    --prefix=/install \
    '.[ollama,fediverse]'

# --- Runtime stage ---
FROM python:3.14-slim

# Runtime deps only (no dev headers)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi8 \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash \
    offscroll
USER offscroll
WORKDIR /home/offscroll

# Copy installed packages from builder
COPY --from=builder /install /home/offscroll/.local

# Copy project source and config example
COPY --chown=offscroll:offscroll pyproject.toml ./
COPY --chown=offscroll:offscroll src/ ./src/
COPY --chown=offscroll:offscroll \
    config.example.yaml ./

# Install the project itself (editable, uses
# already-installed deps)
RUN pip install --user --no-cache-dir \
    --no-deps -e .

# Bundle fonts
COPY --chown=offscroll:offscroll \
    src/offscroll/layout/fonts/ \
    /home/offscroll/.local/share/fonts/
RUN fc-cache -f 2>/dev/null || true

# PATH
ENV PATH="/home/offscroll/.local/bin:${PATH}"

# Volumes
VOLUME ["/home/offscroll/.offscroll"]
VOLUME ["/home/offscroll/data"]

ENTRYPOINT ["offscroll"]
CMD ["run"]
