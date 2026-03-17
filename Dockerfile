# OffScroll Dockerfile
# Multi-stage build, multi-arch (amd64 + arm64)

# --- Build stage ---
FROM python:3.12-slim AS builder

WORKDIR /build

# Build deps for WeasyPrint native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf-2.0-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a clean prefix
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --prefix=/install '.[fediverse]'

# --- Runtime stage ---
FROM python:3.12-slim

# Runtime libs for WeasyPrint (no dev headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    shared-mime-info \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home --shell /bin/bash offscroll

# Copy installed packages from builder into system Python
COPY --from=builder /install/lib/ /usr/local/lib/
COPY --from=builder /install/bin/ /usr/local/bin/

# Bundle fonts for WeasyPrint
COPY --chown=offscroll:offscroll \
    src/offscroll/layout/fonts/ \
    /home/offscroll/.local/share/fonts/
RUN fc-cache -f

# Copy example config
COPY --chown=offscroll:offscroll config.example.yaml \
    /home/offscroll/config.example.yaml

# Switch to non-root
USER offscroll
WORKDIR /home/offscroll

# Default data and config directories
RUN mkdir -p /home/offscroll/.offscroll /home/offscroll/data

VOLUME ["/home/offscroll/.offscroll"]
VOLUME ["/home/offscroll/data"]

ENTRYPOINT ["offscroll"]
CMD ["--help"]
