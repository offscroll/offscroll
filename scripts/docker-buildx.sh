#!/usr/bin/env bash
# docker-buildx.sh — Build multi-arch Docker images for OffScroll.
#
# Builds for amd64 (x86_64) and arm64 (Raspberry Pi 5, Apple Silicon).
# Uses Docker Buildx with QEMU emulation for cross-platform builds.
#
# Usage:
#   ./scripts/docker-buildx.sh              # Build and load locally (current arch only)
#   ./scripts/docker-buildx.sh --push       # Build both arches and push to GHCR
#   ./scripts/docker-buildx.sh --tag v0.1.0 # Tag with a specific version
#
# Prerequisites:
#   docker buildx create --name offscroll-builder --use
#   docker buildx inspect --bootstrap

set -euo pipefail

REGISTRY="ghcr.io"
IMAGE="${REGISTRY}/offscroll/offscroll"
PLATFORMS="linux/amd64,linux/arm64"
TAG="latest"
PUSH=false
LOAD=false

usage() {
    echo "Usage: $0 [--push] [--tag TAG] [--load]"
    echo
    echo "  --push       Build for amd64+arm64 and push to ${REGISTRY}"
    echo "  --tag TAG    Image tag (default: latest)"
    echo "  --load       Load into local Docker (single arch only)"
    echo
    echo "Without --push or --load, performs a build check (no output)."
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)
            PUSH=true
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --load)
            LOAD=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Ensure buildx builder exists
if ! docker buildx inspect offscroll-builder > /dev/null 2>&1; then
    echo "Creating buildx builder 'offscroll-builder'..."
    docker buildx create --name offscroll-builder --use
    docker buildx inspect --bootstrap
else
    docker buildx use offscroll-builder
fi

echo "Building OffScroll Docker image"
echo "  Image:     ${IMAGE}:${TAG}"
echo "  Platforms: ${PLATFORMS}"
echo

BUILD_ARGS=(
    --file Dockerfile
    --platform "${PLATFORMS}"
    --tag "${IMAGE}:${TAG}"
)

# If a version tag is given, also tag as latest
if [[ "${TAG}" != "latest" ]]; then
    BUILD_ARGS+=(--tag "${IMAGE}:latest")
fi

if [[ "${PUSH}" == "true" ]]; then
    echo "Mode: build + push to ${REGISTRY}"
    docker buildx build "${BUILD_ARGS[@]}" --push .
elif [[ "${LOAD}" == "true" ]]; then
    echo "Mode: build + load locally (current architecture only)"
    # --load only works for a single platform
    docker buildx build \
        --file Dockerfile \
        --tag "${IMAGE}:${TAG}" \
        --load .
else
    echo "Mode: build check (no output image)"
    docker buildx build "${BUILD_ARGS[@]}" .
fi

echo
echo "Done."
