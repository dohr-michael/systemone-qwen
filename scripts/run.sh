#!/usr/bin/env bash
# Runs the service against a llama-server: scripts/run.sh http://host:8080 [extra engine args]
set -euo pipefail
backend=${1:?usage: scripts/run.sh <llama-server URL> [extra container args]}
shift
exec "${CONTAINER_ENGINE:-podman}" run --rm -p "127.0.0.1:${SYSTEMONE_PORT:-8000}:8000" \
  -e SYSTEMONE_BACKEND_URL="$backend" -e SYSTEMONE_MODEL="${SYSTEMONE_MODEL:-qwen-decision}" "$@" \
  "${IMAGE_REPOSITORY:-ghcr.io/dohr-michael/systemone-qwen}:${IMAGE_TAG:-latest}"
