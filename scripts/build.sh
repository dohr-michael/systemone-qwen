#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "${CONTAINER_ENGINE:-podman}" build -f "$root/Dockerfile" \
  -t "${IMAGE_REPOSITORY:-localhost/systemone-qwen}:${IMAGE_TAG:-dev}" "$@" "$root"
