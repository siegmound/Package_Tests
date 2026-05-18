#!/usr/bin/env bash
set -euo pipefail
# Backward-compatible build entry point used by the notebooks.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/build_cuda_lib.sh" "$@"
