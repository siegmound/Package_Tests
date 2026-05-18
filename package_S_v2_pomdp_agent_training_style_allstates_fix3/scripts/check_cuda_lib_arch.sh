#!/usr/bin/env bash
set -euo pipefail
LIB="${1:-build/libpomdp_backup_cuda.so}"
if [[ ! -f "$LIB" ]]; then
  echo "ERROR: library not found: $LIB" >&2
  exit 1
fi
if ! command -v cuobjdump >/dev/null 2>&1; then
  echo "ERROR: cuobjdump not found in PATH" >&2
  exit 1
fi
cuobjdump --list-elf "$LIB" | grep -E 'sm_[0-9]+' || {
  echo "No sm_* cubin entries found. The library may contain PTX only or cuobjdump could not parse it." >&2
}
