#!/usr/bin/env bash
set -euo pipefail

ARCHS="${CUDA_ARCHS:-80}"
BUILD_DIR="build"
CLEAN=0
JOBS="${JOBS:-}"

normalize_arch_arg() {
  local a="$1"
  if [[ "$a" == native ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      local cc
      cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .')
      if [[ -n "$cc" ]]; then
        echo "$cc"
        return 0
      fi
    fi
    echo "ERROR: --arch native requested but compute capability could not be inferred. Use --arch 80 for A100 or --arch 120 for RTX 50xx if supported." >&2
    return 2
  fi
  a="${a//sm_/}"
  a="${a//compute_/}"
  echo "$a"
}

usage() {
  cat <<USAGE
Build libpomdp_backup_cuda.so with selectable CUDA architecture.

Default target: sm_80 / compute_80, suitable for NVIDIA A100.

Usage:
  bash scripts/build_cuda_lib.sh [--arch 80] [--build-dir build] [--clean]

Examples:
  # A100 only
  bash scripts/build_cuda_lib.sh --arch 80 --clean

  # RTX 5090 / Blackwell only, if your CUDA toolkit supports sm_120
  bash scripts/build_cuda_lib.sh --arch 120 --clean

  # Multi-arch binary for A100 + RTX 50xx, if your CUDA toolkit supports both
  bash scripts/build_cuda_lib.sh --arch '80;120' --clean

Environment variables:
  CUDA_ARCHS   Same as --arch. Default: 80
  JOBS         Parallel build jobs. Default: cmake decides.

USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch|--cuda-arch|--cuda-archs)
      ARCHS="$2"; shift 2 ;;
    --build-dir)
      BUILD_DIR="$2"; shift 2 ;;
    --clean)
      CLEAN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

ARCHS="$(normalize_arch_arg "$ARCHS")"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake not found in PATH" >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found in PATH. Load a CUDA toolkit module first." >&2
  exit 1
fi

if [[ "$CLEAN" == "1" ]]; then
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

echo "[BUILD] root=$ROOT_DIR"
echo "[BUILD] CUDA_ARCHITECTURES=$ARCHS"
echo "[BUILD] build_dir=$BUILD_DIR"
nvcc --version | sed 's/^/[NVCC] /'

cmake -S cpp -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCHS"

if [[ -n "$JOBS" ]]; then
  cmake --build "$BUILD_DIR" -j "$JOBS"
else
  cmake --build "$BUILD_DIR"
fi

LIB="$BUILD_DIR/libpomdp_backup_cuda.so"
if [[ ! -f "$LIB" ]]; then
  echo "ERROR: expected library not found: $LIB" >&2
  exit 1
fi

echo "[OK] built $LIB"
ls -lh "$LIB"

if command -v cuobjdump >/dev/null 2>&1; then
  echo "[INFO] embedded CUDA code objects:"
  cuobjdump --list-elf "$LIB" 2>/dev/null | grep -E 'sm_[0-9]+' || true
else
  echo "[INFO] cuobjdump not found; skipping architecture inspection."
fi
