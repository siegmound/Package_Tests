#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="${PACKAGE_DIR:-$PWD}"
SO="${SO:-$PACKAGE_DIR/build/libpomdp_backup_cuda.so}"

echo "=== Package ==="
echo "PACKAGE_DIR=$PACKAGE_DIR"
echo "SO=$SO"

echo "=== Python ==="
which python || true
python --version || true

echo "=== CMake ==="
which cmake || true
cmake --version 2>/dev/null | head -n 1 || true

echo "=== NVIDIA ==="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null || nvidia-smi || true

echo "=== nvcc ==="
which nvcc || true
nvcc --version 2>/dev/null | sed -n '1,6p' || true

echo "=== Environment ==="
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDAToolkit_ROOT=${CUDAToolkit_ROOT:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

echo "=== cuBLAS search quick ==="
find /usr/local -name "libcublas.so*" 2>/dev/null | head -20 || true
find /opt/nvidia/hpc_sdk -name "libcublas.so*" 2>/dev/null | head -20 || true

if [[ -f "$SO" ]]; then
  echo "=== ldd CUDA/cuBLAS deps ==="
  ldd "$SO" | grep -E "cublas|cudart|cuda|not found" || true
  echo "=== missing deps ==="
  ldd "$SO" | grep "not found" || echo "No missing deps."
  echo "=== ctypes load ==="
  python - <<PY
import ctypes
lib = r"$SO"
ctypes.CDLL(lib)
print("OK loaded", lib)
PY
else
  echo "[WARN] $SO not found. Build first."
fi
