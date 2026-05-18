#!/usr/bin/env bash
set -euo pipefail

# Strategy 1 — install CMake inside the active Python env, auto-detect CUDA/cuBLAS,
# and build libpomdp_backup_cuda.so for NVIDIA A100 (sm_80 by default).
#
# Usage:
#   cd /path/to/package_S_v2_final_real_cublas_traincuda_v1_clean
#   source /path/to/hpcenv/bin/activate
#   bash /path/to/strategy1_install_cmake_autobuild_a100.sh
#
# Optional overrides:
#   PACKAGE_DIR=/path/to/package bash strategy1_install_cmake_autobuild_a100.sh
#   CUDA_HOME=/usr/local/cuda-11.7 bash strategy1_install_cmake_autobuild_a100.sh
#   CUDA_ARCH=80 bash strategy1_install_cmake_autobuild_a100.sh
#   CLEAN=0 bash strategy1_install_cmake_autobuild_a100.sh

PACKAGE_DIR="${PACKAGE_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-build}"
CLEAN="${CLEAN:-1}"
CUDA_ARCH="${CUDA_ARCH:-}"
REPORT_FILE="${REPORT_FILE:-build_env_report.txt}"

cd "$PACKAGE_DIR"

echo "[INFO] package dir: $PACKAGE_DIR"

if [[ ! -d cpp ]]; then
  echo "[ERROR] cpp/ directory not found in $PACKAGE_DIR" >&2
  echo "        Run this script from the package root or set PACKAGE_DIR=/path/to/package." >&2
  exit 1
fi

if [[ ! -f cpp/CMakeLists.txt ]]; then
  echo "[ERROR] cpp/CMakeLists.txt not found. Cannot build with CMake." >&2
  exit 1
fi

echo "[STEP 1] Install/update CMake inside current Python environment"
python -m pip install --upgrade "cmake>=3.26" ninja
export PATH="$(python - <<'PY'
import os, sys
print(os.path.dirname(sys.executable))
PY
):$PATH"

echo "[INFO] python: $(which python)"
echo "[INFO] cmake: $(which cmake)"
cmake --version | head -n 1

echo "[STEP 2] Detect CUDA toolkit"

find_cuda_from_nvcc() {
  local nvcc_path="$1"
  local bin_dir root
  bin_dir="$(dirname "$nvcc_path")"
  root="$(dirname "$bin_dir")"
  if [[ -x "$root/bin/nvcc" ]]; then
    echo "$root"
  fi
}

candidate_roots=()

# User override has highest priority.
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  candidate_roots+=("$CUDA_HOME")
fi

if command -v nvcc >/dev/null 2>&1; then
  candidate_roots+=("$(find_cuda_from_nvcc "$(command -v nvcc)")")
fi

# Common CUDA installs.
for d in /usr/local/cuda /usr/local/cuda-*; do
  if [[ -x "$d/bin/nvcc" ]]; then
    candidate_roots+=("$d")
  fi
done

# NVIDIA HPC SDK CUDA toolkits.
for d in /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/* /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda; do
  if [[ -x "$d/bin/nvcc" ]]; then
    candidate_roots+=("$d")
  fi
done

# De-duplicate and choose the first candidate that has nvcc and some cuBLAS runtime.
CUDA_CHOSEN=""
for root in "${candidate_roots[@]}"; do
  [[ -n "$root" ]] || continue
  [[ -x "$root/bin/nvcc" ]] || continue
  if compgen -G "$root/lib64/libcublas.so*" >/dev/null || compgen -G "$root/targets/x86_64-linux/lib/libcublas.so*" >/dev/null; then
    CUDA_CHOSEN="$root"
    break
  fi
done

# If no candidate has cuBLAS next to it, accept any nvcc and resolve cuBLAS separately.
if [[ -z "$CUDA_CHOSEN" ]]; then
  for root in "${candidate_roots[@]}"; do
    [[ -n "$root" ]] || continue
    if [[ -x "$root/bin/nvcc" ]]; then
      CUDA_CHOSEN="$root"
      break
    fi
  done
fi

if [[ -z "$CUDA_CHOSEN" ]]; then
  echo "[ERROR] Could not find nvcc automatically." >&2
  echo "        Set CUDA_HOME=/usr/local/cuda-11.7 or CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/13.0" >&2
  exit 1
fi

export CUDA_HOME="$CUDA_CHOSEN"
export CUDAToolkit_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

# Runtime libraries: CUDA lib64, target lib, plus NVHPC math_libs if present.
ld_paths=()
[[ -d "$CUDA_HOME/lib64" ]] && ld_paths+=("$CUDA_HOME/lib64")
[[ -d "$CUDA_HOME/targets/x86_64-linux/lib" ]] && ld_paths+=("$CUDA_HOME/targets/x86_64-linux/lib")

# NVHPC cuBLAS may live outside CUDA_HOME.
for d in /opt/nvidia/hpc_sdk/Linux_x86_64/*/math_libs/*/targets/x86_64-linux/lib; do
  if compgen -G "$d/libcublas.so*" >/dev/null; then
    ld_paths+=("$d")
  fi
done

if [[ ${#ld_paths[@]} -gt 0 ]]; then
  export LD_LIBRARY_PATH="$(IFS=:; echo "${ld_paths[*]}"):${LD_LIBRARY_PATH:-}"
fi

echo "[INFO] CUDA_HOME=$CUDA_HOME"
echo "[INFO] nvcc=$(which nvcc)"
nvcc --version | sed -n '1,6p'

echo "[STEP 3] Detect/choose GPU architecture"
if [[ -z "$CUDA_ARCH" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .')" || true
    if [[ -n "${cc:-}" && "$cc" =~ ^[0-9]+$ ]]; then
      CUDA_ARCH="$cc"
    fi
  fi
fi

# A100 fallback.
CUDA_ARCH="${CUDA_ARCH:-80}"
# Normalize sm_80 / compute_80 into 80.
CUDA_ARCH="${CUDA_ARCH#sm_}"
CUDA_ARCH="${CUDA_ARCH#compute_}"

echo "[INFO] CUDA_ARCH=$CUDA_ARCH"

# A100 sanity warning.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] GPU summary:"
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null || nvidia-smi | head -20
fi

echo "[STEP 4] Clean and configure build"
if [[ "$CLEAN" == "1" ]]; then
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

# Use the CMake installed in the Python environment.
cmake -S cpp -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDAToolkit_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

echo "[STEP 5] Build"
cmake --build "$BUILD_DIR" -j"$(nproc)"

SO="$BUILD_DIR/libpomdp_backup_cuda.so"
if [[ ! -f "$SO" ]]; then
  echo "[ERROR] Build finished but $SO was not created." >&2
  find "$BUILD_DIR" -maxdepth 3 -type f -name '*.so' -print
  exit 1
fi

echo "[STEP 6] Runtime dependency check"
ldd "$SO" | grep -E "cublas|cudart|cuda|not found" || true

if ldd "$SO" | grep -q "not found"; then
  echo "[ERROR] Missing runtime libraries. See ldd output above." >&2
  echo "        Usually fix with LD_LIBRARY_PATH pointing to CUDA/cuBLAS lib directories." >&2
  exit 2
fi

cat > "$REPORT_FILE" <<EOF_REPORT
package_dir=$PACKAGE_DIR
python=$(which python)
cmake=$(which cmake)
cmake_version=$(cmake --version | head -n 1)
CUDA_HOME=$CUDA_HOME
CUDAToolkit_ROOT=$CUDAToolkit_ROOT
nvcc=$(which nvcc)
CUDA_ARCH=$CUDA_ARCH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH
built_so=$PACKAGE_DIR/$SO
EOF_REPORT

echo "[OK] Built and dependency check passed: $PACKAGE_DIR/$SO"
echo "[OK] Report written to: $PACKAGE_DIR/$REPORT_FILE"
