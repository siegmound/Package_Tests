# A100 server build quickstart

This package has been prepared for compilation on an NVIDIA A100 server.

Important: the uploaded `build/` directory was removed from this portable package to avoid stale `CMakeCache.txt` and CUDA/cuBLAS runtime mismatches. Rebuild `build/libpomdp_backup_cuda.so` on the target server.

A100 compute capability is `sm_80`, so the build architecture is:

```bash
--arch 80
```

## Strategy 1 — recommended automatic path

Use this when the server CMake is old or inconsistent. It installs a modern CMake and Ninja inside the active Python environment, detects CUDA/cuBLAS, then builds for A100.

```bash
cd /path/to/this/package
source /home/jlpfritas/HPC-POMDP/hpcenv/bin/activate
bash scripts/strategy1_install_cmake_autobuild_a100.sh
```

Optional overrides:

```bash
CUDA_HOME=/usr/local/cuda-11.7 CUDA_ARCH=80 bash scripts/strategy1_install_cmake_autobuild_a100.sh
```

or, if the admin-provided NVIDIA HPC SDK path is the intended toolchain:

```bash
source scripts/setup_env_nvhpc_cuda13_a100.sh
CUDA_HOME=$CUDA_HOME CUDA_ARCH=80 bash scripts/strategy1_install_cmake_autobuild_a100.sh
```

## Strategy 2 — manual discovery/build

Read and follow:

```bash
less docs/strategies_2_3_manual_and_admin_A100.md
```

Manual short version for `/usr/local/cuda-11.7`:

```bash
cd /path/to/this/package
source /home/jlpfritas/HPC-POMDP/hpcenv/bin/activate

export CUDA_HOME=/usr/local/cuda-11.7
export CUDAToolkit_ROOT=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python -m pip install --upgrade "cmake>=3.26" ninja
export PATH="$(python - <<'PY'
import os, sys
print(os.path.dirname(sys.executable))
PY
):$PATH"

rm -rf build
cmake -S cpp -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCUDAToolkit_ROOT=$CUDAToolkit_ROOT \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j$(nproc)
```

## Strategy 3 — admin NVHPC path is true

If the admin says to use `/opt/nvidia/hpc_sdk/`, use:

```bash
cd /path/to/this/package
source /home/jlpfritas/HPC-POMDP/hpcenv/bin/activate
source scripts/setup_env_nvhpc_cuda13_a100.sh

python -m pip install --upgrade "cmake>=3.26" ninja
export PATH="$(python - <<'PY'
import os, sys
print(os.path.dirname(sys.executable))
PY
):$PATH"

rm -rf build
cmake -S cpp -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCUDAToolkit_ROOT=$CUDAToolkit_ROOT \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j$(nproc)
```

## Validate before Jupyter

```bash
bash scripts/check_runtime_deps_a100.sh
ldd build/libpomdp_backup_cuda.so | grep "not found" || echo "No missing dependencies."
```

If `libcublas.so.13` is missing at runtime and the server has NVHPC 25.11:

```bash
source scripts/setup_env_nvhpc_cuda13_a100.sh
bash scripts/check_runtime_deps_a100.sh
```

Launch Jupyter from the same terminal after setting the environment:

```bash
jupyter lab
```

## Do not mix toolchains

- CUDA 11.7 build should link/run with `libcublas.so.11`.
- NVHPC 25.11 / CUDA 13 build should link/run with `libcublas.so.13`.
- Do not symlink `libcublas.so.11` to `libcublas.so.13` or vice versa.
