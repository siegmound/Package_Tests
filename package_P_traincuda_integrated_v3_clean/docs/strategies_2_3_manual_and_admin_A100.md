# A100 build strategies for `package_S_v2_final_real_cublas_traincuda_v1_clean`

Context:

- Target GPU: NVIDIA A100, so CUDA architecture is `sm_80`, i.e. `CMAKE_CUDA_ARCHITECTURES=80`.
- Package: `package_S_v2_final_real_cublas_traincuda_v1_clean`.
- Goal: build `build/libpomdp_backup_cuda.so` with real cuBLAS v7/v8 support.
- Important rule: do not mix compile-time CUDA/cuBLAS with a different runtime CUDA/cuBLAS.

Two realistic situations exist on the server:

1. The machine really uses `/usr/local/cuda-11.7`.
2. The admin-provided NVIDIA HPC SDK path `/opt/nvidia/hpc_sdk/` is the intended toolchain/runtime.

Use one strategy at a time. After changing toolchain, always remove `build/` and restart the Jupyter kernel.

---

# Strategy 2 — Manual discovery and manual build

Use this when you do **not** fully trust the CUDA/NVHPC path and want to inspect the system first.

## 2.1 Enter package and activate env

```bash
cd /home/jlpfritas/HPC-POMDP/v1train_cuda/package_S_v2_final_real_cublas_traincuda_v1_clean
source /home/jlpfritas/HPC-POMDP/hpcenv/bin/activate
```

## 2.2 Discover CUDA, nvcc and cuBLAS

```bash
echo "=== nvcc in PATH ==="
which nvcc || true
nvcc --version || true

echo "=== nvidia-smi ==="
nvidia-smi

echo "=== CUDA installs under /usr/local ==="
ls -lah /usr/local | grep cuda || true
find /usr/local -maxdepth 3 -name nvcc 2>/dev/null
find /usr/local -name "libcublas.so*" 2>/dev/null

echo "=== NVIDIA HPC SDK installs ==="
find /opt/nvidia/hpc_sdk -maxdepth 6 -name nvcc 2>/dev/null | head -20
find /opt/nvidia/hpc_sdk -name "libcublas.so*" 2>/dev/null | head -50

echo "=== Current env ==="
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDAToolkit_ROOT=$CUDAToolkit_ROOT"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
```

## 2.3 If using `/usr/local/cuda-11.7`

Use this branch if these files exist:

```bash
/usr/local/cuda-11.7/bin/nvcc
/usr/local/cuda-11.7/lib64/libcublas.so*
```

Commands:

```bash
export CUDA_HOME=/usr/local/cuda-11.7
export CUDAToolkit_ROOT=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

which nvcc
nvcc --version
ls -lah $CUDA_HOME/lib64/libcublas.so*
```

Expected runtime after build:

```text
libcublas.so.11 => /usr/local/cuda-11.7/lib64/libcublas.so.11
libcudart.so.11.0 => /usr/local/cuda-11.7/lib64/libcudart.so.11.0
```

## 2.4 Install/force CMake inside the Python env

Do this if the system CMake is too old or inconsistent:

```bash
python -m pip install --upgrade "cmake>=3.26" ninja
export PATH="$(python - <<'PY'
import os, sys
print(os.path.dirname(sys.executable))
PY
):$PATH"

which cmake
cmake --version
```

## 2.5 Manual CMake configure/build for A100

```bash
rm -rf build
mkdir -p build

cmake -S cpp -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCUDAToolkit_ROOT=$CUDAToolkit_ROOT \
  -DCMAKE_CUDA_ARCHITECTURES=80

cmake --build build -j$(nproc)
```

## 2.6 Check runtime dependencies

```bash
ldd build/libpomdp_backup_cuda.so | grep -E "cublas|cudart|cuda|not found"
ldd build/libpomdp_backup_cuda.so | grep "not found" || echo "No missing dependencies."
```

If something is missing, add the corresponding lib directory to `LD_LIBRARY_PATH`, then recheck.

## 2.7 Jupyter launch from same environment

```bash
source /home/jlpfritas/HPC-POMDP/hpcenv/bin/activate
jupyter lab
```

Do not launch Jupyter from a different terminal without the same `CUDA_HOME`, `PATH`, and `LD_LIBRARY_PATH`.

---

# Strategy 3 — Admin configuration is true: use NVIDIA HPC SDK under `/opt/nvidia/hpc_sdk/`

Use this when the server admin expects the NVIDIA HPC SDK to be the CUDA/cuBLAS source.

Admin-provided base setup was:

```bash
export NVARCH=$(uname -s)_$(uname -m)
export NVCOMPILERS=/opt/nvidia/hpc_sdk
export MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/25.11/compilers/man
export PATH=$NVCOMPILERS/$NVARCH/25.11/compilers/bin:$PATH

echo "Ambiente NVIDIA HPC SDK caricato per $NVARCH!"
```

For this package we also need the CUDA and cuBLAS runtime paths.

## 3.1 Enter package and activate env

```bash
cd /home/jlpfritas/HPC-POMDP/v1train_cuda/package_S_v2_final_real_cublas_traincuda_v1_clean
source /home/jlpfritas/HPC-POMDP/hpcenv/bin/activate
```

## 3.2 Configure NVIDIA HPC SDK 25.11

```bash
export NVARCH=$(uname -s)_$(uname -m)
export NVCOMPILERS=/opt/nvidia/hpc_sdk
export NVHPC_VERSION=25.11
export NVHPC_ROOT=$NVCOMPILERS/$NVARCH/$NVHPC_VERSION

export MANPATH=${MANPATH:-}:$NVHPC_ROOT/compilers/man
export PATH=$NVHPC_ROOT/compilers/bin:$PATH
```

Now locate the CUDA toolkit inside NVHPC:

```bash
find $NVHPC_ROOT -maxdepth 4 -name nvcc 2>/dev/null
find $NVHPC_ROOT -name "libcublas.so*" 2>/dev/null | head -20
```

Most likely paths for this server:

```bash
export CUDA_HOME=$NVHPC_ROOT/cuda/13.0
export CUDAToolkit_ROOT=$CUDA_HOME
export CUBLAS13_LIB=$NVHPC_ROOT/math_libs/13.0/targets/x86_64-linux/lib
```

Set paths:

```bash
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUBLAS13_LIB:$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
```

Check:

```bash
echo "NVHPC_ROOT=$NVHPC_ROOT"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUBLAS13_LIB=$CUBLAS13_LIB"
which nvcc
nvcc --version
ls -lah $CUBLAS13_LIB/libcublas.so*
```

## 3.3 Install/force CMake inside env

```bash
python -m pip install --upgrade "cmake>=3.26" ninja
export PATH="$(python - <<'PY'
import os, sys
print(os.path.dirname(sys.executable))
PY
):$PATH"

which cmake
cmake --version
```

## 3.4 Build for A100

```bash
rm -rf build
mkdir -p build

cmake -S cpp -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCUDAToolkit_ROOT=$CUDAToolkit_ROOT \
  -DCMAKE_CUDA_ARCHITECTURES=80

cmake --build build -j$(nproc)
```

## 3.5 Check dependencies

```bash
ldd build/libpomdp_backup_cuda.so | grep -E "cublas|cudart|cuda|not found"
ldd build/libpomdp_backup_cuda.so | grep "not found" || echo "No missing dependencies."
```

Expected if built/linked against NVHPC CUDA/cuBLAS 13:

```text
libcublas.so.13 => /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/math_libs/13.0/targets/x86_64-linux/lib/libcublas.so.13
```

## 3.6 Launch Jupyter

Launch Jupyter from the same shell after all exports:

```bash
jupyter lab
```

If using VSCode remote, start VSCode/Jupyter kernel from a shell that has the same environment or configure the kernel environment explicitly.

---

# Common validation after either strategy

## Check GPU arch and library

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
ldd build/libpomdp_backup_cuda.so | grep "not found" || echo "OK: no missing deps"
```

## Check Python can load the library

```bash
python - <<'PY'
import ctypes
lib = "build/libpomdp_backup_cuda.so"
ctypes.CDLL(lib)
print("OK loaded", lib)
PY
```

## Notebook settings

In the notebook:

```python
PATCH_ROOT = "/home/jlpfritas/HPC-POMDP/v1train_cuda/package_S_v2_final_real_cublas_traincuda_v1_clean"
CUDA_LIB = PATCH_ROOT + "/build/libpomdp_backup_cuda.so"
```

Then use:

```python
version="auto_real"  # automatic v4/v7/v8 dispatcher
# or
version="v7"
# or
version="v8"
```

---

# Decision rule

Prefer Strategy 2 with `/usr/local/cuda-11.7` if the server standard CUDA is really 11.7 and CMake finds it cleanly.

Prefer Strategy 3 if the admin expects NVIDIA HPC SDK and cuBLAS 13 to be used, or if the prebuilt `.so` was linked against `libcublas.so.13`.

Do not mix:

```text
compile with CUDA 11.7 but run with cuBLAS 13
compile with CUDA/cuBLAS 13 but run with CUDA 11.7 only
```
