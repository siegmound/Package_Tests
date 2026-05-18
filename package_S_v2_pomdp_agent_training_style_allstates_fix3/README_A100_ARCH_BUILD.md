# CUDA traincuda runtime — selectable architecture build

This package is the clean notebook/runtime package with CUDA sources included so
`libpomdp_backup_cuda.so` can be rebuilt for the target GPU.

## Important

The included `prebuilt/` library is only a reference copy from the previous
machine and may have been built for `sm_120`. Do **not** use it on A100 unless
`cuobjdump` confirms it contains `sm_80` or compatible PTX. The normal runtime
library must be built as:

```bash
build/libpomdp_backup_cuda.so
```

## A100 build

A100 is Ampere compute capability 8.0, so build with `sm_80`:

```bash
cd /home/jlpfritas/HPC-POMDP/v3/package_S_v2_pomdp_agent_training_style_allstates
bash scripts/build_cuda_lib.sh --arch 80 --clean
```

Equivalent Makefile shortcut:

```bash
make build-a100
```

Check the embedded architecture:

```bash
bash scripts/check_cuda_lib_arch.sh build/libpomdp_backup_cuda.so
# or
make check
```

Expected output should include `sm_80`.

## RTX 5090 / Blackwell build

Only if the installed CUDA toolkit supports it:

```bash
bash scripts/build_cuda_lib.sh --arch 120 --clean
# or
make build-sm120
```

## Multi-arch build

If the CUDA toolkit supports both `sm_80` and `sm_120`:

```bash
bash scripts/build_cuda_lib.sh --arch '80;120' --clean
# or
make build-multi
```

If this fails on the A100 machine, the CUDA toolkit is likely too old for
`sm_120`. In that case use only `--arch 80`.

## Notebook usage after build

```bash
export PYTHONPATH="$PWD/python:$PYTHONPATH"
```

```python
from olfnav_cuda_backend.notebook import enable_cuda_backend

ag_cuda = enable_cuda_backend(
    ag,
    device=0,
    version="auto_real",
    gamma=0.95,
    lib_path="/home/jlpfritas/HPC-POMDP/v3/package_S_v2_pomdp_agent_training_style_allstates/build/libpomdp_backup_cuda.so",
)

result = ag_cuda.traincuda(expansions=1000, use_gpu=True)
```

You can also set the library path once:

```bash
export OLFNAV_CUDA_BACKEND_LIB="$PWD/build/libpomdp_backup_cuda.so"
```

then omit `lib_path` in the notebook.

## Failure mode if wrong architecture is used

If a library built only for `sm_120` is run on A100, CUDA may fail with:

```text
CUDA error: no kernel image is available for execution on the device
```

Rebuild with `--arch 80`.
