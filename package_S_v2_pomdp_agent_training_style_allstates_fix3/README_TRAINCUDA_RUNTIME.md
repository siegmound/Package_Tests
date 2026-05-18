# Package S_v2 real-cuBLAS CUDA backend — clean traincuda runtime

This is the minimal runtime package for using the S_v2 real-cuBLAS CUDA backup backend
from a Jupyter notebook with `olfactory_navigation`.

This architecture-selectable version includes the CUDA source and build scripts,
so the shared library can be rebuilt for A100 (`sm_80`), RTX 50xx (`sm_120`), or
multi-arch when the installed CUDA toolkit supports it.

## Contents

```text
build/                                  # output directory for libpomdp_backup_cuda.so
cpp/                                    # CUDA/C++ backend source
scripts/build_cuda_lib.sh               # configurable build helper
scripts/check_cuda_lib_arch.sh          # inspect embedded sm_* code objects
python/olfnav_cuda_backend/             # Python wrapper and ctypes backend
examples/simplified_cuda_backend_usage.py
pyproject.toml                          # optional editable install metadata
```

## Build the CUDA library first

For A100:

```bash
cd /home/jlpfritas/HPC-POMDP/v3/package_S_v2_pomdp_agent_training_style_allstates
make build-a100
make check
```

Equivalent direct command:

```bash
bash scripts/build_cuda_lib.sh --arch 80 --clean
```

For other architectures:

```bash
make build CUDA_ARCHS=120        # RTX 50xx if CUDA toolkit supports sm_120
make build CUDA_ARCHS='80;120'   # multi-arch if supported
```

See `README_A100_ARCH_BUILD.md` for details.

## Fast notebook usage

From the package root:

```bash
cd /home/jlpfritas/HPC-POMDP/v3/package_S_v2_pomdp_agent_training_style_allstates
export PYTHONPATH="$PWD/python:$PYTHONPATH"
export OLFNAV_CUDA_BACKEND_LIB="$PWD/build/libpomdp_backup_cuda.so"
```

In Jupyter, after creating the normal `FSVI_Agent` as usual:

```python
from olfnav_cuda_backend.notebook import enable_cuda_backend

ag_cuda = enable_cuda_backend(
    ag,
    device=0,
    version="auto_real",
    gamma=0.95,
)

result = ag_cuda.traincuda(
    expansions=1000,
    use_gpu=True,                 # accepted for notebook compatibility
    outdir="tmp/notebook_cuda_train",
    checkpoint_every=100,
)

result.summary
```

Alternative: attach `traincuda` directly to the original agent without changing
`train`:

```python
from olfnav_cuda_backend.notebook import patch_agent_traincuda

patch_agent_traincuda(
    ag,
    device=0,
    version="auto_real",
    gamma=0.95,
)

result = ag.traincuda(expansions=1000, use_gpu=True)
```

The original call remains unchanged:

```python
ag.train(...)       # upstream olfactory_navigation training
ag.traincuda(...)   # custom CUDA-backend training
```

## Optional editable install

```bash
cd /home/jlpfritas/HPC-POMDP/v3/package_S_v2_pomdp_agent_training_style_allstates
pip install -e .
```

## Notes

- This package replaces the expensive FSVI backup step with the S_v2 real-cuBLAS CUDA backend while keeping the olfactory_navigation expand / belief-set / ValueFunction machinery.
- Do not use a library compiled only for `sm_120` on A100. Rebuild with `--arch 80`.
