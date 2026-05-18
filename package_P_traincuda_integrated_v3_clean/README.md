# package_P_traincuda_integrated_v3_1_clean

Clean notebook package for integrating the custom CUDA backup backend into `olfactory_navigation` without modifying upstream `train()` or `SimulationHistory.plot()` from notebooks.

## What changed from integrated_v2

- `enable_cuda_backend(...)` still exposes `traincuda(...)`; native `ag.train(...)` remains untouched.
- Policy-evaluation helpers now normalize non-layered environment metadata automatically.
- `hist.plot()` works for reconstructed non-layered envs where `layers=false` / `environment_layer_labels=False` caused:
  - `TypeError: 'bool' object is not subscriptable`
  - action/position shape mismatches in `SimulationHistory.add_step`
- `run_policy_evaluation(...)`, `run_policy_smoke(...)`, and `run_policy_full_evaluation(...)` return native `SimulationHistory` objects already fixed for plotting.
- The package is cleaned: no temporary campaign output, no `__pycache__`, no CMake build directory noise. Only the CUDA shared library is kept in `build/` if present.

## Quick use in simplified notebook

```python
import os, sys
PATCH_ROOT = "/home/jlpfritas/HPC-POMDP/v1train_cuda/package_P_traincuda_integrated_v3_1_clean"
PATCH_PY = os.path.join(PATCH_ROOT, "python")
if PATCH_PY not in sys.path:
    sys.path.insert(0, PATCH_PY)

from olfnav_cuda_notebook import (
    enable_cuda_backend,
    clean_start_points,
    run_policy_evaluation,
    run_policy_full_evaluation,
    show_cuda_training_report,
)
```

```python
CUDA_LIB = os.path.join(PATCH_ROOT, "build", "libpomdp_backup_cuda.so")

ag_cuda_base = make_agent(partitions=(24, 24))
ag_cuda = enable_cuda_backend(
    ag_cuda_base,
    device=0,
    version="auto",
    gamma=0.95,
    lib_path=CUDA_LIB,
)

res_cuda = ag_cuda.traincuda(
    expansions=100,
    use_gpu=True,
    gamma=0.95,
    outdir="tmp/cuda_traincuda_100",
    checkpoint_every=25,
    visual=True,
)
```

Policy evaluation uses the trained object `ag_cuda`, not the untrained `ag_cuda_base`:

```python
starts = clean_start_points(ag_cuda)
hist_cuda = run_policy_evaluation(
    ag_cuda,
    start_points=starts[:100],
    n=100,
    horizon=1000,
    reward_discount=0.95,
    use_gpu=False,
    time_shift=False,
    time_loop=False,
)

hist_cuda.plot()
```

Full evaluation:

```python
hist_cuda_full = run_policy_full_evaluation(
    ag_cuda,
    horizon=1000,
    reward_discount=0.95,
    use_gpu=False,
    time_shift=False,
    time_loop=False,
)
hist_cuda_full.plot()
```

## Build CUDA library

A prebuilt `build/libpomdp_backup_cuda.so` may be included. Rebuild on the target machine when needed:

```bash
cd package_P_traincuda_integrated_v3_1_clean
bash scripts/31_build_backend_lib.sh --arch 80 --clean      # A100
bash scripts/31_build_backend_lib.sh --arch native --clean  # infer local GPU when supported
bash scripts/31_build_backend_lib.sh --arch 120 --clean     # RTX 50xx if CUDA supports it
```

## Notebook

See:

```text
examples/pomdp_agent_simplified_model.ipynb
```

It contains CPU/native, CuPy/native GPU, CUDA/traincuda training, and policy evaluation histories for all three.

## Important methodological note

This package keeps the comparison separation clear:

- `ag.train(..., use_gpu=False)` = original olfactory CPU/native path.
- `ag.train(..., use_gpu=True)` = original olfactory CuPy/native GPU path.
- `ag_cuda.traincuda(...)` = custom CUDA backup backend integrated into the FSVI pipeline.

The policy-evaluation fixes only normalize metadata for non-layered plotting/simulation compatibility; they do not change the learned value function or the policy computation.
