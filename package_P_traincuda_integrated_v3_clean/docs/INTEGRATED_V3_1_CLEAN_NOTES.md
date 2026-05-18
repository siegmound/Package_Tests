# Integrated v3.1 clean notes

This package is a small corrective release over `package_P_traincuda_integrated_v3_clean`.

## Fixed

- `traincuda(..., visual=True)` no longer fails with:

  ```text
  NameError: name '_show_cuda_training_report' is not defined
  ```

  The visual training report helper is now defined locally inside `olfnav_cuda_backend.notebook`, which is the module that calls it.

- The previous v3 clean non-layered metadata fixes are preserved:
  `environment_layer_labels=False` and related non-layered flags are normalized to `None` for `SimulationHistory.plot()` compatibility.

## Notebook usage

Use:

```python
PATCH_ROOT = "/home/jlpfritas/HPC-POMDP/v1train_cuda/package_P_traincuda_integrated_v3_1_clean"
```

Then import as before:

```python
from olfnav_cuda_notebook import (
    enable_cuda_backend,
    clean_start_points,
    run_policy_evaluation,
    run_policy_full_evaluation,
    show_cuda_training_report,
)
```

`visual=True` is safe again:

```python
res_cuda = ag_cuda.traincuda(..., visual=True, display_rows=10)
```
