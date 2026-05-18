# Integrated v3 clean notes

## Fixed issue

For reconstructed non-layered environments, `env_info.json` can contain:

```json
"layers": false
```

In upstream `olfactory_navigation`, `SimulationHistory` expects `environment_layer_labels=None` for non-layered environments. If it receives `False`, both simulation history tracking and plotting may incorrectly enter the layered branch.

Observed symptoms:

```text
TypeError: 'bool' object is not subscriptable
SimulationHistory.plot -> self.environment_layer_labels[1:]
```

and earlier shape errors in `add_step` because a 2D action array was inserted into a 3-column layered action buffer.

## Package-level fix

The package now normalizes non-layered metadata automatically in:

- `enable_cuda_backend(...)`
- `run_policy_evaluation(...)`
- `run_policy_smoke(...)`
- `run_policy_full_evaluation(...)`
- `hist.plot()` through an idempotent `SimulationHistory` compatibility patch installed by default by `enable_cuda_backend(...)`

The normalization is limited to metadata fields such as:

```text
environment_layer_labels: False -> None
layer_labels: False -> None
layers: False -> None
environment_layers: False -> None
```

## Why this is safe

The fix does not modify:

- actions
- observations
- positions
- rewards
- done flags
- value functions
- alpha vectors
- beliefs
- CUDA backup outputs

It only makes the non-layered representation match the expectation of upstream `SimulationHistory`.

## API

```python
from olfnav_cuda_notebook import (
    normalize_non_layered_environment,
    normalize_simulation_history,
    fix_history_plot_metadata,
    install_simulation_history_patch,
    run_policy_evaluation,
    run_policy_full_evaluation,
)
```

In normal notebook use, you should not need to call these manually.
