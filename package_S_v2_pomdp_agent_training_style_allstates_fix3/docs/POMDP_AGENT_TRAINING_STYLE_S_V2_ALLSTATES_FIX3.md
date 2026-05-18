# S_v2 pomdp_agent_training all-states fix3

This release adds package-level start-point generation for the all-states `pomdp_agent_training` style notebook.

## New APIs

```python
from olfnav_cuda_notebook import (
    raw_start_points_from_environment,
    clean_start_points,
    generate_policy_start_points,
    run_policy_evaluation,
    run_policy_full_evaluation,
)
```

## Generated arrays

The notebook now explicitly creates:

```python
start_points_raw
start_points_full
start_points_eval
```

where:

- `start_points_raw` is produced directly from `env.start_probabilities > 0`.
- `start_points_full` is the clean valid all-start set, with terminal/source points removed when source metadata is available.
- `start_points_eval` is the quick-evaluation subset controlled by `N_EVAL`.

All arrays are saved under `OUT_ROOT`:

```text
start_points_raw.npy
start_points_full.npy
start_points_eval.npy
```

## Evaluation cells

The notebook now contains both:

```python
hist_cuda_eval
hist_cpu_eval
hist_cupy_eval
```

for quick evaluation, and:

```python
hist_cuda_full
hist_cpu_full
hist_cupy_full
```

for full all-clean-starts evaluation.

`RUN_FULL_POLICY_EVAL=False` by default because full evaluation on the all-states model can be slow.

## Notes

This keeps the upstream `pomdp_agent_training` semantics: no partitions, no manual source coordinates, and no notebook-only patches for start-point generation.
