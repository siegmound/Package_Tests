# package_S_v2_pomdp_agent_training_style_allstates_fix3

Notebook/package in the style of upstream `pomdp_agent_training.ipynb`, using all environment states and adding the S_v2 real-cuBLAS `traincuda(...)` path.

## What fix3 adds

Compared with fix2, this version moves start-point generation into the package and notebook:

```python
start_points_raw
start_points_full
start_points_eval
```

and provides package helpers:

```python
raw_start_points_from_environment(...)
clean_start_points(...)
generate_policy_start_points(...)
run_policy_evaluation(...)
run_policy_full_evaluation(...)
```

The notebook now includes quick and full policy-evaluation cells for:

```python
hist_cuda_eval
hist_cpu_eval
hist_cupy_eval
hist_cuda_full
hist_cpu_full
hist_cupy_full
```

`RUN_FULL_POLICY_EVAL=False` by default because full all-starts evaluation can be slow in the all-states model.

## Main notebook

```text
examples/pomdp_agent_training_s_v2_cuda.ipynb
```

Also included as aliases:

```text
examples/pomdp_agent_training.ipynb
examples/pomdp_agent_training_s_v2_cuda_allstates_fix3.ipynb
```

## Usage

```python
PACKAGE_ROOT = Path("/home/jlpfritas/HPC-POMDP/v3/package_S_v2_pomdp_agent_training_style_allstates_fix3")
PACKAGE_PY = PACKAGE_ROOT / "python"
sys.path.insert(0, str(PACKAGE_PY))

from olfnav_cuda_notebook import (
    enable_cuda_backend,
    generate_policy_start_points,
    run_policy_evaluation,
    run_policy_full_evaluation,
)
```

Generate shared start points:

```python
start_point_sets = generate_policy_start_points(ref_agent, n_eval=N_EVAL, out_root=OUT_ROOT)
start_points_raw = start_point_sets["raw"]
start_points_full = start_point_sets["full"]
start_points_eval = start_point_sets["eval"]
```

Quick evaluation:

```python
hist_cuda_eval = run_policy_evaluation(
    ag_cuda.native_agent,
    start_points=start_points_eval,
    n=len(start_points_eval),
    horizon=1000,
    reward_discount=GAMMA,
    use_gpu=False,
)
```

Full evaluation:

```python
hist_cuda_full = run_policy_evaluation(
    ag_cuda.native_agent,
    start_points=start_points_full,
    n=len(start_points_full),
    horizon=1000,
    reward_discount=GAMMA,
    use_gpu=False,
)
```

## Build for A100

```bash
bash scripts/31_build_backend_lib.sh --arch 80 --clean
```

The zip includes `build/libpomdp_backup_cuda.so`, but rebuilding on the target machine is recommended.
