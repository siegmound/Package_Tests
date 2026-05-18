# Simplified notebook guide

Use `examples/pomdp_agent_simplified_model.ipynb` as the reference notebook.

The recommended order is:

1. Add `PATCH_ROOT/python` to `sys.path`.
2. Import from `olfnav_cuda_notebook`.
3. Load the environment with `Environment.load(ENV_PATH)` inside `make_agent(...)`.
4. Build independent agents for CUDA, CPU, and CuPy.
5. Train:
   - CUDA: `ag_cuda.traincuda(...)`
   - CPU: `ag_cpu.train(..., use_gpu=False)`
   - CuPy: `ag_cupy.train(..., use_gpu=True)`
6. Evaluate trained agents with `run_policy_evaluation(...)` or `run_policy_full_evaluation(...)`.
7. Call native `hist.plot()` directly.

Do not evaluate the untrained base object after enabling CUDA. Use `ag_cuda`, not `ag_cuda_base`.
