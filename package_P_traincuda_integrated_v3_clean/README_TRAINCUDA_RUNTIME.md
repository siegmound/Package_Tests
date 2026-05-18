# traincuda runtime notes

`traincuda(...)` replaces only the FSVI backup step with the custom CUDA backend. The upstream environment, expansion, belief union and value-function machinery remain in use.

Recommended notebook call:

```python
ag_cuda = enable_cuda_backend(ag_cuda_base, device=0, version="auto", gamma=0.95, lib_path=CUDA_LIB)
res = ag_cuda.traincuda(expansions=100, use_gpu=True, gamma=0.95, visual=True)
```

Policy evaluation must use the trained object:

```python
hist = run_policy_evaluation(ag_cuda, start_points=starts[:100], n=100, reward_discount=0.95)
hist.plot()
```

Do not evaluate `ag_cuda_base` after enabling CUDA unless it was trained separately.
