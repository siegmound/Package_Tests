# A100 compile instructions for package_P_traincuda_integrated_v3_clean

This package now includes the A100 build helper files:

```text
scripts/strategy1_install_cmake_autobuild_a100.sh
scripts/setup_env_nvhpc_cuda13_a100.sh
scripts/check_runtime_deps_a100.sh
docs/strategies_2_3_manual_and_admin_A100.md
docs/A100_SERVER_BUILD_QUICKSTART.md
A100_SERVER_BUILD.md
```

The uploaded/stale `build/` directory has been removed. Rebuild on the server to avoid CMake cache and CUDA/cuBLAS ABI mismatch problems.

## Fast path

```bash
cd /path/to/package_P_traincuda_integrated_v3_clean
source /home/jlpfritas/HPC-POMDP/hpcenv/bin/activate
bash scripts/strategy1_install_cmake_autobuild_a100.sh
bash scripts/check_runtime_deps_a100.sh
```

## Admin NVHPC path

```bash
cd /path/to/package_P_traincuda_integrated_v3_clean
source /home/jlpfritas/HPC-POMDP/hpcenv/bin/activate
source scripts/setup_env_nvhpc_cuda13_a100.sh
bash scripts/strategy1_install_cmake_autobuild_a100.sh
bash scripts/check_runtime_deps_a100.sh
```

## Manual strategies

Open:

```bash
less docs/strategies_2_3_manual_and_admin_A100.md
```
