# A100 server build entry point for package_S_v2_pomdp_agent_training_style_allstates_fix3

Use this file first when moving the package to the A100 server.

## Recommended one-command build

```bash
cd /path/to/package_S_v2_pomdp_agent_training_style_allstates_fix3
source /home/jlpfritas/HPC-POMDP/hpcenv/bin/activate
bash scripts/strategy1_install_cmake_autobuild_a100.sh
bash scripts/check_runtime_deps_a100.sh
```

## If admin NVHPC configuration is required

```bash
cd /path/to/package_S_v2_pomdp_agent_training_style_allstates_fix3
source /home/jlpfritas/HPC-POMDP/hpcenv/bin/activate
source scripts/setup_env_nvhpc_cuda13_a100.sh
bash scripts/strategy1_install_cmake_autobuild_a100.sh
bash scripts/check_runtime_deps_a100.sh
```

More details:

```text
docs/A100_SERVER_BUILD_QUICKSTART.md
docs/strategies_2_3_manual_and_admin_A100.md
```
