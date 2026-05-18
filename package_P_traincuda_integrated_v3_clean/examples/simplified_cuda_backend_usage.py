"""Minimal simplified-style use of integrated_v3_clean."""

import os
from olfactory_navigation import Environment
from olfactory_navigation.agents import FSVI_Agent
from olfactory_navigation.agents.model_based_util.environment_converter import minimal_converter
from olfnav_cuda_notebook import (
    enable_cuda_backend,
    clean_start_points,
    run_policy_evaluation,
)

PATCH_ROOT = os.environ.get(
    "P_TRAINCUDA_ROOT",
    "/home/jlpfritas/HPC-POMDP/v1train_cuda/package_P_traincuda_integrated_v3_1_clean",
)
CUDA_LIB = os.path.join(PATCH_ROOT, "build", "libpomdp_backup_cuda.so")
ENV_PATH = "/absolute/path/to/Env-..."


def make_agent(partitions=(24, 24)):
    env = Environment.load(ENV_PATH)
    ag = FSVI_Agent(
        env,
        environment_converter=minimal_converter,
        partitions=list(partitions),
        margin_partitions=True,
        seed=123,
    )
    return ag


ag_cuda_base = make_agent()
ag_cuda = enable_cuda_backend(
    ag_cuda_base,
    device=0,
    version="auto",
    gamma=0.95,
    lib_path=CUDA_LIB,
)

result = ag_cuda.traincuda(
    expansions=100,
    use_gpu=True,
    gamma=0.95,
    outdir="tmp/notebook_cuda_train",
    checkpoint_every=25,
    visual=True,
)

starts = clean_start_points(ag_cuda)
hist = run_policy_evaluation(
    ag_cuda,
    start_points=starts[:100],
    n=100,
    horizon=1000,
    reward_discount=0.95,
    use_gpu=False,
    time_shift=False,
    time_loop=False,
)
hist.plot()
