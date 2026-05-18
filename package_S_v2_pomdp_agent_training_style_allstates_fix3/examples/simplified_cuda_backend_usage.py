"""Example notebook-style use of the S_v2 real-cuBLAS CUDA backend.

Copy the relevant cells into the olfactory_navigation simplified notebook/file.
"""

from olfactory_navigation import Environment
from olfactory_navigation.agents import FSVI_Agent
from olfnav_cuda_backend.notebook import enable_cuda_backend

# 1) Build the normal olfactory_navigation agent exactly as before.
env = Environment.load("/absolute/path/to/Env-...")
ag = FSVI_Agent(
    env,
    seed=123,
)

# 2) Wrap it with the CUDA backup backend.
#    The native ag.train(...) remains available on the original agent.
ag_cuda = enable_cuda_backend(
    ag,
    device=1,
    version="auto_real",
    gamma=0.95,
    lib_path="/home/jlpfritas/HPC-POMDP/v3/package_S_v2_pomdp_agent_training_style_allstates/build/libpomdp_backup_cuda.so",
)

# 3) Train explicitly with the CUDA backend.
#    This avoids confusion with the original ag.train(...).
result = ag_cuda.traincuda(
    expansions=1000,
    use_gpu=True,
    outdir="tmp/notebook_cuda_train",
    checkpoint_every=100,
)

print(result.summary)
