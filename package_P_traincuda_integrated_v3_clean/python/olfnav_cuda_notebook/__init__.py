from __future__ import annotations

# Core CUDA backend wrapper.
from olfnav_cuda_backend.notebook import (
    CudaFSVI_Agent,
    CudaNotebookTrainResult,
    enable_cuda_backend as _backend_enable_cuda_backend,
    patch_agent_train,
    patch_agent_traincuda,
    package_root,
    resolve_cuda_lib,
    select_cuda_device,
)

from .fixes import (
    fix_history_plot_metadata,
    install_simulation_history_patch,
    normalize_agent_environment,
    normalize_non_layered_environment,
    normalize_non_layered_metadata,
    normalize_simulation_history,
    simulation_history_patch,
    uninstall_simulation_history_patch,
)
from .evaluation import (
    clean_start_points,
    run_policy_evaluation,
    run_policy_full,
    run_policy_full_evaluation,
    run_policy_smoke,
)
from .visual import show_cuda_training_report


def enable_cuda_backend(agent, *, install_history_patch: bool = True, **kwargs):
    """Enable traincuda and install the safe history metadata patch by default."""
    normalize_agent_environment(agent)
    if install_history_patch:
        globals()["install_simulation_history_patch"](verbose=False)
    return _backend_enable_cuda_backend(agent, **kwargs)


__all__ = [
    "CudaFSVI_Agent",
    "CudaNotebookTrainResult",
    "enable_cuda_backend",
    "patch_agent_train",
    "patch_agent_traincuda",
    "package_root",
    "resolve_cuda_lib",
    "select_cuda_device",
    "clean_start_points",
    "run_policy_smoke",
    "run_policy_evaluation",
    "run_policy_full_evaluation",
    "run_policy_full",
    "show_cuda_training_report",
    "fix_history_plot_metadata",
    "install_simulation_history_patch",
    "uninstall_simulation_history_patch",
    "simulation_history_patch",
    "normalize_non_layered_metadata",
    "normalize_non_layered_environment",
    "normalize_agent_environment",
    "normalize_simulation_history",
]
