from __future__ import annotations

from olfnav_cuda_backend.notebook import (
    CudaFSVI_Agent,
    CudaNotebookTrainResult,
    enable_cuda_backend,
    patch_agent_train,
    patch_agent_traincuda,
    resolve_cuda_lib,
)
from .fixes import (
    install_simulation_history_patch,
    normalize_environment_for_exact_converter,
    read_environment_metadata,
    environment_kwargs_from_metadata,
    construct_environment_from_metadata,
    load_environment_from_metadata,
)
from .visual import show_cuda_training_report
from .evaluation import (
    raw_start_points_from_environment,
    clean_start_points,
    generate_policy_start_points,
    make_policy_start_points,
    run_policy_evaluation,
    run_policy_full_evaluation,
    run_policy_smoke,
)

__all__ = [
    "CudaFSVI_Agent",
    "CudaNotebookTrainResult",
    "enable_cuda_backend",
    "patch_agent_train",
    "patch_agent_traincuda",
    "resolve_cuda_lib",
    "install_simulation_history_patch",
    "normalize_environment_for_exact_converter",
    "read_environment_metadata",
    "environment_kwargs_from_metadata",
    "construct_environment_from_metadata",
    "load_environment_from_metadata",
    "show_cuda_training_report",
    "raw_start_points_from_environment",
    "clean_start_points",
    "generate_policy_start_points",
    "make_policy_start_points",
    "run_policy_evaluation",
    "run_policy_full_evaluation",
    "run_policy_smoke",
]
