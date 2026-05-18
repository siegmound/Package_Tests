from __future__ import annotations

from typing import Any
import numpy as np

from .model_extractor import extract_model_from_agent
from .sparse_ell import compile_sparse_ell
from .backend_ctypes import CudaSparseBackupBackend


def make_backend_from_agent(agent: Any, *, lib_path: str, version: str = "auto", max_nB: int = 0, max_nG: int = 0, **extract_kwargs) -> CudaSparseBackupBackend:
    model = extract_model_from_agent(agent, **extract_kwargs)
    ell = compile_sparse_ell(model.T)
    return CudaSparseBackupBackend.from_model(
        model,
        ell,
        lib_path=lib_path,
        version=version,
        max_nB=max_nB,
        max_nG=max_nG,
    )


def select_kernel_version(nS: int, nB: int, nG: int) -> str:
    """K2-bis pragmatic policy.

    Small alpha sets use v4, intermediate regimes use v7, and very large
    alpha sets use v8. The C++ backend also implements this policy for
    version='auto'; this function is kept for Python-side logging and manual
    dispatch experiments.
    """
    if nG < 64:
        return "v4"
    if nG < 4096:
        return "v7"
    return "v8"


def backup_with_auto_version(backend: CudaSparseBackupBackend, B: np.ndarray, Gamma: np.ndarray):
    # Prefer C++ auto dispatch so that Python and C ABI stay aligned.
    return backend.backup(B, Gamma, version="auto")


def attach_backup_method(agent: Any, backend: CudaSparseBackupBackend, *, method_name: str = "cuda_sparse_backup") -> None:
    """Attach `agent.cuda_sparse_backup(B, Gamma)` without monkey-patching train()."""
    def _method(B, Gamma, version: str = "auto"):
        BKP, actions, info = backend.backup(B, Gamma, version=version)
        return BKP, actions, info
    setattr(agent, method_name, _method)
