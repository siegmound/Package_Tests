from __future__ import annotations

"""Notebook-friendly Package S_v2 real-cuBLAS backend adapter for olfactory_navigation FSVI.

This module is intentionally small and user-facing.  It lets a notebook that
currently does something like

    ag = FSVI_Agent(...)
    _ = ag.train(expansions=1000, use_gpu=True)

switch to the Package S_v2 real-cuBLAS CUDA backup backend without overloading the original
``train`` method:

    from olfnav_cuda_backend.notebook import enable_cuda_backend
    ag_cuda = enable_cuda_backend(ag, device=0)
    _ = ag_cuda.traincuda(expansions=1000, use_gpu=True)

The wrapper keeps the original olfactory_navigation expand / belief union /
ValueFunction machinery, and only replaces the expensive backup step with the
in-process CUDA sparse backend.  This is the same methodological split used in
Package S_v2/K7/K8/K9: full-pipeline FSVI with real-cuBLAS CUDA backup replacement, not a raw
isolated microbenchmark.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable
import json
import os
import time
import types

import numpy as np

from .fsvi_cuda_loop import (
    K3RunConfig,
    _solve_mdp_policy,
    append_backup_to_value_function,
    belief_matrix,
    compute_vnew_and_gstar,
    initial_belief_set,
    initial_value_function,
    maybe_prune,
    vf_gamma,
    write_csv,
    write_md,
)
from .solver_bridge import make_backend_from_agent


@dataclass
class CudaNotebookTrainResult:
    """Return object produced by :meth:`CudaFSVI_Agent.train`.

    The underlying olfactory agent is also updated with:

    - ``agent.value_function``
    - ``agent.belief_set``
    - ``agent.cuda_training_summary``
    - ``agent.cuda_training_rows``

    so notebooks can continue inspecting the agent after training.
    """

    agent: Any
    value_function: Any
    belief_set: Any
    rows: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return dict(self.summary)


def _json_ready(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def package_root() -> Path:
    # .../package_S_v2_pomdp_agent_training_style_allstates/python/olfnav_cuda_backend/notebook.py
    return Path(__file__).resolve().parents[2]


def resolve_cuda_lib(lib_path: str | os.PathLike[str] | None = None) -> Path:
    """Find ``libpomdp_backup_cuda.so`` for notebook usage.

    Search order:
    1. explicit ``lib_path`` argument;
    2. ``OLFNAV_CUDA_BACKEND_LIB`` environment variable;
    3. ``./build/libpomdp_backup_cuda.so`` from the current notebook directory;
    4. ``<package root>/build/libpomdp_backup_cuda.so`` for editable installs.
    """
    candidates: list[Path] = []
    if lib_path:
        candidates.append(Path(lib_path).expanduser())
    env_lib = os.environ.get("OLFNAV_CUDA_BACKEND_LIB")
    if env_lib:
        candidates.append(Path(env_lib).expanduser())
    candidates.append(Path.cwd() / "build" / "libpomdp_backup_cuda.so")
    candidates.append(package_root() / "build" / "libpomdp_backup_cuda.so")

    for p in candidates:
        p = p if p.is_absolute() else (Path.cwd() / p)
        if p.exists():
            return p.resolve()
    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find libpomdp_backup_cuda.so. Pass lib_path=... or set "
        f"OLFNAV_CUDA_BACKEND_LIB. Tried:\n{tried}"
    )


def select_cuda_device(device: int | None = None) -> None:
    """Best-effort CUDA device selection for notebooks.

    This must be called before the CUDA shared library/CuPy context is created.
    """
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(device))


def _set_agent_attr(agent: Any, name: str, value: Any) -> None:
    try:
        setattr(agent, name, value)
    except Exception:
        # Some upstream classes may use slots/properties.  The wrapper result still
        # exposes the objects even if setting an attribute on the original agent fails.
        pass


class CudaFSVI_Agent:
    """Composition wrapper exposing a notebook-friendly ``traincuda`` method.

    The wrapper delegates unknown attributes to the original FSVI agent. Since
    this class intentionally does not override ``train``, ``ag_cuda.train(...)``
    still resolves to the native upstream method through ``__getattr__``, while
    ``ag_cuda.traincuda(...)`` uses the Package S_v2 real-cuBLAS CUDA backend. This avoids
    ambiguity inside simplified notebooks.
    """

    def __init__(
        self,
        agent: Any,
        *,
        lib_path: str | os.PathLike[str] | None = None,
        device: int | None = None,
        version: str = "auto_real",
        gamma: float = 0.95,
        max_belief_growth: int = 10,
        prune_interval: int = 10,
        prune_level: int = 1,
        compact_cuda_backup_before_append: bool = True,
        cuda_backup_compact_prune_level: int = 1,
        mdp_vi_horizon: int = 1000,
        mdp_vi_eps: float = 1e-6,
        mdp_use_gpu: bool = False,
        print_progress: bool = False,
        default_outdir: str | os.PathLike[str] | None = None,
    ) -> None:
        select_cuda_device(device)
        self._agent = agent
        self.cuda_lib_path = resolve_cuda_lib(lib_path)
        self.cuda_device = device
        self.cuda_version = str(version)
        self.gamma = float(gamma)
        self.max_belief_growth = int(max_belief_growth)
        self.prune_interval = int(prune_interval)
        self.prune_level = int(prune_level)
        self.compact_cuda_backup_before_append = bool(compact_cuda_backup_before_append)
        self.cuda_backup_compact_prune_level = int(cuda_backup_compact_prune_level)
        self.mdp_vi_horizon = int(mdp_vi_horizon)
        self.mdp_vi_eps = float(mdp_vi_eps)
        self.mdp_use_gpu = bool(mdp_use_gpu)
        self.print_progress = bool(print_progress)
        self.default_outdir = Path(default_outdir) if default_outdir is not None else None
        self.last_result: CudaNotebookTrainResult | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    @property
    def native_agent(self) -> Any:
        return self._agent

    def _make_cfg(
        self,
        *,
        gamma: float | None = None,
        max_belief_growth: int | None = None,
        prune_interval: int | None = None,
        prune_level: int | None = None,
        mdp_use_gpu: bool | None = None,
        print_progress: bool | None = None,
    ) -> K3RunConfig:
        return K3RunConfig(
            lib_path=str(self.cuda_lib_path),
            gamma=float(self.gamma if gamma is None else gamma),
            mdp_vi_horizon=int(self.mdp_vi_horizon),
            mdp_vi_eps=float(self.mdp_vi_eps),
            max_belief_growth=int(self.max_belief_growth if max_belief_growth is None else max_belief_growth),
            prune_interval=int(self.prune_interval if prune_interval is None else prune_interval),
            prune_level=int(self.prune_level if prune_level is None else prune_level),
            compact_cuda_backup_before_append=bool(self.compact_cuda_backup_before_append),
            cuda_backup_compact_prune_level=int(self.cuda_backup_compact_prune_level),
            lockstep_native_state=False,
            version=str(self.cuda_version),
            use_gpu_trace=bool(self.mdp_use_gpu if mdp_use_gpu is None else mdp_use_gpu),
            print_progress=bool(self.print_progress if print_progress is None else print_progress),
        )

    def traincuda(
        self,
        expansions: int = 1000,
        *,
        use_gpu: bool | None = None,
        gamma: float | None = None,
        max_belief_growth: int | None = None,
        prune_interval: int | None = None,
        prune_level: int | None = None,
        outdir: str | os.PathLike[str] | None = None,
        checkpoint_every: int = 0,
        mdp_use_gpu: bool | None = None,
        return_result: bool = True,
        visual: bool = False,
        display_rows: int = 10,
        **unused_upstream_kwargs: Any,
    ) -> CudaNotebookTrainResult | Any:
        """Run FSVI using the custom CUDA backup backend.

        Parameters intentionally mirror the common upstream notebook call.  The
        ``use_gpu`` argument is accepted for compatibility with existing notebooks;
        the backup always uses the CUDA backend. By default, ``use_gpu`` does not
        move the upstream olfactory objects to CuPy, because this adapter is meant
        to preserve the compact CPU-like pipeline while replacing only backup.
        Set ``mdp_use_gpu=True`` explicitly to request the upstream GPU path only
        for the MDP policy solve.
        """
        cfg = self._make_cfg(
            gamma=gamma,
            max_belief_growth=max_belief_growth,
            prune_interval=prune_interval,
            prune_level=prune_level,
            mdp_use_gpu=mdp_use_gpu,
        )
        target_iter = int(expansions)
        out_path = Path(outdir) if outdir is not None else self.default_outdir
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)

        t_global0 = time.perf_counter()
        backend = make_backend_from_agent(
            self._agent,
            lib_path=str(self.cuda_lib_path),
            version=str(self.cuda_version),
            gamma=float(cfg.gamma),
        )
        rows: list[dict[str, Any]] = []

        try:
            t_policy0 = time.perf_counter()
            mdp_policy, _ = _solve_mdp_policy(self._agent, cfg)
            policy_ms = (time.perf_counter() - t_policy0) * 1000.0

            belief_set = initial_belief_set(self._agent.model)
            value_function = initial_value_function(self._agent.model)
            iteration_counter = 0

            for it in range(1, target_iter + 1):
                t_iter0 = time.perf_counter()
                gamma_in = vf_gamma(value_function)

                t_expand0 = time.perf_counter()
                new_bs = self._agent.expand(
                    belief_set=belief_set,
                    value_function=value_function,
                    max_generation=int(cfg.max_belief_growth),
                    mdp_policy=mdp_policy,
                )
                expand_ms = (time.perf_counter() - t_expand0) * 1000.0
                B = belief_matrix(new_bs)

                t_backup0 = time.perf_counter()
                bkp, actions, info = backend.backup(B, gamma_in, version=str(cfg.version))
                backup_wall_ms = (time.perf_counter() - t_backup0) * 1000.0

                t_update0 = time.perf_counter()
                value_function, append_info = append_backup_to_value_function(
                    self._agent.model,
                    value_function,
                    bkp,
                    actions,
                    compact_before_append=bool(cfg.compact_cuda_backup_before_append),
                    compact_prune_level=int(cfg.cuda_backup_compact_prune_level),
                )
                value_function, prune_info = maybe_prune(
                    value_function,
                    iteration_counter=iteration_counter,
                    prune_interval=int(cfg.prune_interval),
                    prune_level=int(cfg.prune_level),
                )
                update_ms = (time.perf_counter() - t_update0) * 1000.0

                iteration_counter += 1
                belief_set = belief_set.union(new_bs)
                gamma_after = vf_gamma(value_function)
                B_total = belief_matrix(belief_set)
                values_eval, _ = compute_vnew_and_gstar(B_total, gamma_after)
                iter_total_ms = (time.perf_counter() - t_iter0) * 1000.0

                row = {
                    "iter": int(it),
                    "nB": int(B.shape[0]),
                    "nG_in": int(gamma_in.shape[0]),
                    "expand_ms": float(expand_ms),
                    "backup_ms": float(info.elapsed_ms),
                    "backup_wall_ms": float(backup_wall_ms),
                    "update_ms": float(update_ms),
                    "iter_total_ms": float(iter_total_ms),
                    "actual_version": str(info.version_used),
                    "alpha_after": int(gamma_after.shape[0]),
                    "belief_total_after": int(B_total.shape[0]),
                    "value_mean_on_union": float(np.mean(values_eval)) if values_eval.size else 0.0,
                    "value_max_on_union": float(np.max(values_eval)) if values_eval.size else 0.0,
                    "value_min_on_union": float(np.min(values_eval)) if values_eval.size else 0.0,
                    "pruned": bool(prune_info.get("pruned", False)),
                    "alpha_before_prune": int(prune_info.get("alpha_vectors_before", gamma_after.shape[0])),
                    "alpha_after_prune": int(prune_info.get("alpha_vectors_after", gamma_after.shape[0])),
                    "cuda_bkp_rows_raw": int(append_info.get("cuda_bkp_rows_raw", bkp.shape[0])),
                    "cuda_bkp_rows_compact": int(append_info.get("cuda_bkp_rows_compact", bkp.shape[0])),
                    "cuda_backup_compacted": bool(append_info.get("cuda_backup_compacted", False)),
                }
                rows.append(row)

                if checkpoint_every > 0 and (it % int(checkpoint_every) == 0 or it == target_iter):
                    if out_path is not None:
                        np.save(out_path / f"iter_{it:04d}_gamma.npy", gamma_after)
                        np.save(out_path / f"iter_{it:04d}_beliefs.npy", B_total)

                if out_path is not None and (it == target_iter or it % max(1, int(checkpoint_every or target_iter)) == 0):
                    write_csv(out_path / "cuda_fsvi_rows.csv", rows)

            total_wall_s = time.perf_counter() - t_global0
            summary = {
                "mode": "s_v2_real_cublas_traincuda_notebook",
                "completed": True,
                "requested_expansions": int(target_iter),
                "completed_expansions": int(len(rows)),
                "cuda_lib_path": str(self.cuda_lib_path),
                "cuda_version_requested": str(self.cuda_version),
                "real_cublas_v7v8_pipeline": True,
                "cuda_device_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gamma": float(cfg.gamma),
                "max_belief_growth": int(cfg.max_belief_growth),
                "prune_interval": int(cfg.prune_interval),
                "prune_level": int(cfg.prune_level),
                "compact_cuda_backup_before_append": bool(cfg.compact_cuda_backup_before_append),
                "use_gpu_argument_accepted_for_compatibility": None if use_gpu is None else bool(use_gpu),
                "mdp_use_gpu": bool(cfg.use_gpu_trace),
                "policy_ms": float(policy_ms),
                "total_wall_s": float(total_wall_s),
                "sum_expand_ms": float(sum(r["expand_ms"] for r in rows)),
                "sum_backup_ms": float(sum(r["backup_ms"] for r in rows)),
                "sum_backup_wall_ms": float(sum(r["backup_wall_ms"] for r in rows)),
                "sum_update_ms": float(sum(r["update_ms"] for r in rows)),
                "sum_iter_total_ms": float(sum(r["iter_total_ms"] for r in rows)),
                "last_alpha_after": int(rows[-1]["alpha_after"]) if rows else None,
                "last_belief_total_after": int(rows[-1]["belief_total_after"]) if rows else None,
            }

            _set_agent_attr(self._agent, "value_function", value_function)
            _set_agent_attr(self._agent, "belief_set", belief_set)
            _set_agent_attr(self._agent, "cuda_training_rows", rows)
            _set_agent_attr(self._agent, "cuda_training_summary", summary)

            result = CudaNotebookTrainResult(
                agent=self._agent,
                value_function=value_function,
                belief_set=belief_set,
                rows=rows,
                summary=summary,
            )
            self.last_result = result

            if out_path is not None:
                columns = [
                    "iter", "nB", "nG_in", "expand_ms", "backup_ms", "backup_wall_ms",
                    "update_ms", "iter_total_ms", "actual_version", "alpha_after",
                    "belief_total_after", "pruned", "cuda_bkp_rows_raw", "cuda_bkp_rows_compact",
                    "cuda_backup_compacted",
                ]
                write_csv(out_path / "cuda_fsvi_rows.csv", rows)
                write_md(out_path / "cuda_fsvi_rows.md", "Notebook CUDA FSVI rows", rows, columns)
                (out_path / "cuda_fsvi_summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True, default=_json_ready) + "\n",
                    encoding="utf-8",
                )

            if visual:
                try:
                    from olfnav_cuda_notebook.visual import show_cuda_training_report
                    show_cuda_training_report(result, display_rows=int(display_rows))
                except Exception as exc:
                    print(f"[traincuda visual report skipped] {exc}")

            return result if return_result else value_function
        finally:
            try:
                backend.close()
            except Exception:
                pass


def enable_cuda_backend(agent: Any, **kwargs: Any) -> CudaFSVI_Agent:
    """Return a wrapper exposing ``traincuda`` for the CUDA backend.

    Example:

        ag = FSVI_Agent(...)
        ag_cuda = enable_cuda_backend(ag, device=0)
        result = ag_cuda.traincuda(expansions=1000, use_gpu=True)
    """
    return CudaFSVI_Agent(agent, **kwargs)


def patch_agent_traincuda(agent: Any, **kwargs: Any) -> Any:
    """Attach ``agent.traincuda(...)`` without changing native ``agent.train``.

    This is the safest option for simplified notebooks: ``train`` remains the
    upstream olfactory_navigation method, and ``traincuda`` is the CUDA-backend
    training method.

    Example:

        from olfnav_cuda_backend.notebook import patch_agent_traincuda
        patch_agent_traincuda(ag, device=0)
        result = ag.traincuda(expansions=1000, use_gpu=True)
    """
    runner = CudaFSVI_Agent(agent, **kwargs)
    _set_agent_attr(agent, "cuda_backend_runner", runner)

    def _cuda_traincuda(self: Any, *args: Any, **train_kwargs: Any) -> Any:
        return self.cuda_backend_runner.traincuda(*args, **train_kwargs)

    try:
        setattr(agent, "traincuda", types.MethodType(_cuda_traincuda, agent))
    except Exception as exc:
        raise RuntimeError(
            "Could not attach traincuda to this agent instance. Use "
            "ag_cuda = enable_cuda_backend(ag, ...) instead."
        ) from exc
    return agent


def patch_agent_train(agent: Any, **kwargs: Any) -> Any:
    """Deprecated compatibility alias for older notebooks.

    It now calls :func:`patch_agent_traincuda` and does *not* override
    ``agent.train``. Use ``agent.traincuda(...)`` afterwards.
    """
    return patch_agent_traincuda(agent, **kwargs)


__all__ = [
    "CudaFSVI_Agent",
    "CudaNotebookTrainResult",
    "enable_cuda_backend",
    "patch_agent_traincuda",
    "patch_agent_train",
    "package_root",
    "resolve_cuda_lib",
    "select_cuda_device",
]
