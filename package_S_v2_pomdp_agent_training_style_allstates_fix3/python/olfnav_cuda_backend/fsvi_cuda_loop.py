from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import time

import numpy as np

from .model_extractor import extract_model_from_agent
from .sparse_ell import compile_sparse_ell
from .backend_ctypes import CudaSparseBackupBackend
from .validation import compare_arrays


def _to_cpu_obj(x: Any) -> Any:
    if hasattr(x, "is_on_gpu") and bool(getattr(x, "is_on_gpu")):
        try:
            return x.to_cpu()
        except Exception:
            return x
    return x


def vf_gamma(vf: Any) -> np.ndarray:
    vf = _to_cpu_obj(vf)
    arr = np.asarray(getattr(vf, "alpha_vector_array"), dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected alpha_vector_array shape: {arr.shape}")
    return np.ascontiguousarray(arr)


def vf_actions(vf: Any) -> np.ndarray | None:
    vf = _to_cpu_obj(vf)
    for name in ["actions", "action_list", "_actions", "_action_list"]:
        if not hasattr(vf, name):
            continue
        try:
            arr = np.asarray(getattr(vf, name), dtype=np.int64).ravel()
        except Exception:
            continue
        if arr.size > 0:
            return arr
    return None


def belief_matrix(bs: Any) -> np.ndarray:
    bs = _to_cpu_obj(bs)
    arr = np.asarray(getattr(bs, "belief_array"), dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected belief_array shape: {arr.shape}")
    return np.ascontiguousarray(arr)


def initial_belief_set(model: Any) -> Any:
    from olfactory_navigation.agents.model_based_util.belief import Belief, BeliefSet
    return BeliefSet(model, [Belief(model)])


def initial_value_function(model: Any) -> Any:
    from olfactory_navigation.agents.model_based_util.value_function import ValueFunction
    return ValueFunction(model, model.expected_rewards_table.T, model.actions)


def compute_vnew_and_gstar(B: np.ndarray, Gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(B, dtype=np.float64) @ np.asarray(Gamma, dtype=np.float64).T
    gstar = np.argmax(scores, axis=1).astype(np.int64)
    vnew = scores[np.arange(scores.shape[0]), gstar]
    return vnew.astype(np.float64), gstar


def selected_actions_for_beliefs(actions: np.ndarray | None, gstar: np.ndarray, n_beliefs: int) -> np.ndarray | None:
    """Map a ValueFunction action array to one action per evaluated belief.

    Native olfactory_navigation backups may return a compact ValueFunction
    containing fewer alpha vectors than the number of beliefs because duplicate
    or dominated alpha rows are merged. The CUDA backend usually returns one
    candidate row per belief. Therefore comparing raw action arrays is invalid;
    the correct comparison is after selecting the alpha index that maximizes each
    belief.
    """
    if actions is None:
        return None
    arr = np.asarray(actions, dtype=np.int64).ravel()
    if arr.size == 0:
        return None
    gi = np.asarray(gstar, dtype=np.int64).ravel()
    if gi.size != int(n_beliefs):
        return None
    if int(gi.max(initial=0)) >= arr.size:
        return None
    return arr[gi].astype(np.int64)


def value_equivalence_on_beliefs(
    B_native: np.ndarray,
    Gamma_native: np.ndarray,
    B_cuda: np.ndarray,
    Gamma_cuda: np.ndarray,
    *,
    tol: float = 1e-10,
) -> tuple[bool, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compare two value functions semantically on a belief matrix.

    This is intentionally weaker than coefficient equality and is the right K3
    criterion, because native ValueFunction construction may compact, reorder,
    or merge alpha vectors while preserving values and selected policies.
    """
    native_v, native_g = compute_vnew_and_gstar(B_native, Gamma_native)
    cuda_v, cuda_g = compute_vnew_and_gstar(B_cuda, Gamma_cuda)
    ok, max_abs = _cmp(native_v, cuda_v, tol=tol)
    return ok, max_abs, native_v, native_g, cuda_v, cuda_g


def _candidate_action_payloads(model: Any, action_indices: np.ndarray) -> list[Any]:
    idx = np.asarray(action_indices, dtype=np.int64).ravel()
    out: list[Any] = [idx.astype(np.int64), idx.astype(np.int32)]
    actions = getattr(model, "actions", None)
    if actions is not None:
        try:
            seq = list(actions)
            out.append([seq[int(i)] for i in idx])
        except Exception:
            pass
    return out


def make_value_function_from_arrays(model: Any, alpha: np.ndarray, actions: np.ndarray) -> Any:
    from olfactory_navigation.agents.model_based_util.value_function import ValueFunction
    alpha = np.ascontiguousarray(np.asarray(alpha, dtype=np.float64))
    last_err: Exception | None = None
    for candidate_actions in _candidate_action_payloads(model, actions):
        try:
            return ValueFunction(model, alpha, candidate_actions)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not construct ValueFunction from CUDA arrays; last error={last_err}")


def compact_backup_arrays(
    model: Any,
    bkp: np.ndarray,
    actions: np.ndarray,
    *,
    prune_level: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compact CUDA backup candidates before appending to the solver value function.

    The CUDA backend naturally returns one candidate alpha vector per belief.
    The native olfactory_navigation backup often returns a compact ValueFunction
    with dominated/duplicate/equivalent rows removed. If the CUDA branch appends
    all raw rows, future iterations can diverge even when the current backup is
    semantically equivalent. This helper applies the same ValueFunction-level
    pruning machinery to the CUDA backup candidates before they are appended.
    """
    raw_bkp = np.ascontiguousarray(np.asarray(bkp, dtype=np.float64))
    raw_actions = np.ascontiguousarray(np.asarray(actions, dtype=np.int64).ravel())
    info: dict[str, Any] = {
        "cuda_bkp_rows_raw": int(raw_bkp.shape[0]),
        "cuda_bkp_rows_compact": int(raw_bkp.shape[0]),
        "cuda_backup_compacted": False,
    }
    try:
        cand_vf = make_value_function_from_arrays(model, raw_bkp, raw_actions)
        before = int(len(cand_vf))
        if int(prune_level) >= 0:
            cand_vf.prune(int(prune_level))
        compact_bkp = vf_gamma(cand_vf)
        compact_actions = vf_actions(cand_vf)
        if compact_actions is None or compact_actions.size != compact_bkp.shape[0]:
            compact_actions = raw_actions[: compact_bkp.shape[0]]
            if compact_actions.size != compact_bkp.shape[0]:
                compact_actions = np.zeros((compact_bkp.shape[0],), dtype=np.int64)
        after = int(compact_bkp.shape[0])
        info.update({
            "cuda_bkp_rows_raw": int(raw_bkp.shape[0]),
            "cuda_bkp_rows_before_compact_vf": before,
            "cuda_bkp_rows_compact": after,
            "cuda_backup_compacted": bool(after != raw_bkp.shape[0]),
        })
        return np.ascontiguousarray(compact_bkp), np.ascontiguousarray(compact_actions, dtype=np.int64), info
    except Exception as e:
        info["cuda_backup_compact_error"] = repr(e)
        return raw_bkp, raw_actions, info


def append_backup_to_value_function(
    model: Any,
    old_vf: Any,
    bkp: np.ndarray,
    actions: np.ndarray,
    *,
    compact_before_append: bool = True,
    compact_prune_level: int = 1,
) -> tuple[Any, dict[str, Any]]:
    old_gamma = vf_gamma(old_vf)
    old_actions = vf_actions(old_vf)
    if old_actions is None or old_actions.size != old_gamma.shape[0]:
        # This should rarely happen, but avoids crashing the bridge when the
        # library stores action objects that numpy cannot coerce. For old vectors
        # only the alpha values matter for future backup selection; actions matter
        # mainly for final policy reporting.
        old_actions = np.zeros((old_gamma.shape[0],), dtype=np.int64)

    info: dict[str, Any] = {
        "cuda_bkp_rows_raw": int(np.asarray(bkp).shape[0]),
        "cuda_bkp_rows_compact": int(np.asarray(bkp).shape[0]),
        "cuda_backup_compacted": False,
    }
    append_bkp = np.asarray(bkp, dtype=np.float64)
    append_actions = np.asarray(actions, dtype=np.int64).ravel()
    if compact_before_append:
        append_bkp, append_actions, info = compact_backup_arrays(
            model, append_bkp, append_actions, prune_level=int(compact_prune_level)
        )

    new_gamma = np.ascontiguousarray(np.vstack([old_gamma, append_bkp]))
    new_actions = np.ascontiguousarray(np.concatenate([old_actions.astype(np.int64), append_actions.astype(np.int64).ravel()]))
    return make_value_function_from_arrays(model, new_gamma, new_actions), info


def maybe_prune(value_function: Any, *, iteration_counter: int, prune_interval: int, prune_level: int) -> tuple[Any, dict[str, Any]]:
    info = {
        "pruned": False,
        "prune_iteration_counter": int(iteration_counter),
        "alpha_vectors_before": int(len(value_function)),
        "alpha_vectors_after": int(len(value_function)),
    }
    if prune_interval > 0 and (iteration_counter % prune_interval) == 0 and iteration_counter > 0:
        value_function.prune(prune_level)
        info["pruned"] = True
        info["alpha_vectors_after"] = int(len(value_function))
    return value_function, info


@dataclass
class K3RunConfig:
    lib_path: str
    gamma: float = 0.95
    mdp_vi_horizon: int = 1000
    mdp_vi_eps: float = 1e-6
    max_belief_growth: int = 10
    update_passes: int = 1
    prune_interval: int = 10
    prune_level: int = 1
    compact_cuda_backup_before_append: bool = True
    cuda_backup_compact_prune_level: int = 1
    # Keep CUDA backup comparison lockstep with native solver state by default.
    # This validates the replacement backup operator without accumulating
    # representation-level ValueFunction pruning differences.
    lockstep_native_state: bool = True
    version: str = "auto"
    use_gpu_trace: bool = False
    print_progress: bool = False


def _solve_mdp_policy(agent: Any, cfg: K3RunConfig):
    from olfactory_navigation.agents.model_based_util import vi_solver
    policy, hist = vi_solver.solve(
        model=agent.model,
        horizon=int(cfg.mdp_vi_horizon),
        initial_value_function=None,
        gamma=float(cfg.gamma),
        eps=float(cfg.mdp_vi_eps),
        use_gpu=bool(cfg.use_gpu_trace),
        history_tracking_level=1,
        print_progress=bool(cfg.print_progress),
    )
    return policy, hist


def _cmp(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> tuple[bool, float]:
    if a.shape != b.shape:
        return False, float("inf")
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    max_abs = float(diff.max()) if diff.size else 0.0
    return bool(max_abs <= tol), max_abs


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(path: str | Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows.")
    else:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for r in rows:
            vals = []
            for c in columns:
                v = r.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_shadow_fsvi_pair(build_agent, *, targets: Iterable[int], cfg: K3RunConfig, outdir: str | Path) -> list[dict[str, Any]]:
    """Run native FSVI and CUDA-backup FSVI side-by-side.

    This is K3's conservative bridge: it does not monkey-patch the upstream
    library. It reproduces the FSVI expand/backup/update loop explicitly, exactly
    as the previous shadow packages did, but replaces the CUDA branch backup with
    CudaSparseBackupBackend.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    targets = sorted({int(x) for x in targets})
    if not targets:
        raise ValueError("targets must not be empty")
    max_target = max(targets)

    # Build two agents so native and CUDA branches can evolve independently.
    agent_native = build_agent()
    agent_cuda = build_agent()
    if cfg.use_gpu_trace:
        agent_native = agent_native.to_gpu()
        agent_cuda = agent_cuda.to_gpu()

    model_static = extract_model_from_agent(agent_cuda, gamma=cfg.gamma)
    ell = compile_sparse_ell(model_static.T)
    backend = CudaSparseBackupBackend.from_model(
        model_static,
        ell,
        lib_path=cfg.lib_path,
        version=cfg.version,
    )

    native_policy, _ = _solve_mdp_policy(agent_native, cfg)
    cuda_policy, _ = _solve_mdp_policy(agent_cuda, cfg)

    native_belief_set = initial_belief_set(agent_native.model)
    cuda_belief_set = initial_belief_set(agent_cuda.model)
    native_vf = initial_value_function(agent_native.model)
    cuda_vf = initial_value_function(agent_cuda.model)

    rows: list[dict[str, Any]] = []
    native_iter_counter = 0
    cuda_iter_counter = 0

    meta = {
        "gamma": cfg.gamma,
        "targets": targets,
        "model_hash": model_static.model_hash,
        "nS": model_static.nS,
        "nA": model_static.nA,
        "nO": model_static.nO,
        "max_nnz": ell.max_nnz,
        "version": cfg.version,
        "mode": "explicit_shadow_fsvi_native_vs_cuda_backend",
        "note": "K3 explicit FSVI loop; native branch uses agent.backup, CUDA branch uses in-process backend and manual ValueFunction append. K3-compact compacts CUDA backup candidates before append to mimic native ValueFunction semantics.",
        "compact_cuda_backup_before_append": bool(cfg.compact_cuda_backup_before_append),
        "cuda_backup_compact_prune_level": int(cfg.cuda_backup_compact_prune_level),
        "lockstep_native_state": bool(cfg.lockstep_native_state),
    }
    (outdir / "k3_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for expansion_i in range(1, max_target + 1):
        # Native branch.
        native_gamma_in = vf_gamma(native_vf)
        t_expand_native = time.perf_counter()
        native_new_bs = agent_native.expand(
            belief_set=native_belief_set,
            value_function=native_vf,
            max_generation=int(cfg.max_belief_growth),
            mdp_policy=native_policy,
        )
        native_expand_ms = (time.perf_counter() - t_expand_native) * 1000.0
        native_B = belief_matrix(native_new_bs)

        t_native_raw = time.perf_counter()
        native_raw_vf = agent_native.backup(
            native_new_bs,
            native_vf,
            gamma=float(cfg.gamma),
            append=False,
            belief_dominance_prune=False,
        )
        native_backup_raw_ms = (time.perf_counter() - t_native_raw) * 1000.0
        native_bkp = vf_gamma(native_raw_vf)
        native_actions = vf_actions(native_raw_vf)
        if native_actions is None:
            native_actions = np.full((native_bkp.shape[0],), -1, dtype=np.int64)

        t_native_update = time.perf_counter()
        for _ in range(int(cfg.update_passes)):
            native_vf = agent_native.backup(
                native_new_bs,
                native_vf,
                gamma=float(cfg.gamma),
                append=True,
                belief_dominance_prune=False,
            )
            native_vf, native_prune_info = maybe_prune(
                native_vf,
                iteration_counter=native_iter_counter,
                prune_interval=int(cfg.prune_interval),
                prune_level=int(cfg.prune_level),
            )
            native_iter_counter += 1
        native_update_ms = (time.perf_counter() - t_native_update) * 1000.0
        native_belief_set = native_belief_set.union(native_new_bs)

        # CUDA branch.
        #
        # In lockstep mode, CUDA is evaluated on the exact native solver state
        # at every iteration. This prevents representation-level pruning
        # differences from accumulating and producing false negatives in later
        # iterations.
        if bool(cfg.lockstep_native_state):
            cuda_gamma_in = native_gamma_in.copy()
            cuda_new_bs = native_new_bs
            cuda_B = native_B.copy()
            cuda_expand_ms = 0.0
        else:
            cuda_gamma_in = vf_gamma(cuda_vf)
            t_expand_cuda = time.perf_counter()
            cuda_new_bs = agent_cuda.expand(
                belief_set=cuda_belief_set,
                value_function=cuda_vf,
                max_generation=int(cfg.max_belief_growth),
                mdp_policy=cuda_policy,
            )
            cuda_expand_ms = (time.perf_counter() - t_expand_cuda) * 1000.0
            cuda_B = belief_matrix(cuda_new_bs)

        t_cuda_backup = time.perf_counter()
        cuda_bkp, cuda_actions, info = backend.backup(cuda_B, cuda_gamma_in, version=cfg.version)
        cuda_backup_wall_ms = (time.perf_counter() - t_cuda_backup) * 1000.0

        t_cuda_update = time.perf_counter()
        for _ in range(int(cfg.update_passes)):
            cuda_vf, cuda_append_info = append_backup_to_value_function(
                agent_cuda.model,
                cuda_vf,
                cuda_bkp,
                cuda_actions,
                compact_before_append=bool(cfg.compact_cuda_backup_before_append),
                compact_prune_level=int(cfg.cuda_backup_compact_prune_level),
            )
            cuda_vf, cuda_prune_info = maybe_prune(
                cuda_vf,
                iteration_counter=cuda_iter_counter,
                prune_interval=int(cfg.prune_interval),
                prune_level=int(cfg.prune_level),
            )
            cuda_iter_counter += 1
        cuda_update_ms = (time.perf_counter() - t_cuda_update) * 1000.0
        cuda_belief_set = cuda_belief_set.union(cuda_new_bs)

        # Comparisons.
        #
        # Important K3 distinction:
        # olfactory_navigation.ValueFunction may compact/reorder equivalent
        # alpha-vectors. Therefore coefficient-level equality of BKP/full Gamma
        # is diagnostic only. The pass/fail criterion is semantic equivalence on
        # the traced belief rows plus selected-action equivalence where actions
        # are representable.
        beliefs_ok, beliefs_max_abs = _cmp(native_B, cuda_B, tol=1e-10)
        gamma_in_exact_ok, gamma_in_max_abs = _cmp(native_gamma_in, cuda_gamma_in, tol=1e-10)
        gamma_in_value_ok, gamma_in_value_max_abs, _, _, _, _ = value_equivalence_on_beliefs(
            native_B, native_gamma_in, cuda_B, cuda_gamma_in, tol=1e-10
        )

        bkp_coeff_ok, bkp_coeff_max_abs = _cmp(native_bkp, cuda_bkp, tol=1e-10)
        bkp_ok, bkp_value_max_abs, native_vnew, native_gstar, cuda_vnew, cuda_gstar = value_equivalence_on_beliefs(
            native_B, native_bkp, cuda_B, cuda_bkp, tol=1e-10
        )
        native_selected_actions = selected_actions_for_beliefs(native_actions, native_gstar, native_B.shape[0])
        cuda_selected_actions = selected_actions_for_beliefs(cuda_actions, cuda_gstar, cuda_B.shape[0])
        actions_comparable = native_selected_actions is not None and cuda_selected_actions is not None
        actions_ok = bool(actions_comparable and np.array_equal(native_selected_actions, cuda_selected_actions))
        vnew_ok, vnew_max_abs = _cmp(native_vnew, cuda_vnew, tol=1e-10)
        # Exact gstar indices are only comparable if the two candidate alpha sets
        # have the same shape/order. Otherwise gstar_ok is diagnostic only.
        gstar_ok = bool(native_bkp.shape == cuda_bkp.shape and np.array_equal(native_gstar, cuda_gstar))

        full_gamma_native = vf_gamma(native_vf)
        full_gamma_cuda = vf_gamma(cuda_vf)
        full_gamma_ok, full_gamma_max_abs = _cmp(full_gamma_native, full_gamma_cuda, tol=1e-10)
        native_union_B = belief_matrix(native_belief_set)
        cuda_union_B = belief_matrix(cuda_belief_set)
        union_beliefs_ok, union_beliefs_max_abs = _cmp(native_union_B, cuda_union_B, tol=1e-10)
        full_value_ok, full_value_max_abs, native_full_values, native_full_gstar, cuda_full_values, cuda_full_gstar = value_equivalence_on_beliefs(
            native_union_B, full_gamma_native, cuda_union_B, full_gamma_cuda, tol=1e-10
        )

        row = {
            "iter": expansion_i,
            "is_target": int(expansion_i in targets),
            "nS": model_static.nS,
            "nB": int(native_B.shape[0]),
            "nG_in": int(native_gamma_in.shape[0]),
            "native_expand_ms": native_expand_ms,
            "cuda_expand_ms": cuda_expand_ms,
            "native_backup_raw_ms": native_backup_raw_ms,
            "native_update_backup_ms": native_update_ms,
            "cuda_backend_elapsed_ms": float(info.elapsed_ms),
            "cuda_backup_wall_ms": cuda_backup_wall_ms,
            "cuda_update_append_ms": cuda_update_ms,
            "cuda_actual_version": info.version_used,
            "lockstep_native_state": bool(cfg.lockstep_native_state),
            "beliefs_ok": beliefs_ok,
            "beliefs_max_abs": beliefs_max_abs,
            "union_beliefs_ok": union_beliefs_ok,
            "union_beliefs_max_abs": union_beliefs_max_abs,
            "gamma_in_ok": gamma_in_value_ok,
            "gamma_in_exact_ok": gamma_in_exact_ok,
            "gamma_in_max_abs": gamma_in_max_abs,
            "gamma_in_value_max_abs": gamma_in_value_max_abs,
            "bkp_ok": bkp_ok,
            "bkp_value_max_abs": bkp_value_max_abs,
            "bkp_coeff_ok": bkp_coeff_ok,
            "bkp_coeff_max_abs": bkp_coeff_max_abs,
            "native_bkp_rows": int(native_bkp.shape[0]),
            "cuda_bkp_rows": int(cuda_bkp.shape[0]),
            "cuda_bkp_rows_compact": int(cuda_append_info.get("cuda_bkp_rows_compact", int(cuda_bkp.shape[0]))),
            "cuda_backup_compacted": bool(cuda_append_info.get("cuda_backup_compacted", False)),
            "actions_ok": actions_ok,
            "actions_comparable": actions_comparable,
            "native_action_rows": int(np.asarray(native_actions).size),
            "cuda_action_rows": int(np.asarray(cuda_actions).size),
            "vnew_ok": vnew_ok,
            "vnew_max_abs": vnew_max_abs,
            "gstar_ok": gstar_ok,
            "full_gamma_ok": full_gamma_ok,
            "full_gamma_max_abs": full_gamma_max_abs,
            "full_value_ok": full_value_ok,
            "full_value_max_abs": full_value_max_abs,
            "current_backup_semantic_ok": bool(beliefs_ok and gamma_in_value_ok and bkp_ok and vnew_ok),
            "solver_state_semantic_ok": bool(full_value_ok),
            "native_alpha_after": int(full_gamma_native.shape[0]),
            "cuda_alpha_after": int(full_gamma_cuda.shape[0]),
        }
        rows.append(row)
        print(
            "[K3] "
            f"iter={expansion_i} nB={row['nB']} nG={row['nG_in']} "
            f"native_backup_ms={native_backup_raw_ms:.6f} "
            f"cuda_ms={float(info.elapsed_ms):.6f} actual={info.version_used} "
            f"bkp_ok={bkp_ok} actions_ok={actions_ok} full_value_ok={full_value_ok} "
            f"lockstep={bool(cfg.lockstep_native_state)} coeff_ok={bkp_coeff_ok}"
        )

        if expansion_i in targets:
            prefix = outdir / f"iter_{expansion_i:04d}"
            np.save(str(prefix) + "_native_B.npy", native_B)
            np.save(str(prefix) + "_cuda_B.npy", cuda_B)
            np.save(str(prefix) + "_native_gamma_in.npy", native_gamma_in)
            np.save(str(prefix) + "_cuda_gamma_in.npy", cuda_gamma_in)
            np.save(str(prefix) + "_native_bkp.npy", native_bkp)
            np.save(str(prefix) + "_cuda_bkp.npy", cuda_bkp)
            np.save(str(prefix) + "_native_actions.npy", native_actions)
            np.save(str(prefix) + "_cuda_actions.npy", cuda_actions)

    write_csv(outdir / "k3_shadow_rows.csv", rows)
    target_rows = [r for r in rows if int(r.get("is_target", 0)) == 1]
    write_csv(outdir / "k3_shadow_targets.csv", target_rows)
    columns = [
        "iter", "nB", "nG_in", "native_backup_raw_ms", "cuda_backend_elapsed_ms",
        "cuda_actual_version", "lockstep_native_state", "bkp_ok", "actions_ok", "vnew_ok", "current_backup_semantic_ok", "full_value_ok",
        "bkp_value_max_abs", "full_value_max_abs", "bkp_coeff_ok", "bkp_coeff_max_abs",
        "native_bkp_rows", "cuda_bkp_rows", "cuda_bkp_rows_compact", "cuda_backup_compacted", "gamma_in_ok", "gamma_in_exact_ok", "full_gamma_ok",
    ]
    write_md(outdir / "k3_shadow_targets.md", "K3 shadow FSVI CUDA-backup targets", target_rows, columns)
    backend.close()
    return rows
