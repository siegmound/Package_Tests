from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from .fixes import install_simulation_history_patch, normalize_environment_for_exact_converter


def _native_if_cuda(agent: Any) -> Any:
    if agent is None:
        return None
    return getattr(agent, "native_agent", agent)


def _environment_from_agent_or_env(agent_or_env: Any) -> Any:
    obj = _native_if_cuda(agent_or_env)
    env = getattr(obj, "environment", None)
    if env is not None:
        return env
    return obj


def _as_numpy_start_points(start_points: Any) -> np.ndarray:
    if start_points is None:
        return None
    try:
        if hasattr(start_points, "to_numpy"):
            start_points = start_points.to_numpy()
        return np.asarray(start_points, dtype=int)
    except Exception:
        return np.asarray(start_points)


def raw_start_points_from_environment(agent_or_env: Any) -> np.ndarray:
    """Return raw start coordinates from ``environment.start_probabilities``.

    This mirrors the upstream all-starts idea, but it is explicit and reusable
    for CPU/CuPy/CUDA comparisons.  No hard-coded source coordinates are used.
    """
    env = _environment_from_agent_or_env(agent_or_env)
    if env is None or not hasattr(env, "start_probabilities"):
        raise AttributeError("Could not find environment.start_probabilities")
    normalize_environment_for_exact_converter(env, verbose=False)
    return np.argwhere(np.asarray(env.start_probabilities) > 0)


def _is_source_or_terminal(env: Any, point: np.ndarray) -> bool:
    for method_name in ("is_source", "is_at_source", "reached_source", "is_terminal", "is_done"):
        fn = getattr(env, method_name, None)
        if callable(fn):
            try:
                return bool(fn(point))
            except Exception:
                try:
                    return bool(fn(tuple(np.asarray(point).tolist())))
                except Exception:
                    pass

    source_position = getattr(env, "source_position", None)
    source_radius = getattr(env, "source_radius", None)
    if source_position is not None and source_radius is not None:
        try:
            source_position = np.asarray(source_position, dtype=float).reshape(-1)
            point = np.asarray(point, dtype=float).reshape(-1)
            dims = min(len(point), len(source_position))
            distance = np.linalg.norm(point[-dims:] - source_position[-dims:])
            return bool(distance <= float(source_radius))
        except Exception:
            pass
    return False


def clean_start_points(
    agent_or_env: Any,
    *,
    max_points: int | None = None,
    remove_source_points: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Generate clean start points for policy evaluation.

    Parameters
    ----------
    agent_or_env:
        Either an agent/wrapper or an Environment instance.
    max_points:
        Optional prefix limit for quick tests.  ``None`` returns all valid starts.
    remove_source_points:
        Remove points already inside the source radius when source metadata is
        available.  This avoids simulations that are terminal at step zero.
    verbose:
        Print a compact audit.

    Returns
    -------
    np.ndarray
        Plain integer array of start coordinates, normally shape ``(N, 2)`` for
        the olfactory 2D envs.
    """
    env = _environment_from_agent_or_env(agent_or_env)
    if env is None:
        raise AttributeError("Could not infer environment from agent_or_env")

    normalize_environment_for_exact_converter(env, verbose=False)

    raw = raw_start_points_from_environment(env)
    starts = np.asarray(raw, dtype=int)

    dims = int(getattr(env, "dimensions", starts.shape[1] if starts.ndim == 2 else 1))

    # For non-layered envs, keep only spatial coordinates if an upstream array
    # accidentally includes extra indexing columns.
    if starts.ndim == 1:
        starts = starts.reshape(-1, dims)
    elif starts.ndim == 2 and starts.shape[1] > dims:
        starts = starts[:, -dims:]

    if remove_source_points and starts.size:
        keep = []
        for point in starts:
            keep.append(not _is_source_or_terminal(env, point))
        starts = starts[np.asarray(keep, dtype=bool)]

    starts = np.ascontiguousarray(starts, dtype=int)
    if max_points is not None:
        starts = starts[: int(max_points)]

    if verbose:
        print("[START_POINTS] raw:", raw.shape)
        print("[START_POINTS] clean:", starts.shape)
        print("[START_POINTS] first clean points:")
        print(starts[:5])

    return starts


def generate_policy_start_points(
    agent_or_env: Any,
    *,
    n_eval: int | None = 100,
    out_root: str | Path | None = None,
    remove_source_points: bool = True,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Generate raw/full/eval start point arrays and optionally save them.

    Returns a dictionary with:

    - ``raw``: all positive start-probability coordinates.
    - ``full``: cleaned full start set.
    - ``eval``: quick-evaluation subset, controlled by ``n_eval``.
    """
    env = _environment_from_agent_or_env(agent_or_env)
    raw = raw_start_points_from_environment(env)
    full = clean_start_points(
        env,
        max_points=None,
        remove_source_points=remove_source_points,
        verbose=False,
    )
    eval_points = full if n_eval is None else full[: int(n_eval)]

    if verbose:
        print("[START_POINTS] start_probabilities shape:", np.asarray(env.start_probabilities).shape)
        print("[START_POINTS] raw start points:", raw.shape)
        print("[START_POINTS] clean/full start points:", full.shape)
        print("[START_POINTS] n_eval:", n_eval)
        print("[START_POINTS] eval start points:", eval_points.shape)
        print("[START_POINTS] first eval points:")
        print(eval_points[:5])

    if out_root is not None:
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        np.save(out_root / "start_points_raw.npy", raw)
        np.save(out_root / "start_points_full.npy", full)
        np.save(out_root / "start_points_eval.npy", eval_points)
        if verbose:
            print("[START_POINTS] saved to:", out_root)

    return {"raw": raw, "full": full, "eval": eval_points}


# Backward-compatible spelling used in earlier notebooks.
def make_policy_start_points(*args, **kwargs):
    return generate_policy_start_points(*args, **kwargs)


def run_policy_evaluation(
    agent: Any,
    *,
    n: int | None = 100,
    start_points=None,
    patch_history: bool = True,
    **kwargs,
):
    """Run upstream olfactory_navigation policy simulation with safe start points."""
    if patch_history:
        install_simulation_history_patch(verbose=False)
    from olfactory_navigation.simulation import run_test

    native_agent = _native_if_cuda(agent)

    if start_points is None:
        start_points = clean_start_points(native_agent, max_points=n)
    else:
        start_points = _as_numpy_start_points(start_points)
        if n is not None:
            start_points = start_points[: int(n)]

    if n is None:
        n_run = len(start_points)
    else:
        n_run = min(int(n), len(start_points))
        start_points = start_points[:n_run]

    return run_test(agent=native_agent, n=int(n_run), start_points=start_points, **kwargs)


def run_policy_smoke(agent: Any, *, n: int = 100, start_points=None, **kwargs):
    """Compatibility alias. Prefer ``run_policy_evaluation`` for official runs."""
    return run_policy_evaluation(agent, n=n, start_points=start_points, **kwargs)


def run_policy_full_evaluation(agent: Any, *, start_points=None, **kwargs):
    """Run policy evaluation on all clean start points."""
    native_agent = _native_if_cuda(agent)
    if start_points is None:
        start_points = clean_start_points(native_agent, max_points=None)
    else:
        start_points = _as_numpy_start_points(start_points)
    return run_policy_evaluation(native_agent, n=len(start_points), start_points=start_points, **kwargs)
