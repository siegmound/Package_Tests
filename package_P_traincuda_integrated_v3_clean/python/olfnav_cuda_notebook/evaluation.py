from __future__ import annotations

from typing import Any
import numpy as np

from .fixes import (
    normalize_agent_environment,
    normalize_non_layered_environment,
    normalize_simulation_history,
    simulation_history_patch,
)


def _native_agent(agent: Any) -> Any:
    # The CUDA wrapper delegates attributes, but upstream run_test is happier
    # when it receives the actual FSVI_Agent object after traincuda has populated it.
    try:
        return agent.native_agent
    except Exception:
        return agent


def _agent_environment(agent: Any) -> Any:
    try:
        return agent.environment
    except Exception:
        return _native_agent(agent).environment


def clean_start_points(agent_or_env: Any, *, limit: int | None = None, verbose: bool = False) -> np.ndarray:
    """Return valid start coordinates in the shape expected by run_test.

    For non-layered 2D environments this returns an (N, 2) integer array.  If a
    reconstructed object accidentally stores layer metadata as False, it is
    normalized before extracting the starts.
    """
    env = agent_or_env
    if hasattr(agent_or_env, "environment") or hasattr(agent_or_env, "native_agent"):
        env = _agent_environment(agent_or_env)
    normalize_non_layered_environment(env, verbose=verbose)

    starts = np.argwhere(np.asarray(env.start_probabilities) > 0)
    dims = int(getattr(env, "dimensions", starts.shape[1]))

    is_non_layered = True
    for attr in ("environment_layer_labels", "layer_labels", "layers"):
        try:
            val = getattr(env, attr)
            if val not in (None, False):
                is_non_layered = False
                break
        except Exception:
            pass

    if is_non_layered and starts.ndim == 2 and starts.shape[1] > dims:
        starts = starts[:, -dims:]

    starts = np.asarray(starts, dtype=int)
    if limit is not None:
        starts = starts[: int(limit)]
    return starts


def run_policy_evaluation(
    agent: Any,
    *,
    start_points: np.ndarray | None = None,
    n: int | None = None,
    environment: Any | None = None,
    patch_history: bool = True,
    verbose_fixes: bool = False,
    **kwargs: Any,
):
    """Run upstream run_test and return a normalized SimulationHistory.

    This wrapper intentionally preserves the upstream SimulationHistory object;
    after return, hist.plot() is still the original olfactory plot method.
    """
    from olfactory_navigation.simulation import run_test

    normalize_agent_environment(agent, verbose=verbose_fixes)
    native = _native_agent(agent)
    normalize_agent_environment(native, verbose=verbose_fixes)

    if environment is None:
        environment = _agent_environment(native)
    normalize_non_layered_environment(environment, verbose=verbose_fixes)

    if start_points is None:
        start_points = clean_start_points(environment)
    else:
        start_points = np.asarray(start_points, dtype=int)

    if n is None:
        n = len(start_points)
    else:
        n = int(n)
        start_points = start_points[:n]

    call_kwargs = dict(
        agent=native,
        n=n,
        start_points=start_points,
        environment=environment,
    )
    call_kwargs.update(kwargs)

    with simulation_history_patch(enabled=patch_history, verbose=verbose_fixes):
        hist = run_test(**call_kwargs)
    return normalize_simulation_history(hist, verbose=verbose_fixes, name="hist")


def run_policy_smoke(agent: Any, *, n: int = 100, start_points: np.ndarray | None = None, **kwargs: Any):
    """Quick policy sanity check. For official full runs use run_policy_full_evaluation."""
    if start_points is None:
        start_points = clean_start_points(agent, limit=n)
    else:
        start_points = np.asarray(start_points, dtype=int)[: int(n)]
    return run_policy_evaluation(agent, start_points=start_points, n=int(n), **kwargs)


def run_policy_full_evaluation(agent: Any, *, start_points: np.ndarray | None = None, **kwargs: Any):
    """Evaluate all valid clean start points by default."""
    if start_points is None:
        start_points = clean_start_points(agent)
    else:
        start_points = np.asarray(start_points, dtype=int)
    return run_policy_evaluation(agent, start_points=start_points, n=len(start_points), **kwargs)


# Backward compatible explicit name.
run_policy_full = run_policy_full_evaluation
