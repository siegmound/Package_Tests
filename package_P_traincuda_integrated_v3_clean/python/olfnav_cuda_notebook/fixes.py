from __future__ import annotations

"""Safe integration fixes for olfactory_navigation notebooks.

The fixes are intentionally narrow:
- non-layered environment metadata represented as False is normalized to None;
- returned SimulationHistory objects are normalized before hist.plot();
- an optional scoped monkey patch protects direct run_test usage.

No policy, value function, rewards, positions, observations or actions are
changed by the metadata normalization itself.
"""

from contextlib import contextmanager
from typing import Any, Iterator
import numpy as np


def _is_false_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value) is False


def normalize_non_layered_metadata(obj: Any, *, verbose: bool = False, name: str = "object") -> Any:
    """Normalize common non-layered metadata fields in-place.

    Upstream olfactory_navigation distinguishes non-layered environments using
    None.  Some reconstructed env_info files store the same information as
    False.  False is not None, so SimulationHistory.add_step()/plot() can enter
    the layered branch and produce shape or bool-subscript errors.
    """
    if obj is None:
        return None
    for attr in (
        "environment_layer_labels",
        "layer_labels",
        "layers",
        "environment_layers",
    ):
        try:
            if hasattr(obj, attr) and _is_false_bool(getattr(obj, attr)):
                if verbose:
                    print(f"[fix] {name}.{attr}: False -> None")
                setattr(obj, attr, None)
        except Exception:
            pass
    return obj


def normalize_non_layered_environment(env: Any, *, verbose: bool = False) -> Any:
    return normalize_non_layered_metadata(env, verbose=verbose, name="env")


def normalize_agent_environment(agent: Any, *, verbose: bool = False) -> Any:
    normalize_non_layered_metadata(agent, verbose=verbose, name="agent")
    try:
        normalize_non_layered_environment(agent.environment, verbose=verbose)
    except Exception:
        pass
    try:
        normalize_non_layered_environment(agent._agent.environment, verbose=verbose)
    except Exception:
        pass
    return agent


def normalize_simulation_history(hist: Any, *, verbose: bool = False, name: str = "hist") -> Any:
    """Normalize a SimulationHistory object so hist.plot() works for non-layered envs."""
    normalize_non_layered_metadata(hist, verbose=verbose, name=name)
    try:
        normalize_non_layered_environment(hist.environment, verbose=verbose)
    except Exception:
        pass
    return hist


def fix_history_plot_metadata(hist: Any, *, verbose: bool = True, name: str = "hist") -> Any:
    """Public alias used when a user already has an existing hist object."""
    return normalize_simulation_history(hist, verbose=verbose, name=name)


_PATCH_STATE: dict[str, Any] = {
    "installed": False,
    "orig_init": None,
    "orig_add_step": None,
    "orig_plot": None,
}


def install_simulation_history_patch(*, verbose: bool = False) -> bool:
    """Install a narrow upstream SimulationHistory metadata patch.

    This is idempotent. It patches __init__, add_step and plot only to normalize
    non-layered metadata before the original implementation runs.
    """
    if _PATCH_STATE["installed"]:
        return True

    try:
        from olfactory_navigation.simulation import SimulationHistory
    except Exception as exc:
        if verbose:
            print(f"[fix] could not import SimulationHistory: {type(exc).__name__}: {exc}")
        return False

    orig_init = SimulationHistory.__init__
    orig_add_step = SimulationHistory.add_step
    orig_plot = SimulationHistory.plot

    def patched_init(self, *args, **kwargs):
        out = orig_init(self, *args, **kwargs)
        normalize_simulation_history(self, verbose=False, name="SimulationHistory")
        return out

    def patched_add_step(self, *args, **kwargs):
        normalize_simulation_history(self, verbose=False, name="SimulationHistory")
        return orig_add_step(self, *args, **kwargs)

    def patched_plot(self, *args, **kwargs):
        normalize_simulation_history(self, verbose=False, name="SimulationHistory")
        return orig_plot(self, *args, **kwargs)

    _PATCH_STATE.update(
        installed=True,
        orig_init=orig_init,
        orig_add_step=orig_add_step,
        orig_plot=orig_plot,
    )
    SimulationHistory.__init__ = patched_init
    SimulationHistory.add_step = patched_add_step
    SimulationHistory.plot = patched_plot
    if verbose:
        print("[fix] installed SimulationHistory non-layered metadata patch")
    return True


def uninstall_simulation_history_patch(*, verbose: bool = False) -> bool:
    if not _PATCH_STATE["installed"]:
        return True
    try:
        from olfactory_navigation.simulation import SimulationHistory
        SimulationHistory.__init__ = _PATCH_STATE["orig_init"]
        SimulationHistory.add_step = _PATCH_STATE["orig_add_step"]
        SimulationHistory.plot = _PATCH_STATE["orig_plot"]
    except Exception as exc:
        if verbose:
            print(f"[fix] could not uninstall SimulationHistory patch: {type(exc).__name__}: {exc}")
        return False
    finally:
        _PATCH_STATE.update(installed=False, orig_init=None, orig_add_step=None, orig_plot=None)
    if verbose:
        print("[fix] uninstalled SimulationHistory patch")
    return True


@contextmanager
def simulation_history_patch(*, enabled: bool = True, verbose: bool = False) -> Iterator[None]:
    """Scoped compatibility patch.

    Evaluation wrappers use this automatically. Users generally do not need to
    call it from notebooks anymore.
    """
    if enabled:
        install_simulation_history_patch(verbose=verbose)
    yield
