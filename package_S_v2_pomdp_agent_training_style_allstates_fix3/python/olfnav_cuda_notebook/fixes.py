from __future__ import annotations

"""Small compatibility fixes for notebook use.

The upstream non-layered environments may store layer metadata as ``False``.
Some plotting paths expect either ``None`` or a sequence of layer labels.  The
patch below is intentionally narrow: it only normalizes metadata for plotting;
it does not alter actions, observations, rewards, trajectories, beliefs, or
value functions.
"""

from typing import Any


def _normalize_layer_metadata(obj: Any) -> None:
    for attr in ("environment_layer_labels", "layer_labels"):
        if hasattr(obj, attr):
            try:
                if getattr(obj, attr) is False:
                    setattr(obj, attr, None)
            except Exception:
                pass
    env = getattr(obj, "environment", None)
    if env is not None:
        _normalize_layer_metadata(env)


def normalize_environment_for_exact_converter(env: Any, verbose: bool = False) -> Any:
    """Normalize upstream ``Environment`` metadata before ``FSVI_Agent(env)``.

    Some environments loaded from JSON metadata store ``shape`` as a Python
    ``list``.  The upstream exact converter concatenates ``environment.shape``
    with a tuple, so list-valued shapes fail with:

        TypeError: can only concatenate list (not "tuple") to list

    This helper is intentionally narrow: it only converts shape-like metadata
    lists to tuples and normalizes non-layered metadata from ``False`` to
    ``None``.  It does not alter data arrays, source position, thresholds,
    actions, rewards, transitions, beliefs, or value functions.
    """
    if env is None:
        return env

    changed = []

    def _tupleize_attr(obj: Any, attr: str) -> None:
        if not hasattr(obj, attr):
            return
        try:
            value = getattr(obj, attr)
        except Exception:
            return
        if isinstance(value, list):
            new_value = tuple(value)
            try:
                setattr(obj, attr, new_value)
                changed.append(attr)
            except Exception:
                # Fallback for classes that expose a read-only property backed
                # by a private attribute.  This is harmless if the private
                # field does not exist.
                private_attr = f"_{attr}"
                if hasattr(obj, private_attr):
                    try:
                        setattr(obj, private_attr, new_value)
                        changed.append(private_attr)
                    except Exception:
                        pass

    for attr in ("shape", "data_shape"):
        _tupleize_attr(env, attr)

    _normalize_layer_metadata(env)

    if verbose:
        shape = getattr(env, "shape", None)
        data_shape = getattr(env, "data_shape", None)
        print(
            "[olfnav_cuda_notebook] normalized environment for exact_converter: "
            f"shape={shape} ({type(shape).__name__}), "
            f"data_shape={data_shape} ({type(data_shape).__name__}), "
            f"changed={changed or 'none'}"
        )

    return env


def install_simulation_history_patch(verbose: bool = False) -> bool:
    """Install a safe patch for ``SimulationHistory.plot``.

    Returns ``True`` if the patch was installed or was already installed.
    """
    try:
        from olfactory_navigation.simulation import SimulationHistory
    except Exception as exc:  # pragma: no cover - depends on user env
        if verbose:
            print(f"[olfnav_cuda_notebook] SimulationHistory patch skipped: {exc}")
        return False

    if getattr(SimulationHistory.plot, "_s_v2_non_layered_patch", False):
        return True

    original_plot = SimulationHistory.plot

    def patched_plot(self, *args, **kwargs):
        _normalize_layer_metadata(self)
        return original_plot(self, *args, **kwargs)

    patched_plot._s_v2_non_layered_patch = True
    patched_plot._s_v2_original_plot = original_plot
    SimulationHistory.plot = patched_plot
    if verbose:
        print("[olfnav_cuda_notebook] installed non-layered SimulationHistory.plot patch")
    return True


# -----------------------------------------------------------------------------
# Metadata-driven environment loading
# -----------------------------------------------------------------------------

from pathlib import Path
import json


def _first_existing_metadata_file(env_dir: Path, metadata_file: str | None = None) -> Path | None:
    env_dir = Path(env_dir)
    if metadata_file is not None:
        candidate = env_dir / metadata_file
        if candidate.exists():
            return candidate
        candidate = Path(metadata_file)
        if candidate.exists():
            return candidate

    preferred = [
        "env_info.json",
        "environment_info.json",
        "environment_metadata.json",
        "metadata.json",
        "config.json",
    ]
    for name in preferred:
        candidate = env_dir / name
        if candidate.exists():
            return candidate

    json_files = sorted(env_dir.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0]

    # Prefer files containing canonical environment keys.
    for candidate in json_files:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        keys = set(data.keys())
        if {"source_radius", "margins"}.issubset(keys) or {"data_source_position", "source_position"} & keys:
            return candidate
    return None


def read_environment_metadata(env_dir: Path | str, metadata_file: str | None = None, *, verbose: bool = False) -> dict:
    """Read the JSON metadata stored in an olfactory environment directory.

    The function intentionally supports several common names because generated
    env folders across experiments may use different filenames.
    """
    env_dir = Path(env_dir)
    meta_path = _first_existing_metadata_file(env_dir, metadata_file)
    if meta_path is None:
        raise FileNotFoundError(f"No JSON metadata file found in {env_dir}")
    metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    metadata["_metadata_path"] = str(meta_path)
    if verbose:
        print(f"[olfnav_cuda_notebook] loaded environment metadata: {meta_path}")
    return metadata


def _resolve_data_file_from_metadata(env_dir: Path, metadata: dict, *, verbose: bool = False) -> Path:
    raw = (
        metadata.get("data_file_path")
        or metadata.get("data_file")
        or metadata.get("data_path")
        or metadata.get("data_filename")
    )
    candidates = []
    if raw:
        p = Path(str(raw))
        candidates.append(p if p.is_absolute() else env_dir / p)
        candidates.append(env_dir / p.name)
    candidates.append(env_dir / "data.npy")
    candidates.append(env_dir / "data.h5")

    for candidate in candidates:
        if candidate.exists():
            if verbose:
                print(f"[olfnav_cuda_notebook] metadata data_file -> {candidate}")
            return candidate

    # Return the first candidate for a clear error from Environment if none exists.
    return candidates[0]


def _as_list_or_none(x):
    if x is None:
        return None
    try:
        return list(x)
    except TypeError:
        return x


def _lower_margins(margins):
    if margins is None:
        return None
    lows = []
    for row in margins:
        try:
            lows.append(row[0])
        except Exception:
            lows.append(0)
    return lows


def _derive_data_source_position(metadata: dict):
    dsp = metadata.get("data_source_position")
    if dsp is not None:
        return _as_list_or_none(dsp)
    dsp = metadata.get("original_data_source_position")
    if dsp is not None:
        return _as_list_or_none(dsp)

    source_position = metadata.get("source_position")
    margins = metadata.get("margins")
    lows = _lower_margins(margins)
    if source_position is not None and lows is not None:
        try:
            return [int(s) - int(m) for s, m in zip(source_position, lows)]
        except Exception:
            try:
                return [float(s) - float(m) for s, m in zip(source_position, lows)]
            except Exception:
                pass
    return None


def environment_kwargs_from_metadata(env_dir: Path | str, metadata: dict, *, verbose: bool = False) -> dict:
    """Convert env metadata into ``Environment(...)`` constructor kwargs."""
    env_dir = Path(env_dir)
    data_file = _resolve_data_file_from_metadata(env_dir, metadata, verbose=verbose)
    data_source_position = _derive_data_source_position(metadata)

    if data_source_position is None:
        raise KeyError(
            "metadata does not contain data_source_position/original_data_source_position "
            "and it cannot be derived from source_position - margins"
        )

    kwargs = {
        "data_file": str(data_file),
        "data_source_position": data_source_position,
        "source_radius": metadata.get("source_radius", 2),
        "margins": metadata.get("margins", [[0, 0], [0, 0]]),
        "boundary_condition": metadata.get("boundary_condition", "wrap_vertical"),
        "start_zone": metadata.get("start_zone", metadata.get("start_type", "odor_present")),
        "odor_present_threshold": metadata.get("odor_present_threshold", 1e-4),
    }

    # Only pass values accepted by the old constructor path used in the current notebooks.
    if verbose:
        print("[olfnav_cuda_notebook] Environment fallback kwargs from metadata:")
        for k, v in kwargs.items():
            print(f"  {k}={v}")
    return kwargs


def construct_environment_from_metadata(env_dir: Path | str, EnvironmentClass, *, metadata_file: str | None = None, verbose: bool = False):
    """Construct ``Environment`` using only metadata from the env folder."""
    env_dir = Path(env_dir)
    metadata = read_environment_metadata(env_dir, metadata_file=metadata_file, verbose=verbose)
    kwargs = environment_kwargs_from_metadata(env_dir, metadata, verbose=verbose)
    env = EnvironmentClass(**kwargs)
    normalize_environment_for_exact_converter(env, verbose=verbose)
    return env


def load_environment_from_metadata(
    env_dir: Path | str,
    EnvironmentClass,
    *,
    metadata_file: str | None = None,
    prefer_environment_load: bool = True,
    verbose: bool = False,
):
    """Load an olfactory environment with a metadata-driven fallback.

    Preferred path:
        ``Environment.load(env_dir)``

    Fallback path:
        Read JSON metadata from ``env_dir`` and build ``Environment(...)`` using
        the stored data file, source position, margins, boundary condition,
        start type and threshold. This avoids hard-coded source coordinates in
        notebooks when switching to a new generated environment.
    """
    env_dir = Path(env_dir)
    if prefer_environment_load:
        try:
            env = EnvironmentClass.load(str(env_dir))
            normalize_environment_for_exact_converter(env, verbose=verbose)
            if verbose:
                print(f"[olfnav_cuda_notebook] Environment.load succeeded: {env_dir}")
            return env
        except Exception as exc:
            if verbose:
                print("[olfnav_cuda_notebook] Environment.load failed; using metadata fallback:", repr(exc))

    return construct_environment_from_metadata(
        env_dir,
        EnvironmentClass,
        metadata_file=metadata_file,
        verbose=verbose,
    )
