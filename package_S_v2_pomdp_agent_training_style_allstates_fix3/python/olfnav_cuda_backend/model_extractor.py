from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class PomdpModel:
    nS: int
    nA: int
    nO: int
    gamma: float
    T: np.ndarray  # [nA, nS, nS], float64
    O: np.ndarray  # [nO, nA, nS], float64
    R: np.ndarray  # [nA, nS], float64
    metadata: dict

    @property
    def model_hash(self) -> str:
        h = hashlib.sha256()
        for arr in (self.T, self.O, self.R):
            c = np.ascontiguousarray(arr)
            h.update(str(c.shape).encode())
            h.update(str(c.dtype).encode())
            h.update(c.view(np.uint8))
        h.update(repr(float(self.gamma)).encode())
        return h.hexdigest()


def _as_f64_contig(x: Any, name: str) -> np.ndarray:
    # cupy arrays expose .get(); use it if present, but do not require cupy import.
    if hasattr(x, "get") and callable(getattr(x, "get")):
        try:
            x = x.get()
        except Exception:
            pass
    arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return np.ascontiguousarray(arr)


def _candidate_get(obj: Any, path: str) -> Optional[Any]:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            return None
    return cur


def _candidate_iter(obj: Any, paths: Iterable[str]):
    """Yield (path, value) for existing dotted attribute paths / dict keys."""
    seen_ids: set[int] = set()
    for path in paths:
        val = _candidate_get(obj, path)
        if val is None:
            continue
        vid = id(val)
        if vid in seen_ids:
            continue
        seen_ids.add(vid)
        yield path, val


def _shape_of(x: Any) -> str:
    try:
        return str(np.asarray(x).shape)
    except Exception:
        return f"<unavailable:{type(x).__name__}>"


def _normalize_T(T_raw: Any) -> np.ndarray:
    T = _as_f64_contig(T_raw, "T")
    if T.ndim != 3:
        raise ValueError(f"T must be 3D [nA,nS,nS] or [nS,nA,nS], got shape={T.shape}")
    # Accept [nA,nS,nS] or [nS,nA,nS]. Prefer the one where the two state axes match.
    if T.shape[1] == T.shape[2]:
        return np.ascontiguousarray(T)
    if T.shape[0] == T.shape[2]:
        return np.ascontiguousarray(np.transpose(T, (1, 0, 2)))
    raise ValueError(f"Cannot infer T layout from shape={T.shape}")


def _normalize_O(O_raw: Any, nA: int, nS: int) -> np.ndarray:
    O = _as_f64_contig(O_raw, "O")
    if O.ndim != 3:
        raise ValueError(f"O must be 3D, got shape={O.shape}")
    # Supported layouts converted to kernel layout [nO, nA, nS]:
    #   kernel/native kernel: [nO, nA, nS]
    #   olfactory_navigation: [nS, nA, nO]
    #   alternative: [nA, nS, nO]
    #   alternative: [nA, nO, nS]
    if O.shape[1] == nA and O.shape[2] == nS:
        return np.ascontiguousarray(O)
    if O.shape[0] == nS and O.shape[1] == nA:
        return np.ascontiguousarray(np.transpose(O, (2, 1, 0)))
    if O.shape[0] == nA and O.shape[1] == nS:
        return np.ascontiguousarray(np.transpose(O, (2, 0, 1)))
    if O.shape[0] == nA and O.shape[2] == nS:
        return np.ascontiguousarray(np.transpose(O, (1, 0, 2)))
    raise ValueError(f"Cannot infer O layout from shape={O.shape}, nA={nA}, nS={nS}")


def _normalize_R(R_raw: Any, nA: int, nS: int) -> np.ndarray:
    R = _as_f64_contig(R_raw, "R")
    if R.ndim != 2:
        raise ValueError(f"R must be 2D, got shape={R.shape}")
    if R.shape == (nA, nS):
        return np.ascontiguousarray(R)
    if R.shape == (nS, nA):
        return np.ascontiguousarray(R.T)
    raise ValueError(f"Cannot infer R layout from shape={R.shape}, nA={nA}, nS={nS}")




def _get_expected_rewards_table(model: Any) -> np.ndarray:
    """Return olfactory_navigation reward table in native [nS,nA] layout when available."""
    for name in ("expected_rewards_table", "expected_reward_table", "reward_table", "rewards", "reward"):
        if hasattr(model, name):
            val = getattr(model, name)
            if val is not None:
                arr = _as_f64_contig(val, f"model.{name}")
                if arr.ndim == 2:
                    return arr
    raise AttributeError("Could not find a 2D expected reward table on agent.model")


def _get_dense_transition_table_olfnav(model: Any) -> tuple[np.ndarray, str]:
    """Return transition table in olfactory_navigation native [nS,nA,nS] layout."""
    if hasattr(model, "transition_table") and getattr(model, "transition_table") is not None:
        T = _as_f64_contig(getattr(model, "transition_table"), "model.transition_table")
        if T.ndim == 3:
            return T, "model.transition_table"

    rs = getattr(model, "reachable_states", None)
    rp = getattr(model, "reachable_probabilities", None)
    if rs is None or rp is None:
        raise AttributeError("no dense transition_table and no reachable_states/reachable_probabilities")

    rs_arr = np.asarray(rs)
    rp_arr = _as_f64_contig(rp, "model.reachable_probabilities")
    if rs_arr.shape != rp_arr.shape or rs_arr.ndim != 3:
        raise ValueError(f"reachable arrays mismatch: states={rs_arr.shape}, probs={rp_arr.shape}")

    nS, nA, width = rs_arr.shape
    T = np.zeros((nS, nA, nS), dtype=np.float64)
    for s in range(nS):
        for a in range(nA):
            for k in range(width):
                sp = int(rs_arr[s, a, k])
                prob = float(rp_arr[s, a, k])
                if prob != 0.0:
                    T[s, a, sp] += prob
    return np.ascontiguousarray(T), "model.reachable_states+reachable_probabilities"


def _try_extract_olfnav_model(agent: Any, gamma: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict] | None:
    """Fast path for olfactory_navigation Model objects.

    This mirrors the exporter used in packages A-J:
      T_sas = model.transition_table or reconstruction from reachable_*  -> [S,A,S]
      O_sao = model.observation_table                                    -> [S,A,O]
      R_sa  = model.expected_rewards_table                                -> [S,A]
    and converts to kernel layout:
      T -> [A,S,S], O -> [O,A,S], R -> [A,S]
    """
    model = getattr(agent, "model", None)
    if model is None or not hasattr(model, "observation_table"):
        return None
    try:
        T_sas, T_path = _get_dense_transition_table_olfnav(model)
        O_sao = _as_f64_contig(getattr(model, "observation_table"), "model.observation_table")
        R_sa = _get_expected_rewards_table(model)

        if T_sas.ndim != 3:
            raise ValueError(f"olfnav T must be 3D, got {T_sas.shape}")
        nS, nA, nS2 = T_sas.shape
        if nS != nS2:
            raise ValueError(f"olfnav T must be [S,A,S], got {T_sas.shape}")
        if O_sao.ndim != 3 or O_sao.shape[0] != nS or O_sao.shape[1] != nA:
            raise ValueError(f"olfnav O must be [S,A,O], got {O_sao.shape}; expected first axes {(nS,nA)}")
        if R_sa.shape != (nS, nA):
            raise ValueError(f"olfnav R must be [S,A], got {R_sa.shape}; expected {(nS,nA)}")

        T_kernel = np.ascontiguousarray(np.transpose(T_sas, (1, 0, 2)))
        O_kernel = np.ascontiguousarray(np.transpose(O_sao, (2, 1, 0)))
        R_kernel = np.ascontiguousarray(np.transpose(R_sa, (1, 0)))

        gamma_raw = gamma
        gamma_path = "explicit:gamma" if gamma is not None else None
        if gamma_raw is None:
            for path, val in _candidate_iter(agent, GAMMA_CANDIDATES):
                gamma_raw = val
                gamma_path = path
                break
        if gamma_raw is None:
            gamma_raw = 0.95
            gamma_path = "fallback:0.95"

        meta = {
            "extractor_fast_path": "olfactory_navigation",
            "extractor_paths": {
                "T": T_path,
                "O": "model.observation_table",
                "R": "model.expected_rewards_table/expected_reward_table",
                "gamma": gamma_path,
            },
            "native_shapes": {
                "T_sas": list(T_sas.shape),
                "O_sao": list(O_sao.shape),
                "R_sa": list(R_sa.shape),
            },
        }
        return T_kernel, O_kernel, R_kernel, float(gamma_raw), meta
    except Exception as e:
        setattr(agent, "_phasek_olfnav_fast_path_error", f"{type(e).__name__}: {e}")
        return None

def _validate_model(T: np.ndarray, O: np.ndarray, R: np.ndarray, gamma: float, *, tol: float = 1e-7) -> None:
    nA, nS, nS2 = T.shape
    if nS != nS2:
        raise ValueError("T must be [nA,nS,nS]")
    if O.shape[1:] != (nA, nS):
        raise ValueError(f"O must be [nO,nA,nS], got {O.shape}")
    if R.shape != (nA, nS):
        raise ValueError(f"R must be [nA,nS], got {R.shape}")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0,1], got {gamma}")
    row_sums = T.sum(axis=2)
    bad = np.abs(row_sums - 1.0) > tol
    if np.any(bad):
        max_dev = float(np.max(np.abs(row_sums - 1.0)))
        if max_dev > 1e-3:
            raise ValueError(f"T row sums deviate from 1.0 too much: max_dev={max_dev}")


# IMPORTANT: in olfactory_navigation, attributes called "observations" can be just
# the vector/list of observation labels, e.g. shape=(3,), not the observation-probability tensor.
# Therefore the extractor now tries candidates and validates their shape before accepting them.
T_CANDIDATES = [
    "transition_probabilities", "transition_table", "transition_model", "transition_matrix", "T", "transitions",
    "model.transition_probabilities", "model.transition_table", "model.transition_model", "model.transition_matrix", "model.T", "model.transitions",
    "environment_converter.transition_probabilities", "converter.transition_probabilities",
    "pomdp.T", "problem.T",
]

O_CANDIDATES = [
    "observation_probabilities", "observation_table", "observation_model", "observation_matrix", "O",
    "model.observation_probabilities", "model.observation_table", "model.observation_model", "model.observation_matrix", "model.O",
    "model.observation_probability_table", "model.observation_probabilities_table",
    "environment_converter.observation_probabilities", "converter.observation_probabilities",
    "pomdp.O", "problem.O",
    # Last-resort names only: these are often label arrays, not tensors.
    "observations", "model.observations",
]

R_CANDIDATES = [
    "expected_rewards_table", "reward_table", "rewards", "reward", "R",
    "model.expected_rewards_table", "model.reward_table", "model.rewards", "model.reward", "model.R",
    "environment_converter.rewards", "converter.rewards", "pomdp.R", "problem.R",
]

GAMMA_CANDIDATES = [
    "gamma", "discount", "discount_factor", "model.gamma", "problem.gamma",
]


def _extract_normalized(agent: Any, paths: list[str], normalizer, name: str):
    errors: list[str] = []
    for path, val in _candidate_iter(agent, paths):
        try:
            arr = normalizer(val)
            return arr, path, errors
        except Exception as e:
            errors.append(f"{path}: shape={_shape_of(val)} -> {type(e).__name__}: {e}")
    raise AttributeError(f"Could not extract valid {name}. Tried candidates:\n" + "\n".join(errors))


def extract_model_from_agent(
    agent: Any,
    *,
    T: Any | None = None,
    O: Any | None = None,
    R: Any | None = None,
    gamma: float | None = None,
    strict: bool = True,
) -> PomdpModel:
    """Extract a canonical POMDP model from an FSVI agent or explicit arrays.

    This version is robust against olfactory_navigation objects that expose both:
    - observation label arrays, often `model.observations` with shape=(nO,)
    - real observation-probability tensors under another name.

    The extractor accepts a candidate only after shape/layout normalization succeeds.
    """
    fast_meta: dict = {}
    try:
        if T is None and O is None and R is None:
            fast = _try_extract_olfnav_model(agent, gamma=gamma)
        else:
            fast = None

        if fast is not None:
            Tn, On, Rn, gamma_f, fast_meta = fast
            nA, nS, _ = Tn.shape
            T_path = fast_meta["extractor_paths"]["T"]
            O_path = fast_meta["extractor_paths"]["O"]
            R_path = fast_meta["extractor_paths"]["R"]
            gamma_path = fast_meta["extractor_paths"]["gamma"]
            T_errors = O_errors = R_errors = []
        else:
            if T is not None:
                Tn = _normalize_T(T)
                T_path = "explicit:T"
                T_errors: list[str] = []
            else:
                Tn, T_path, T_errors = _extract_normalized(agent, T_CANDIDATES, _normalize_T, "T")
            nA, nS, _ = Tn.shape

            if O is not None:
                On = _normalize_O(O, nA=nA, nS=nS)
                O_path = "explicit:O"
                O_errors: list[str] = []
            else:
                On, O_path, O_errors = _extract_normalized(agent, O_CANDIDATES, lambda x: _normalize_O(x, nA=nA, nS=nS), "O")

            if R is not None:
                Rn = _normalize_R(R, nA=nA, nS=nS)
                R_path = "explicit:R"
                R_errors: list[str] = []
            else:
                Rn, R_path, R_errors = _extract_normalized(agent, R_CANDIDATES, lambda x: _normalize_R(x, nA=nA, nS=nS), "R")

            gamma_raw = gamma if gamma is not None else None
            gamma_path = "explicit:gamma" if gamma is not None else None
            if gamma_raw is None:
                for path, val in _candidate_iter(agent, GAMMA_CANDIDATES):
                    gamma_raw = val
                    gamma_path = path
                    break
            if gamma_raw is None:
                gamma_raw = 0.95
                gamma_path = "fallback:0.95"

            gamma_f = float(gamma_raw)

        _validate_model(Tn, On, Rn, gamma_f)

    except Exception:
        if strict:
            raise
        raise

    metadata = {
        "source": "extract_model_from_agent",
        "nS": int(nS),
        "nA": int(nA),
        "nO": int(On.shape[0]),
        "T_shape": list(Tn.shape),
        "O_shape": list(On.shape),
        "R_shape": list(Rn.shape),
        "T_density_pct": float(100.0 * np.count_nonzero(Tn) / Tn.size),
        "extractor_paths": {"T": T_path, "O": O_path, "R": R_path, "gamma": gamma_path},
        "skipped_candidate_errors": {"T": T_errors[:20], "O": O_errors[:20], "R": R_errors[:20]},
        "olfnav_fast_path_error": getattr(agent, "_phasek_olfnav_fast_path_error", None),
    }
    metadata.update(fast_meta)
    model = PomdpModel(nS=int(nS), nA=int(nA), nO=int(On.shape[0]), gamma=gamma_f, T=Tn, O=On, R=Rn, metadata=metadata)
    model.metadata["model_hash"] = model.model_hash
    return model


def save_model_snapshot(model: PomdpModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        T=model.T,
        O=model.O,
        R=model.R,
        gamma=np.array([model.gamma], dtype=np.float64),
        metadata_json=np.array([json.dumps(model.metadata, sort_keys=True)], dtype=object),
    )


def load_model_snapshot(path: str | Path) -> PomdpModel:
    data = np.load(path, allow_pickle=True)
    T = _normalize_T(data["T"])
    nA, nS, _ = T.shape
    O = _normalize_O(data["O"], nA=nA, nS=nS)
    R = _normalize_R(data["R"], nA=nA, nS=nS)
    gamma = float(np.asarray(data["gamma"]).ravel()[0])
    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).ravel()[0]))
    model = PomdpModel(nS=nS, nA=nA, nO=int(O.shape[0]), gamma=gamma, T=T, O=O, R=R, metadata=metadata)
    model.metadata.setdefault("model_hash", model.model_hash)
    return model
