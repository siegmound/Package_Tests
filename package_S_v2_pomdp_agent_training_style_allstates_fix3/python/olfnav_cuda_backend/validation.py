from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .sparse_ell import SparseELL, ell_to_dense


@dataclass(frozen=True)
class CompareResult:
    ok: bool
    max_abs: float
    max_rel: float
    shape_a: tuple
    shape_b: tuple


def compare_arrays(a, b, *, atol=1e-10, rtol=1e-10) -> CompareResult:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        return CompareResult(False, float("inf"), float("inf"), aa.shape, bb.shape)
    diff = np.abs(aa - bb)
    max_abs = float(diff.max(initial=0.0))
    denom = np.maximum(np.abs(aa), np.abs(bb))
    rel = diff / np.maximum(denom, 1e-300)
    max_rel = float(rel.max(initial=0.0))
    return CompareResult(bool(np.allclose(aa, bb, atol=atol, rtol=rtol)), max_abs, max_rel, aa.shape, bb.shape)


def validate_sparse_equivalence(T_dense: np.ndarray, ell: SparseELL, *, atol=1e-12) -> CompareResult:
    return compare_arrays(np.asarray(T_dense, dtype=np.float64), ell_to_dense(ell), atol=atol, rtol=0.0)
