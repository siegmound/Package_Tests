from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SparseELL:
    nnz: np.ndarray       # int32 [nA,nS]
    idx: np.ndarray       # int32 [nA,nS,max_nnz]
    val: np.ndarray       # float64 [nA,nS,max_nnz]
    max_nnz: int
    density_pct: float


def compile_sparse_ell(T: np.ndarray, *, zero_tol: float = 0.0) -> SparseELL:
    T = np.asarray(T, dtype=np.float64)
    if T.ndim != 3 or T.shape[1] != T.shape[2]:
        raise ValueError(f"T must be [nA,nS,nS], got {T.shape}")
    nA, nS, _ = T.shape
    masks = np.abs(T) > zero_tol
    counts = masks.sum(axis=2).astype(np.int32)
    max_nnz = int(counts.max(initial=0))
    if max_nnz < 1:
        raise ValueError("T has no non-zero transitions")
    idx = np.zeros((nA, nS, max_nnz), dtype=np.int32)
    val = np.zeros((nA, nS, max_nnz), dtype=np.float64)
    for a in range(nA):
        for s in range(nS):
            cols = np.nonzero(masks[a, s])[0].astype(np.int32)
            c = len(cols)
            idx[a, s, :c] = cols
            val[a, s, :c] = T[a, s, cols]
    density_pct = float(100.0 * int(counts.sum()) / T.size)
    return SparseELL(nnz=np.ascontiguousarray(counts), idx=np.ascontiguousarray(idx), val=np.ascontiguousarray(val), max_nnz=max_nnz, density_pct=density_pct)


def ell_to_dense(ell: SparseELL) -> np.ndarray:
    nA, nS = ell.nnz.shape
    T = np.zeros((nA, nS, nS), dtype=np.float64)
    for a in range(nA):
        for s in range(nS):
            c = int(ell.nnz[a, s])
            T[a, s, ell.idx[a, s, :c]] = ell.val[a, s, :c]
    return T
