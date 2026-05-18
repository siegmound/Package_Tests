from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ctypes as C
import time
import numpy as np

from .model_extractor import PomdpModel
from .sparse_ell import SparseELL, compile_sparse_ell


class BackendError(RuntimeError):
    pass


def _arr(a, dtype, ndim, name):
    out = np.ascontiguousarray(np.asarray(a, dtype=dtype))
    if out.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape={out.shape}")
    return out


@dataclass
class BackupInfo:
    elapsed_ms: float
    nB: int
    nG: int
    version_used: str
    requested_version: str


class CudaSparseBackupBackend:
    """ctypes wrapper around libpomdp_backup_cuda.so.

    K2 ABI:
      pomdp_backup_create(..., void** handle)
      pomdp_backup_run(handle, nB, nG, B, Gamma, BKP, actions, version_hint)
      pomdp_backup_get_last_version(handle, char*, int)  [optional but present in K2]
      pomdp_backup_destroy(handle)

    Version hints:
      generic / k1 : correctness path from K1
      v4 / v4_sparse_precompute       : custom sparse precompute path, no cuBLAS
      v7 / v7_cublas_all               : real cuBLAS global GEMM selection path
      v8 / v8_cublas_by_action         : real cuBLAS per-action GEMM selection path
      auto / auto_real                 : real dispatcher over the above paths
    """

    def __init__(
        self,
        lib_path: str | Path,
        model: PomdpModel,
        ell: SparseELL | None = None,
        *,
        version: str = "auto",
        max_nB: int = 0,
        max_nG: int = 0,
    ):
        self.lib_path = str(lib_path)
        self.model = model
        self.ell = ell or compile_sparse_ell(model.T)
        self.version = version
        self.max_nB = int(max_nB)
        self.max_nG = int(max_nG)
        self._lib = C.CDLL(self.lib_path)
        self._handle = C.c_void_p()
        self._has_get_last_version = False
        self._configure_symbols()
        self._create()

    @classmethod
    def from_model(
        cls,
        model: PomdpModel,
        ell: SparseELL | None = None,
        *,
        lib_path: str | Path,
        version: str = "auto",
        max_nB: int = 0,
        max_nG: int = 0,
    ):
        return cls(lib_path=lib_path, model=model, ell=ell, version=version, max_nB=max_nB, max_nG=max_nG)

    def _configure_symbols(self) -> None:
        dbl_p = C.POINTER(C.c_double)
        int_p = C.POINTER(C.c_int)
        void_pp = C.POINTER(C.c_void_p)
        self._lib.pomdp_backup_create.argtypes = [
            C.c_int, C.c_int, C.c_int, C.c_int, C.c_double,
            int_p, int_p, dbl_p, dbl_p, dbl_p,
            C.c_int, C.c_int,
            void_pp,
        ]
        self._lib.pomdp_backup_create.restype = C.c_int
        self._lib.pomdp_backup_run.argtypes = [
            C.c_void_p, C.c_int, C.c_int,
            dbl_p, dbl_p, dbl_p, int_p,
            C.c_char_p,
        ]
        self._lib.pomdp_backup_run.restype = C.c_int
        self._lib.pomdp_backup_destroy.argtypes = [C.c_void_p]
        self._lib.pomdp_backup_destroy.restype = None
        try:
            self._lib.pomdp_backup_get_last_version.argtypes = [C.c_void_p, C.c_char_p, C.c_int]
            self._lib.pomdp_backup_get_last_version.restype = C.c_int
            self._has_get_last_version = True
        except AttributeError:
            self._has_get_last_version = False

    def _create(self) -> None:
        m, e = self.model, self.ell
        rc = self._lib.pomdp_backup_create(
            C.c_int(m.nS), C.c_int(m.nA), C.c_int(m.nO), C.c_int(e.max_nnz), C.c_double(m.gamma),
            e.nnz.ctypes.data_as(C.POINTER(C.c_int)),
            e.idx.ctypes.data_as(C.POINTER(C.c_int)),
            e.val.ctypes.data_as(C.POINTER(C.c_double)),
            m.O.ctypes.data_as(C.POINTER(C.c_double)),
            m.R.ctypes.data_as(C.POINTER(C.c_double)),
            C.c_int(self.max_nB), C.c_int(self.max_nG),
            C.byref(self._handle),
        )
        if rc != 0 or not self._handle.value:
            raise BackendError(f"pomdp_backup_create failed rc={rc}")

    def _last_version(self, fallback: str) -> str:
        if not self._has_get_last_version:
            return fallback
        buf = C.create_string_buffer(128)
        rc = self._lib.pomdp_backup_get_last_version(self._handle, buf, C.c_int(len(buf)))
        if rc != 0:
            return fallback
        try:
            out = buf.value.decode("utf-8", errors="replace")
            return out or fallback
        except Exception:
            return fallback

    def backup(self, B: np.ndarray, Gamma: np.ndarray, *, version: str | None = None):
        m = self.model
        Bc = _arr(B, np.float64, 2, "B")
        Gc = _arr(Gamma, np.float64, 2, "Gamma")
        if Bc.shape[1] != m.nS:
            raise ValueError(f"B second dim must be nS={m.nS}, got {Bc.shape}")
        if Gc.shape[1] != m.nS:
            raise ValueError(f"Gamma second dim must be nS={m.nS}, got {Gc.shape}")
        nB, nG = Bc.shape[0], Gc.shape[0]
        BKP = np.empty((nB, m.nS), dtype=np.float64)
        actions = np.empty((nB,), dtype=np.int32)
        requested = version or self.version
        v = requested.encode()
        t0 = time.perf_counter()
        rc = self._lib.pomdp_backup_run(
            self._handle, C.c_int(nB), C.c_int(nG),
            Bc.ctypes.data_as(C.POINTER(C.c_double)),
            Gc.ctypes.data_as(C.POINTER(C.c_double)),
            BKP.ctypes.data_as(C.POINTER(C.c_double)),
            actions.ctypes.data_as(C.POINTER(C.c_int)),
            C.c_char_p(v),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if rc != 0:
            raise BackendError(f"pomdp_backup_run failed rc={rc}")
        actual = self._last_version(requested)
        return BKP, actions, BackupInfo(elapsed_ms=elapsed_ms, nB=nB, nG=nG, version_used=actual, requested_version=requested)

    def close(self):
        if getattr(self, "_handle", None) is not None and self._handle.value:
            self._lib.pomdp_backup_destroy(self._handle)
            self._handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
