from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import os
import time


@dataclass
class CupyStatus:
    cupy_available: bool
    cupy_smoke_ok: bool
    device_id: int | None = None
    device_name: str | None = None
    runtime_version: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def import_and_check_cupy(device: int = 0, *, require: bool = True) -> tuple[Any | None, CupyStatus]:
    """Import CuPy, select the CUDA device, and run a tiny synchronized operation.

    This function is intentionally called before importing `olfactory_navigation` in
    benchmark scripts. Some olfactory versions decide GPU availability at import time;
    importing CuPy first makes failures explicit and avoids silent CPU fallbacks.
    """
    try:
        import cupy as cp  # type: ignore
        cp.cuda.Device(device).use()
        x = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float64)
        y = float(cp.sum(x).get())
        cp.cuda.Stream.null.synchronize()
        props = cp.cuda.runtime.getDeviceProperties(device)
        raw_name = props.get("name", b"")
        if isinstance(raw_name, bytes):
            name = raw_name.decode("utf-8", errors="replace")
        else:
            name = str(raw_name)
        status = CupyStatus(
            cupy_available=True,
            cupy_smoke_ok=(abs(y - 6.0) < 1e-12),
            device_id=int(device),
            device_name=name,
            runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
        )
        return cp, status
    except Exception as exc:  # pragma: no cover - executed on target workstation
        status = CupyStatus(
            cupy_available=False,
            cupy_smoke_ok=False,
            device_id=int(device),
            error=f"{type(exc).__name__}: {exc}",
        )
        if require:
            raise RuntimeError(
                "CuPy/GPU baseline requested, but CuPy could not be initialized. "
                f"Details: {status.error}"
            ) from exc
        return None, status


def synchronize_if_cupy(cp: Any | None) -> None:
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


def to_gpu_if_possible(obj: Any, *, name: str) -> Any:
    """Call `.to_gpu()` when available and return the converted object.

    The olfactory_navigation classes are not fully stable across local editable installs;
    this helper keeps the benchmark script defensive while still failing loudly when the
    top-level agent cannot be moved to GPU.
    """
    if hasattr(obj, "to_gpu"):
        return obj.to_gpu()
    raise RuntimeError(f"{name} has no .to_gpu() method; cannot build a native GPU baseline")


def maybe_to_gpu(obj: Any) -> Any:
    if hasattr(obj, "to_gpu"):
        return obj.to_gpu()
    return obj


def object_reports_gpu(obj: Any) -> bool | None:
    for attr in ("is_on_gpu", "on_gpu", "gpu"):
        if hasattr(obj, attr):
            try:
                return bool(getattr(obj, attr))
            except Exception:
                pass
    return None


def current_cupy_memory_mb(cp: Any | None) -> dict[str, float | None]:
    if cp is None:
        return {"cupy_mempool_used_mb": None, "cupy_mempool_total_mb": None}
    try:
        pool = cp.get_default_memory_pool()
        return {
            "cupy_mempool_used_mb": float(pool.used_bytes()) / (1024.0 * 1024.0),
            "cupy_mempool_total_mb": float(pool.total_bytes()) / (1024.0 * 1024.0),
        }
    except Exception:
        return {"cupy_mempool_used_mb": None, "cupy_mempool_total_mb": None}


class TimedBlock:
    def __init__(self, cp: Any | None = None):
        self.cp = cp
        self.elapsed_ms = 0.0
        self._t0 = 0.0

    def __enter__(self):
        synchronize_if_cupy(self.cp)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        synchronize_if_cupy(self.cp)
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        return False


def force_cuda_visible_device(device: int) -> None:
    """Set CUDA_VISIBLE_DEVICES only if the parent did not already set it.

    Campaign scripts normally set this in the child environment. For direct runs, this
    provides a simple default without overriding explicit user choices.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(device))
