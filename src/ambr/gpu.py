"""Optional GPU backend support for AMBER (CuPy).

AMBER's columnar / vectorized execution path can run on the GPU when CuPy is
installed. The idea is the same as the resident tensor lane (see
``tensor_lane``), but the resident buffer lives in **device** memory: agent
state is held in CuPy arrays, the per-step kernels run on the GPU, and we sync
back to the Polars system-of-record only at relational / logging boundaries.
Host<->device transfer is the bottleneck (PCIe), so it is minimised --
state stays resident across steps.

CuPy is NumPy-compatible, so a vectorized rule written against ``numpy`` ports
to the GPU by swapping the array module (:func:`get_array_module`). This module
never hard-requires CuPy: import it unconditionally and branch on
:data:`GPU_AVAILABLE`.
"""

import numpy as _np

try:
    import cupy as _cp

    _cp.cuda.runtime.getDeviceCount()  # raises if no usable device/driver
    GPU_AVAILABLE = True
except Exception:  # pragma: no cover - depends on host CUDA
    _cp = None
    GPU_AVAILABLE = False


def device_name() -> str:
    """Human-readable name of the active CUDA device, or 'cpu'."""
    if not GPU_AVAILABLE:
        return "cpu"
    return _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()


def get_array_module(prefer_gpu: bool = True):
    """Return the array module to write backend-agnostic kernels against.

    ``xp = get_array_module()`` yields ``cupy`` when a GPU is available, else
    ``numpy`` -- so the same kernel code runs on either backend.
    """
    return _cp if (prefer_gpu and GPU_AVAILABLE) else _np


def to_device(array):
    """Move a host array onto the GPU (no-op when no GPU is available)."""
    return _cp.asarray(array) if GPU_AVAILABLE else _np.asarray(array)


def to_host(array):
    """Move a device array back to host as a NumPy array."""
    if GPU_AVAILABLE and isinstance(array, _cp.ndarray):
        return _cp.asnumpy(array)
    return _np.asarray(array)


def synchronize() -> None:
    """Block until all queued GPU work has finished (for honest timing)."""
    if GPU_AVAILABLE:
        _cp.cuda.Stream.null.synchronize()


def require_gpu() -> None:
    """Raise a clear error if CuPy/CUDA is not available."""
    if not GPU_AVAILABLE:
        raise RuntimeError(
            "GPU requested but CuPy/CUDA is not available. "
            "Install a CuPy wheel matching your CUDA toolkit, e.g.\n"
            "  pip install cupy-cuda12x\n"
            "Then verify with: import ambr; ambr.print_status()"
        )
