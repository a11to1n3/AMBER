"""Optional GPU backend support for AMBER (CuPy / CUDA only).

AMBER's array kernels can run on the GPU when **CuPy + NVIDIA CUDA** are
available. Apple Metal/MPS is **not** used — on Mac, prefer the vectorized
CPU path with Numba (see :mod:`ambr.lanes` / :mod:`ambr.performance`).

Design notes
------------
* Device-resident buffers (like the tensor lane, but on GPU): host↔device
  transfer is the bottleneck (PCIe), so state stays on device across steps.
* CuPy is NumPy-compatible: write kernels against ``xp = get_array_module()``.
* This module never hard-requires CuPy; branch on :data:`GPU_AVAILABLE`.
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
