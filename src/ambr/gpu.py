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


def host_value(value):
    """Copy a CuPy array to host for host-side column writes.

    Returns a NumPy array, or a NumPy scalar of the same dtype for a 0-d
    array (``np.float32`` stays ``np.float32``), which is what the same code
    produces under ``cpu()``. Any other value is returned unchanged. Without
    this, host write paths that accept lists and NumPy arrays treat a device
    array as one scalar value.
    """
    if GPU_AVAILABLE and isinstance(value, _cp.ndarray):
        host = _cp.asnumpy(value)
        return host[()] if host.ndim == 0 else host
    return value


def synchronize() -> None:
    """Block until all queued GPU work has finished (for honest timing)."""
    if GPU_AVAILABLE:
        _cp.cuda.Stream.null.synchronize()


class DeviceRNG:
    """CuPy RNG with NumPy-compatible :meth:`choice` for model ``step()`` code."""

    def __init__(self, rng):
        self._rng = rng

    def choice(self, a, size=None, replace=True, p=None):
        xp = _cp
        pool = xp.asarray(a)
        n = int(pool.size)
        if n == 0:
            raise ValueError("choice requires a non-empty array")
        if p is not None:
            raise NotImplementedError("weighted choice is not supported on GPU yet")
        if size is None:
            idx = self._rng.integers(0, n)
            return pool[idx]
        size = int(size)
        if not replace and size > n:
            raise ValueError(f"cannot sample {size} unique items from {n}")
        if not replace:
            idx = self._rng.permutation(n)[:size]
        else:
            idx = self._rng.integers(0, n, size=size)
        return pool[idx]

    def __getattr__(self, name):
        return getattr(self._rng, name)

    @staticmethod
    def _state_buffer(bit_generator):
        # CuPy's state() returns a pointer into its cuRAND byte buffer.
        # Guard the layout so an unsupported generator leaves replay inconclusive.
        buffer = getattr(bit_generator, "_state", None)
        if (not isinstance(buffer, _cp.ndarray) or buffer.dtype != _np.int8
                or buffer.ndim != 1 or bit_generator.state() != buffer.data.ptr):
            raise TypeError("unsupported CuPy RNG state")
        return buffer

    def _snapshot_state(self):
        rng = self._rng
        bit_generator = rng.bit_generator
        with bit_generator.lock:
            buffer = self._state_buffer(bit_generator)
            cache = rng._binomial_state
            _cp.cuda.runtime.deviceSynchronize()
            return rng, bit_generator, buffer.get(), None if cache is None else cache.get()

    def _restore_state(self, snapshot):
        rng, bit_generator, state, cache = snapshot
        with bit_generator.lock:
            buffer = self._state_buffer(bit_generator)
            _cp.cuda.runtime.deviceSynchronize()
            buffer.set(state)
            rng._binomial_state = None if cache is None else _cp.asarray(cache)
            rng.bit_generator = bit_generator
            self._rng = rng


def make_device_rng(seed=None) -> DeviceRNG:
    """Create a GPU random generator with NumPy-shaped helpers."""
    require_gpu()
    return DeviceRNG(_cp.random.default_rng(seed))


def scatter_add(base, indices, values):
    """Scatter-add *values* into *base* at *indices* (1-D NumPy or CuPy arrays).

    NumPy arrays use :func:`~ambr.performance.apply_scatter_add` (Numba when
    installed); CuPy arrays use :func:`cupy.add.at`. Returns the array that
    holds the result (may be a new buffer after a NumPy dtype upcast).
    """
    if GPU_AVAILABLE and isinstance(base, _cp.ndarray):
        _cp.add.at(base, _cp.asarray(indices, dtype=_cp.int64), values)
        return base
    from .performance import apply_scatter_add

    return apply_scatter_add(_np.asarray(base), _np.asarray(indices), _np.asarray(values))


def require_gpu() -> None:
    """Raise a clear error if CuPy/CUDA is not available."""
    if not GPU_AVAILABLE:
        raise RuntimeError(
            "GPU requested but CuPy/CUDA is not available. "
            "Install a CuPy wheel matching your CUDA toolkit, e.g.\n"
            "  pip install cupy-cuda12x\n"
            "Then verify with: import ambr; ambr.print_status()"
        )
