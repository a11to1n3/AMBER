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


def _is_cupy_array(array) -> bool:
    return GPU_AVAILABLE and isinstance(array, _cp.ndarray)


def to_host(array):
    """Move a device array back to host as a NumPy array."""
    if _is_cupy_array(array):
        return _cp.asnumpy(array)
    return _np.asarray(array)


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


# --- Keyed (counter-based) draws --------------------------------------------
#
# A keyed draw is a pure function of (seed, step, event, key, partner, draw),
# computed with SplitMix64 exactly as the SIR infection kernel and
# tests/test_sir_counter_tape.py specify (spec 2: the seed is mixed before the
# step is folded in).  Nothing in it depends on the CUDA device, the launch
# configuration or the order in which threads run, so one seed yields the same
# values on every GPU and on the NumPy reference below.

_MASK64 = (1 << 64) - 1
_SPLITMIX_GOLDEN = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_TWO_POW_M53 = 1.0 / (1 << 53)
_FLOAT32_BELOW_ONE = _np.nextafter(_np.float32(1), _np.float32(0))

#: Event tag of draws made before the first :meth:`KeyedDeviceRNG.begin_step`.
EVT_INIT = 8
#: Event tag of :meth:`KeyedDeviceRNG.permutation`.
EVT_PRIORITY = 7


def _mix64(z: int) -> int:
    z = (z + _SPLITMIX_GOLDEN) & _MASK64
    z = ((z ^ (z >> 30)) * _SPLITMIX_M1) & _MASK64
    z = ((z ^ (z >> 27)) * _SPLITMIX_M2) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _keyed_prefix(seed: int, step: int, event: int) -> int:
    x = _mix64(int(seed) & _MASK64)
    x = _mix64(x ^ (int(step) & _MASK64))
    return _mix64(x ^ (int(event) & _MASK64))


def keyed_rng64(seed, step, event, key, partner=0, draw=0) -> int:
    """64-bit keyed draw for one key (pure-Python reference)."""
    x = _keyed_prefix(seed, step, event)
    for v in (key, partner, draw):
        x = _mix64(x ^ (int(v) & _MASK64))
    return _mix64(x)


def keyed_u01(seed, step, event, key, partner=0, draw=0) -> float:
    """Keyed U[0, 1) draw from the top 53 bits of :func:`keyed_rng64`."""
    return (keyed_rng64(seed, step, event, key, partner, draw) >> 11) * _TWO_POW_M53


def _mix64_np(z):
    z = z + _np.uint64(_SPLITMIX_GOLDEN)
    z = (z ^ (z >> _np.uint64(30))) * _np.uint64(_SPLITMIX_M1)
    z = (z ^ (z >> _np.uint64(27))) * _np.uint64(_SPLITMIX_M2)
    return z ^ (z >> _np.uint64(31))


def _keyed64_np(size: int, keys, prefix: int, partner: int, draw: int):
    """NumPy reference of the device kernel: uint64 draws for indices 0..size-1."""
    k = _np.arange(size, dtype=_np.uint64) if keys is None else keys.view(_np.uint64)
    with _np.errstate(over="ignore"):
        x = _mix64_np(_np.uint64(prefix) ^ k)
        x = _mix64_np(x ^ _np.uint64(partner & _MASK64))
        x = _mix64_np(x ^ _np.uint64(draw & _MASK64))
        return _mix64_np(x)


_KEYED_PREAMBLE = r"""
__device__ __forceinline__ unsigned long long ambr_mix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
__device__ __forceinline__ unsigned long long ambr_keyed64(
        unsigned long long key, unsigned long long prefix,
        unsigned long long partner, unsigned long long draw) {
    unsigned long long x = ambr_mix64(prefix ^ key);
    x = ambr_mix64(x ^ partner);
    x = ambr_mix64(x ^ draw);
    return ambr_mix64(x);
}
"""
_KEYED_INPUTS = "raw int64 keys, bool has_keys, uint64 prefix, uint64 partner, uint64 draw"
_KEYED_KEY = (
    "unsigned long long k = has_keys ? (unsigned long long)keys[i] "
    ": (unsigned long long)i; "
    "unsigned long long r = ambr_keyed64(k, prefix, partner, draw); "
)
_KEYED_U = "double u = (double)(r >> 11) * (1.0 / 9007199254740992.0); "
_keyed_kernels: dict = {}


def _keyed_kernel(kind: str, clamp_float32: bool):
    cache_key = (kind, clamp_float32)
    kernel = _keyed_kernels.get(cache_key)
    if kernel is not None:
        return kernel
    clamp = "if (out >= 1.0f) out = nextafterf(1.0f, 0.0f);" if clamp_float32 else ""
    if kind == "u01":
        kernel = _cp.ElementwiseKernel(
            _KEYED_INPUTS, "T out", _KEYED_KEY + _KEYED_U + "out = (T)u; " + clamp,
            "ambr_keyed_u01", preamble=_KEYED_PREAMBLE,
        )
    elif kind == "uniform":
        # __dmul_rn/__dadd_rn forbid fused multiply-add, so the device value
        # equals NumPy's low + span * u bit for bit.
        kernel = _cp.ElementwiseKernel(
            _KEYED_INPUTS + ", float64 low, float64 span", "T out",
            _KEYED_KEY + _KEYED_U + "out = (T)__dadd_rn(low, __dmul_rn(span, u)); " + clamp,
            "ambr_keyed_uniform", preamble=_KEYED_PREAMBLE,
        )
    elif kind == "int":
        kernel = _cp.ElementwiseKernel(
            _KEYED_INPUTS + ", int64 low, uint64 span", "int64 out",
            _KEYED_KEY + "out = low + (long long)(r % span);",
            "ambr_keyed_int", preamble=_KEYED_PREAMBLE,
        )
    else:  # pragma: no cover - internal misuse
        raise ValueError(kind)
    _keyed_kernels[cache_key] = kernel
    return kernel


def _size_tuple(size):
    if isinstance(size, (int, _np.integer)):
        return (int(size),)
    return tuple(int(s) for s in size)


class KeyedDeviceRNG(DeviceRNG):
    """Counter-based random generator whose draws do not depend on the GPU.

    Element ``i`` of a bulk draw is keyed by ``(seed, step, event, key_i,
    partner, stream)``, where ``key_i`` is the agent identity bound by
    :meth:`set_agent_keys` (or ``i`` when no identity is bound) and ``stream``
    advances for each bulk call, keyed scalar call, and run of unkeyed scalar
    draws. Unkeyed scalar draws within a run use successive positions in
    their reserved stream. Integer and choice draws reduce the 64-bit value
    modulo the range, as the counter tape does.

    With ``xp=numpy`` the same draws are computed on the host; tests use that
    as the reference for the CUDA kernels.
    """

    #: :meth:`set_agent_keys` accepts device arrays without a host copy.
    device_keys = True

    def __init__(self, seed, xp=None):
        self._rng = None
        self.global_seed = int(seed) & _MASK64
        self._xp = _cp if xp is None else xp
        self.step = 0
        self._event = EVT_INIT
        self._stream = 0
        self._scalar_stream = None
        self._cursor = 0
        self._keys = None

    def __getattr__(self, name):
        raise AttributeError(f"KeyedDeviceRNG has no attribute {name!r}")

    # --- step and key control ----------------------------------------------

    def begin_step(self, step: int, event: int = EVT_INIT) -> None:
        """Start the draws of logical step ``step`` under event tag ``event``."""
        self.step = int(step)
        self._event = int(event)
        self._stream = 0
        self._scalar_stream = None
        self._cursor = 0
        self._keys = None

    def set_agent_keys(self, agent_ids, event=None) -> None:
        """Bind one agent ID per draw position until the next binding or step.

        Draws raise ``ValueError`` if the number of bound IDs differs from the
        number of values drawn, including a single value for a scalar draw.
        Pass ``None`` to clear the binding and use positional keys.
        """
        self._keys = (
            None if agent_ids is None else self._xp.asarray(agent_ids, dtype=self._xp.int64).ravel()
        )
        self._scalar_stream = None
        self._cursor = 0
        if event is not None:
            self._event = int(event)

    # --- internals ---------------------------------------------------------

    def _on_device(self) -> bool:
        return GPU_AVAILABLE and self._xp is _cp

    def _prefix(self, event=None) -> int:
        return _keyed_prefix(
            self.global_seed, self.step, self._event if event is None else event
        )

    @staticmethod
    def _require_key_count(keys, size: int) -> None:
        if keys is not None and keys.size != size:
            raise ValueError(f"expected {size} agent keys for draw size {size}, got {keys.size}")

    def _next_scalar_counter(self) -> tuple[int, int]:
        if self._keys is not None:
            self._require_key_count(self._keys, 1)
            key, stream = int(self._keys[0]), self._stream
            self._advance(1)
            return key, stream
        if self._scalar_stream is None:
            self._scalar_stream = self._stream
            self._stream += 1
        key = self._cursor
        self._cursor += 1
        return key, self._scalar_stream

    def _launch(self, kind, shape, dtype, *, event=None, keys=None, partner=0, extra=(),
                clamp_float32=False):
        """Draw ``shape`` keyed values of ``kind`` from the current stream."""
        keys = self._keys if keys is None and event is None else keys
        prefix = self._prefix(event)
        draw = self._stream
        n = 1
        for s in shape:
            n *= s
        self._require_key_count(keys, n)
        if self._on_device():
            out = _cp.empty(shape, dtype=dtype)
            keys_arg = _cp.zeros(1, dtype=_cp.int64) if keys is None else keys
            _keyed_kernel(kind, clamp_float32)(
                keys_arg, keys is not None, _np.uint64(prefix), _np.uint64(partner),
                _np.uint64(draw), *extra, out,
            )
            return out
        r = _keyed64_np(n, keys, prefix, partner, draw)
        if kind == "int":
            low, span = extra
            out = low + (r % _np.uint64(span)).astype(_np.int64)
        else:
            u = (r >> _np.uint64(11)).astype(_np.float64) * _TWO_POW_M53
            if kind == "uniform":
                low, span = extra
                u = low + span * u
            out = u
        out = out.astype(dtype, copy=False)
        if clamp_float32:
            _np.minimum(out, _FLOAT32_BELOW_ONE, out=out)
        return out.reshape(shape)

    def _advance(self, n: int) -> None:
        self._cursor += n
        self._stream += 1
        self._scalar_stream = None

    # --- generator surface -------------------------------------------------

    def random(self, size=None, dtype=None):
        dtype = _np.dtype(_np.float64 if dtype is None else dtype)
        clamp_float32 = dtype.type is _np.float32
        if size is None:
            key, stream = self._next_scalar_counter()
            out = dtype.type(keyed_u01(self.global_seed, self.step, self._event, key, 0, stream))
            return min(out, _FLOAT32_BELOW_ONE) if clamp_float32 else out
        shape = _size_tuple(size)
        out = self._launch("u01", shape, dtype, clamp_float32=clamp_float32)
        self._advance(out.size)
        return out

    def uniform(self, low=0.0, high=1.0, size=None, dtype=None):
        dtype = _np.dtype(_np.float64 if dtype is None else dtype)
        span = float(high - low)
        clamp_float32 = dtype.type is _np.float32 and high == 1.0 and span > 0.0
        if size is None:
            u = self.random()
            out = dtype.type(float(low) + span * float(u))
            return min(out, _FLOAT32_BELOW_ONE) if clamp_float32 else out
        shape = _size_tuple(size)
        out = self._launch(
            "uniform", shape, dtype, extra=(float(low), span), clamp_float32=clamp_float32,
        )
        self._advance(out.size)
        return out

    def integers(self, low, high=None, size=None, dtype=None, endpoint=False):
        if high is None:
            low, high = 0, low
        low, high = int(low), int(high) + (1 if endpoint else 0)
        if high <= low:
            raise ValueError("high must be greater than low")
        span = high - low
        if size is None:
            key, stream = self._next_scalar_counter()
            r = keyed_rng64(self.global_seed, self.step, self._event, key, 0, stream)
            return low + r % span
        shape = _size_tuple(size)
        out = self._launch("int", shape, _np.int64, extra=(low, _np.uint64(span)))
        self._advance(out.size)
        return out if dtype is None else out.astype(dtype, copy=False)

    def choice(self, a, size=None, replace=True, p=None):
        """Sample from ``a`` using one key per draw or candidate.

        Sampling without replacement ranks the entire pool, so it needs one
        key per candidate when agent keys are bound.
        """
        if p is not None:
            raise NotImplementedError("weighted choice is not supported for keyed draws")
        pool = self._xp.asarray(a)
        n = int(pool.size)
        if n == 0:
            raise ValueError("choice requires a non-empty array")
        if size is None:
            return pool[self.integers(0, n)]
        size = int(size)
        if not replace:
            if size > n:
                raise ValueError(f"cannot sample {size} unique items from {n}")
            order = self._xp.argsort(self.random(n))
            return pool[order[:size]]
        return pool[self.integers(0, n, size=size)]

    def permutation(self, x):
        """Permutation keyed by position under :data:`EVT_PRIORITY`."""
        m = int(x) if isinstance(x, (int, _np.integer)) else int(len(x))
        u = self._launch("u01", (m,), _np.float64, event=EVT_PRIORITY)
        self._advance(0)
        order = self._xp.argsort(u)
        if isinstance(x, (int, _np.integer)):
            return order
        return self._xp.asarray(x)[order]

    # --- replay hooks used by Model ----------------------------------------

    def _snapshot_state(self):
        keys = self._keys
        if keys is not None:
            keys = to_host(keys) if self._on_device() else keys.copy()
        return (self.step, self._event, self._stream, self._scalar_stream, self._cursor, keys)

    def _restore_state(self, snapshot):
        step, event, stream, scalar_stream, cursor, keys = snapshot
        restored_keys = None if keys is None else self._xp.array(keys, copy=True)
        self.step, self._event, self._stream = step, event, stream
        self._scalar_stream, self._cursor, self._keys = scalar_stream, cursor, restored_keys


def make_keyed_device_rng(seed=None, xp=None) -> KeyedDeviceRNG:
    """Create a :class:`KeyedDeviceRNG`; ``seed=None`` draws a fresh 64-bit seed."""
    if xp is None:
        require_gpu()
    if seed is None:
        import secrets

        seed = secrets.randbits(64)
    return KeyedDeviceRNG(seed, xp=xp)


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
