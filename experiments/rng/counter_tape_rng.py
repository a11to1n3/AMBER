"""RNG adapters that implement NumPy/CuPy-shaped APIs over the counter tape.

Production kernels call ``rng.choice``, ``rng.uniform``, ``rng.random``, and
``rng.integers``.  For attestation we inject this adapter as ``device_rng`` so
the *same* ``_run_gpu_fast`` / fused kernels run, but every draw is keyed by
``(global_seed, step, event_type, agent_id, partner_id, draw_index)``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rng.counter_rng import (  # noqa: E402
    EVT_DISPLACE_X,
    EVT_DISPLACE_Y,
    EVT_INIT,
    EVT_PRIORITY,
    EVT_PROPOSAL,
    EVT_RECIPIENT,
    int_range,
    rng64,
    u01,
)

try:
    import cupy as cp

    _HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CUPY = False


class CounterTapeRNG:
    """Host-side counter tape with a DeviceRNG-compatible surface.

    Call :meth:`begin_step` before each logical step.  Optional
    :meth:`set_agent_keys` binds the next bulk draw to explicit agent ids
    (required so compacted donor arrays still key by stable identity).
    """

    def __init__(self, global_seed: int, *, prefer_cupy: bool = False):
        self.global_seed = int(global_seed)
        self.step = 0
        self._agent_keys: Optional[np.ndarray] = None
        self._draw_cursor = 0
        self._stream = 0
        self._event = EVT_INIT
        self._prefer_cupy = bool(prefer_cupy and _HAS_CUPY)
        self._xp = cp if self._prefer_cupy else np

    # --- step / key control -------------------------------------------------

    def begin_step(self, step: int, event: int = EVT_INIT) -> None:
        self.step = int(step)
        self._event = int(event)
        self._draw_cursor = 0
        self._stream = 0  # separates consecutive bulk draws (dx vs dy, …)
        self._agent_keys = None

    def set_agent_keys(self, agent_ids: Sequence[int] | np.ndarray, event: int | None = None) -> None:
        self._agent_keys = np.asarray(agent_ids, dtype=np.int64).ravel()
        self._draw_cursor = 0
        if event is not None:
            self._event = int(event)

    def _key_for(self, i: int) -> int:
        if self._agent_keys is not None and i < self._agent_keys.size:
            return int(self._agent_keys[i])
        return int(i)

    # --- bulk generators ----------------------------------------------------

    def random(self, size=None, dtype=None):
        stream = self._stream
        if size is None:
            v = u01(
                self.global_seed, self.step, self._event,
                self._key_for(self._draw_cursor), 0, stream,
            )
            self._draw_cursor += 1
            # keep scalar host-side; bulk path converts
            return self._cast_scalar(v, dtype)
        n = int(np.prod(size)) if not isinstance(size, int) else int(size)
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = u01(
                self.global_seed, self.step, self._event,
                self._key_for(i), 0, stream,
            )
        self._draw_cursor += n
        self._stream += 1  # next bulk call (e.g. dy after dx) gets a new stream
        arr = out.reshape(size) if not isinstance(size, int) else out
        return self._to_backend(arr, dtype)

    def uniform(self, low=0.0, high=1.0, size=None, dtype=None):
        u = self.random(size=size, dtype=None)
        # always scale on host to avoid cupy->__array__ bans
        if size is None:
            return self._cast_scalar(float(low + (high - low) * float(u)), dtype)
        if _HAS_CUPY and hasattr(u, "get"):
            u_host = u.get()
        else:
            u_host = np.asarray(u, dtype=np.float64)
        scaled = low + (high - low) * np.asarray(u_host, dtype=np.float64)
        return self._to_backend(scaled, dtype)

    def integers(self, low, high=None, size=None, dtype=None, endpoint=False):
        if high is None:
            low, high = 0, low
        if endpoint:
            high = high + 1
        stream = self._stream
        if size is None:
            # int_range has no draw_index; fold stream into partner_id
            v = int_range(
                int(low), int(high), self.global_seed, self.step, self._event,
                self._key_for(self._draw_cursor), stream, 0,
            )
            self._draw_cursor += 1
            return int(v)
        n = int(np.prod(size)) if not isinstance(size, int) else int(size)
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            out[i] = int_range(
                int(low), int(high), self.global_seed, self.step, self._event,
                self._key_for(i), stream, 0,
            )
        self._draw_cursor += n
        self._stream += 1
        arr = out.reshape(size) if not isinstance(size, int) else out
        if self._prefer_cupy:
            return cp.asarray(arr)
        return arr

    def choice(self, a, size=None, replace=True, p=None):
        if p is not None:
            raise NotImplementedError("weighted choice not supported on counter tape")
        xp = self._xp
        pool = xp.asarray(a)
        n = int(pool.size)
        if n == 0:
            raise ValueError("choice requires a non-empty array")
        stream = self._stream
        if size is None:
            idx = int_range(
                0, n, self.global_seed, self.step, self._event,
                self._key_for(self._draw_cursor), stream, 0,
            )
            self._draw_cursor += 1
            return pool[idx]
        size = int(size)
        if not replace and size > n:
            raise ValueError(f"cannot sample {size} unique items from {n}")
        if not replace:
            keys = np.array(
                [
                    u01(self.global_seed, self.step, self._event, self._key_for(i), 0, stream)
                    for i in range(n)
                ],
                dtype=np.float64,
            )
            order = np.argsort(keys)[:size]
            self._draw_cursor += n
            self._stream += 1
            if self._prefer_cupy:
                return pool[cp.asarray(order)]
            return pool[order]
        idxs = np.empty(size, dtype=np.int64)
        for i in range(size):
            idxs[i] = int_range(
                0, n, self.global_seed, self.step, self._event,
                self._key_for(i), stream, 0,
            )
        self._draw_cursor += size
        self._stream += 1
        if self._prefer_cupy:
            return pool[cp.asarray(idxs)]
        return pool[idxs]

    def permutation(self, x):
        """NumPy-compatible permutation for Schelling-style shuffles."""
        if isinstance(x, (int, np.integer)):
            m = int(x)
            keys = np.array(
                [u01(self.global_seed, self.step, EVT_PRIORITY, i) for i in range(m)],
                dtype=np.float64,
            )
            order = np.argsort(keys)
            return self._xp.asarray(order) if self._prefer_cupy else order
        arr = np.asarray(x)
        m = arr.shape[0]
        keys = np.array(
            [u01(self.global_seed, self.step, EVT_PRIORITY, i) for i in range(m)],
            dtype=np.float64,
        )
        order = np.argsort(keys)
        return arr[order]

    # wealth helpers ---------------------------------------------------------

    def recipients_for_donors(self, donor_agent_ids: Sequence[int], n_agents: int) -> np.ndarray:
        """Identity-keyed recipient draws for each donor agent id."""
        donors = np.asarray(donor_agent_ids, dtype=np.int64)
        out = np.empty(donors.size, dtype=np.int64)
        for i, d in enumerate(donors):
            out[i] = int_range(0, n_agents, self.global_seed, self.step, EVT_RECIPIENT, int(d), 0, 0)
        return out

    def displace(self, n: int, speed: float) -> tuple[np.ndarray, np.ndarray]:
        dx = np.empty(n, dtype=np.float64)
        dy = np.empty(n, dtype=np.float64)
        for i in range(n):
            dx[i] = (2.0 * u01(self.global_seed, self.step, EVT_DISPLACE_X, i) - 1.0) * speed
            dy[i] = (2.0 * u01(self.global_seed, self.step, EVT_DISPLACE_Y, i) - 1.0) * speed
        return dx, dy

    # backend helpers --------------------------------------------------------

    def _to_backend(self, arr: np.ndarray, dtype) -> Any:
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        if self._prefer_cupy:
            return cp.asarray(arr)
        return arr

    def _cast_scalar(self, v: float, dtype) -> Any:
        if dtype is None:
            return float(v)
        return np.dtype(dtype).type(v)

    def __getattr__(self, name: str):
        # Fall through unknown attrs to numpy Generator for rare APIs.
        raise AttributeError(f"CounterTapeRNG has no attribute {name!r}")
