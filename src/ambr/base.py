from typing import Any, Dict, Optional, List, Type
import polars as pl
import numpy as np
import random


def _coerce_param(typ, value):
    """Coerce a raw parameter to ``typ``, with string-aware bool handling.

    ``bool("false")`` would otherwise be ``True``; here the common string
    spellings of false ("0"/"false"/"no"/"off") map to ``False``.
    """
    if value is None:
        return None
    if typ is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on", "t", "y")
        return bool(value)
    return typ(value)


class AttrDict(dict):
    """
    Dictionary that allows attribute-style access to its keys.

    This provides AgentPy compatibility where parameters are accessed as:
        self.p.param_name instead of self.p['param_name']

    Typed accessors (``get_int``/``get_float``/``get_bool``) coerce on read so
    callers don't litter ``int(self.p.get(...))`` everywhere; a class-level
    ``params`` schema (see :class:`~ambr.model.Model`) pre-coerces at init.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def get_int(self, key, default=None):
        """Return ``self[key]`` coerced to ``int`` (``default`` if absent/None)."""
        v = self.get(key, default)
        return default if v is None else int(v)

    def get_float(self, key, default=None):
        """Return ``self[key]`` coerced to ``float`` (``default`` if absent/None)."""
        v = self.get(key, default)
        return default if v is None else float(v)

    def get_bool(self, key, default=None):
        """Return ``self[key]`` as ``bool`` (string-aware: 'false' -> False)."""
        v = self.get(key, default)
        return default if v is None else _coerce_param(bool, v)


class NPRandomCompat:
    """Adapter exposing the legacy ``model.nprandom`` surface over ``model.rng``.

    Delegates everything to the wrapped ``numpy.random.Generator`` and adds the
    old-style ``randint`` (exclusive high) for AgentPy-shaped code. ``nprandom``
    is retained for compatibility; new code should use ``model.rng`` directly.
    """

    def __init__(self, rng):
        self._rng = rng

    def __getattr__(self, name):
        return getattr(self._rng, name)

    def randint(self, low, high=None, size=None, dtype=int):
        return self._rng.integers(low, high, size=size, dtype=dtype, endpoint=False)


class BaseModel:
    """Base class for all simulation models, using DataFrames for data storage."""
    def __init__(self, parameters: Dict[str, Any]):
        self.p = AttrDict(parameters)  # Wrap parameters for attribute-style access
        # Optional class-level typed schema: ``params = {'n': (int, 200), ...}``.
        # Pre-coerces declared parameters so ``self.p.n`` is already typed and a
        # missing one falls back to the declared default.
        schema = getattr(type(self), 'params', None)
        if schema:
            for name, spec in schema.items():
                typ, default = spec if isinstance(spec, tuple) else (spec, None)
                self.p[name] = _coerce_param(typ, parameters.get(name, default))
        self.t = 0
        self.agents_df = pl.DataFrame({'id': [], 't': []})
        self.model_df = pl.DataFrame({'t': [0], **{k: [v] for k, v in parameters.items()}})
        # One coherent RNG story: `random` (stdlib) and `rng` (the canonical
        # numpy Generator). `nprandom` is the legacy alias over `rng`.
        seed = parameters.get('seed', None)
        self.random = random.Random(seed)
        self.rng = np.random.default_rng(seed)
        self.nprandom = NPRandomCompat(self.rng)

class BaseAgent:
    """Base class for all agents in the simulation, using DataFrames for data storage."""
    def __init__(self, model: 'BaseModel', agent_id: int):
        self.model = model
        self.id = agent_id
        self.p = model.p if model is not None else AttrDict({})