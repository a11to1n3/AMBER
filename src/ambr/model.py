"""Core simulation model: population store, write flush, and run loop.

Write architecture (single source of truth = Polars ``agents_df``)
------------------------------------------------------------------
* **OOP path** — ``Agent.__setattr__`` queues into ``_pending_writes``;
  :meth:`_flush_pending_writes` applies them (uses :mod:`ambr._id_index`).
* **Vectorized path** — view API in :mod:`ambr.sequences` calls
  :meth:`_set_frame` directly (scatter / set / column assign).
* **Tensor path** — :mod:`ambr.tensor_lane` borrow/commit also uses
  :meth:`_set_frame`.

Id-position caches (``_id_pos_cache``, ``_ids_arange_cache``) are owned here
but filled by :mod:`ambr._id_index`; :meth:`_bump_id_version` invalidates them.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, NamedTuple, Type, Optional, Tuple, TYPE_CHECKING
from collections import Counter, OrderedDict, defaultdict, deque
from contextlib import contextmanager
import abc
import copy
import datetime as _datetime
import decimal
import enum
import fractions
import logging
import math
import pathlib
import random
import types
from secrets import randbits
import polars as pl
import warnings
import numpy as np
import time
from datetime import timedelta
from .base import AttrDict, BaseModel
from .population import Population
from ._deprecation import warn_deprecated
from ._id_index import resolve_positions
from .gpu import DeviceRNG
from .contract import (
    ContractCertificate,
    ContractMonitor,
    ContractViolationError,
)
from .results import RunResults
from .provenance import build_run_info
from .execution import (
    active_rng,
    active_xp,
    begin_execution,
    begin_fast_execution,
    end_execution,
    prepare_end_hook,
    reconcile_host_frame,
    refresh_tracked_device_columns,
    resolve_config,
    sync_agent_writes,
    sync_all_device_columns,
    sync_device_column_to_frame,
)

if TYPE_CHECKING:
    from .agent import Agent


class _RngSnapshot(NamedTuple):
    """Live stream objects, their states, and the attributes that point at them."""

    streams: Tuple[Tuple[str, int, Any, Any], ...]
    bindings: Tuple[Tuple[str, str, int], ...]


class _ReplayOutcome(NamedTuple):
    """Witness value plus an optional reason when the swap was inconclusive."""

    value: Optional[bool]
    reason: Optional[str] = None


class _ReplayAliasMismatch(Exception):
    """The two state graphs do not have a one-to-one node mapping."""


class _ReplayUnsupported(Exception):
    """A reachable value cannot be restored, so the swap cannot be isolated."""


# Type flags (CPython ``Py_TPFLAGS_*``). A static type, or a heap type marked
# immutable, rejects attribute assignment, so it holds no state a step can
# change.
_TPFLAGS_IMMUTABLETYPE = 1 << 8
_TPFLAGS_HEAPTYPE = 1 << 9

# Scalar types whose == is exact for snapshot comparison (no NaN).
_REPLAY_EXACT_EQUAL = frozenset({type(None), bool, int, str, bytes})


def _replay_failure_reason(exc: BaseException) -> str:
    text = f"swapped execution raised {type(exc).__name__}: {exc}"
    if len(text) > 240:
        return text[:237] + "..."
    return text


try:  # Python 3.9+
    from zoneinfo import ZoneInfo as _ZoneInfo
except ImportError:  # pragma: no cover
    _ZoneInfo = None

# Time zones whose comparison runs no user code.
_REPLAY_TZ_TYPES = tuple(t for t in (_datetime.timezone, _ZoneInfo) if t is not None)


def _replay_tz_ok(value: Any) -> bool:
    tz = value.tzinfo
    return tz is None or type(tz) in _REPLAY_TZ_TYPES


# Immutable standard-library value types, matched by exact type. Each maps to
# a function giving an exact, comparable key that runs no user code. They
# need no restore: the snapshot rebinds the same object.
_REPLAY_VALUE_KEYS = {
    _datetime.date: lambda v: v.__reduce__()[1],
    _datetime.datetime: lambda v: v.__reduce__()[1],
    _datetime.time: lambda v: v.__reduce__()[1],
    _datetime.timedelta: lambda v: (v.days, v.seconds, v.microseconds),
    _datetime.timezone: lambda v: v.__reduce__()[1],
    decimal.Decimal: lambda v: v.as_tuple(),
    fractions.Fraction: lambda v: (v.numerator, v.denominator),
    range: lambda v: (v.start, v.stop, v.step),
    pathlib.PurePosixPath: str,
    pathlib.PureWindowsPath: str,
    pathlib.PosixPath: str,
    pathlib.WindowsPath: str,
}

# Objects kept by identity only. Logging objects reach process-wide logging
# state, so their attributes are not snapshotted either.
_REPLAY_OPAQUE_TYPES = (
    logging.Logger,
    logging.LoggerAdapter,
    logging.Handler,
    logging.Manager,
    logging.PlaceHolder,
)

# Metaclasses whose classes the snapshot treats as plain classes.
_REPLAY_PLAIN_METACLASSES = (type, abc.ABCMeta)


def _replay_float_same(left: float, right: float) -> bool:
    """Exact float identity: NaN matches NaN, and ``-0.0`` differs from ``0.0``."""
    if left != left:
        return right != right
    return left == right and math.copysign(1.0, left) == math.copysign(1.0, right)


def _replay_numpy_scalar_same(left: Any, right: Any) -> bool:
    """Compare two NumPy scalars of the same type.

    NaN matches NaN in each real and imaginary part, ``-0.0`` differs from
    ``0.0``, and NaT matches NaT.
    """
    if left.dtype != right.dtype:
        return False
    kind = left.dtype.kind
    if kind == "c":
        return (
            _replay_numpy_scalar_same(left.real, right.real)
            and _replay_numpy_scalar_same(left.imag, right.imag)
        )
    if kind == "f":
        if np.isnan(left):
            return bool(np.isnan(right))
        return bool(left == right) and bool(np.signbit(left) == np.signbit(right))
    if kind in "mM":
        if np.isnat(left) or np.isnat(right):
            return bool(np.isnat(left) and np.isnat(right))
        return bool(left == right)
    return bool(left == right)


def _replay_typed_set(items: Any) -> set:
    """Set items paired with their types, so ``{1}`` and ``{True}`` differ."""
    return {(type(item), item) for item in items}


def _replay_frames_same(left: Any, right: Any) -> bool:
    """Exact equality of two Polars frames or series (null and NaN match)."""
    if type(left) is not type(right):
        return False
    if type(left) is pl.DataFrame:
        return left.schema == right.schema and bool(left.equals(right))
    return (
        left.name == right.name
        and left.dtype == right.dtype
        and bool(left.equals(right, check_dtypes=True, check_names=True))
    )


def _replay_arrays_same(left: np.ndarray, right: np.ndarray) -> bool:
    """Element-wise equality with NaN matching NaN and NaT matching NaT."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    kind = left.dtype.kind
    if kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    if kind in "mM":
        # NaT is one bit pattern, so equal bits are equal values.
        return bool(np.array_equal(left.view(np.int64), right.view(np.int64)))
    return bool(np.array_equal(left, right))


class Model(BaseModel):
    """Base class for all simulation models (Polars-backed agent table)."""

    #: Declarative model-level metrics, evaluated once per step into the
    #: ``'model'`` results frame. Maps a column name to a ``callable(model)``,
    #: the name of a model attribute/method, or a constant. Override on the
    #: *subclass* (do not mutate this class attribute -- default is ``None``
    #: to avoid shared mutable state). Complements imperative ``record_model``.
    model_reporters: Optional[Dict[str, Any]] = None
    #: Declarative per-agent columns to snapshot each step into the (opt-in)
    #: ``'agent_vars'`` long-format results frame. Empty / ``None`` = no cost.
    #: Override on the subclass; do not mutate the base attribute.
    agent_reporters: Optional[List[str]] = None
    #: When True, capture a ``t=0`` row of the reporters before the first step.
    record_initial: bool = False
    #: Optional typed parameter schema ``{'name': (type, default)}``. When set
    #: on the subclass, ``self.p.name`` is pre-coerced to ``type`` at init
    #: (missing -> default). Default ``None`` avoids shared mutable state.
    params: Optional[Dict[str, Any]] = None

    def __init__(self, parameters: Dict[str, Any]):
        """Initialize a new model.

        Args:
            parameters: Dictionary of model parameters
        """
        # Population must exist before super().__init__ (which triggers the
        # agents_df setter).
        self.population = Population(schema={})

        # {column: {agent_id: value}} buffer flushed on the next agents_df read.
        self._pending_writes: Dict[str, Dict[Any, Any]] = {}
        # Monotonic counter bumped whenever the id column changes. Views use
        # it to invalidate cached id→row-position lookups.
        self._id_version: int = 0

        # Snapshot-view contract monitor (see contract.py). Mode 'off' adds
        # zero per-write overhead.
        self._contract = ContractMonitor()
        self._auto_batch_enabled = False
        self._auto_batches = 0
        self._auto_batched_agent_calls = 0
        # Pre-step agent frame used by activation replay, and the order a
        # replay asks stage_agents / activate to follow.
        self._pre_step_state = None
        self._pre_step_rng = None
        self._pre_step_world = None
        self._pre_step_world_reason = None
        self._pre_step_roster = None
        self._pre_step_globals = None
        self._activation_override_order = None

        super().__init__(parameters)  # sets self.random / self.rng / self.nprandom
        seed = parameters.get("seed")
        self._keyed_seed = randbits(128) if seed is None else int(seed)
        self.t = 0
        self._start_time = None
        self._last_progress_time = None
        # Quiet by default (library-friendly); set show_progress=True for CLI demos.
        self._show_progress = parameters.get('show_progress', False)
        self._model_data = []
        self._agent_vars = []  # per-step agent snapshots when agent_reporters is set
        # User placement (``cpu()`` / ``gpu()``) vs active runtime during :meth:`run`.
        self._device: str = (
            parameters.get("device") or parameters.get("backend") or "cpu"
        ).lower()
        self._execution_mode: str = (parameters.get("mode") or "vectorized").lower()
        self._execution = None
        # Optional OOP activation policy (see :mod:`ambr.scheduling`). ``None``
        # means "caller manages order" — default :meth:`step_oop` still falls
        # back to :meth:`step` and does not auto-activate agents.
        raw_act = parameters.get("activation")
        self._activation: Optional[str] = (
            None if raw_act in (None, "") else str(raw_act).lower().strip()
        )
        # Private model-specific GPU loops are opt-in per model instance.
        # This is an explicit deployment declaration, not a claim that AMBER
        # has independently verified the supplied evidence label.
        self._fast_path_approval: Optional[str] = None

        from .sequences import AgentList
        self.agents = AgentList(self, [])

    def __getattribute__(self, name: str) -> Any:
        if name in ("random", "rng", "nprandom"):
            monitor = object.__getattribute__(self, "__dict__").get("_contract")
            if monitor is not None and monitor._current_activation is not None:
                if monitor._in_replay:
                    monitor._replay_used_shared_rng = True
                elif monitor.active:
                    monitor._used_shared_rng = True
                    monitor._shared_rng_activations.add(monitor._current_activation)
        return super().__getattribute__(name)

    def cpu(self, mode: Optional[str] = None) -> "Model":
        """Place the next :meth:`run` on CPU. Returns ``self`` for chaining.

        Args:
            mode: optional execution style — ``'vectorized'`` (view API, default
                when omitted on the model) or ``'oop'`` (per-agent objects).
                Same as ``model.cpu().run(mode=...)`` when passed here.

        Examples::

            model.cpu().run()
            model.cpu(mode="vectorized").run()
            model.cpu(mode="oop").run()
        """
        self._device = "cpu"
        if mode is not None:
            self._set_execution_mode(mode)
        return self

    def vectorize(self) -> "Model":
        """Select vectorized execution on the current device. Returns ``self``.

        A model defined with ordinary ``setup()`` and ``step()`` methods can
        use this selector. AMBER batches eligible agent calls automatically.
        Models can also provide an explicit ``step_vectorized()`` hook.
        """
        self._set_execution_mode("vectorized")
        return self

    def gpu(self, mode: Optional[str] = None) -> "Model":
        """Place the next :meth:`run` on GPU (device-resident columns). Returns ``self``.

        Args:
            mode: defaults to ``'vectorized'`` for ordinary agent methods and
                the vectorized view API, including after CPU OOP placement.

        Examples::

            model.gpu().run()
            model.gpu(mode="vectorized").run()
        """
        self._set_execution_mode("vectorized" if mode is None else mode)
        self._device = "gpu"
        return self

    def approve_fast_path(self, evidence: str) -> "Model":
        """Allow this instance to use a private optimized GPU loop.

        ``evidence`` must be a non-empty provenance label chosen by the caller
        (for example, a test report or experiment identifier).  AMBER records
        only the explicit declaration; it does not validate the evidence.
        Without approval, :meth:`run` uses the general native runner even when
        private ``_setup_gpu_fast`` / ``_run_gpu_fast`` hooks are present.
        """
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("fast-path approval requires a non-empty evidence label")
        self._fast_path_approval = evidence.strip()
        return self

    def revoke_fast_path_approval(self) -> "Model":
        """Remove this instance's private-fast-path deployment approval."""
        self._fast_path_approval = None
        return self

    @property
    def fast_path_approval(self) -> Optional[str]:
        """Caller-supplied evidence label, or ``None`` when not approved."""
        return self._fast_path_approval

    def _set_execution_mode(self, mode: str) -> None:
        from .execution import EXECUTION_MODES

        resolved = mode.lower()
        if resolved not in EXECUTION_MODES:
            raise ValueError(
                f"mode must be one of {EXECUTION_MODES}, got {mode!r}"
            )
        self._execution_mode = resolved

    @property
    def device(self) -> str:
        """Selected execution device for the next :meth:`run` — ``'cpu'`` or ``'gpu'``."""
        return self._device

    @property
    def mode(self) -> str:
        """Execution style — ``'vectorized'`` (view API) or ``'oop'`` (agent objects)."""
        return self._execution_mode

    # --- public contract surface (stable for tests / callers) ---------------

    @property
    def _contract_mode(self) -> str:
        return self._contract.mode

    @_contract_mode.setter
    def _contract_mode(self, value: str) -> None:
        self._contract.mode = value

    @property
    def contract_certificates(self) -> List[ContractCertificate]:
        return self._contract.certificates

    @contract_certificates.setter
    def contract_certificates(self, value: List[ContractCertificate]) -> None:
        self._contract.certificates = value

    @property
    def _contract_active(self) -> bool:
        return self._contract.active

    @property
    def xp(self):
        """NumPy or CuPy array module for the active run (CPU when idle)."""
        return active_xp(self)

    @property
    def rng(self):
        """Step RNG — device RNG during ``gpu()`` runs, NumPy otherwise."""
        return active_rng(self)

    @rng.setter
    def rng(self, value) -> None:
        self._host_rng = value

    @property
    def agents_df(self) -> pl.DataFrame:
        """The current agent table, with pending writes applied.

        Under contract checking, a read inside an activation counts as a
        whole-column read of every column, as with a column view.
        """
        frame = self._agents_frame()
        monitor = self._contract
        if monitor.active:
            monitor.record_frame_read(frame.columns)
        return frame

    def _agents_frame(self) -> pl.DataFrame:
        """Return :attr:`agents_df` without recording a contract read.

        Library paths that report their own reads, writes, or reductions use
        this accessor.
        """
        self._frame_for_columns()
        sync_all_device_columns(self, dirty_only=True)
        frame = self.population.data
        self._exposed_agents_frame_id = id(frame)
        execution = self._execution
        if execution is not None and execution.config.device == "gpu":
            exposed = execution.exposed_frame
            if exposed is None or exposed[0] is not frame:
                execution.exposed_frame = (frame, frame.clone())
        return frame

    def _frame_for_columns(self) -> pl.DataFrame:
        """Read host schema and buffered state for backend-aware view operations."""
        self._flush_pending_writes()
        return self.population.data

    @agents_df.setter
    def agents_df(self, value):
        sync_agent_writes(self)
        self._pending_writes = {}
        self._set_frame(value)
        self._bump_id_version()
        self._contract.record_frame_replacement()

    def _set_frame(
        self,
        df: pl.DataFrame,
        *,
        written_columns: Optional[List[str]] = None,
        target_ids: Optional[Iterable[Any]] = None,
    ) -> None:
        """Internal single write seam for the agent table.

        All columnar writes (the view API in ``sequences.py`` and the tensor
        lane) route here instead of assigning ``population.data`` directly, so
        the backing store stays encapsulated. Does not bump the id-version
        (column-value writes keep the id set stable); callers that change the
        id set (e.g. the ``agents_df`` setter, ``add_agents``) bump explicitly.

        When ``written_columns`` is provided and the contract is active, those
        columns are recorded as lane/view commits (``scatter_add`` deliberately
        omits this -- it is the sanctioned multi-write reducer).
        """
        self.population.replace_frame(df)
        refresh_tracked_device_columns(self, written_columns)
        if written_columns is not None and self._contract.active:
            self._contract.record_commit(written_columns, target_ids)

    def _queue_write(self, column: str, agent_id: Any, value: Any) -> None:
        self._pending_writes.setdefault(column, {})[agent_id] = value
        # Contract: count ordinary writes per (column, id) within a step so a
        # second write to an already-written cell (which the buffer would
        # silently clobber) is detectable as a partial-map conflict.
        contract = self._contract
        if contract.active:
            contract.record_buffered_write(column, agent_id)

    def _bump_id_version(self) -> None:
        """Invalidate id→row caches after the agent id set changes."""
        self._id_version += 1
        # Caches filled by ambr._id_index.resolve_positions / ids_are_arange.
        self._id_pos_cache = None
        self._ids_arange_cache = None

    def _flush_pending_writes(self, *, columns=None) -> None:
        """Apply queued ``Agent`` writes, optionally for selected columns.

        Fast path: map ids → rows once via :func:`~ambr._id_index.resolve_positions`
        and scatter into Python lists (preserves ``None``; cheaper than a wide
        Polars update). Fallback: ``df.update(on='id')`` hash join.
        """
        execution = getattr(self, "_execution", None)
        if execution is not None and execution.exposed_frame is not None:
            reconcile_host_frame(self, execution)
        pending = self._pending_writes
        selected = pending if columns is None else {
            name: pending[name] for name in columns if name in pending
        }
        if not selected:
            return
        if execution is not None and execution.tracked_agents:
            for name in selected.keys() & execution.dirty_columns:
                sync_device_column_to_frame(self, name)
        # Clear before the write so an exception can't leave a stale buffer
        # that would double-apply on the next flush.
        if columns is None:
            self._pending_writes = {}
        else:
            for name in selected:
                del pending[name]
        self._apply_pending_writes(selected)
        if execution is not None and execution.tracked_agents:
            refresh_tracked_device_columns(self, selected)

    def _apply_pending_writes(self, pending) -> None:
        df = self.population.data

        # Empty / missing id: build the frame entirely from the buffer.
        if df.is_empty() or "id" not in df.columns:
            touched_ids = list({aid for col_map in pending.values() for aid in col_map})
            data_cols: Dict[str, list] = {"id": touched_ids}
            for col, id_to_val in pending.items():
                data_cols[col] = [id_to_val.get(aid, None) for aid in touched_ids]
            self.population.replace_frame(pl.DataFrame(
                [pl.Series(k, v, strict=False) for k, v in data_cols.items()]
            ))
            self._bump_id_version()
            return

        # Polars update requires matching join-key types.
        if df["id"].dtype == pl.Null:
            df = df.with_columns(pl.col("id").cast(pl.Int64))

        missing = [c for c in pending if c not in df.columns]
        if missing:
            df = df.with_columns(
                [pl.Series(c, [None] * df.height, strict=False) for c in missing]
            )

        touched_ids = list({aid for col_map in pending.values() for aid in col_map})

        # Fast path: shared id→row index + list scatter (keeps None, not nan).
        try:
            ids_np = np.asarray(touched_ids)
            positions = resolve_positions(self, df, ids_np)
            new_cols = []
            for col, id_to_val in pending.items():
                base = (
                    df[col].to_list()
                    if col in df.columns
                    else [None] * df.height
                )
                for aid, pos in zip(touched_ids, positions):
                    if aid in id_to_val:
                        base[int(pos)] = id_to_val[aid]
                new_cols.append(pl.Series(col, base, strict=False))
            self.population.replace_frame(df.with_columns(new_cols))
            return
        except Exception:
            pass

        # Fallback: df.update(on='id') hash join.
        update_cols: Dict[str, list] = {"id": touched_ids}
        for col, id_to_val in pending.items():
            update_cols[col] = [id_to_val.get(aid, None) for aid in touched_ids]
        update_df = pl.DataFrame(
            [pl.Series(k, v, strict=False) for k, v in update_cols.items()]
        )
        self.population.replace_frame(df.update(update_df, on="id", how="left"))

    # --- snapshot-view contract seams ---------------------------------------

    def _contract_snapshot(self) -> Tuple[Dict[str, str], Counter[Any]]:
        """Return column types and ID multiplicities of the committed population.

        The schema carries dtypes (not just names) so that a mid-step dtype
        change -- e.g. an ``Int64`` attribute silently rewritten as ``Float64``
        -- surfaces as a schema mutation rather than passing unnoticed.
        """
        df = self._agents_frame()  # flushes pending writes
        schema = {name: str(dt) for name, dt in zip(df.columns, df.dtypes)}
        return schema, Counter(df["id"]) if "id" in df.columns else Counter()

    def _contract_record_borrow(self, column: str) -> None:
        """Record a lane borrow of ``column``; flag it if already committed.

        Commits go through :meth:`_set_frame` (``written_columns=``); only
        borrows need this separate seam.
        """
        self._contract.record_borrow(column)

    def _contract_record_mutable_borrow(self, column: str) -> None:
        """Record exposure of a mutable NumPy/CuPy column buffer."""
        self._contract.record_mutable_borrow(column)

    def _contract_record_reduction(
        self,
        columns: Iterable[str],
        domains: Optional[Dict[str, str]] = None,
        operator: str = "+",
    ) -> None:
        """Record columns written through a commutative reduction.

        ``domains`` maps each column to the dtype of the executed fold.
        """
        self._contract.record_reduction(columns, domains=domains, operator=operator)

    def setup(self): pass
    def step(self): pass

    def step_vectorized(self):
        """Execute one vectorized step.

        Models with a distinct vectorized implementation can override this
        hook. The default runs ``step()``, or ``step_oop()`` if that is the
        model's only step implementation.
        """
        step = self.step
        if getattr(step, "__func__", None) is Model.step:
            return self.step_oop()
        return step()

    def step_oop(self):
        """Execute one object-oriented step.

        Models with tracked :class:`~ambr.agent.Agent` objects can override
        this hook. The default preserves backwards compatibility for models
        that have only a single ``step()`` implementation.

        If ``parameters['activation']`` is set and this method is **not**
        overridden, the default still calls :meth:`step` (not auto-activate),
        so existing models keep working. Call :meth:`activate_agents` explicitly
        inside ``step`` / ``step_oop`` to use activation helpers.
        """
        return self.step()

    def activate_agents(
        self,
        mode: Optional[str] = None,
        method: str = "step",
    ) -> None:
        """Run OOP agents under a thin activation policy (Mesa-inspired).

        Args:
            mode: ``'sequential'``, ``'random'``, or ``'simultaneous'``.
                Defaults to ``parameters['activation']`` or ``'sequential'``.
            method: Per-agent method name (default ``'step'``).

        Notes:
            * Vectorized models should encode order inside
              :meth:`step_vectorized` (see :func:`~ambr.scheduling.shuffled_ids`).
            * Under contract checking, AMBER can certify order independence
              for supported sequential and shuffled single-phase sweeps.
        """
        from .scheduling import activate, normalize_activation

        chosen = mode if mode is not None else self._activation
        execution = self._execution
        rng = self.rng
        if execution is not None and execution.tracked_agents and rng is execution.device_rng:
            rng = self._host_rng
        activate(
            self.agents,
            mode=normalize_activation(chosen),
            method=method,
            rng=rng,
        )

    def stage_agents(self, stage: str = "stage0") -> Iterable[Any]:
        """Yield each agent inside one monitored stage.

        Conflicts found before the stage closes are reported with an
        inconclusive witness. A swap replay, when one runs, is started from
        :meth:`run_step` only after the whole step has finished.
        """
        mon = self._contract
        active = bool(getattr(mon, "active", False)) and not getattr(
            mon, "_in_replay", False
        )
        if active:
            mon.begin_stage(stage)
        ordered = list(self.agents)
        try:
            for agent in ordered:
                aid = getattr(agent, "id", None)
                if active and aid is not None:
                    mon.begin_activation(aid)
                try:
                    yield agent
                finally:
                    if active and aid is not None:
                        mon.end_activation()
        finally:
            self._flush_pending_writes()
            if active:
                mon.end_stage(model=self)

    def update(self):
        """Per-step hook, called after :meth:`step` with ``t`` already advanced.

        Override to record model metrics imperatively (``self.record_model(...)``)
        or for any post-step bookkeeping. This is a *pure hook*: overriding it no
        longer requires ``super().update()`` -- the step counter and step-data
        lifecycle are owned by :meth:`run_step`. Calling ``super().update()``
        remains legal (it is a no-op) so existing models keep working.
        """
        pass

    def record_model(self, key: str, value: Any):
        """Record a model-level variable for the current step."""
        if not hasattr(self, '_current_step_data'):
            self._current_step_data = {'t': self.t}
        self._current_step_data[key] = value

    def record(self, key: str, value: Any):
        """Deprecated AgentPy alias for :meth:`record_model` / ``model_reporters``."""
        warn_deprecated("Model.record(key, value)", "record_model(key, value) or model_reporters")
        self.record_model(key, value)

    def _collect_model_reporters(self) -> None:
        """Evaluate declarative ``model_reporters`` into the current step row.

        Each spec is a ``callable(model)``, a model attribute/method name, or a
        constant. Runs after :meth:`step` and before :meth:`update`. On
        duplicate keys the later stage wins::

            step() record_model  <  model_reporters  <  update() record_model
        """
        reporters = type(self).model_reporters
        if not reporters:
            return
        for name, spec in reporters.items():
            if callable(spec):
                value = spec(self)
            elif isinstance(spec, str):
                value = getattr(self, spec)
                if callable(value):
                    value = value()
            else:
                value = spec
            self._current_step_data[name] = value

    def _snapshot_agent_reporters(self) -> None:
        """Append a per-agent snapshot of ``agent_reporters`` columns (opt-in)."""
        cols = type(self).agent_reporters
        if not cols:
            return
        df = self._agents_frame()
        have = [c for c in cols if c in df.columns]
        if not have:
            return
        self._agent_vars.append(
            df.select(['id', *have]).with_columns(pl.lit(self.t).alias('t'))
        )

    def _finalize_step_data(self):
        data = getattr(self, '_current_step_data', None)
        if data:
            self._model_data.append(data.copy())

    def _append_fast_step(self, **data: Any) -> None:
        """Append a model-data row for a private optimized execution loop.

        This is an internal seam for backends that keep state on an accelerator
        and must avoid routing every step through the general AgentList view
        machinery.  It does not change the public run/result contract.
        """
        self.t += 1
        row = {'t': self.t}
        row.update(data)
        self._model_data.append(row)

    def _begin_step_data(self) -> None:
        """Allocate the step row *before* :meth:`step` so recordings are kept.

        The row's ``t`` is the post-step time (``self.t + 1``). Call sites must
        not finalize this row if the step body fails.
        """
        self._current_step_data = {'t': self.t + 1}

    def _advance_and_record(self) -> None:
        """Advance ``t`` and finalize the current step row.

        Owns the step-counter increment and step-data lifecycle so :meth:`update`
        can be a pure hook. Values already present in ``_current_step_data``
        (e.g. from :meth:`record_model` during :meth:`step`) are preserved.
        Declarative reporters are evaluated next, then the imperative
        :meth:`update` hook (which wins on duplicate keys), then the row is
        finalized.

        When called without a prior :meth:`_begin_step_data` (tests / legacy),
        starts a fresh row for the new ``t``.
        """
        self.t += 1
        row = getattr(self, '_current_step_data', None)
        if not row or row.get('t') != self.t:
            self._current_step_data = {'t': self.t}
        else:
            self._current_step_data['t'] = self.t
        self._collect_model_reporters()
        self._snapshot_agent_reporters()
        self.update()
        self._finalize_step_data()

    def _record_initial_state(self) -> None:
        """Capture a ``t=0`` row of the declarative reporters (record_initial)."""
        self._current_step_data = {'t': self.t}
        self._collect_model_reporters()
        self._snapshot_agent_reporters()
        self._finalize_step_data()

    def end(self): pass

    def _ensure_setup(self) -> None:
        """Run model setup once before simulation steps execute."""
        if not getattr(self, "_setup_done", False):
            self.setup()
            self._setup_done = True

    def run_step(self) -> None:
        """Execute one simulation step. The first call also runs ``setup``.

        When a contract mode other than ``'off'`` is active (set via
        :meth:`run`), the step body is bracketed by the snapshot-view
        conformance checker and a :class:`~ambr.contract.ContractCertificate`
        is appended to ``self.contract_certificates``.

        The step-data row is allocated before :meth:`step` so values recorded
        inside ``step()`` survive into the model frame. A failed step discards
        the partial row and does not append it to ``_model_data``.
        """
        self._ensure_setup()
        execution = getattr(self, "_execution", None)
        mode = execution.config.mode if execution is not None else self._execution_mode
        step_fn = self.step_oop if mode == "oop" else self.step_vectorized
        mon = self._contract
        self._begin_step_data()
        try:
            if mon.mode == "off":
                step_fn()
            else:
                mon.begin_step(self._contract_snapshot())
                step_finished = False
                try:
                    self._pre_step_state = self._agents_frame().clone()
                    self._pre_step_rng = self._capture_rng_state()
                    entry_reasons: List[str] = []
                    self._pre_step_world = self._capture_replay_world(None, entry_reasons)
                    self._pre_step_world_reason = entry_reasons[0] if entry_reasons else None
                    self._pre_step_roster = self._snapshot_replay_roster()
                    self._pre_step_globals = self._capture_global_random()
                    step_fn()
                    step_finished = True
                finally:
                    cert = mon.end_step(
                        self.t, self._contract_snapshot(),
                        model=self if step_finished else None,
                    )
                    self._pre_step_state = None
                    self._pre_step_rng = None
                    self._pre_step_world = None
                    self._pre_step_world_reason = None
                    self._pre_step_roster = None
                    self._pre_step_globals = None

                if mon.mode == "warn":
                    for v in cert.violations:
                        warnings.warn(
                            f"[step {cert.step}] {v.kind}: {v.detail}",
                            UserWarning,
                            stacklevel=2,
                        )
                elif mon.mode == "raise" and not cert.ok:
                    raise ContractViolationError(cert)

            self._advance_and_record()
        except Exception:
            self._current_step_data = {}
            self._pre_step_state = None
            self._pre_step_rng = None
            self._pre_step_world = None
            self._pre_step_world_reason = None
            self._pre_step_roster = None
            self._pre_step_globals = None
            raise

    def replay_activation_swap(self, a: Any, b: Any) -> _ReplayOutcome:
        """Rerun this step with activations ``a`` and ``b`` swapped.

        ``value`` is True when the agent frame or restored model state
        diverges, False when both match, and None when the swap is not a
        sound witness. ``reason`` explains an inconclusive result. The replay
        restores the step-entry random streams and the supported graph of
        model attributes and private agent attributes, then puts the forward
        objects back. When that graph reaches a value the replay cannot
        restore, the swap is not run and the witness is None, so monitoring
        cannot change the forward run. Stream restoration and model-state
        restoration are attempted separately, so one failure does not skip the
        other. An exception from the swapped execution is caught here. The
        forward frame, pending writes, per-agent attribute cache, activation
        override, and monitor mode are restored either way.
        """
        mon = self._contract
        if getattr(mon, "_in_replay", False) or getattr(mon, "_stage_open", False):
            return _ReplayOutcome(None, None)
        pre = getattr(self, "_pre_step_state", None)
        if pre is None:
            return _ReplayOutcome(None, None)
        if mon._activation_sweeps != 1:
            return _ReplayOutcome(None, "replay requires a single recorded activation sweep")
        order = list(getattr(mon, "_activation_order", []))
        if a not in order or b not in order:
            return _ReplayOutcome(None, None)
        if mon._sweep_rng_reason is not None:
            return _ReplayOutcome(None, mon._sweep_rng_reason)
        if mon._used_shared_rng:
            return _ReplayOutcome(None, "shared random draws were accessed during activation")
        forward_roster = self._snapshot_replay_roster()
        if forward_roster is None or not self._replay_roster_matches(self._pre_step_roster):
            return _ReplayOutcome(None, "population or agent roster changed during the step")
        entry_rng = getattr(self, "_pre_step_rng", None)
        post_rng = self._capture_rng_state()
        if (
            entry_rng is None
            or post_rng is None
            or not self._rng_same_stream_ids(entry_rng, post_rng)
        ):
            return _ReplayOutcome(None, None)
        entry_world = getattr(self, "_pre_step_world", None)
        reasons: List[str] = []
        post_world = self._capture_replay_world(
            self._replay_world_objects(entry_world), reasons, entry_world
        )
        if entry_world is None:
            reasons.insert(0, getattr(self, "_pre_step_world_reason", None))
        if entry_world is None or post_world is None:
            reason = next((text for text in reasons if text), None)
            return _ReplayOutcome(None, (
                reason or "step-entry model or agent state could not be captured"
            ) + "; the swap was not run")
        entry_globals = getattr(self, "_pre_step_globals", None)
        forward_globals = self._capture_global_random()

        original_post = self._agents_frame().clone()
        saved_attrs = self._snapshot_agent_attrs()
        forward_step_data = self._current_step_data
        try:
            recorded = copy.deepcopy(forward_step_data)
        except Exception:
            recorded = None
        swapped = list(order)
        index_a = swapped.index(a)
        index_b = swapped.index(b)
        swapped[index_a], swapped[index_b] = swapped[index_b], swapped[index_a]

        saved_mode = mon.mode
        saved_active = mon.active
        saved_current = mon._current_activation
        saved_order = list(mon._activation_order)
        saved_reads = {key: set(value) for key, value in mon._activation_reads.items()}
        saved_writes = {
            key: set(value) for key, value in mon._activation_writes.items()
        }
        saved_stage_open = mon._stage_open
        saved_stage = mon._current_stage
        saved_stage_violations = list(mon._stage_violations)
        mon._in_replay = True
        mon._replay_used_shared_rng = False
        mon._replay_sweep_rng_reason = None
        mon.mode = "off"
        mon.active = False
        witness: Optional[bool] = None
        reason: Optional[str] = None
        try:
            entry_rng_ok = self._restore_rng_state(entry_rng)
            failures: List[str] = []
            entry_world_ok = self._restore_replay_world(entry_world, failures)
            # The global ``random`` and legacy ``numpy.random`` streams are
            # not model state, but agent code often draws from them.
            entry_globals_ok = self._restore_global_random(entry_globals)
            if not entry_rng_ok:
                reason = "step-entry random streams could not be restored"
            elif not entry_world_ok:
                reason = "step-entry model or agent state could not be restored" + (
                    f" ({failures[0]})" if failures else ""
                )
            elif not entry_globals_ok:
                reason = "the step-entry global random states could not be restored"
            else:
                self._set_frame(pre.clone())
                self._pending_writes = {}
                self._refresh_agent_cache(pre)
                self._activation_override_order = swapped
                self._begin_step_data()
                execution = getattr(self, "_execution", None)
                mode = (
                    execution.config.mode
                    if execution is not None
                    else self._execution_mode
                )
                step_fn = self.step_oop if mode == "oop" else self.step_vectorized
                try:
                    step_fn()
                    self._flush_pending_writes()
                except Exception as exc:
                    reason = _replay_failure_reason(exc)
                else:
                    if mon._replay_used_shared_rng:
                        reason = "shared random draws were accessed during replay activation"
                    elif mon._replay_sweep_rng_reason is not None:
                        reason = mon._replay_sweep_rng_reason
                    elif self._activation_override_order is not None:
                        reason = "the recorded activation sweep did not run during replay"
                    elif not self._replay_roster_matches(forward_roster):
                        reason = "replay changed the population or agent roster"
                    else:
                        witness, reason = self._replay_compare(
                            original_post, post_world, entry_world, recorded
                        )
        finally:
            self._current_step_data = forward_step_data
            post_rng_ok = self._restore_rng_state(post_rng)
            post_failures: List[str] = []
            post_world_ok = self._restore_replay_world(post_world, post_failures)
            post_globals_ok = self._restore_global_random(forward_globals)
            if not post_rng_ok:
                witness = None
                reason = "forward random streams could not be restored"
            elif not post_world_ok:
                witness = None
                reason = "forward model state could not be restored" + (
                    f" ({post_failures[0]})" if post_failures else ""
                )
            elif not post_globals_ok:
                witness = None
                reason = "the forward global random states could not be restored"
            root, objects, bindings, agent_type = forward_roster
            self.agents = root
            root._agent_objects[:] = objects
            root._agents_by_id.clear()
            root._agents_by_id.update(bindings)
            root.agent_type = agent_type
            self._set_frame(original_post.clone())
            self._bump_id_version()
            self._pending_writes = {}
            self._activation_override_order = None
            if not self._restore_agent_attrs(saved_attrs):
                witness = None
                reason = "forward agent state could not be restored"
            mon.mode = saved_mode
            mon.active = saved_active
            mon._in_replay = False
            mon._replay_used_shared_rng = False
            mon._replay_sweep_rng_reason = None
            mon._current_activation = saved_current
            mon._activation_order = saved_order
            mon._activation_reads = saved_reads
            mon._activation_writes = saved_writes
            mon._stage_open = saved_stage_open
            mon._current_stage = saved_stage
            mon._stage_violations = saved_stage_violations
        return _ReplayOutcome(witness, reason)

    @staticmethod
    def _capture_global_random() -> Optional[Tuple[Any, Any]]:
        """The states of the global ``random`` and legacy ``numpy.random`` streams."""
        try:
            return random.getstate(), np.random.get_state()
        except Exception:
            return None

    @staticmethod
    def _restore_global_random(saved: Optional[Tuple[Any, Any]]) -> bool:
        if saved is None:
            return False
        try:
            random.setstate(saved[0])
            np.random.set_state(saved[1])
            return True
        except Exception:
            return False

    def _step_changed_replay_state(self) -> Optional[bool]:
        """Whether model or private agent state differs from step entry.

        None when either state cannot be captured or compared.
        """
        status, _where = self._step_replay_state_check()
        if status == "unknown":
            return None
        return status == "changed"

    def _step_replay_state_check(self) -> Tuple[str, Optional[str]]:
        """Compare model and private agent state with its step-entry snapshot.

        Returns ``("same", None)``, ``("changed", where)``, or
        ``("unknown", where)``. ``where`` names the attribute that differs or
        could not be represented, when one is known.
        """
        entry = getattr(self, "_pre_step_world", None)
        if entry is None:
            return "unknown", getattr(self, "_pre_step_world_reason", None)
        if self._replay_world_matches_live(entry):
            return "same", None
        reasons: List[str] = []
        current = self._capture_replay_world(self._replay_world_objects(entry), reasons, entry)
        if current is None:
            return "unknown", reasons[0] if reasons else None
        differs: List[str] = []
        try:
            same = self._replay_worlds_equal(entry, current, differs)
        except _ReplayAliasMismatch as exc:
            return "changed", str(exc) or None
        if same is None:
            return "unknown", differs[0] if differs else None
        if same:
            return "same", None
        return "changed", differs[0] if differs else None

    def _replay_identity_notes(self) -> Tuple[str, ...]:
        """Describe step-entry state that is compared by identity only.

        These objects do not withhold the certificate; their notes are added
        to its assumptions.
        """
        world = getattr(self, "_pre_step_world", None)
        if world is None or len(world) < 4 or not world[3]:
            return ()
        owners = world[2]
        notes: List[str] = []
        items = list(world[3].items())
        for (on_model, name, type_name, mode), (count, index) in items[:8]:
            if on_model:
                where = f"Model attribute {name!r}"
            else:
                where = f"Private attribute {name!r} of agent {self._replay_owner_id(owners, index)!r}"
                if count > 1:
                    where += f" (and {count - 1} other agent(s))"
            if mode == "attributes":
                notes.append(
                    f"{where} holds a {type_name} kept by identity: its own attributes "
                    f"are compared with step entry and restored for replay, but the "
                    f"objects it refers to are compared by identity only."
                )
            else:
                notes.append(
                    f"{where} holds a {type_name} kept by identity: its internal state "
                    f"is not compared with step entry or restored for replay."
                )
        if len(items) > 8:
            notes.append(f"{len(items) - 8} more attribute(s) hold objects kept by identity.")
        return tuple(notes)

    @staticmethod
    def _replay_owner_id(owners: Tuple[Any, ...], index: int) -> Any:
        try:
            return object.__getattribute__(owners[index], "__dict__").get("id")
        except Exception:
            return index

    def _replay_binding_label(self, world: Any, key: Tuple[int, str, str]) -> str:
        index, _where, name = key
        if index < 0:
            return f"model attribute {name!r}"
        return f"attribute {name!r} of agent {self._replay_owner_id(world[2], index)!r}"

    @staticmethod
    def _replay_children(node: Tuple[str, Any, Any]) -> List[Tuple[str, Any]]:
        """The references held by one snapshot node."""
        kind, _obj, payload = node
        if kind in ("list", "tuple", "function", "method", "wrapper"):
            return list(payload)
        if kind == "dict":
            pairs, factory = payload
            refs = [ref for _key, ref in pairs]
            if factory is not None:
                refs.append(factory)
            return refs
        if kind == "view":
            return [payload[0]]
        if kind == "cell":
            return [] if payload is None else [payload]
        if kind == "class":
            namespace, bases, _name, _qualname = payload
            return [ref for _name, ref in namespace] + [bases]
        if kind == "object":
            return [ref for _key, ref in payload[0]]
        return []

    def _replay_node_label(self, world: Any, oid: int) -> Optional[str]:
        """Name the attribute through which ``world`` reaches node ``oid``."""
        try:
            bindings, nodes = world[0], world[1]
            for key, ref in bindings:
                stack = [ref]
                seen = set()
                while stack:
                    kind, payload = stack.pop()
                    if kind != "node" or payload in seen:
                        continue
                    if payload == oid:
                        return self._replay_binding_label(world, key)
                    seen.add(payload)
                    stack.extend(self._replay_children(nodes[payload]))
        except Exception:
            return None
        return None

    def _replay_world_matches_live(self, world: Any) -> bool:
        """True when the live state still matches ``world`` exactly.

        Checks that every binding and every snapshotted container holds the
        same objects, and scalars of the same type and value, as recorded.
        This proves the state unchanged in one pass, without building a
        second snapshot. False means "not proven": the caller then compares
        a fresh snapshot, which also accepts an equal state built from new
        objects.
        """
        try:
            bindings, nodes, owners = world[0], world[1], world[2]
            live_owners = self._agent_objects()
            if len(live_owners) != len(owners) or any(
                a is not b for a, b in zip(live_owners, owners)
            ):
                return False
            exact = _REPLAY_EXACT_EQUAL
            values_equal = self._replay_values_equal

            def matches(value: Any, ref: Tuple[str, Any]) -> bool:
                kind = ref[0]
                if kind == "node":
                    return value is nodes[ref[1]][1]
                if kind == "anchor" or kind == "external":
                    return value is ref[1]
                recorded = ref[1]
                if type(value) is not type(recorded):
                    return False
                if type(value) in exact:
                    return value == recorded
                return values_equal(value, recorded)

            refs = dict(bindings)
            found = 0
            slots_cache: Dict[int, Tuple[Tuple[str, Any], ...]] = {}
            for index, owner in enumerate((self, *owners), start=-1):
                attrs = object.__getattribute__(owner, "__dict__")
                owned = self._is_replay_model_attr if index < 0 else Model._is_replay_agent_attr
                for name, value in attrs.items():
                    if owned(name):
                        ref = refs.get((index, "dict", name))
                        if ref is None or not matches(value, ref):
                            return False
                        found += 1
                for name, descriptor in Model._replay_slots(type(owner), slots_cache):
                    if owned(name):
                        try:
                            value = descriptor.__get__(owner, type(owner))
                        except AttributeError:
                            continue
                        ref = refs.get((index, "slot", name))
                        if ref is None or not matches(value, ref):
                            return False
                        found += 1
            if found != len(refs):
                return False
            for kind, obj, payload in nodes.values():
                if kind == "list":
                    if len(obj) != len(payload) or not all(
                        map(matches, obj, payload)
                    ):
                        return False
                elif kind == "dict":
                    pairs, factory = payload
                    if len(obj) != len(pairs):
                        return False
                    for (key, value), (recorded, ref) in zip(obj.items(), pairs):
                        if type(key) is not type(recorded) or not values_equal(key, recorded):
                            return False
                        if not matches(value, ref):
                            return False
                    if factory is not None and not matches(obj.default_factory, factory):
                        return False
                elif kind == "set":
                    if len(obj) != len(payload) or _replay_typed_set(obj) != _replay_typed_set(payload):
                        return False
                elif kind == "ndarray":
                    if (
                        obj.flags.writeable != payload.flags.writeable
                        or not values_equal(obj, payload)
                    ):
                        return False
                elif kind == "view":
                    if (
                        obj.flags.writeable != payload[1]
                        or obj.shape != payload[2]
                        or obj.strides != payload[3]
                    ):
                        return False
                elif kind == "function":
                    if not all(
                        matches(getattr(obj, name), ref)
                        for name, ref in zip(self._REPLAY_FUNCTION_FIELDS, payload)
                    ):
                        return False
                elif kind == "cell":
                    try:
                        contents = obj.cell_contents
                    except ValueError:
                        if payload is not None:
                            return False
                    else:
                        if payload is None or not matches(contents, payload):
                            return False
                elif kind == "class":
                    namespace, bases, name, qualname = payload
                    current = list(obj.__dict__.items())
                    if (
                        obj.__name__ != name
                        or obj.__qualname__ != qualname
                        or not matches(obj.__bases__, bases)
                        or len(current) != len(namespace)
                        or any(
                            key != recorded or not matches(value, ref)
                            for (key, value), (recorded, ref) in zip(current, namespace)
                        )
                    ):
                        return False
                elif kind == "object":
                    attrs, _external = payload
                    live = self._replay_instance_items(obj)
                    if live is None or len(live) != len(attrs) or any(
                        key != recorded or not matches(value, ref)
                        for (key, value), (recorded, ref) in zip(live, attrs)
                    ):
                        return False
                elif kind == "stream":
                    if not self._replay_values_equal(
                        self._replay_stream_state(obj, payload[0]), payload[1]
                    ):
                        return False
                elif kind == "frame":
                    if not _replay_frames_same(obj, payload):
                        return False
                # Tuples, methods, and descriptor wrappers cannot change;
                # their mutable children are nodes of their own.
            return True
        except Exception:
            return False

    def _snapshot_replay_roster(self):
        from .sequences import AgentList

        root = self.agents
        if type(root) is not AgentList:
            return None
        return root, tuple(root._agent_objects), dict(root._agents_by_id), root.agent_type

    def _replay_roster_matches(self, snapshot) -> bool:
        if snapshot is None:
            return False
        root, objects, bindings, agent_type = snapshot
        return (
            self.agents is root
            and root.agent_type is agent_type
            and len(root._agent_objects) == len(objects)
            and all(a is b for a, b in zip(root._agent_objects, objects))
            and root._agents_by_id.keys() == bindings.keys()
            and all(root._agents_by_id[aid] is agent and agent.id == aid for aid, agent in bindings.items())
        )

    # Public model attributes owned by the framework, not by the user's step.
    _REPLAY_FRAMEWORK_ATTRS = frozenset({
        "population",
        "t",
        "agents",
        "random",
        "rng",
        "nprandom",
        "model_df",
        "agents_df",
    })

    # Private model attributes owned by the framework. Every other private
    # model attribute is user state, which a replay snapshots like public
    # state.
    _REPLAY_FRAMEWORK_PRIVATE = frozenset({
        "_activation",
        "_activation_override_order",
        "_agent_vars",
        "_auto_batch_enabled",
        "_auto_batched_agent_calls",
        "_auto_batches",
        "_contract",
        "_current_step_data",
        "_device",
        "_execution",
        "_execution_mode",
        "_exposed_agents_frame_id",
        "_fast_path_approval",
        "_host_rng",
        "_id_pos_cache",
        "_id_version",
        "_ids_arange_cache",
        "_keyed_seed",
        "_last_progress_time",
        "_model_data",
        "_pending_writes",
        "_pre_step_globals",
        "_pre_step_rng",
        "_pre_step_roster",
        "_pre_step_state",
        "_pre_step_world",
        "_pre_step_world_reason",
        "_setup_done",
        "_show_progress",
        "_start_time",
    })

    def _is_replay_model_attr(self, name: str) -> bool:
        """True for a model attribute that holds user state."""
        return (
            name not in self._REPLAY_FRAMEWORK_ATTRS
            and name not in self._REPLAY_FRAMEWORK_PRIVATE
        )

    @staticmethod
    def _is_replay_agent_attr(name: str) -> bool:
        """True for a private agent attribute. Public ones live in the frame."""
        return name.startswith("_") and name != "_ambr_container_cells"

    def _replay_anchors(self) -> Dict[int, Any]:
        """Objects the replay snapshot keeps by identity and does not copy.

        The model and its agents are restored through their attribute
        bindings and the agent frame, and the random streams through
        :meth:`_capture_rng_state`.
        """
        objects = [self, self.__dict__.get("agents"), self.__dict__.get("population")]
        objects.extend(self._agent_objects())
        execution = self.__dict__.get("_execution")
        for stream in (
            self.__dict__.get("random"),
            self.__dict__.get("_host_rng"),
            self.__dict__.get("nprandom"),
            getattr(execution, "device_rng", None) if execution is not None else None,
        ):
            objects.append(stream)
            objects.append(getattr(stream, "_rng", None))
        return {id(obj): obj for obj in objects if obj is not None}

    # Built-in descriptors found in class dicts. They cannot be modified.
    _REPLAY_FIXED_DESCRIPTORS = (
        types.GetSetDescriptorType,
        types.MemberDescriptorType,
        types.WrapperDescriptorType,
        types.MethodDescriptorType,
        types.ClassMethodDescriptorType,
    )

    @staticmethod
    def _is_replay_code(value: Any) -> bool:
        """True for code that holds no state a step can change.

        Built-in types such as ``int`` or ``list`` and built-in descriptors
        cannot be modified, a built-in function bound to no object or to a
        module keeps no state of its own, and AMBER's own classes are not
        user state. Python functions and other classes are snapshotted with
        their own state; modules are not supported.
        """
        if isinstance(value, type):
            flags = value.__flags__
            if not flags & _TPFLAGS_HEAPTYPE or flags & _TPFLAGS_IMMUTABLETYPE:
                return True
            module = value.__dict__.get("__module__")
            return isinstance(module, str) and module.split(".")[0] == "ambr"
        if isinstance(value, (types.CodeType, *Model._REPLAY_FIXED_DESCRIPTORS)):
            return True
        if isinstance(value, types.BuiltinFunctionType):
            owner = value.__self__
            return owner is None or isinstance(owner, types.ModuleType)
        return False

    # Function attributes a step can rebind or mutate.
    _REPLAY_FUNCTION_FIELDS = (
        "__code__", "__defaults__", "__kwdefaults__", "__dict__", "__closure__",
        "__name__", "__qualname__", "__module__", "__doc__",
    )

    @staticmethod
    def _replay_slots(
        cls: type, cache: Optional[Dict[int, Tuple[Tuple[str, Any], ...]]] = None
    ) -> Tuple[Tuple[str, Any], ...]:
        """``(name, member descriptor)`` for each ``__slots__`` entry of ``cls``.

        Slot values are not in the instance ``__dict__``, so the snapshot
        reads and writes them through their descriptors. ``cache`` holds the
        results of one snapshot pass, keyed by class ID; no global cache
        keeps classes alive.
        """
        if cache is not None:
            found_before = cache.get(id(cls))
            if found_before is not None:
                return found_before
        found: List[Tuple[str, Any]] = []
        seen = set()
        for klass in cls.__mro__:
            slots = klass.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in ("__dict__", "__weakref__"):
                    continue
                name = slot
                if slot.startswith("__") and not slot.endswith("__"):
                    name = f"_{klass.__name__.lstrip('_')}{slot}"
                descriptor = klass.__dict__.get(name)
                if isinstance(descriptor, types.MemberDescriptorType) and name not in seen:
                    seen.add(name)
                    found.append((name, descriptor))
        result = tuple(found)
        if cache is not None:
            cache[id(cls)] = result
        return result

    # Immutable built-in scalars. Only these exact types are restored by
    # rebinding: a subclass can carry instance state or override comparison.
    _REPLAY_SCALAR_TYPES = frozenset({
        type(None), bool, int, float, complex, str, bytes,
    })

    # Dicts whose whole state is their items (plus a defaultdict's
    # default_factory). Exact types only, so restoring them runs no user code.
    _REPLAY_DICT_TYPES = frozenset({dict, OrderedDict, Counter, defaultdict, AttrDict})

    # Mutable built-in containers whose contents the snapshot cannot restore
    # without running a subclass's methods, or at all. State that reaches one
    # (other than the exact types handled above) is not snapshotted.
    _REPLAY_UNSUPPORTED_CONTAINERS = (
        list, dict, set, frozenset, deque, bytearray, memoryview, np.ndarray,
    )

    @staticmethod
    def _is_numpy_scalar(value: Any) -> bool:
        return (
            isinstance(value, np.generic)
            and type(value).__module__ == "numpy"
            and not isinstance(value, (np.void, np.object_))
        )

    @staticmethod
    def _is_replay_leaf(value: Any) -> bool:
        """True when ``value`` is immutable and compared by value.

        Covers exact built-in scalars, NumPy scalars other than void and
        object scalars, enum members, the immutable standard-library values
        in ``_REPLAY_VALUE_KEYS`` (dates and times with a standard time
        zone, ``Decimal``, ``Fraction``, ``range``, and paths), and tuples
        and frozensets made only of such values. Leaves need no restore.
        """
        kind = type(value)
        if kind in Model._REPLAY_SCALAR_TYPES:
            return True
        if kind is tuple or kind is frozenset:
            return all(Model._is_replay_leaf(item) for item in value)
        if Model._is_numpy_scalar(value):
            return True
        if kind in _REPLAY_VALUE_KEYS:
            return kind not in (_datetime.datetime, _datetime.time) or _replay_tz_ok(value)
        return isinstance(kind, enum.EnumMeta)

    @staticmethod
    def _is_replay_tuple(value: Any) -> bool:
        """A tuple or a named tuple without per-instance attributes."""
        kind = type(value)
        if kind is tuple:
            return True
        return (
            isinstance(value, tuple)
            and hasattr(kind, "_fields")
            and getattr(value, "__dict__", None) is None
        )

    @staticmethod
    def _ndarray_replay_supported(value: np.ndarray) -> bool:
        dtype = value.dtype
        return not dtype.hasobject and dtype.kind in "biufcmMUS"

    @classmethod
    def _replay_array_root(cls, value: np.ndarray) -> np.ndarray:
        """Return the array that owns ``value``'s memory.

        A view is restored through the array that owns its memory, because
        a write through another view of the same memory changes it too.
        Raises :class:`_ReplayUnsupported` when the owner is not a plain
        numeric array that owns its data.
        """
        if type(value) is not np.ndarray:
            raise _ReplayUnsupported(f"a {type(value).__name__} array subclass")
        root = value
        while isinstance(root.base, np.ndarray):
            root = root.base
        if type(root) is not np.ndarray or root.base is not None or not root.flags.owndata:
            raise _ReplayUnsupported("a NumPy array whose memory is owned by another object")
        if not cls._ndarray_replay_supported(root):
            raise _ReplayUnsupported(f"a {root.dtype} array")
        return root

    @staticmethod
    def _is_replay_record(kind: type) -> bool:
        """A ``SimpleNamespace`` or a dataclass instance's type."""
        if kind is types.SimpleNamespace:
            return True
        return any("__dataclass_fields__" in klass.__dict__ for klass in kind.__mro__)

    @staticmethod
    def _replay_instance_state(
        value: Any, cache: Optional[Dict[int, Any]] = None
    ) -> Optional[Tuple[Optional[Dict[str, Any]], Tuple[Tuple[str, Any], ...]]]:
        """The instance ``__dict__`` and slots of a plain Python object.

        A plain object is a ``SimpleNamespace`` or an instance of a class
        whose bases are all Python classes (no built-in base other than
        ``object``) with an ordinary metaclass, so its whole state is its
        ``__dict__`` and slots. Reads them through the built-in descriptors
        only, so no user code runs. None for any other object.
        """
        kind = type(value)
        if kind is types.SimpleNamespace:
            return object.__getattribute__(value, "__dict__"), ()
        if type(kind) not in _REPLAY_PLAIN_METACLASSES:
            return None
        mro = kind.__mro__
        if mro[-1] is not object:
            return None
        for klass in mro[:-1]:
            if not Model._is_python_class(klass, cache):
                return None
        instance_dict = None
        for klass in mro:
            descriptor = klass.__dict__.get("__dict__")
            if descriptor is not None:
                if type(descriptor) is not types.GetSetDescriptorType:
                    return None
                instance_dict = descriptor.__get__(value, kind)
                if type(instance_dict) is not dict:
                    return None
                break
        return instance_dict, Model._replay_slots(kind, cache)

    # Class-dict entries that only a class implemented in C has.
    _REPLAY_C_CLASS_ENTRIES = (
        types.WrapperDescriptorType,
        types.MethodDescriptorType,
        types.ClassMethodDescriptorType,
        types.BuiltinFunctionType,
    )

    @staticmethod
    def _is_python_class(klass: type, cache: Optional[Dict[Any, Any]] = None) -> bool:
        """Whether ``klass`` was written in Python, so it adds no C-level state.

        Extension types can be heap types too, so the class dict is checked:
        C methods, C slots, and C members exist only on extension types.
        """
        key = ("python", id(klass))
        if cache is not None and key in cache:
            return cache[key]
        found = bool(klass.__flags__ & _TPFLAGS_HEAPTYPE)
        if found:
            slots = klass.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            try:
                declared = set(slots)
            except TypeError:
                declared = set()
            for name, entry in klass.__dict__.items():
                kind = type(entry)
                if issubclass(kind, Model._REPLAY_C_CLASS_ENTRIES):
                    found = False
                elif kind is types.GetSetDescriptorType:
                    found = name in ("__dict__", "__weakref__")
                elif kind is types.MemberDescriptorType:
                    found = name in declared or any(
                        name.endswith(slot) for slot in declared if slot.startswith("__")
                    )
                if not found:
                    break
        if cache is not None:
            cache[key] = found
        return found

    @staticmethod
    def _replay_instance_items(
        value: Any, cache: Optional[Dict[int, Any]] = None
    ) -> Optional[List[Tuple[Tuple[str, str], Any]]]:
        """``((place, name), value)`` for each attribute of a plain object."""
        state = Model._replay_instance_state(value, cache)
        if state is None:
            return None
        instance_dict, slots = state
        items: List[Tuple[Tuple[str, str], Any]] = []
        if instance_dict is not None:
            for name, item in instance_dict.items():
                if type(name) is not str:
                    return None
                items.append((("dict", name), item))
        kind = type(value)
        for name, descriptor in slots:
            try:
                item = descriptor.__get__(value, kind)
            except AttributeError:
                continue
            items.append((("slot", name), item))
        return items

    @staticmethod
    def _restore_instance_items(
        obj: Any, items: List[Tuple[Tuple[str, str], Any]], cache: Dict[int, Any]
    ) -> None:
        """Put back the attributes :meth:`_replay_instance_items` returned."""
        state = Model._replay_instance_state(obj, cache)
        if state is None:
            raise ValueError("the object is no longer a plain object")
        instance_dict, slots = state
        desired = [(name, value) for (place, name), value in items if place == "dict"]
        if instance_dict is None:
            if desired:
                raise ValueError("the object lost its __dict__")
        else:
            current = list(instance_dict.items())
            if len(current) != len(desired) or any(
                key != name or value is not wanted
                for (key, value), (name, wanted) in zip(current, desired)
            ):
                dict.clear(instance_dict)
                dict.update(instance_dict, desired)
        wanted_slots = {name: value for (place, name), value in items if place == "slot"}
        kind = type(obj)
        for name, descriptor in slots:
            if name in wanted_slots:
                try:
                    present = descriptor.__get__(obj, kind) is wanted_slots[name]
                except AttributeError:
                    present = False
                if not present:
                    descriptor.__set__(obj, wanted_slots[name])
            else:
                try:
                    descriptor.__delete__(obj)
                except AttributeError:
                    pass

    @staticmethod
    def _replay_stream_kind(value: Any) -> Optional[str]:
        """The kind of a random stream whose state can be saved, or None."""
        kind = type(value)
        if kind is random.Random:
            return "random"
        if kind is np.random.RandomState:
            return "randomstate"
        if kind is np.random.Generator and type(value.bit_generator).__module__.startswith("numpy.random"):
            return "generator"
        return None

    @staticmethod
    def _replay_stream_state(obj: Any, kind: str) -> Any:
        if kind == "random":
            return obj.getstate()
        if kind == "generator":
            return copy.deepcopy(obj.bit_generator.state)
        return obj.get_state()

    def _replay_stream_same(self, kind: str, left: Any, right: Any) -> bool:
        if kind == "random":
            return left == right
        return self._replay_values_equal(left, right)

    @staticmethod
    def _replay_set_stream_state(obj: Any, kind: str, state: Any) -> None:
        if kind == "random":
            obj.setstate(state)
        elif kind == "generator":
            obj.bit_generator.state = copy.deepcopy(state)
        else:
            obj.set_state(state)

    def _capture_replay_world(
        self,
        remembered: Optional[Dict[int, Any]] = None,
        reasons: Optional[List[str]] = None,
        remembered_from: Any = None,
    ) -> Optional[Tuple[Any, Dict[int, Any], Tuple[Any, ...], Dict[Any, List[int]]]]:
        """Snapshot the user state a swapped replay can change.

        The snapshot covers the model's public and user-owned private
        attributes and the private attributes of every tracked agent (public
        agent attributes live in the frame), with the objects they reach.
        Containers, dataclass instances, namespaces, random streams, and
        Polars frames are recorded as the live objects plus their state, so
        a later restore can update them in place and keep aliases. Immutable
        values are kept as they are. The model, its agents, and its random
        streams are kept by identity.

        Any other object is kept by identity too. When it is a plain Python
        object, its own attributes are recorded one level deep (containers
        it holds are followed, other objects it refers to are kept by
        identity); otherwise its internal state is not recorded. These
        objects are listed in the fourth item of the result, so the
        certificate can name them.

        Returns None when the state reaches a mutable container the replay
        cannot restore (a container or array subclass, a ``bytearray``, an
        array that does not own its memory, a set or dict key that is not an
        immutable value) or a module, and appends the cause, naming the
        attribute, to ``reasons``. ``remembered`` holds the objects of an
        earlier snapshot ``remembered_from``, which are captured even when
        no attribute reaches them any more.
        """
        nodes: Dict[int, Any] = {}
        anchors = self._replay_anchors()
        # Locals, not attribute lookups: this runs for every private value of
        # every agent, and the model's __getattribute__ is not free.
        scalar_types = Model._REPLAY_SCALAR_TYPES
        dict_types = Model._REPLAY_DICT_TYPES
        is_leaf = Model._is_replay_leaf
        is_code = Model._is_replay_code
        slots_cache: Dict[int, Any] = {}
        # (on the model, attribute, type name, mode) -> [count, first owner].
        identity_notes: Dict[Tuple[bool, str, str, str], List[int]] = {}
        current: List[Any] = [None]

        def note(value: Any, mode: str) -> None:
            key = current[0]
            if key is None:
                return
            index, _place, name = key
            entry = identity_notes.setdefault(
                (index < 0, name, type(value).__name__, mode), [0, index]
            )
            entry[0] += 1

        def capture_child(child: Any) -> Tuple[str, Any]:
            """Capture an attribute of an object kept by identity."""
            mark = len(nodes)
            try:
                return capture(child, True)
            except (_ReplayUnsupported, RecursionError):
                if len(nodes) > mark:
                    for oid in list(nodes)[mark:]:
                        del nodes[oid]
                return ("external", child)

        def capture(value: Any, opaque: bool = False) -> Tuple[str, Any]:
            kind = type(value)
            if kind in scalar_types:
                return ("leaf", value)
            oid = id(value)
            if oid in nodes:
                return ("node", oid)
            if oid in anchors:
                return ("anchor", value)
            # Containers are matched by exact type: a subclass can hold state
            # outside its items or override the methods a restore would call.
            if kind is list or kind is deque:
                nodes[oid] = ("list", value, ())
                nodes[oid] = ("list", value, tuple([
                    ("leaf", item) if type(item) in scalar_types else capture(item, opaque)
                    for item in value
                ]))
            elif kind in dict_types:
                if kind is AttrDict or kind is Counter or kind is OrderedDict:
                    if object.__getattribute__(value, "__dict__"):
                        raise _ReplayUnsupported(f"a {kind.__name__} with instance attributes")
                nodes[oid] = ("dict", value, ((), None))
                pairs = []
                for key, item in value.items():
                    if not is_leaf(key):
                        raise _ReplayUnsupported("a dict key that is not an immutable value")
                    pairs.append((key, capture(item, opaque)))
                factory = None
                if kind is defaultdict:
                    try:
                        factory = capture(value.default_factory, opaque)
                    except _ReplayUnsupported as exc:
                        raise _ReplayUnsupported(
                            f"a defaultdict whose default_factory is {exc}"
                        ) from None
                nodes[oid] = ("dict", value, (tuple(pairs), factory))
            elif kind is set:
                if not all(is_leaf(item) for item in value):
                    raise _ReplayUnsupported("a set with an item that is not an immutable value")
                nodes[oid] = ("set", value, frozenset(value))
            elif is_leaf(value) or is_code(value):
                return ("leaf", value)
            elif Model._is_replay_tuple(value):
                nodes[oid] = ("tuple", value, ())
                nodes[oid] = ("tuple", value, tuple(capture(item, opaque) for item in value))
            elif isinstance(value, np.ndarray):
                root = self._replay_array_root(value)
                if root is value:
                    nodes[oid] = ("ndarray", value, self._array_snapshot(value))
                else:
                    nodes[oid] = ("view", value, (
                        capture(root, opaque), value.flags.writeable, value.shape, value.strides
                    ))
            elif kind is types.FunctionType:
                # A function's own state: code, defaults, attributes, closure
                # cells, and names. The module globals it reaches are not
                # part of the snapshot.
                nodes[oid] = ("function", value, ())
                nodes[oid] = ("function", value, tuple(
                    capture(getattr(value, name), opaque)
                    for name in self._REPLAY_FUNCTION_FIELDS
                ))
            elif kind is types.CellType:
                nodes[oid] = ("cell", value, None)
                try:
                    contents = value.cell_contents
                except ValueError:
                    nodes[oid] = ("cell", value, None)
                else:
                    nodes[oid] = ("cell", value, capture(contents, opaque))
            elif kind is types.MethodType or kind is types.BuiltinMethodType:
                owner = value.__self__
                func = value.__func__ if kind is types.MethodType else ("leaf", value.__name__)
                nodes[oid] = ("method", value, ())
                nodes[oid] = ("method", value, (
                    capture(owner, opaque),
                    capture(func, opaque) if kind is types.MethodType else func,
                ))
            elif kind is staticmethod or kind is classmethod:
                nodes[oid] = ("wrapper", value, ())
                nodes[oid] = ("wrapper", value, (
                    capture(value.__func__, opaque),
                    capture(object.__getattribute__(value, "__dict__"), opaque),
                ))
            elif kind is property:
                nodes[oid] = ("wrapper", value, ())
                nodes[oid] = ("wrapper", value, (
                    capture(value.fget, opaque),
                    capture(value.fset, opaque),
                    capture(value.fdel, opaque),
                ))
            elif isinstance(value, type):
                # A class's own state: its namespace, bases, and names. Only
                # plain classes: a metaclass can keep state elsewhere or
                # intercept the restore, so such a class is kept by identity.
                if type(value) is not type:
                    if not opaque:
                        note(value, "identity")
                    return ("external", value)
                nodes[oid] = ("class", value, ((), ("leaf", ()), "", ""))
                namespace = tuple(
                    (name, capture(item, opaque)) for name, item in list(value.__dict__.items())
                )
                nodes[oid] = ("class", value, (
                    namespace, capture(value.__bases__, opaque), value.__name__, value.__qualname__
                ))
            elif isinstance(value, types.ModuleType):
                raise _ReplayUnsupported("a module")
            elif issubclass(kind, Model._REPLAY_UNSUPPORTED_CONTAINERS):
                raise _ReplayUnsupported(f"a {kind.__name__} object")
            else:
                stream = Model._replay_stream_kind(value)
                if stream is not None:
                    nodes[oid] = ("stream", value, (stream, Model._replay_stream_state(value, stream)))
                elif kind is pl.DataFrame or kind is pl.Series:
                    nodes[oid] = ("frame", value, value.clone())
                elif opaque:
                    return ("external", value)
                else:
                    items = (
                        None if issubclass(kind, _REPLAY_OPAQUE_TYPES)
                        else Model._replay_instance_items(value, slots_cache)
                    )
                    if items is None:
                        note(value, "identity")
                        return ("external", value)
                    if Model._is_replay_record(kind):
                        # Dataclasses and namespaces are values: their fields
                        # are snapshotted like the fields of any container.
                        nodes[oid] = ("object", value, ((), False))
                        nodes[oid] = ("object", value, (
                            tuple((key, capture(item)) for key, item in items), False
                        ))
                    else:
                        note(value, "attributes")
                        nodes[oid] = ("object", value, ((), True))
                        nodes[oid] = ("object", value, (
                            tuple((key, capture_child(item)) for key, item in items), True
                        ))
            return ("node", oid)

        bindings: List[Tuple[Tuple[int, str, str], Tuple[str, Any]]] = []
        owners = tuple(self._agent_objects())

        def label(key: Tuple[int, str, str]) -> str:
            index, _place, name = key
            if index < 0:
                return f"model attribute {name!r}"
            return f"agent {self._replay_owner_id(owners, index)!r} attribute {name!r}"

        def bind(key: Tuple[int, str, str], value: Any) -> None:
            if type(value) in scalar_types:
                bindings.append((key, ("leaf", value)))
                return
            current[0] = key
            try:
                bindings.append((key, capture(value)))
            except _ReplayUnsupported as exc:
                raise _ReplayUnsupported(f"{label(key)} holds {exc}") from None
            except RecursionError:
                raise _ReplayUnsupported(
                    f"{label(key)} is nested too deeply to snapshot"
                ) from None
            except Exception as exc:
                raise _ReplayUnsupported(
                    f"{label(key)} could not be snapshotted ({type(exc).__name__})"
                ) from None

        try:
            for index, owner in enumerate((self, *owners), start=-1):
                attrs = object.__getattribute__(owner, "__dict__")
                owned = self._is_replay_model_attr if index < 0 else Model._is_replay_agent_attr
                for name, value in list(attrs.items()):
                    if owned(name):
                        bind((index, "dict", name), value)
                for name, descriptor in Model._replay_slots(type(owner), slots_cache):
                    if owned(name):
                        try:
                            value = descriptor.__get__(owner, type(owner))
                        except AttributeError:
                            continue
                        bind((index, "slot", name), value)
            current[0] = None
            if remembered:
                for oid, obj in remembered.items():
                    if oid not in nodes:
                        try:
                            capture(obj)
                        except (_ReplayUnsupported, RecursionError) as exc:
                            where = (
                                self._replay_node_label(remembered_from, oid)
                                if remembered_from is not None else None
                            )
                            raise _ReplayUnsupported(
                                f"{where or 'model or agent state'} held an object at "
                                f"step entry that now holds {exc or 'state nested too deeply'}"
                            ) from None
            return (
                tuple(sorted(bindings, key=lambda item: item[0])), nodes, owners, identity_notes
            )
        except _ReplayUnsupported as exc:
            if reasons is not None:
                reasons.append(f"{exc}, which the replay snapshot cannot represent")
            return None
        except RecursionError:
            if reasons is not None:
                reasons.append("model or agent state is nested too deeply to snapshot")
            return None
        except Exception as exc:
            if reasons is not None:
                reasons.append(f"model or agent state could not be snapshotted ({type(exc).__name__})")
            return None

    @staticmethod
    def _replay_world_objects(world: Any) -> Dict[int, Any]:
        if world is None:
            return {}
        return {oid: node[1] for oid, node in world[1].items()}

    @staticmethod
    def _replay_resolve(ref: Tuple[str, Any], nodes: Dict[int, Any]) -> Any:
        kind, payload = ref
        if kind in ("leaf", "anchor", "external"):
            return payload
        return nodes[payload][1]

    def _restore_replay_world(self, world: Any, failures: Optional[List[str]] = None) -> bool:
        """Rebind attributes and update supported objects in place.

        In-place updates keep aliases, including objects stored in tuples.
        Objects kept by identity are rebound, not restored, except for the
        own attributes of a plain object. On failure, the first attribute
        that could not be restored is appended to ``failures``.
        """
        if world is None:
            return False
        bindings, nodes, owners = world[0], world[1], world[2]

        def fail(oid: Optional[int]) -> None:
            if failures is not None and not failures:
                found = self._replay_node_label(world, oid) if oid is not None else None
                failures.append(found or "model or agent attribute bindings")

        resolved: Dict[int, Any] = {}
        try:
            for oid, node in nodes.items():
                kind, _obj, payload = node
                if kind == "list":
                    resolved[oid] = [
                        self._replay_resolve(ref, nodes) for ref in payload
                    ]
                elif kind == "dict":
                    pairs, factory = payload
                    resolved[oid] = (
                        [(key, self._replay_resolve(ref, nodes)) for key, ref in pairs],
                        None if factory is None else self._replay_resolve(factory, nodes),
                    )
                elif kind == "function":
                    resolved[oid] = [self._replay_resolve(ref, nodes) for ref in payload]
                elif kind == "cell" and payload is not None:
                    resolved[oid] = self._replay_resolve(payload, nodes)
                elif kind == "class":
                    namespace, bases, _name, _qualname = payload
                    resolved[oid] = (
                        {name: self._replay_resolve(ref, nodes) for name, ref in namespace},
                        self._replay_resolve(bases, nodes),
                    )
                elif kind == "object":
                    resolved[oid] = [
                        (key, self._replay_resolve(ref, nodes)) for key, ref in payload[0]
                    ]
        except Exception:
            fail(None)
            return False

        ok = True
        slots_cache: Dict[int, Any] = {}
        # Views go last: their contents come back with the owning array, and
        # a view's write flag can only be set once its owner is writable.
        ordered = sorted(nodes.items(), key=lambda item: item[1][0] == "view")
        for oid, node in ordered:
            kind, obj, payload = node
            try:
                # Every container here has an exact built-in type, so these
                # calls run no user code.
                if kind == "list":
                    obj.clear()
                    obj.extend(resolved[oid])
                elif kind == "dict":
                    items, factory = resolved[oid]
                    if type(obj) is OrderedDict:
                        obj.clear()
                        obj.update(items)
                    else:
                        # Counter.update would add counts instead of replacing them.
                        dict.clear(obj)
                        dict.update(obj, items)
                    if payload[1] is not None:
                        obj.default_factory = factory
                elif kind == "set":
                    obj.clear()
                    obj.update(payload)
                elif kind == "ndarray":
                    if not self._restore_array_in_place(obj, payload):
                        ok = False
                        fail(oid)
                elif kind == "view":
                    _root, writeable, shape, strides = payload
                    if obj.shape != shape:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", DeprecationWarning)
                            obj.shape = shape
                    if obj.strides != strides:
                        raise ValueError("a view's strides cannot be restored")
                    if obj.flags.writeable != writeable:
                        obj.setflags(write=writeable)
                elif kind == "function":
                    for name, value in zip(self._REPLAY_FUNCTION_FIELDS, resolved[oid]):
                        if name != "__closure__" and getattr(obj, name) is not value:
                            setattr(obj, name, value)
                elif kind == "cell":
                    if payload is not None:
                        obj.cell_contents = resolved[oid]
                    else:
                        try:
                            del obj.cell_contents
                        except ValueError:
                            pass
                elif kind == "class":
                    namespace, bases = resolved[oid]
                    _namespace, _bases, name, qualname = payload
                    current = obj.__dict__
                    for key in [key for key in current if key not in namespace]:
                        type.__delattr__(obj, key)
                    for key, value in namespace.items():
                        if key not in current or current[key] is not value:
                            type.__setattr__(obj, key, value)
                    if obj.__bases__ != bases:
                        obj.__bases__ = bases
                    if obj.__name__ != name:
                        obj.__name__ = name
                    if obj.__qualname__ != qualname:
                        obj.__qualname__ = qualname
                elif kind == "object":
                    self._restore_instance_items(obj, resolved[oid], slots_cache)
                elif kind == "stream":
                    stream, state = payload
                    if not self._replay_stream_same(
                        stream, self._replay_stream_state(obj, stream), state
                    ):
                        self._replay_set_stream_state(obj, stream, state)
                elif kind == "frame":
                    if not _replay_frames_same(obj, payload):
                        inner = "_df" if type(obj) is pl.DataFrame else "_s"
                        object.__setattr__(obj, inner, getattr(payload.clone(), inner))
                elif kind not in ("tuple", "method", "wrapper"):
                    ok = False
                    fail(oid)
            except Exception:
                ok = False
                fail(oid)
        try:
            desired = {
                key: self._replay_resolve(ref, nodes) for key, ref in bindings
            }
            targets = (self, *owners)
            for index, owner in enumerate(targets, start=-1):
                live = object.__getattribute__(owner, "__dict__")
                owned = self._is_replay_model_attr if index < 0 else self._is_replay_agent_attr
                for name in [
                    name for name in live
                    if owned(name) and (index, "dict", name) not in desired
                ]:
                    del live[name]
                for name, descriptor in self._replay_slots(type(owner), slots_cache):
                    if owned(name) and (index, "slot", name) not in desired:
                        try:
                            descriptor.__delete__(owner)
                        except AttributeError:
                            pass
            for (index, where, name), value in desired.items():
                owner = targets[index + 1]
                if where == "slot":
                    dict(self._replay_slots(type(owner), slots_cache))[name].__set__(owner, value)
                else:
                    object.__getattribute__(owner, "__dict__")[name] = value
        except Exception:
            fail(None)
            return False
        return ok

    @staticmethod
    def _array_snapshot(value: np.ndarray) -> np.ndarray:
        """Copy ``value`` with its write flag for :meth:`_restore_array_in_place`."""
        saved = value.copy()
        saved.setflags(write=value.flags.writeable)
        return saved

    def _restore_array_in_place(self, obj: Any, saved: Any) -> bool:
        """Write ``saved`` back into ``obj``. False when the dtype changed."""
        if (
            not isinstance(obj, np.ndarray)
            or not isinstance(saved, np.ndarray)
            or obj.dtype != saved.dtype
        ):
            return False
        if not self._replay_values_equal(obj, saved):
            if not obj.flags.writeable:
                obj.setflags(write=True)
            if obj.shape != saved.shape:
                obj.resize(saved.shape, refcheck=False)
            obj[...] = saved
        if obj.flags.writeable != saved.flags.writeable:
            obj.setflags(write=saved.flags.writeable)
        return True

    def _replay_values_equal(self, left: Any, right: Any) -> bool:
        """Compare leaves and containers without using an array's truth value.

        Values of different types differ, so ``1 -> True`` is a change. A
        NaN equals a NaN (also in each part of a complex number), NaT equals
        NaT, and ``-0.0`` differs from ``0.0``, so an untouched value is
        never a change. Enum members compare by identity, and the immutable
        standard-library values by an exact key, so no user ``__eq__`` runs.
        """
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
                return False
            return _replay_arrays_same(left, right)
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict):
                return False
            if set(left) != set(right):
                return False
            return all(
                self._replay_values_equal(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                return False
            return all(
                self._replay_values_equal(item, other)
                for item, other in zip(left, right)
            )
        kind = type(left)
        if kind is not type(right):
            return False
        if kind in _REPLAY_EXACT_EQUAL:
            return left == right
        if kind is float:
            return _replay_float_same(left, right)
        if kind is complex:
            return (
                _replay_float_same(left.real, right.real)
                and _replay_float_same(left.imag, right.imag)
            )
        if self._is_numpy_scalar(left):
            return _replay_numpy_scalar_same(left, right)
        value_key = _REPLAY_VALUE_KEYS.get(kind)
        if value_key is not None:
            return value_key(left) == value_key(right)
        if isinstance(kind, enum.EnumMeta):
            return left is right
        if kind is frozenset:
            return len(left) == len(right) and _replay_typed_set(left) == _replay_typed_set(right)
        compared = left == right
        if isinstance(compared, np.ndarray):
            return bool(np.array_equal(left, right, equal_nan=True))
        return bool(compared)

    def _replay_compare(
        self,
        original_post: pl.DataFrame,
        post_world: Any,
        entry_world: Any,
        recorded: Any,
    ) -> Tuple[Optional[bool], Optional[str]]:
        try:
            frame_differs = not bool(original_post.equals(self._agents_frame()))
        except Exception as exc:
            return None, _replay_failure_reason(exc)
        reasons: List[str] = []
        replay_world = self._capture_replay_world(
            self._replay_world_objects(entry_world), reasons, entry_world
        )
        if replay_world is None:
            return None, "replay state could not be captured" + (
                f" ({reasons[0]})" if reasons else ""
            )
        differs: List[str] = []
        try:
            same_world = self._replay_worlds_equal(post_world, replay_world, differs)
        except _ReplayAliasMismatch as exc:
            where = str(exc)
            return None, "alias relationships are not one-to-one" + (
                f" ({where})" if where else ""
            )
        if same_world is None:
            return None, "replay state could not be compared" + (
                f" ({differs[0]})" if differs else ""
            )
        if frame_differs or not same_world:
            return True, None
        if recorded is None:
            return None, "recorded step values could not be captured"
        try:
            return not self._replay_values_equal(recorded, self._current_step_data), None
        except Exception as exc:
            return None, _replay_failure_reason(exc)

    def _replay_refs_equal(
        self,
        left: Tuple[str, Any],
        left_nodes: Dict[int, Any],
        right: Tuple[str, Any],
        right_nodes: Dict[int, Any],
        forward: Dict[int, int],
        inverse: Dict[int, int],
    ) -> bool:
        if left[0] in ("anchor", "external") or right[0] in ("anchor", "external"):
            return left[0] == right[0] and left[1] is right[1]
        if left[0] == "leaf" and right[0] == "leaf":
            left_value, right_value = left[1], right[1]
            kind = type(left_value)
            if kind is type(right_value) and kind in _REPLAY_EXACT_EQUAL:
                return left_value == right_value
            return self._replay_values_equal(left_value, right_value)
        if left[0] != "node" or right[0] != "node":
            return False
        left_id = left[1]
        right_id = right[1]
        previous = forward.get(left_id)
        if previous is not None:
            if previous != right_id:
                raise _ReplayAliasMismatch()
            return True
        if right_id in inverse:
            raise _ReplayAliasMismatch()
        forward[left_id] = right_id
        inverse[right_id] = left_id
        left_node = left_nodes[left_id]
        right_node = right_nodes[right_id]
        if left_node[0] != right_node[0] or type(left_node[1]) is not type(right_node[1]):
            return False
        kind = left_node[0]
        if kind == "ndarray":
            left_array, right_array = left_node[2], right_node[2]
            return (left_array.flags.writeable == right_array.flags.writeable
                    and self._replay_values_equal(left_array, right_array))
        if kind == "set":
            return _replay_typed_set(left_node[2]) == _replay_typed_set(right_node[2])
        if kind == "view":
            return left_node[2][1:] == right_node[2][1:] and self._replay_refs_equal(
                left_node[2][0], left_nodes, right_node[2][0], right_nodes, forward, inverse
            )
        if kind == "cell":
            if left_node[2] is None or right_node[2] is None:
                return left_node[2] is None and right_node[2] is None
            return self._replay_refs_equal(
                left_node[2], left_nodes, right_node[2], right_nodes, forward, inverse
            )
        if kind == "class":
            (left_space, left_bases, *left_names), (right_space, right_bases, *right_names) = (
                left_node[2], right_node[2]
            )
            return (
                left_names == right_names
                and [name for name, _ in left_space] == [name for name, _ in right_space]
                and self._replay_refs_equal(
                    left_bases, left_nodes, right_bases, right_nodes, forward, inverse
                )
                and all(
                    self._replay_refs_equal(ref, left_nodes, other, right_nodes, forward, inverse)
                    for (_, ref), (_, other) in zip(left_space, right_space)
                )
            )
        if kind == "object":
            (left_items, left_kept), (right_items, right_kept) = left_node[2], right_node[2]
            return (
                left_kept == right_kept
                and [key for key, _ in left_items] == [key for key, _ in right_items]
                and all(
                    self._replay_refs_equal(ref, left_nodes, other, right_nodes, forward, inverse)
                    for (_, ref), (_, other) in zip(left_items, right_items)
                )
            )
        if kind == "stream":
            (left_kind, left_state), (right_kind, right_state) = left_node[2], right_node[2]
            return left_kind == right_kind and self._replay_stream_same(
                left_kind, left_state, right_state
            )
        if kind == "frame":
            return _replay_frames_same(left_node[2], right_node[2])
        if kind in ("list", "tuple", "method", "function", "wrapper"):
            left_children = left_node[2]
            right_children = right_node[2]
            if len(left_children) != len(right_children):
                return False
            return all(
                self._replay_refs_equal(
                    item, left_nodes, other, right_nodes, forward, inverse
                )
                for item, other in zip(left_children, right_children)
            )
        if kind == "dict":
            (left_pairs, left_factory), (right_pairs, right_factory) = left_node[2], right_node[2]
            if (left_factory is None) != (right_factory is None):
                return False
            if left_factory is not None and not self._replay_refs_equal(
                left_factory, left_nodes, right_factory, right_nodes, forward, inverse
            ):
                return False
            return len(left_pairs) == len(right_pairs) and all(
                self._replay_values_equal(key, other_key)
                and self._replay_refs_equal(ref, left_nodes, other_ref, right_nodes, forward, inverse)
                for (key, ref), (other_key, other_ref) in zip(left_pairs, right_pairs)
            )
        return False

    def _replay_worlds_equal(
        self, left: Any, right: Any, differs: Optional[List[str]] = None
    ) -> Optional[bool]:
        """Compare two snapshots. None when they cannot be compared.

        ``differs`` receives the attribute where they first differ. An alias
        mismatch raises :class:`_ReplayAliasMismatch` naming the attribute.
        """
        try:
            if len(left[2]) != len(right[2]) or any(
                a is not b for a, b in zip(left[2], right[2])
            ):
                return None
            left_bindings = dict(left[0])
            right_bindings = dict(right[0])
            if set(left_bindings) != set(right_bindings):
                if differs is not None:
                    key = min(set(left_bindings) ^ set(right_bindings))
                    differs.append(self._replay_binding_label(left, key))
                return False
            forward: Dict[int, int] = {}
            inverse: Dict[int, int] = {}
            for name in left_bindings:
                try:
                    same = self._replay_refs_equal(
                        left_bindings[name],
                        left[1],
                        right_bindings[name],
                        right[1],
                        forward,
                        inverse,
                    )
                except _ReplayAliasMismatch:
                    raise _ReplayAliasMismatch(self._replay_binding_label(left, name)) from None
                if not same:
                    if differs is not None:
                        differs.append(self._replay_binding_label(left, name))
                    return False
            shared = set(left[1]) & set(right[1])
            for oid in shared:
                try:
                    same = self._replay_refs_equal(
                        ("node", oid),
                        left[1],
                        ("node", oid),
                        right[1],
                        forward,
                        inverse,
                    )
                except _ReplayAliasMismatch:
                    raise _ReplayAliasMismatch(
                        self._replay_node_label(left, oid) or ""
                    ) from None
                if not same:
                    if differs is not None:
                        differs.append(
                            self._replay_node_label(left, oid)
                            or "an object the state held at step entry"
                        )
                    return False
            return True
        except _ReplayAliasMismatch:
            raise
        except Exception:
            return None

    def _agent_objects(self) -> List[Any]:
        return list(getattr(self.agents, "_agent_objects", ()) or ())

    def _snapshot_agent_attrs(
        self,
    ) -> List[Tuple[Any, Dict[str, Any], List[Tuple[Any, Any]]]]:
        """Save every instance attribute of every tracked agent.

        Public values are deep-copied. Private values and the framework's
        bindings keep the forward objects, so a replay that adds, rebinds, or
        deletes a private attribute leaves no trace in the forward run.
        Private lists, dicts, sets, and numeric arrays also keep a copy of
        their top-level contents, which the restore writes back in place.
        """
        saved: List[Tuple[Any, Dict[str, Any], List[Tuple[Any, Any]]]] = []
        for agent in self._agent_objects():
            attrs: Dict[str, Any] = {}
            contents: List[Tuple[Any, Any]] = []
            for key, value in object.__getattribute__(agent, "__dict__").items():
                if key == "_ambr_container_cells":
                    attrs[key] = value.copy()
                    continue
                if key in ("model", "id", "p"):
                    attrs[key] = value
                    continue
                if key.startswith("_"):
                    attrs[key] = value
                    self._keep_private_contents(value, contents)
                    continue
                try:
                    attrs[key] = copy.deepcopy(value)
                except Exception:
                    attrs[key] = value
            saved.append((agent, attrs, contents))
        return saved

    def _restore_agent_attrs(
        self, saved: List[Tuple[Any, Dict[str, Any], List[Tuple[Any, Any]]]]
    ) -> bool:
        """Put back what :meth:`_snapshot_agent_attrs` saved.

        Returns False when a private array could not be written back.
        """
        ok = True
        for agent, attrs, contents in saved:
            live = object.__getattribute__(agent, "__dict__")
            live.clear()
            live.update(attrs)
            ok = self._write_back_contents(contents) and ok
        return ok

    def _keep_private_contents(self, value: Any, contents: List[Tuple[Any, Any]]) -> None:
        """Copy the top-level contents of a container that can be restored in place."""
        if type(value) in (list, dict, set):
            contents.append((value, value.copy()))
        elif isinstance(value, np.ndarray) and self._ndarray_replay_supported(value):
            contents.append((value, self._array_snapshot(value)))

    def _write_back_contents(self, contents: List[Tuple[Any, Any]]) -> bool:
        """Write saved top-level contents back into the same objects."""
        ok = True
        for obj, items in contents:
            try:
                if isinstance(obj, np.ndarray):
                    ok = self._restore_array_in_place(obj, items) and ok
                elif isinstance(obj, list):
                    obj[:] = items
                else:
                    obj.clear()
                    obj.update(items)
            except Exception:
                ok = False
        return ok

    def _refresh_agent_cache(self, df: pl.DataFrame) -> None:
        """Load already-present agent attributes from ``df`` without queueing."""
        if "id" not in df.columns:
            return
        rows = {agent_id: index for index, agent_id in enumerate(df["id"].to_list())}
        for agent in self._agent_objects():
            row = rows.get(agent.id)
            if row is None:
                continue
            attrs = object.__getattribute__(agent, "__dict__")
            attrs.pop("_ambr_container_cells", None)
            for key in list(attrs):
                if key in ("model", "id", "p") or str(key).startswith("_"):
                    continue
                if key not in df.columns:
                    continue
                value = df[key][row]
                if isinstance(value, pl.Series):
                    value = value.to_list()
                object.__setattr__(agent, key, value)

    def _underlying_generator(self, obj: Any) -> Any:
        """Return a NumPy/CuPy Generator hidden behind a thin wrapper, if any."""
        if obj is None:
            return None
        if hasattr(obj, "bit_generator"):
            return obj
        inner = getattr(obj, "_rng", None)
        if inner is not None and hasattr(inner, "bit_generator"):
            return inner
        return None

    def _capture_rng_state(self) -> Optional[_RngSnapshot]:
        """Snapshot every random stream the step can draw from.

        The snapshot keeps the stream objects and the bindings that point at
        them, so a later step can put those objects back before restoring
        state. The legacy ``nprandom`` adapter is one of those bindings, along
        with its inner generator. Returns None when a stream has no
        restorable state. The snapshot does not advance any generator.
        """
        try:
            streams: Dict[int, Tuple[str, int, Any, Any]] = {}
            bindings: List[Tuple[str, str, int]] = []

            def keep(kind: str, obj: Any, state: Any) -> int:
                oid = id(obj)
                if oid not in streams:
                    streams[oid] = (kind, oid, obj, state)
                return oid

            def keep_generator(obj: Any) -> int:
                return keep(
                    "generator", obj, copy.deepcopy(obj.bit_generator.state)
                )

            random_obj = self.__dict__.get("random")
            if not isinstance(random_obj, random.Random):
                return None
            bindings.append((
                "model",
                "random",
                keep("random", random_obj, copy.deepcopy(random_obj.getstate())),
            ))

            host = self.__dict__.get("_host_rng")
            if host is not None:
                if not hasattr(host, "bit_generator"):
                    return None
                bindings.append(("model", "_host_rng", keep_generator(host)))

            execution = getattr(self, "_execution", None)
            device = (
                getattr(execution, "device_rng", None) if execution is not None else None
            )
            if isinstance(device, DeviceRNG):
                oid = keep("device_rng", device, device._snapshot_state())
                bindings.append(("execution", "device_rng", oid))
            elif device is not None:
                device_gen = self._underlying_generator(device)
                if device_gen is None:
                    return None
                gen_oid = keep_generator(device_gen)
                dev_oid = keep("binding", device, None)
                bindings.append(("execution", "device_rng", dev_oid))
                if getattr(device, "_rng", None) is device_gen and dev_oid != gen_oid:
                    bindings.append(("device_inner", "_rng", gen_oid))

            nprandom = getattr(self, "nprandom", None)
            if nprandom is not None:
                # The adapter attribute has to be restored before its inner
                # generator. Once the attribute is missing, the inner binding
                # has nowhere to go.
                npr_inner = getattr(nprandom, "_rng", None)
                if npr_inner is not None and not hasattr(npr_inner, "bit_generator"):
                    return None
                adapter_oid = keep("binding", nprandom, None)
                bindings.append(("model", "nprandom", adapter_oid))
                if npr_inner is not None:
                    gen_oid = keep_generator(npr_inner)
                    if gen_oid != adapter_oid:
                        bindings.append(("nprandom", "_rng", gen_oid))
            return _RngSnapshot(tuple(streams.values()), tuple(bindings))
        except Exception:
            return None

    def _rng_same_stream_ids(
        self,
        entry: Optional[_RngSnapshot],
        post: Optional[_RngSnapshot],
    ) -> bool:
        if entry is None or post is None:
            return False
        return (
            [item[1] for item in entry.streams] == [item[1] for item in post.streams]
            and entry.bindings == post.bindings
        )

    def _rng_states_equal(
        self, before: Optional[_RngSnapshot], after: Optional[_RngSnapshot]
    ) -> Optional[bool]:
        """Whether two stream snapshots hold the same states. None if unknown."""
        if before is None or after is None or not self._rng_same_stream_ids(before, after):
            return None
        try:
            return all(
                self._replay_values_equal(left[3], right[3])
                for left, right in zip(before.streams, after.streams)
            )
        except Exception:
            return None

    @contextmanager
    def _watch_sweep_rng(self):
        """Check model-owned streams around agent calls, excluding scheduler draws.

        Also records the state at the sweep's exit for the certificate.
        """
        monitor = self._contract
        if not (monitor.active or monitor._in_replay) or monitor._current_activation is not None:
            yield
            return
        before = self._capture_rng_state()
        try:
            yield
        finally:
            after = self._capture_rng_state()
            monitor.record_sweep_rng_check(self._rng_states_equal(before, after))
            monitor._end_sweep(self, after)

    def _sweep_boundary(self, rng_state: Any = None, *, capture_rng: bool = True) -> Any:
        """The committed frame, pending writes, and streams at a sweep boundary."""
        try:
            frame = self.population.data.clone()
            pending = {
                column: dict(values) for column, values in self._pending_writes.items()
            }
        except Exception:
            return None
        if rng_state is None and capture_rng:
            rng_state = self._capture_rng_state()
        return frame, pending, rng_state

    @staticmethod
    def _frame_with_pending(frame: Any, pending: Dict[str, Dict[Any, Any]]) -> Any:
        """``frame`` with buffered writes applied, as a flush would apply them.

        None when a write targets a column or ID the frame does not have.
        """
        if not pending:
            return frame
        if "id" not in frame.columns:
            return None
        rows = {aid: index for index, aid in enumerate(frame["id"].to_list())}
        columns = []
        for column, values in pending.items():
            if column not in frame.columns:
                return None
            cells = frame[column].to_list()
            for aid, value in values.items():
                row = rows.get(aid)
                if row is None:
                    return None
                cells[row] = value
            columns.append(pl.Series(column, cells, strict=False))
        return frame.with_columns(columns)

    @staticmethod
    def _frames_equal(left: Any, right: Any) -> bool:
        try:
            return (
                left is not None
                and right is not None
                and left.schema == right.schema
                and bool(left.equals(right))
            )
        except Exception:
            return False

    def _step_effects_outside_sweep(self, entry: Any, exit: Any) -> Optional[str]:
        """Why the step did more than its verified sweep, or None.

        Compares the agent frame and the random streams at step entry with
        the sweep's entry, and at the sweep's exit with step end, so an
        in-place frame edit, a ``Population`` write, or a draw before or after
        the sweep is found whichever path made it. Also checks the agent
        roster and the values recorded with :meth:`record_model`.
        """
        if entry is None or exit is None:
            return "The state at the verified sweep's entry or exit was not recorded."
        if not self._replay_roster_matches(getattr(self, "_pre_step_roster", None)):
            return "The agent roster or an agent's ID binding changed during the step."
        recorded = sorted(str(key) for key in (self._current_step_data or {}) if key != "t")
        if recorded:
            return (
                f"The step recorded model output ({', '.join(recorded)}) outside the "
                f"verified activation sweep."
            )
        entry_frame, entry_pending, entry_rng = entry
        exit_frame, exit_pending, exit_rng = exit
        final = self._frame_with_pending(self.population.data, self._pending_writes)
        if (
            entry_pending
            or not self._frames_equal(entry_frame, getattr(self, "_pre_step_state", None))
            or not self._frames_equal(self._frame_with_pending(exit_frame, exit_pending), final)
        ):
            return "The agent frame changed outside the verified activation sweep."
        if not (
            self._rng_states_equal(getattr(self, "_pre_step_rng", None), entry_rng)
            and self._rng_states_equal(exit_rng, self._capture_rng_state())
        ):
            return "A shared random stream changed outside the verified activation sweep."
        return None

    def _rng_bound(self, place: str, name: str) -> Any:
        if place == "model":
            return self.__dict__.get(name)
        if place == "execution":
            execution = getattr(self, "_execution", None)
            if execution is None:
                return None
            return getattr(execution, name, None)
        if place == "nprandom":
            nprandom = getattr(self, "nprandom", None)
            if nprandom is None:
                return None
            return getattr(nprandom, name, None)
        if place == "device_inner":
            execution = getattr(self, "_execution", None)
            device = (
                getattr(execution, "device_rng", None) if execution is not None else None
            )
            if device is None:
                return None
            return getattr(device, name, None)
        return None

    def _restore_rng_state(self, saved: Optional[_RngSnapshot]) -> bool:
        """Rebind the snapshotted stream objects, then restore their states.

        Returns False when a binding cannot be put back or a state cannot be
        applied. Does not draw.
        """
        if saved is None:
            return False
        objects = {oid: obj for _kind, oid, obj, _state in saved.streams}
        try:
            for place, name, oid in saved.bindings:
                obj = objects.get(oid)
                if obj is None:
                    return False
                if place == "model":
                    self.__dict__[name] = obj
                elif place == "execution":
                    execution = getattr(self, "_execution", None)
                    if execution is None:
                        return False
                    setattr(execution, name, obj)
                elif place == "nprandom":
                    nprandom = getattr(self, "nprandom", None)
                    if nprandom is None:
                        return False
                    setattr(nprandom, name, obj)
                elif place == "device_inner":
                    execution = getattr(self, "_execution", None)
                    device = (
                        getattr(execution, "device_rng", None)
                        if execution is not None else None
                    )
                    if device is None:
                        return False
                    setattr(device, name, obj)
                else:
                    return False
            for kind, _oid, obj, state in saved.streams:
                if kind == "binding":
                    continue
                if kind == "device_rng":
                    obj._restore_state(state)
                    continue
                payload = copy.deepcopy(state)
                if kind == "random":
                    obj.setstate(payload)
                elif kind == "generator":
                    obj.bit_generator.state = payload
                else:
                    return False
            for place, name, oid in saved.bindings:
                if self._rng_bound(place, name) is not objects[oid]:
                    return False
            return True
        except Exception:
            return False

    def _fast_path_is_eligible(
        self,
        config: Any,
        contract: str,
        fast_runner: Any,
        fast_setup: Any,
    ) -> bool:
        """Return whether the private loop may run under this configuration."""
        return bool(
            config.device == "gpu"
            and config.mode == "vectorized"
            and contract == "off"
            and not self._show_progress
            and not self.record_initial
            and not self.agent_reporters
            and not self.model_reporters
            and callable(fast_runner)
            and callable(fast_setup)
            and self._fast_path_approval is not None
        )

    def run(
        self,
        steps: Optional[int] = None,
        contract: str = "off",
        mode: Optional[str] = None,
        device: Optional[str] = None,
        backend: Optional[str] = None,
        *,
        auto_batch: bool = True,
    ) -> RunResults:
        """Run the simulation.

        Returns a :class:`~ambr.results.RunResults` mapping (dict subclass).
        Use ``results['agents']`` or ``results.agents`` interchangeably.

        Device placement is Keras-style: call :meth:`cpu` or :meth:`gpu` on the
        model (or pass ``device=`` / legacy ``backend=`` here). Use
        :meth:`vectorize` to select vectorized execution on the chosen device,
        or pass ``mode=`` here. On CPU, ``mode='vectorized'`` (default) dispatches
        :meth:`step_vectorized`; ``mode='oop'`` dispatches :meth:`step_oop` and
        expects tracked per-agent objects. Models that only implement
        :meth:`step` retain backwards-compatible fallback behavior.

        Args:
            steps: number of steps to run (defaults to ``self.p['steps']`` or 100).
            contract: snapshot-view conformance checking mode. One of
                ``'off'`` (default, zero overhead), ``'check'`` (record a
                per-step :class:`~ambr.contract.ContractCertificate` in
                ``self.contract_certificates`` and the ``'contract'`` results
                key), ``'warn'`` (also emit a warning per violation), or
                ``'raise'`` (raise :class:`~ambr.contract.ContractViolationError`
                on the first step with an error-severity violation).
            mode: ``'vectorized'`` (default) or ``'oop'``. Overrides
                fluent placement and ``parameters['mode']``.
            device: ``'cpu'`` or ``'gpu'``. Overrides :meth:`cpu` / :meth:`gpu`.
                Legacy alias: ``backend=``.
            auto_batch: batch eligible own-field integer updates on the selected device.
                Set to ``False`` to execute every agent call in Python.
        """
        self._contract.reset_run(contract)
        self._auto_batch_enabled = auto_batch
        self._auto_batches = 0
        self._auto_batched_agent_calls = 0

        start_time = time.time()
        max_steps = steps if steps is not None else self.p.get('steps', 100)
        config = resolve_config(
            self, device=device, backend=backend, mode=mode
        )
        self._device = config.device
        self._execution_mode = config.mode

        fast_runner = getattr(self, "_run_gpu_fast", None)
        fast_setup = getattr(self, "_setup_gpu_fast", None)
        can_fast_run = self._fast_path_is_eligible(
            config, contract, fast_runner, fast_setup
        )

        if self._show_progress:
            self._start_time = start_time
            self._print_start_info(max_steps)
            self._print_progress(0, max_steps, force=True)

        if can_fast_run:
            device_columns, device_rng = fast_setup()
            self._setup_done = True
            begin_fast_execution(
                self, config, device_columns, device_rng=device_rng
            )
        else:
            self._ensure_setup()
            begin_execution(self, config)

        run_status = "completed"
        run_error: Optional[BaseException] = None
        try:
            if can_fast_run:
                # The model-specific private loop owns only the hot step
                # execution and appends ordinary model-data rows.  Setup,
                # device placement, teardown, and result assembly remain the
                # same public Model lifecycle.
                fast_runner(max_steps)
            else:
                if self.record_initial:
                    self._record_initial_state()

                # Use run_step() to execute exactly one model step per loop iteration.
                while self.t < max_steps:
                    self.run_step()
                    if self._show_progress:
                        self._print_progress(self.t, max_steps)

            prepare_end_hook(self)
            self.end()
        except Exception as exc:
            # Failures stay visible to the caller; no silent swallow.
            run_status = "failed"
            run_error = exc
            raise
        finally:
            # Teardown must not mask the original simulation exception.
            try:
                end_execution(self)
            except Exception as teardown_exc:
                if run_error is not None:
                    try:
                        run_error.add_note(  # type: ignore[attr-defined]
                            f"GPU/execution teardown also failed: {teardown_exc!r}"
                        )
                    except Exception:
                        pass
                    # Keep raising the simulation error (already active).
                else:
                    raise

        if self._show_progress:
            self._print_progress(max_steps, max_steps, force=True)
            self._print_end_info(start_time, max_steps)

        return self._collect_results(
            start_time,
            max_steps,
            device=config.device,
            mode=config.mode,
            status=run_status,
        )

    # --- Helper methods ---
    def _print_start_info(self, max_steps):
        print(f"Simulation: {self.__class__.__name__}")
        print(f"Steps: {max_steps:,}")

    def _print_end_info(self, start_time, max_steps):
        total_time = time.time() - start_time
        print(f"\nDone. Time: {timedelta(seconds=int(total_time))}")
        if total_time > 0:
            print(f"Rate: {max_steps/total_time:.1f} steps/s")
        else:
            print("Rate: Inf steps/s")

    def _collect_results(
        self,
        start_time,
        max_steps,
        *,
        device: str = "cpu",
        mode: str = "vectorized",
        status: str = "completed",
    ):
        if self._model_data:
            # Column-oriented construction to avoid Polars concat ShapeErrors with sparse data
            all_keys = sorted(list(set().union(*(d.keys() for d in self._model_data))))
            data_dict = {k: [] for k in all_keys}

            for d in self._model_data:
                for k in all_keys:
                    data_dict[k].append(d.get(k, None))

            series_list = []
            for k, v in data_dict.items():
                try:
                    s = pl.Series(k, v, strict=False)
                except (TypeError, ValueError):
                    # Fallback to Object type for columns with mixed None/Arrays which confuses Polars inference
                    try:
                        s = pl.Series(k, v, dtype=pl.Object, strict=False)
                    except Exception:
                        # thorough fallback
                        s = pl.Series(k, v, dtype=pl.Object)
                series_list.append(s)

            model_df = pl.DataFrame(series_list)
        else:
            model_df = pl.DataFrame({'t': []})

        end_time = time.time()
        info = build_run_info(
            self,
            steps=self.t,
            start_time=start_time,
            end_time=end_time,
            device=device,
            mode=mode,
            status=status,
            extra={
                # Keep requested step count alongside completed ``steps``.
                "steps_requested": max_steps,
                "auto_batch": {
                    "enabled": self._auto_batch_enabled,
                    "batches": self._auto_batches,
                    "agent_calls": self._auto_batched_agent_calls,
                },
            },
        )
        results = RunResults(
            info=info,
            # agents_df flushes the buffered write queue so OOP /
            # update_agent_data writes land in the returned frame.
            agents=self._agents_frame(),
            model=model_df,
        )
        if self._contract.mode != "off":
            results['contract'] = self._contract.certificates
        if self._agent_vars:
            results['agent_vars'] = pl.concat(self._agent_vars, how='vertical_relaxed')
        return results

    # --- Agent Management Delegates ---
    def add_agent(self, agent: Agent):
        """Add a single agent. Prefer :meth:`add_agents` for bulk creation."""
        execution = getattr(self, "_execution", None)
        if execution is not None and execution.tracked_agents:
            sync_all_device_columns(self, dirty_only=True)
        # Forward Python attributes set on the instance (e.g. ``agent.wealth = 5``
        # before this call) into the population row.
        attributes = {
            k: v
            for k, v in vars(agent).items()
            if k not in {"model", "id", "p"} and not k.startswith("_")
        }
        self.population.add_agent(agent.id, self.t, **attributes)
        self._bump_id_version()
        refresh_tracked_device_columns(self)

        from .sequences import AgentList
        if isinstance(self.agents, AgentList):
            self.agents.append(agent)
            if self.agents.agent_type is None:
                self.agents.agent_type = type(agent)

    def add_agents(
        self,
        n: int,
        *,
        agent_class: Optional[Type] = None,
        **columns: Any,
    ):
        """Bulk-create ``n`` agents with columnar initial state::

            self.add_agents(100, wealth=self.rng.integers(1, 10, 100),
                                 status='S')

        Scalar kwargs broadcast; list / ``np.ndarray`` / ``pl.Series`` values
        must have length ``n``. Pass ``agent_class=`` to also spin up Python
        instances so ``AgentList.call`` and per-agent iteration work.
        """
        from .sequences import AgentList

        execution = getattr(self, "_execution", None)
        if execution is not None and execution.tracked_agents:
            sync_all_device_columns(self, dirty_only=True)

        start_id = (
            self.population.data['id'].max() + 1
            if not self.population.data.is_empty() and 'id' in self.population.data.columns
            else 0
        )

        # If an agent_class is given, create Python instances first so
        # setup() can write extra columns via record(), then merge those
        # columns into the batch-add call.
        if agent_class is not None and n > 0:
            agent_columns: Dict[str, Any] = dict(columns)
            extras: Dict[str, list] = {}
            for i in range(n):
                aid = start_id + i
                agent = agent_class(self, aid)
                agent.setup()
                # Forward non-internal Python attrs set in setup()
                for k, v in vars(agent).items():
                    if k in {"model", "id", "p"} or k.startswith("_"):
                        continue
                    extras.setdefault(k, []).append(v)
                if isinstance(self.agents, AgentList):
                    self.agents.append(agent)
                    if self.agents.agent_type is None:
                        self.agents.agent_type = agent_class
            # Merge extras into columns, with explicit columns taking priority
            for k, vals in extras.items():
                if k not in agent_columns:
                    agent_columns[k] = vals
            self.population.batch_add_agents(n, step=self.t, **agent_columns)
            # Explicit initial values take precedence over agent setup defaults.
            for column in columns:
                pending = self._pending_writes.get(column)
                if pending:
                    for agent_id in range(start_id, start_id + n):
                        pending.pop(agent_id, None)
                    if not pending:
                        self._pending_writes.pop(column)
            self._bump_id_version()
            refresh_tracked_device_columns(self)
            return self.agents

        self.population.batch_add_agents(n, step=self.t, **columns)
        self._bump_id_version()
        refresh_tracked_device_columns(self)

        if not isinstance(self.agents, AgentList):
            return self.agents  # type: ignore[return-value]
        return self.agents

    def get_agent_data(self, agent_id: Any) -> pl.DataFrame:
        """Return a 1-row DataFrame with the current state of ``agent_id``.

        Uses :attr:`agents_df` so pending buffered writes are flushed first.
        """
        return self.agents_df.filter(pl.col('id') == agent_id)

    def update_agent_data(self, agent_id: int, data: Dict[str, Any]):
        """Deprecated: use ``agent.<col> = value`` or ``agents.at[id].set(...)``.

        Still routes through :meth:`_queue_write` so the snapshot-view contract
        can witness the writes.
        """
        warn_deprecated(
            "Model.update_agent_data(...)",
            "agent.<col> = value or agents.at[id].set(**cols)",
        )
        monitor = self._contract
        for key, value in data.items():
            self._queue_write(key, agent_id, value)
            if monitor.active:
                monitor._record_model_buffered_write(key, agent_id, value)

    def batch_update_agents(self, agent_ids: list, data: dict):
        """Deprecated: use ``agents.at[ids].set(**data)`` (or column assign).

        Still equivalent to ``self.agents.at[agent_ids].set(**data)`` so
        multi-column updates stay atomic and contract-observed.
        """
        warn_deprecated(
            "Model.batch_update_agents(...)",
            "agents.at[ids].set(**cols) or agents.where(...).set(**cols)",
        )
        self.agents.at[agent_ids].set(**data)

    def _print_progress(self, current_step: int, total_steps: int, force: bool = False):
        pass
