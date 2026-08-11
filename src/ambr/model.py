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

from typing import Any, Dict, Iterable, List, Type, Optional, Set, Tuple, TYPE_CHECKING
import polars as pl
import warnings
import numpy as np
import time
from datetime import timedelta
from .base import BaseModel
from .population import Population
from ._deprecation import warn_deprecated
from ._id_index import resolve_positions
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
    resolve_config,
)

if TYPE_CHECKING:
    from .agent import Agent


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

        super().__init__(parameters)  # sets self.random / self.rng / self.nprandom
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

    def gpu(self, mode: Optional[str] = None) -> "Model":
        """Place the next :meth:`run` on GPU (device-resident columns). Returns ``self``.

        Args:
            mode: optional execution style. GPU runs support the
                ``'vectorized'`` view API; Python Agent objects use CPU OOP mode.

        Examples::

            model.gpu().run()
            model.gpu(mode="vectorized").run()
        """
        self._device = "gpu"
        if mode is not None:
            self._set_execution_mode(mode)
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
        self._flush_pending_writes()
        return self.population.data

    @agents_df.setter
    def agents_df(self, value):
        self._pending_writes = {}
        self._set_frame(value)
        self._bump_id_version()

    def _set_frame(
        self,
        df: pl.DataFrame,
        *,
        written_columns: Optional[List[str]] = None,
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
        if written_columns is not None and self._contract.active:
            self._contract.record_commit(written_columns)

    def _queue_write(self, column: str, agent_id: Any, value: Any) -> None:
        self._pending_writes.setdefault(column, {})[agent_id] = value
        # Contract: count ordinary writes per (column, id) within a step so a
        # second write to an already-written cell (which the buffer would
        # silently clobber) is detectable as a partial-map conflict.
        self._contract.record_buffered_write(column, agent_id)

    def _bump_id_version(self) -> None:
        """Invalidate id→row caches after the agent id set changes."""
        self._id_version += 1
        # Caches filled by ambr._id_index.resolve_positions / ids_are_arange.
        self._id_pos_cache = None
        self._ids_arange_cache = None

    def _flush_pending_writes(self) -> None:
        """Apply all queued ``Agent`` attribute writes into ``population.data``.

        Fast path: map ids → rows once via :func:`~ambr._id_index.resolve_positions`
        and scatter into Python lists (preserves ``None``; cheaper than a wide
        Polars update). Fallback: ``df.update(on='id')`` hash join.
        """
        if not self._pending_writes:
            return
        # Clear before the write so an exception can't leave a stale buffer
        # that would double-apply on the next flush.
        pending = self._pending_writes
        self._pending_writes = {}

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

    def _contract_snapshot(self) -> Tuple[Dict[str, str], Set[Any]]:
        """Return ({column: dtype-str}, id-set) of the committed population.

        The schema carries dtypes (not just names) so that a mid-step dtype
        change -- e.g. an ``Int64`` attribute silently rewritten as ``Float64``
        -- surfaces as a schema mutation rather than passing unnoticed.
        """
        from .device_columns import model_uses_device_columns, sync_all_device_columns

        if model_uses_device_columns(self):
            sync_all_device_columns(self)
        df = self.agents_df  # flushes pending writes
        schema = {name: str(dt) for name, dt in zip(df.columns, df.dtypes)}
        if "id" in df.columns and df.height:
            ids = set(df["id"].to_list())
        else:
            ids = set()
        return schema, ids

    def _contract_record_borrow(self, column: str) -> None:
        """Record a lane borrow of ``column``; flag it if already committed.

        Commits go through :meth:`_set_frame` (``written_columns=``); only
        borrows need this separate seam.
        """
        self._contract.record_borrow(column)

    def _contract_record_mutable_borrow(self, column: str) -> None:
        """Record exposure of a mutable NumPy/CuPy column buffer."""
        self._contract.record_mutable_borrow(column)

    def _contract_record_reduction(self, columns: Iterable[str]) -> None:
        """Record columns written through a commutative reduction."""
        self._contract.record_reduction(columns)

    def setup(self): pass
    def step(self): pass

    def step_vectorized(self):
        """Execute one vectorized step.

        Models with a distinct vectorized implementation can override this
        hook. The default preserves the pre-mode API, where ``step()`` is the
        model's single implementation.
        """
        return self.step()

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
            * Does not prove schedule equivalence; the contract monitor is
              separate and operational only.
        """
        from .scheduling import activate, normalize_activation

        chosen = mode if mode is not None else self._activation
        activate(
            self.agents,
            mode=normalize_activation(chosen),
            method=method,
            rng=self.rng,
        )

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
        from .device_columns import model_uses_device_columns, sync_all_device_columns

        if model_uses_device_columns(self):
            sync_all_device_columns(self)
        df = self.agents_df
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
                try:
                    step_fn()
                finally:
                    cert = mon.end_step(self.t, self._contract_snapshot())

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
            raise

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
    ) -> RunResults:
        """Run the simulation.

        Returns a :class:`~ambr.results.RunResults` mapping (dict subclass).
        Use ``results['agents']`` or ``results.agents`` interchangeably.

        Device placement is Keras-style: call :meth:`cpu` or :meth:`gpu` on the
        model (or pass ``device=`` / legacy ``backend=`` here). Mode can be set
        on those fluent methods (``model.cpu(mode="vectorized")``) or here via
        ``mode=``. On CPU, ``mode='vectorized'`` (default) dispatches
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
                :meth:`cpu` / :meth:`gpu` ``mode=`` and ``parameters['mode']``.
            device: ``'cpu'`` or ``'gpu'``. Overrides :meth:`cpu` / :meth:`gpu`.
                Legacy alias: ``backend=``.
        """
        self._contract.reset_run(contract)

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
        try:
            if can_fast_run:
                # The model-specific private loop owns only the hot step
                # execution and appends ordinary model-data rows.  Setup,
                # device placement, teardown, and result assembly remain the
                # same public Model lifecycle.
                fast_runner(max_steps)
                self.end()
            else:
                if self.record_initial:
                    self._record_initial_state()

                # Use run_step() to execute exactly one model step per loop iteration.
                while self.t < max_steps:
                    self.run_step()
                    if self._show_progress:
                        self._print_progress(self.t, max_steps)

                self.end()
        except Exception:
            # Failures stay visible to the caller; no silent swallow.
            run_status = "failed"
            raise
        finally:
            end_execution(self)

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
            },
        )
        results = RunResults(
            info=info,
            # agents_df flushes the buffered write queue so OOP /
            # update_agent_data writes land in the returned frame.
            agents=self.agents_df,
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
        # Forward Python attributes set on the instance (e.g. ``agent.wealth = 5``
        # before this call) into the population row.
        attributes = {
            k: v
            for k, v in vars(agent).items()
            if k not in {"model", "id", "p"} and not k.startswith("_")
        }
        self.population.add_agent(agent.id, self.t, **attributes)
        self._bump_id_version()

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
            self._bump_id_version()
            return self.agents

        self.population.batch_add_agents(n, step=self.t, **columns)
        self._bump_id_version()

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
        for key, value in data.items():
            self._queue_write(key, agent_id, value)

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
