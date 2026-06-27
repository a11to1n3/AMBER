from typing import Any, Dict, List, Type, Optional
import polars as pl
import random
import warnings
import numpy as np
import time
from datetime import datetime, timedelta
from .base import BaseModel
from .population import Population
from ._deprecation import warn_deprecated
from .contract import (
    CONTRACT_MODES,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    ContractCertificate,
    ContractViolation,
    ContractViolationError,
)

class Model(BaseModel):
    """Base class for all simulation models, using DataFrames for data storage."""

    #: Declarative model-level metrics, evaluated once per step into the
    #: ``'model'`` results frame. Maps a column name to a ``callable(model)``,
    #: the name of a model attribute/method, or a constant. Overridable per
    #: subclass; complements imperative ``record_model``.
    model_reporters: Dict[str, Any] = {}
    #: Declarative per-agent columns to snapshot each step into the (opt-in)
    #: ``'agent_vars'`` long-format results frame. Empty by default (no cost).
    agent_reporters: List[str] = []
    #: When True, capture a ``t=0`` row of the reporters before the first step.
    record_initial: bool = False
    #: Optional typed parameter schema ``{'name': (type, default)}``. When set,
    #: ``self.p.name`` is pre-coerced to ``type`` at init (missing -> default),
    #: replacing scattered ``int(self.p.get('name', ...))`` boilerplate.
    params: Dict[str, Any] = {}

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

        # --- snapshot-view contract conformance checking (see contract.py) ---
        # Mode is one of CONTRACT_MODES; 'off' adds zero per-write overhead.
        self._contract_mode: str = "off"
        self.contract_certificates: List[ContractCertificate] = []
        # Active only while a step body runs under a non-'off' mode.
        self._contract_active: bool = False
        # Per-step bookkeeping for duplicate-write detection on the buffered
        # (OOP) write path, plus the step-entry snapshot for lifecycle diffing.
        self._contract_write_counts: Dict[Any, int] = {}
        self._contract_dup_cells: set = set()
        self._contract_entry_schema: Optional[Dict[str, str]] = None
        self._contract_entry_ids: Optional[set] = None
        # Per-step bookkeeping for the vectorized lane/view write path
        # (``population.data = df.with_columns(...)``), which bypasses
        # ``_queue_write``. The tensor lane reports its commits/borrows here so
        # duplicate commits and read-after-write are visible to the contract.
        self._contract_commit_counts: Dict[str, int] = {}
        self._contract_committed: set = set()
        self._contract_raw_cols: set = set()

        super().__init__(parameters)  # sets self.random / self.rng / self.nprandom
        self.t = 0
        self._start_time = None
        self._last_progress_time = None
        self._show_progress = parameters.get('show_progress', True)
        self._model_data = []
        self._agent_vars = []  # per-step agent snapshots when agent_reporters is set

        from .sequences import AgentList
        self.agents = AgentList(self, [])
    
    @property
    def agents_df(self) -> pl.DataFrame:
        self._flush_pending_writes()
        return self.population.data

    @agents_df.setter
    def agents_df(self, value):
        self._pending_writes = {}
        self._set_frame(value)
        self._bump_id_version()

    def _set_frame(self, df: pl.DataFrame) -> None:
        """Internal single write seam for the agent table.

        All columnar writes (the view API in ``sequences.py`` and the tensor
        lane) route here instead of assigning ``population.data`` directly, so
        the backing store stays encapsulated. Does not bump the id-version
        (column-value writes keep the id set stable); callers that change the
        id set (e.g. the ``agents_df`` setter, ``add_agents``) bump explicitly.
        """
        self.population.data = df

    def _queue_write(self, column: str, agent_id: Any, value: Any) -> None:
        self._pending_writes.setdefault(column, {})[agent_id] = value
        # Contract: count ordinary writes per (column, id) within a step so a
        # second write to an already-written cell (which the buffer would
        # silently clobber) is detectable as a partial-map conflict.
        if self._contract_active:
            key = (column, agent_id)
            count = self._contract_write_counts.get(key, 0) + 1
            self._contract_write_counts[key] = count
            if count == 2:
                self._contract_dup_cells.add(key)

    def _bump_id_version(self) -> None:
        self._id_version += 1

    def _flush_pending_writes(self) -> None:
        """Apply all queued ``Agent.record()`` / ``update_data()`` writes
        as a single ``df.update(on='id')`` hash join."""
        if not self._pending_writes:
            return
        # Clear before the write so an exception can't leave a stale buffer
        # that would double-apply on the next flush.
        pending = self._pending_writes
        self._pending_writes = {}

        df = self.population.data

        # If the DataFrame is empty or 'id' is null-typed, initialise it
        # with an Int64 id column so the update join key types match.
        if df.is_empty() or 'id' not in df.columns:
            # Build the full DataFrame from the pending writes instead.
            touched_ids = list({aid for col_map in pending.values() for aid in col_map})
            data_cols: Dict[str, list] = {'id': touched_ids}
            for col, id_to_val in pending.items():
                data_cols[col] = [id_to_val.get(aid, None) for aid in touched_ids]
            self.population.data = pl.DataFrame(
                [pl.Series(k, v, strict=False) for k, v in data_cols.items()]
            )
            self._bump_id_version()
            return

        # Ensure id column is not null-typed (Polars update requires matching types).
        if df['id'].dtype == pl.Null:
            df = df.with_columns(pl.col('id').cast(pl.Int64))

        missing = [c for c in pending if c not in df.columns]
        if missing:
            df = df.with_columns(
                [pl.Series(c, [None] * df.height, strict=False) for c in missing]
            )

        # df.update(on='id') skips null cells, so untouched (col, id) pairs
        # retain their current value.
        touched_ids = list({aid for col_map in pending.values() for aid in col_map})
        update_cols: Dict[str, list] = {'id': touched_ids}
        for col, id_to_val in pending.items():
            update_cols[col] = [id_to_val.get(aid, None) for aid in touched_ids]
        update_df = pl.DataFrame(
            [pl.Series(k, v, strict=False) for k, v in update_cols.items()]
        )
        self.population.data = df.update(update_df, on='id', how='left')

    # --- snapshot-view contract conformance checking ------------------------

    def _contract_snapshot(self):
        """Return ({column: dtype-str}, id-set) of the committed population.

        The schema carries dtypes (not just names) so that a mid-step dtype
        change -- e.g. an ``Int64`` attribute silently rewritten as ``Float64``
        -- surfaces as a schema mutation rather than passing unnoticed.
        """
        df = self.agents_df  # flushes pending writes
        schema = {name: str(dt) for name, dt in zip(df.columns, df.dtypes)}
        if "id" in df.columns and df.height:
            ids = set(df["id"].to_list())
        else:
            ids = set()
        return schema, ids

    def _contract_begin_step(self) -> None:
        """Snapshot pre-step structure and arm per-write duplicate tracking."""
        self._contract_entry_schema, self._contract_entry_ids = self._contract_snapshot()
        self._contract_write_counts = {}
        self._contract_dup_cells = set()
        self._contract_commit_counts = {}
        self._contract_committed = set()
        self._contract_raw_cols = set()
        self._contract_active = True

    def _contract_record_commit(self, columns) -> None:
        """Record a vectorized lane/view commit of ``columns`` this step.

        Called by the tensor lane (``commit_columns`` / ``TensorLane.commit`` /
        ``write_result``) when the contract is active, so the conformance check
        can witness writes that bypass the ``_queue_write`` seam.
        """
        if not self._contract_active:
            return
        for c in columns:
            self._contract_commit_counts[c] = self._contract_commit_counts.get(c, 0) + 1
            self._contract_committed.add(c)

    def _contract_record_borrow(self, column: str) -> None:
        """Record a lane borrow of ``column``; flag it if already committed."""
        if not self._contract_active:
            return
        if column in self._contract_committed:
            self._contract_raw_cols.add(column)

    def _contract_end_step(self, cert: "ContractCertificate") -> None:
        """Finalise ``cert`` by comparing the committed end-of-step state to
        the step-entry snapshot and folding in any duplicate-write findings."""
        self._contract_active = False
        exit_schema, exit_ids = self._contract_snapshot()

        # (1) Duplicate unreduced writes on the buffered (OOP) path.
        if self._contract_dup_cells:
            cols = sorted({c for c, _ in self._contract_dup_cells})
            ids = sorted({i for _, i in self._contract_dup_cells})
            cert.add(ContractViolation(
                "duplicate_write",
                f"{len(self._contract_dup_cells)} (column, id) cell(s) received more "
                f"than one ordinary write within the step; under simultaneous "
                f"activation the later write clobbers the earlier (partial-map "
                f"conflict). Use agents.at[ids].scatter_add(...) to accumulate.",
                severity=SEVERITY_ERROR,
                columns=cols,
                ids=ids,
            ))

        # (2) Schema mutation across the step (names added/removed or dtype drift).
        if self._contract_entry_schema is not None:
            entry = self._contract_entry_schema
            added = set(exit_schema) - set(entry)
            removed = set(entry) - set(exit_schema)
            retyped = sorted(
                c for c in (set(entry) & set(exit_schema)) if entry[c] != exit_schema[c]
            )
            if removed:
                cert.add(ContractViolation(
                    "schema_mutation",
                    f"column(s) removed mid-step ({sorted(removed)}); reads issued "
                    f"during the step may reference a column that no longer exists.",
                    severity=SEVERITY_ERROR,
                    columns=sorted(removed),
                ))
            if retyped:
                cert.add(ContractViolation(
                    "schema_mutation",
                    f"column dtype(s) changed mid-step "
                    f"({[(c, entry[c], exit_schema[c]) for c in retyped]}); a write "
                    f"silently altered a column's type, breaking schema stability.",
                    severity=SEVERITY_ERROR,
                    columns=retyped,
                ))
            if added:
                cert.add(ContractViolation(
                    "schema_mutation",
                    f"column(s) added mid-step ({sorted(added)}); a vectorized read "
                    f"of these earlier in the step would see no stable snapshot.",
                    severity=SEVERITY_WARNING,
                    columns=sorted(added),
                ))

        # (2b) Vectorized lane/view conflicts (writes that bypass _queue_write).
        dup_commits = sorted(c for c, n in self._contract_commit_counts.items() if n > 1)
        if dup_commits:
            cert.add(ContractViolation(
                "duplicate_write",
                f"column(s) {dup_commits} were committed more than once within the "
                f"step via the lane/view path; the later commit clobbers the earlier "
                f"under simultaneous activation. Commit each column exactly once.",
                severity=SEVERITY_ERROR,
                columns=dup_commits,
            ))
        if self._contract_raw_cols:
            raw = sorted(self._contract_raw_cols)
            cert.add(ContractViolation(
                "read_after_write",
                f"column(s) {raw} were borrowed after being committed within the same "
                f"step; a snapshot rule must read step-entry state, not post-write "
                f"state. Borrow all inputs before committing any output.",
                severity=SEVERITY_ERROR,
                columns=raw,
            ))

        # (3) Population mutation (births/deaths) across the step.
        if self._contract_entry_ids is not None:
            born = exit_ids - self._contract_entry_ids
            died = self._contract_entry_ids - exit_ids
            if born or died:
                cert.add(ContractViolation(
                    "population_mutation",
                    f"agent population changed mid-step (added={len(born)}, "
                    f"removed={len(died)}); verify lifecycle rules read the "
                    f"step-entry snapshot rather than partially-updated state.",
                    severity=SEVERITY_WARNING,
                    ids=sorted(born | died),
                ))

    def setup(self): pass
    def step(self): pass
    
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
        constant. Runs before :meth:`update`, so an imperative ``record_model``
        of the same key wins.
        """
        reporters = getattr(self, 'model_reporters', None)
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
        cols = getattr(self, 'agent_reporters', None)
        if not cols:
            return
        df = self.agents_df
        have = [c for c in cols if c in df.columns]
        if not have:
            return
        self._agent_vars.append(
            df.select(['id', *have]).with_columns(pl.lit(self.t).alias('t'))
        )

    def _finalize_step_data(self):
        if hasattr(self, '_current_step_data'):
            self._model_data.append(self._current_step_data.copy())

    def _advance_and_record(self) -> None:
        """Advance ``t`` and capture this step's recorded data.

        Owns the step-counter increment and step-data lifecycle so :meth:`update`
        can be a pure hook. Declarative reporters are evaluated first, then the
        imperative :meth:`update` hook, then the row is finalized.
        """
        self.t += 1
        self._current_step_data = {'t': self.t}
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
        """
        self._ensure_setup()
        if self._contract_mode == "off":
            self.step()
            self._advance_and_record()
            return

        cert = ContractCertificate(self.t)
        self._contract_begin_step()
        try:
            self.step()
        finally:
            self._contract_end_step(cert)
        self.contract_certificates.append(cert)

        if self._contract_mode == "warn":
            for v in cert.violations:
                warnings.warn(
                    f"[step {cert.step}] {v.kind}: {v.detail}",
                    UserWarning,
                    stacklevel=2,
                )
        elif self._contract_mode == "raise" and not cert.ok:
            raise ContractViolationError(cert)

        self._advance_and_record()

    def run(
        self,
        steps: Optional[int] = None,
        contract: str = "off",
    ) -> Dict[str, pl.DataFrame]:
        """Run the simulation.

        Args:
            steps: number of steps to run (defaults to ``self.p['steps']`` or 100).
            contract: snapshot-view conformance checking mode. One of
                ``'off'`` (default, zero overhead), ``'check'`` (record a
                per-step :class:`~ambr.contract.ContractCertificate` in
                ``self.contract_certificates`` and the ``'contract'`` results
                key), ``'warn'`` (also emit a warning per violation), or
                ``'raise'`` (raise :class:`~ambr.contract.ContractViolationError`
                on the first step with an error-severity violation).
        """
        if contract not in CONTRACT_MODES:
            raise ValueError(
                f"contract must be one of {CONTRACT_MODES}, got {contract!r}"
            )
        self._contract_mode = contract
        if contract != "off":
            self.contract_certificates = []

        start_time = time.time()
        max_steps = steps if steps is not None else self.p.get('steps', 100)

        if self._show_progress:
            self._start_time = start_time
            self._print_start_info(max_steps)
            self._print_progress(0, max_steps, force=True)

        self._ensure_setup()

        if getattr(self, 'record_initial', False):
            self._record_initial_state()

        # Use run_step() to execute exactly one model step per loop iteration.
        while self.t < max_steps:
            self.run_step()
            if self._show_progress:
                self._print_progress(self.t, max_steps)

        self.end()

        if self._show_progress:
            self._print_progress(max_steps, max_steps, force=True)
            self._print_end_info(start_time, max_steps)

        return self._collect_results(start_time, max_steps)

    # --- Helper methods ---
    def _print_start_info(self, max_steps):
        print(f"🚀 Simulation: {self.__class__.__name__}")
        print(f"⏱️  Steps: {max_steps:,}")

    def _print_end_info(self, start_time, max_steps):
        total_time = time.time() - start_time
        print(f"\n✅ Done. Time: {timedelta(seconds=int(total_time))}")
        if total_time > 0:
            print(f"📈 Rate: {max_steps/total_time:.1f} steps/s")
        else:
            print(f"📈 Rate: Inf steps/s")

    def _collect_results(self, start_time, max_steps):
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
            
        results = {
            'info': {'steps': self.t, 'run_time': time.time() - start_time},
            'agents': self.population.data,
            'model': model_df
        }
        if self._contract_mode != "off":
            results['contract'] = self.contract_certificates
        if getattr(self, '_agent_vars', None):
            results['agent_vars'] = pl.concat(self._agent_vars, how='vertical_relaxed')
        return results

    # --- Agent Management Delegates ---
    def add_agent(self, agent: 'Agent'):
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
        """Return a 1-row DataFrame with the current state of ``agent_id``."""
        return self.population.data.filter(pl.col('id') == agent_id)

    def update_agent_data(self, agent_id: int, data: Dict[str, Any]):
        """Update data for a single agent."""
        for key, value in data.items():
            self.population.set_agent_value(agent_id, key, value)
        
    def batch_update_agents(self, agent_ids: list, data: dict):
        """Batch update multiple agents at once for better performance.
        
        Args:
            agent_ids: List of agent IDs to update
            data: Dictionary of column names and values (or lists of values)
        """
        self.population.batch_update_by_ids(agent_ids, data)

    def _print_progress(self, current_step: int, total_steps: int, force: bool = False):
        pass
