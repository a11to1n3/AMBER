"""Runtime conformance checking for AMBER's snapshot-view contract.

This module turns simultaneous-update discipline into a falsifiable per-step
artifact.  A :class:`ContractCertificate` reports conflicts observed at AMBER's
read/write seams; it is a runtime diagnostic, not a proof that arbitrary user
array code is schedule-independent.

It witnesses violations on *two* write paths:

* the **buffered (OOP) path** -- per-agent ``Agent.__setattr__`` writes routed
  through ``Model._queue_write``; and
* the **vectorized lane/view path** -- whole-column writes via
  ``agents.col = ...`` / ``agents.commit(...)`` / :class:`~ambr.tensor_lane.TensorLane`,
  which bypass ``_queue_write`` but report commits to :class:`ContractMonitor`.

The checks:

* **duplicate unreduced write** -- the same ``(column, agent_id)`` cell received
  more than one *ordinary* (non-scatter) write within a step (buffered path), or
  the same column was committed more than once within a step (lane/view path).
  Under simultaneous activation the later write silently clobbers the earlier
  one (a "partial map" conflict). The sanctioned fix is a commutative reducer
  (``agents.at[ids].scatter_add(col=delta)``) or committing each column once.
* **cross-path write** -- the same column received both a buffered (OOP)
  write and a whole-column lane/view commit within one step. The flush order
  makes the later path win; under simultaneous activation that is still a
  partial-map conflict.
* **read-after-write (lane path)** -- a column borrowed *after* it was committed
  in the same step; a snapshot rule must read step-entry state.
* **uncertified mutable borrow** -- ``agents.array(...)`` exposed a mutable
  NumPy/CuPy buffer.  AMBER observes the borrow, but cannot infer whether or how
  user code subsequently mutates or indexes that raw array.
* **schema / population mutation** -- the column set, a column's *dtype*, or the
  agent id-set changed between step entry and exit, so reads issued during the
  step no longer see a stable snapshot. Column/population *additions* and
  births/deaths are reported as *warnings*; column removal and dtype change are
  *errors*.

This is a conformance *monitor*, not a prover, with two known limits: (1) the
schema/population diff is **endpoint-only** -- a mutation that is introduced and
reverted within a single step is not caught; (2) raw ``population.data = ...``
writes that bypass both ``_queue_write`` and the reported lane/view seams are
invisible to the conflict check (``Population.data = ...`` is deprecated and
warns; prefer ``agents.col = ...`` / ``agents.set`` / ``agents.commit``).
Cell-level dependency and commutativity reasoning remains the job of model
analysis or tests; a clean certificate is evidence only for the operations the
runtime seams can observe.

A conflict closed while a stage is still open is reported as
``activation_race`` with ``divergence_witness is None``. Automatic replay
runs only after the step body has finished, and a replay restores monitor
mode before it returns.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

#: Valid values for ``Model.run(contract=...)``.
CONTRACT_MODES = ("off", "check", "warn", "raise")

# Snapshot type: ({column: dtype-str}, id-set)
Snapshot = Tuple[Dict[str, str], Set[Any]]


class ContractViolation:
    """A single admissibility violation observed during one step."""

    def __init__(
        self,
        kind: str,
        detail: str,
        severity: str = SEVERITY_ERROR,
        columns: Optional[Iterable[str]] = None,
        ids: Optional[Iterable[Any]] = None,
        divergence_witness: Optional[bool] = None,
    ):
        self.kind = kind
        self.detail = detail
        self.severity = severity
        self.columns: List[str] = list(columns) if columns else []
        self.ids: List[Any] = list(ids) if ids else []
        # True/False only after a completed step-level swap. None means the
        # witness was not run, or it was not conclusive.
        self.divergence_witness = divergence_witness

    def __repr__(self) -> str:
        loc = ""
        if self.columns:
            loc += f" columns={self.columns}"
        if self.ids:
            shown = self.ids[:8]
            more = f" (+{len(self.ids) - 8} more)" if len(self.ids) > 8 else ""
            loc += f" ids={shown}{more}"
        if self.divergence_witness is not None:
            loc += f" divergence_witness={self.divergence_witness}"
        return f"<ContractViolation {self.severity}:{self.kind}{loc}: {self.detail}>"


class ContractCertificate:
    """Per-step record of contract conformance.

    ``ok`` is ``True`` when no *error*-severity violations were found (warnings
    do not flip it). ``clean`` is ``True`` only when there are no violations of
    any severity.
    """

    def __init__(self, step: int):
        self.step = step
        self.violations: List[ContractViolation] = []

    @property
    def ok(self) -> bool:
        return not any(v.severity == SEVERITY_ERROR for v in self.violations)

    @property
    def clean(self) -> bool:
        return not self.violations

    def add(self, violation: ContractViolation) -> None:
        self.violations.append(violation)

    def errors(self) -> List[ContractViolation]:
        return [v for v in self.violations if v.severity == SEVERITY_ERROR]

    def warnings(self) -> List[ContractViolation]:
        return [v for v in self.violations if v.severity == SEVERITY_WARNING]

    def __repr__(self) -> str:
        if self.clean:
            return f"<ContractCertificate step={self.step} ok (no violations)>"
        n_err = sum(1 for v in self.violations if v.severity == SEVERITY_ERROR)
        n_warn = len(self.violations) - n_err
        return (
            f"<ContractCertificate step={self.step} "
            f"errors={n_err} warnings={n_warn}>"
        )


class ContractViolationError(RuntimeError):
    """Raised by ``Model.run(contract='raise')`` on the first non-conforming step."""

    def __init__(self, certificate: ContractCertificate):
        self.certificate = certificate
        errs = certificate.errors()
        msg = "; ".join(f"{v.kind}: {v.detail}" for v in errs) or "contract violation"
        super().__init__(
            f"[step {certificate.step}] snapshot-contract violation -- {msg}"
        )


class ContractMonitor:
    """Per-model runtime monitor for the snapshot-view contract.

    Owns per-step bookkeeping and certificate construction so :class:`~ambr.model.Model`
    only needs a thin seam (``_queue_write``, view/lane commits, ``run_step``).
    """

    def __init__(self) -> None:
        self.mode: str = "off"
        self.certificates: List[ContractCertificate] = []
        self.active: bool = False
        # Buffered (OOP) path
        self._write_counts: Dict[Any, int] = {}
        self._dup_cells: Set[Any] = set()
        self._buffered_cols: Set[str] = set()
        # Lane / whole-column path
        self._commit_counts: Dict[str, int] = {}
        self._committed: Set[str] = set()
        self._raw_cols: Set[str] = set()
        self._borrowed: Set[str] = set()
        self._mutable_borrows: Set[str] = set()
        self._reduced_cols: Set[str] = set()
        # Columns touched by both buffered and lane/view paths in this step
        self._cross_path_cols: Set[str] = set()
        # Step-entry snapshot for schema / population diff
        self._entry_schema: Optional[Dict[str, str]] = None
        self._entry_ids: Optional[Set[Any]] = None
        # Activation trace. A stage closes this trace without replay.
        self._current_stage: Optional[str] = None
        self._stage_open: bool = False
        self._stage_violations: List[ContractViolation] = []
        self._current_activation: Any = None
        self._activation_order: List[Any] = []
        self._activation_reads: Dict[Any, Set[Tuple[str, Any]]] = {}
        self._activation_writes: Dict[Any, Set[Tuple[str, Any]]] = {}
        self._in_replay: bool = False

    def reset_run(self, mode: str) -> None:
        """Configure mode for a new ``Model.run`` and clear prior certificates."""
        if mode not in CONTRACT_MODES:
            raise ValueError(
                f"contract must be one of {CONTRACT_MODES}, got {mode!r}"
            )
        self.mode = mode
        if mode != "off":
            self.certificates = []

    def begin_step(self, snapshot: Snapshot) -> None:
        """Arm tracking for a step body; ``snapshot`` is step-entry state."""
        self._entry_schema, self._entry_ids = snapshot
        self._write_counts = {}
        self._dup_cells = set()
        self._buffered_cols = set()
        self._commit_counts = {}
        self._committed = set()
        self._raw_cols = set()
        self._borrowed = set()
        self._mutable_borrows = set()
        self._reduced_cols = set()
        self._cross_path_cols = set()
        self._current_stage = None
        self._stage_open = False
        self._stage_violations = []
        self._current_activation = None
        self._activation_order = []
        self._activation_reads = {}
        self._activation_writes = {}
        self.active = True

    def end_step(
        self, step: int, snapshot: Snapshot, model: Optional[Any] = None
    ) -> ContractCertificate:
        """Disarm tracking and return the certificate for ``step``.

        Replay is allowed only after the step body has finished and no stage
        is open. An exception unwinding through this method is left alone.
        """
        self.active = False
        exit_schema, exit_ids = snapshot
        cert = ContractCertificate(step)
        self._check_duplicate_buffered(cert)
        self._check_cross_path(cert)
        self._check_schema(cert, exit_schema)
        self._check_lane_conflicts(cert)
        self._check_population(cert, exit_ids)
        if sys.exc_info()[0] is None:
            allow_replay = not self._stage_open and not self._in_replay
            self._check_activation_races(
                cert, model=model, allow_replay=allow_replay
            )
        if self._stage_violations:
            cert.violations.extend(self._stage_violations)
        self.certificates.append(cert)
        return cert

    def record_buffered_write(self, column: str, agent_id: Any) -> None:
        """Count an ordinary OOP-path write; second write to a cell is a conflict."""
        if not self.active:
            return
        key = (column, agent_id)
        count = self._write_counts.get(key, 0) + 1
        self._write_counts[key] = count
        if count == 2:
            self._dup_cells.add(key)
        self._buffered_cols.add(column)
        if column in self._committed or column in self._reduced_cols:
            self._cross_path_cols.add(column)

    def record_commit(self, columns: Iterable[str]) -> None:
        """Record a whole-column lane/view commit of ``columns`` this step."""
        if not self.active:
            return
        for c in columns:
            self._commit_counts[c] = self._commit_counts.get(c, 0) + 1
            self._committed.add(c)
            if c in self._buffered_cols or c in self._reduced_cols:
                self._cross_path_cols.add(c)

    def record_reduction(self, columns: Iterable[str]) -> None:
        """Record a sanctioned commutative reduction write.

        Repeated reductions are not duplicate-write errors, but mixing a
        reduction with an ordinary write to the same column remains ambiguous
        and is reported as a cross-path conflict.
        """
        if not self.active:
            return
        for c in columns:
            self._reduced_cols.add(c)
            self._committed.add(c)
            if c in self._buffered_cols or c in self._commit_counts:
                self._cross_path_cols.add(c)

    def record_borrow(self, column: str) -> None:
        """Record a lane borrow of ``column``; flag if already committed."""
        if not self.active:
            return
        self._borrowed.add(column)
        if column in self._committed:
            self._raw_cols.add(column)

    def record_mutable_borrow(self, column: str) -> None:
        """Record raw mutable array access that cannot be fully traced."""
        if not self.active:
            return
        self.record_borrow(column)
        self._mutable_borrows.add(column)

    # --- activation trace ---------------------------------------------------

    def begin_stage(self, stage: str = "stage0") -> None:
        """Start a named stage. An open trace is closed without replay."""
        if not self.active or self._in_replay:
            return
        if self._activation_order:
            self.end_stage()
        self._current_stage = stage
        self._stage_open = True

    def end_stage(self, model: Optional[Any] = None) -> None:
        """Close the stage and keep any conflict witness inconclusive.

        Replaying here would compare this stage's frame with a full-step
        rerun, and that rerun would enter this method again. ``model`` is
        accepted so callers can pass the host; it is not used.
        """
        _ = model
        if not self.active or self._in_replay:
            return
        self._stage_open = False
        if not self._activation_order:
            self._current_activation = None
            return
        temp = ContractCertificate(0)
        self._check_activation_races(temp, model=None, allow_replay=False)
        self._stage_violations.extend(temp.violations)
        self._current_activation = None
        self._activation_order = []
        self._activation_reads = {}
        self._activation_writes = {}

    def begin_activation(self, agent_id: Any) -> None:
        """Mark the start of one agent activation inside the open step."""
        if not self.active or self._in_replay:
            return
        self._current_activation = agent_id
        if agent_id not in self._activation_reads:
            self._activation_order.append(agent_id)
            self._activation_reads[agent_id] = set()
            self._activation_writes[agent_id] = set()

    def end_activation(self) -> None:
        """Mark the end of the current activation."""
        self._current_activation = None

    def record_activation_read(self, column: str, agent_id: Any) -> None:
        """Record an ordinary attribute read by the active activation."""
        if not self.active or self._in_replay or self._current_activation is None:
            return
        act = self._current_activation
        self._activation_reads.setdefault(act, set()).add((column, agent_id))

    def record_activation_write(self, column: str, agent_id: Any) -> None:
        """Record an ordinary attribute write by the active activation."""
        if not self.active or self._in_replay or self._current_activation is None:
            return
        act = self._current_activation
        self._activation_writes.setdefault(act, set()).add((column, agent_id))

    # --- internal checks ----------------------------------------------------

    def _check_duplicate_buffered(self, cert: ContractCertificate) -> None:
        if not self._dup_cells:
            return
        cols = sorted({c for c, _ in self._dup_cells})
        ids = sorted({i for _, i in self._dup_cells})
        cert.add(ContractViolation(
            "duplicate_write",
            f"{len(self._dup_cells)} (column, id) cell(s) received more "
            f"than one ordinary write within the step; under simultaneous "
            f"activation the later write clobbers the earlier (partial-map "
            f"conflict). Use agents.at[ids].scatter_add(...) to accumulate.",
            severity=SEVERITY_ERROR,
            columns=cols,
            ids=ids,
        ))

    def _check_cross_path(self, cert: ContractCertificate) -> None:
        if not self._cross_path_cols:
            return
        cols = sorted(self._cross_path_cols)
        cert.add(ContractViolation(
            "cross_path_write",
            f"column(s) {cols} were written through incompatible ordinary, "
            f"buffered, or reduction mechanisms within the same step; the "
            f"result depends on commit ordering. Use one write mechanism per "
            f"column per step (or a single commutative reduction).",
            severity=SEVERITY_ERROR,
            columns=cols,
        ))

    def _check_schema(
        self, cert: ContractCertificate, exit_schema: Dict[str, str]
    ) -> None:
        if self._entry_schema is None:
            return
        entry = self._entry_schema
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

    def _check_lane_conflicts(self, cert: ContractCertificate) -> None:
        dup_commits = sorted(c for c, n in self._commit_counts.items() if n > 1)
        if dup_commits:
            cert.add(ContractViolation(
                "duplicate_write",
                f"column(s) {dup_commits} were committed more than once within the "
                f"step via the lane/view path; the later commit clobbers the earlier "
                f"under simultaneous activation. Commit each column exactly once.",
                severity=SEVERITY_ERROR,
                columns=dup_commits,
            ))
        if self._raw_cols:
            raw = sorted(self._raw_cols)
            cert.add(ContractViolation(
                "read_after_write",
                f"column(s) {raw} were borrowed after being committed within the same "
                f"step; a snapshot rule must read step-entry state, not post-write "
                f"state. Borrow all inputs before committing any output.",
                severity=SEVERITY_ERROR,
                columns=raw,
            ))
        if self._mutable_borrows:
            mutable = sorted(self._mutable_borrows)
            cert.add(ContractViolation(
                "uncertified_mutable_borrow",
                f"column(s) {mutable} were exposed through a mutable raw "
                f"NumPy/CuPy array. The runtime observes this access but "
                f"cannot infer subsequent in-place writes or gather indices; "
                f"use immutable column expressions / TensorLane borrows for "
                f"a fully observed trace.",
                severity=SEVERITY_WARNING,
                columns=mutable,
            ))

    def _check_population(
        self, cert: ContractCertificate, exit_ids: Set[Any]
    ) -> None:
        if self._entry_ids is None:
            return
        born = exit_ids - self._entry_ids
        died = self._entry_ids - exit_ids
        if born or died:
            cert.add(ContractViolation(
                "population_mutation",
                f"agent population changed mid-step (added={len(born)}, "
                f"removed={len(died)}); verify lifecycle rules read the "
                f"step-entry snapshot rather than partially-updated state.",
                severity=SEVERITY_WARNING,
                ids=sorted(born | died),
            ))

    def _check_activation_races(
        self,
        cert: ContractCertificate,
        model: Optional[Any] = None,
        allow_replay: bool = False,
    ) -> None:
        """Report two activations writing one cell.

        ``allow_replay`` is false while a stage is open. The diagnostic is
        still recorded, and ``divergence_witness`` stays ``None``.
        """
        if self._in_replay or not self._activation_order:
            return
        writers_by_cell: Dict[Tuple[str, Any], List[Any]] = {}
        for act in self._activation_order:
            for col, aid in self._activation_writes.get(act, ()):
                writers = writers_by_cell.setdefault((col, aid), [])
                if act not in writers:
                    writers.append(act)
        replay_fn = (
            getattr(model, "replay_activation_swap", None) if allow_replay else None
        )
        replayed: Set[Tuple[Any, Any]] = set()
        max_replays = 5
        for cell, writers in writers_by_cell.items():
            if len(writers) < 2:
                continue
            col, _aid = cell
            a, b = writers[0], writers[1]
            witness = None
            note = (
                " Witness replay was skipped because a stage is open or the "
                "swap is not available; divergence_witness is inconclusive."
            )
            pair = (a, b)
            if (
                allow_replay
                and callable(replay_fn)
                and pair not in replayed
                and len(replayed) < max_replays
            ):
                witness = replay_fn(a, b)
                replayed.add(pair)
                if witness is True:
                    note = (
                        " Witness replay under swapped order confirmed "
                        "schedule divergence."
                    )
                elif witness is False:
                    note = (
                        " Witness replay under swapped order produced "
                        "identical final state."
                    )
                else:
                    note = " Witness replay was inconclusive."
            cert.add(ContractViolation(
                "activation_race",
                f"Distinct activations {a} and {b} wrote the same cell {cell}."
                f"{note}",
                severity=SEVERITY_ERROR,
                columns=[col],
                ids=[a, b],
                divergence_witness=witness,
            ))
