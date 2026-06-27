"""Runtime conformance checking for AMBER's snapshot-view contract.

This module implements the *runtime* half of the view contract described in the
AMBER paper (the propositions classifying which agent-rule classes preserve
their schedule semantics under columnar / simultaneous-update execution). It
turns that contract from prose into a falsifiable, per-step artifact: a
:class:`ContractCertificate` that witnesses whether the writes committed during
a step are admissible under simultaneous-activation semantics.

It witnesses violations on *two* write paths:

* the **buffered (OOP) path** -- per-agent ``Agent.__setattr__`` writes routed
  through ``Model._queue_write``; and
* the **vectorized lane/view path** -- ``population.data = df.with_columns(...)``,
  which bypasses ``_queue_write`` but is reported to the model by the tensor
  lane (``commit_columns`` / ``TensorLane``).

The checks:

* **duplicate unreduced write** -- the same ``(column, agent_id)`` cell received
  more than one *ordinary* (non-scatter) write within a step (buffered path), or
  the same column was committed more than once within a step (lane path). Under
  simultaneous activation the later write silently clobbers the earlier one (a
  "partial map" conflict). The sanctioned fix is a commutative reducer
  (``agents.at[ids].scatter_add(col=delta)``) or committing each column once.
* **read-after-write (lane path)** -- a column borrowed *after* it was committed
  in the same step; a snapshot rule must read step-entry state.
* **schema / population mutation** -- the column set, a column's *dtype*, or the
  agent id-set changed between step entry and exit, so reads issued during the
  step no longer see a stable snapshot. Column/population *additions* and
  births/deaths are reported as *warnings*; column removal and dtype change are
  *errors*.

This is a conformance *monitor*, not a prover, with two known limits: (1) the
schema/population diff is **endpoint-only** -- a mutation that is introduced and
reverted within a single step is not caught; (2) raw ``population.data = ...``
writes that bypass both ``_queue_write`` *and* the tensor lane are invisible to
the conflict check. Cell-level read-after-write within the buffered view API
remains the job of the (separate) static analyzer.
"""

from typing import Any, Iterable, List, Optional

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

#: Valid values for ``Model.run(contract=...)``.
CONTRACT_MODES = ("off", "check", "warn", "raise")


class ContractViolation:
    """A single admissibility violation observed during one step."""

    def __init__(
        self,
        kind: str,
        detail: str,
        severity: str = SEVERITY_ERROR,
        columns: Optional[Iterable[str]] = None,
        ids: Optional[Iterable[Any]] = None,
    ):
        self.kind = kind
        self.detail = detail
        self.severity = severity
        self.columns: List[str] = list(columns) if columns else []
        self.ids: List[Any] = list(ids) if ids else []

    def __repr__(self) -> str:
        loc = ""
        if self.columns:
            loc += f" columns={self.columns}"
        if self.ids:
            shown = self.ids[:8]
            more = f" (+{len(self.ids) - 8} more)" if len(self.ids) > 8 else ""
            loc += f" ids={shown}{more}"
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
        return (
            f"<ContractCertificate step={self.step} "
            f"errors={len(self.errors())} warnings={len(self.warnings())}>"
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
