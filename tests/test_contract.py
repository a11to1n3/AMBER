"""Tests for the runtime snapshot-view contract conformance checker.

Covers the four ``Model.run(contract=...)`` modes and the three violation
classes the runtime monitor can observe at the buffered commit seam:
duplicate unreduced writes (OOP path), schema mutation, and population
mutation. Also locks in backward compatibility of the default ('off') path.
"""

import numpy as np
import polars as pl
import pytest

import ambr as am
from ambr.contract import (
    CONTRACT_MODES,
    ContractCertificate,
    ContractViolation,
    ContractViolationError,
)


# --- model fixtures --------------------------------------------------------

class CleanVectorModel(am.Model):
    """Row-local, snapshot-safe vectorized update: no conflict possible."""

    def setup(self):
        self.add_agents(20, wealth=np.ones(20, dtype=int))

    def step(self):
        self.agents.wealth = self.agents.wealth + 1


class _Walker(am.Agent):
    def setup(self):
        self.x = 0


class DuplicateWriteModel(am.Model):
    """OOP model that writes the same (column, id) twice per step."""

    def setup(self):
        self.add_agents(4, agent_class=_Walker)

    def step(self):
        for a in self.agents:
            a.x = a.x + 1
            a.x = a.x + 10  # second ordinary write to the same cell this step


class ScatterModel(am.Model):
    """Accumulation through the sanctioned commutative reducer (no conflict)."""

    def setup(self):
        self.add_agents(5, wealth=np.zeros(5, dtype=int))

    def step(self):
        # Agent 0 receives +2 (duplicate id), agent 2 receives +1.
        self.agents.at[[0, 0, 2]].scatter_add(wealth=1)


class RawArrayGatherModel(am.Model):
    """A raw array gather is observable but cannot be fully interpreted."""

    def setup(self):
        self.add_agents(3, x=np.array([1, 2, 3], dtype=np.int64))

    def step(self):
        x = self.agents.array("x")
        self.agents.x = np.roll(x, 1)


class BorrowBeforeCommitGatherModel(am.Model):
    """A one-borrow gather documents the monitor's provenance boundary."""

    def setup(self):
        self.add_agents(3, x=np.array([1, 2, 3], dtype=np.int64))

    def step(self):
        # The public column read is observed, but the monitor cannot infer that
        # np.roll introduces cross-agent dependence before the one commit.
        x = self.agents.x.to_numpy()
        self.agents.x = np.roll(x, 1)


class BirthModel(am.Model):
    def setup(self):
        self.add_agents(3, wealth=np.ones(3, dtype=int))

    def step(self):
        self.agents.wealth = self.agents.wealth + 1
        self.add_agents(1, wealth=[0])  # mid-step birth


class NewColumnModel(am.Model):
    def setup(self):
        self.add_agents(3, wealth=np.ones(3, dtype=int))

    def step(self):
        # Introduce a brand-new column mid-step (only happens on step 0).
        self.agents.energy = np.arange(len(self.agents))


def _params(**extra):
    base = {"steps": 5, "show_progress": False}
    base.update(extra)
    return base


# --- mode plumbing ---------------------------------------------------------

def test_off_mode_is_backward_compatible():
    res = CleanVectorModel(_params(steps=3)).run()
    assert "contract" not in res
    assert set(res) == {"info", "agents", "model"}
    assert res["agents"]["wealth"].to_list() == [4] * 20  # 1 + 3 steps


def test_off_mode_records_no_certificates():
    m = CleanVectorModel(_params(steps=3))
    m.run()
    assert m.contract_certificates == []
    assert m._contract_mode == "off"


def test_invalid_contract_mode_raises():
    with pytest.raises(ValueError):
        CleanVectorModel(_params(steps=1)).run(contract="bogus")


def test_check_mode_emits_one_certificate_per_step():
    m = CleanVectorModel(_params(steps=5))
    res = m.run(contract="check")
    assert "contract" in res
    assert len(res["contract"]) == 5
    assert res["contract"] is m.contract_certificates
    assert all(isinstance(c, ContractCertificate) for c in res["contract"])


# --- clean / sanctioned paths produce no errors ----------------------------

def test_clean_vectorized_model_is_conforming():
    res = CleanVectorModel(_params(steps=5)).run(contract="check")
    certs = res["contract"]
    assert all(c.clean for c in certs)
    assert all(c.ok for c in certs)


def test_scatter_add_is_not_flagged_as_duplicate():
    res = ScatterModel(_params(steps=3)).run(contract="check")
    certs = res["contract"]
    # scatter_add never touches the buffered write path -> no duplicate_write.
    assert all(c.clean for c in certs)
    # And it actually accumulated: agent 0 got +2 per step over 3 steps.
    assert res["agents"].filter(pl.col("id") == 0)["wealth"].item() == 6


def test_raw_mutable_array_borrow_cannot_receive_clean_certificate():
    res = RawArrayGatherModel(_params(steps=1)).run(contract="check")
    cert = res["contract"][0]
    assert cert.ok  # warning: the observed trace contains no definite conflict
    assert not cert.clean
    warning = next(
        v for v in cert.violations if v.kind == "uncertified_mutable_borrow"
    )
    assert warning.severity == "warning"
    assert warning.columns == ["x"]
    assert res["agents"]["x"].to_list() == [3, 1, 2]


def test_column_read_after_commit_is_detected_without_tensor_lane():
    class M(am.Model):
        def setup(self):
            self.add_agents(3, x=np.arange(3), y=np.zeros(3, dtype=int))

        def step(self):
            self.agents.x = self.agents.x + 1
            self.agents.y = self.agents.x

    cert = M(_params(steps=1)).run(contract="check")["contract"][0]
    assert not cert.ok
    raw = next(v for v in cert.violations if v.kind == "read_after_write")
    assert raw.columns == ["x"]


def test_borrow_before_commit_gather_documents_monitor_limit():
    result = BorrowBeforeCommitGatherModel(_params(steps=1)).run(
        contract="check"
    )
    cert = result["contract"][0]
    assert cert.clean
    # Snapshot roll gives [3, 1, 2]. Under sequential order 0,1,2, each cell
    # reads the already-updated predecessor and the result is [3, 3, 3]. The
    # clean certificate therefore coexists with a matched-reference mismatch.
    assert result["agents"]["x"].to_list() == [3, 1, 2]
    assert result["agents"]["x"].to_list() != [3, 3, 3]


def test_raise_mode_allows_clean_model_to_complete():
    res = CleanVectorModel(_params(steps=4)).run(contract="raise")
    assert len(res["contract"]) == 4


# --- duplicate write detection ---------------------------------------------

def test_duplicate_write_is_detected():
    m = DuplicateWriteModel(_params(steps=2))
    res = m.run(contract="check")
    certs = res["contract"]
    assert all(not c.ok for c in certs), "each step double-writes the same cell"
    kinds = {v.kind for c in certs for v in c.violations}
    assert "duplicate_write" in kinds
    # All four agents' x column should be named among the offending ids.
    first = certs[0]
    dup = next(v for v in first.violations if v.kind == "duplicate_write")
    assert dup.severity == "error"
    assert "x" in dup.columns
    assert set(dup.ids) == {0, 1, 2, 3}


class DoubleColumnCommitModel(am.Model):
    """View-path model that commits the same column twice per step."""

    def setup(self):
        self.add_agents(5, wealth=np.ones(5, dtype=int))

    def step(self):
        self.agents.wealth = self.agents.wealth + 1
        self.agents.wealth = self.agents.wealth + 1  # second whole-column commit


def test_view_path_double_commit_is_detected():
    """Whole-column view writes report commits; a second write is a conflict."""
    res = DoubleColumnCommitModel(_params(steps=1)).run(contract="check")
    cert = res["contract"][0]
    assert not cert.ok
    dup = next(v for v in cert.violations if v.kind == "duplicate_write")
    assert "wealth" in dup.columns
    assert "lane/view" in dup.detail


class CrossPathModel(am.Model):
    """Buffered OOP write then whole-column view write on the same column."""

    def setup(self):
        self.add_agents(3, agent_class=_Walker)

    def step(self):
        for a in self.agents:
            a.x = a.x + 1  # buffered path
        self.agents.x = self.agents.x + 10  # lane/view path


def test_cross_path_write_is_detected():
    res = CrossPathModel(_params(steps=1)).run(contract="check")
    cert = res["contract"][0]
    assert not cert.ok
    cross = next(v for v in cert.violations if v.kind == "cross_path_write")
    assert "x" in cross.columns
    assert cross.severity == "error"


def test_agents_set_is_atomic_single_commit_per_column():
    """One agents.set(...) must not flag duplicate_write for multi-column."""

    class M(am.Model):
        def setup(self):
            self.add_agents(4, x=np.zeros(4), y=np.zeros(4))

        def step(self):
            self.agents.set(x=self.agents.x + 1, y=self.agents.y + 1)

    res = M(_params(steps=3)).run(contract="check")
    assert all(c.clean for c in res["contract"])
    assert res["agents"]["x"].to_list() == [3, 3, 3, 3]
    assert res["agents"]["y"].to_list() == [3, 3, 3, 3]


def test_raise_mode_raises_on_duplicate_write():
    m = DuplicateWriteModel(_params(steps=5))
    with pytest.raises(ContractViolationError) as exc:
        m.run(contract="raise")
    # Fails fast on the very first step.
    assert exc.value.certificate.step == 0
    assert len(m.contract_certificates) == 1


def test_warn_mode_emits_warnings_for_duplicate_write():
    m = DuplicateWriteModel(_params(steps=1))
    with pytest.warns(UserWarning, match="duplicate_write"):
        m.run(contract="warn")
    # warn mode does not stop the run.
    assert len(m.contract_certificates) == 1


# --- lifecycle: population + schema mutation -------------------------------

def test_population_mutation_is_warned_not_errored():
    m = BirthModel(_params(steps=3))
    res = m.run(contract="check")
    certs = res["contract"]
    kinds = {v.kind for c in certs for v in c.violations}
    assert "population_mutation" in kinds
    # Births are a warning, not an error: the certificate stays ok.
    assert all(c.ok for c in certs)
    pop_viol = next(
        v for c in certs for v in c.violations if v.kind == "population_mutation"
    )
    assert pop_viol.severity == "warning"


def test_schema_addition_is_warned_on_first_step_only():
    m = NewColumnModel(_params(steps=3))
    res = m.run(contract="check")
    certs = res["contract"]
    first_kinds = {v.kind for v in certs[0].violations}
    assert "schema_mutation" in first_kinds
    assert certs[0].ok  # adding a column is a warning, not an error
    # 'energy' exists from step 1 onward, so no further schema mutations.
    later_kinds = {v.kind for c in certs[1:] for v in c.violations}
    assert "schema_mutation" not in later_kinds


# --- certificate object semantics ------------------------------------------

def test_certificate_ok_vs_clean_semantics():
    cert = ContractCertificate(step=7)
    assert cert.clean and cert.ok
    cert.add(ContractViolation("population_mutation", "birth", severity="warning"))
    assert cert.ok and not cert.clean  # warnings don't flip ok
    cert.add(ContractViolation("duplicate_write", "conflict", severity="error"))
    assert not cert.ok and not cert.clean
    assert len(cert.errors()) == 1
    assert len(cert.warnings()) == 1


def test_contract_modes_constant():
    assert CONTRACT_MODES == ("off", "check", "warn", "raise")
