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
