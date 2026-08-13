"""Step-data lifecycle: preserve step() recordings, update wins, failed steps.

Locks in the precedence contract:

    step() record_model  <  model_reporters  <  update() record_model

and that a raised step never appends a partial model-data row. Contract modes
must share the same recording behaviour as ``contract='off'``.
"""

from __future__ import annotations

import pytest

import ambr as am
from ambr.contract import ContractViolationError


class _StepRecorder(am.Model):
    """Records ``from_step`` inside step() using the pre-increment ``t``."""

    def setup(self):
        self.add_agents(2, x=0)

    def step(self):
        # During step, self.t is still the pre-step counter; the row's ``t``
        # key is already the post-step value (self.t + 1).
        self.record_model("from_step", 100 + self.t)
        self.agents.x = self.agents.x + 1


class _StepAndUpdate(am.Model):
    def setup(self):
        pass

    def step(self):
        self.record_model("metric", "from_step")

    def update(self):
        self.record_model("metric", "from_update")


class _StepReporterUpdate(am.Model):
    model_reporters = {"metric": lambda m: "from_reporter"}

    def setup(self):
        pass

    def step(self):
        self.record_model("metric", "from_step")

    def update(self):
        self.record_model("metric", "from_update")


class _FailingStep(am.Model):
    def setup(self):
        self.n = 0

    def step(self):
        self.n += 1
        self.record_model("partial", self.n)
        if self.n == 2:
            raise RuntimeError("boom at step 2")


class _DupWrite(am.Agent):
    def setup(self):
        self.x = 0


class _ContractConflict(am.Model):
    """OOP model that writes the same cell twice (contract violation)."""

    def setup(self):
        self.add_agents(2, agent_class=_DupWrite)

    def step(self):
        self.record_model("from_step", 42)
        for a in self.agents:
            a.x = a.x + 1
            a.x = a.x + 10


@pytest.mark.unit
def test_record_model_inside_step_is_retained():
    res = _StepRecorder({"show_progress": False, "seed": 0}).run(3)
    md = res["model"]
    assert "from_step" in md.columns
    # step saw t=0,1,2 → recorded 100,101,102; row t is 1,2,3
    assert md["from_step"].to_list() == [100, 101, 102]
    assert md["t"].to_list() == [1, 2, 3]


@pytest.mark.unit
def test_update_overwrites_same_metric():
    res = _StepAndUpdate({"show_progress": False}).run(2)
    assert res["model"]["metric"].to_list() == ["from_update", "from_update"]


@pytest.mark.unit
def test_precedence_step_reporter_update():
    """Later stages win: step < reporters < update."""
    res = _StepReporterUpdate({"show_progress": False}).run(1)
    assert res["model"]["metric"].to_list() == ["from_update"]

    class ReporterWins(am.Model):
        model_reporters = {"metric": lambda m: "from_reporter"}

        def setup(self):
            pass

        def step(self):
            self.record_model("metric", "from_step")

    res2 = ReporterWins({"show_progress": False}).run(1)
    assert res2["model"]["metric"].to_list() == ["from_reporter"]


@pytest.mark.unit
def test_failed_step_does_not_append_partial_row():
    m = _FailingStep({"show_progress": False})
    m.run_step()  # step 1 ok
    assert len(m._model_data) == 1
    assert m._model_data[0]["partial"] == 1

    with pytest.raises(RuntimeError, match="boom"):
        m.run_step()  # step 2 fails

    # Partial row discarded; model data unchanged; t not advanced
    assert len(m._model_data) == 1
    assert m.t == 1
    assert m._current_step_data == {}


@pytest.mark.unit
@pytest.mark.parametrize("contract", ["off", "check", "warn"])
def test_contract_modes_retain_step_recordings(contract):
    m = _StepRecorder({"show_progress": False, "seed": 1})
    res = m.run(2, contract=contract)
    md = res["model"]
    assert md["from_step"].to_list() == [100, 101]
    assert md["t"].to_list() == [1, 2]


@pytest.mark.unit
def test_contract_raise_discards_partial_row_on_violation():
    m = _ContractConflict({"show_progress": False})
    with pytest.raises(ContractViolationError):
        m.run(1, contract="raise")
    # Violation aborts before finalize — no model row for the failed step
    assert m._model_data == []
    assert m.t == 0
    assert m._current_step_data == {}
