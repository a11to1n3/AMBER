"""CI-oriented MultiObjectiveSMAC smoke (requires smac extra)."""

import pytest
import numpy as np

from ambr.optimization import HAS_SMAC, MultiObjectiveSMAC, SMACParameterSpace
import ambr as am


pytestmark = pytest.mark.skipif(not HAS_SMAC, reason="SMAC3 not installed")


class _TinyModel(am.Model):
    """Cheap model: two metrics from a single parameter."""

    # Fixed short horizon so SMAC does not need a degenerate steps HP.
    params = {"steps": (int, 2), "show_progress": (bool, False)}

    def setup(self):
        self.add_agents(4, x=np.zeros(4))

    def step(self):
        # Drive both metrics from x so objectives are meaningful.
        x = float(self.p.get("x", 0.0))
        self.agents.x = self.agents.x + x

    def update(self):
        if self.t > 0:
            mean_x = float(self.agents.x.mean())
            self.record_model("a", abs(mean_x - 1.0))
            self.record_model("b", abs(mean_x - 2.0))


def _obj_a(model: am.Model) -> float:
    return float(model.results["model"]["a"].tail(1).item())


def _obj_b(model: am.Model) -> float:
    return float(model.results["model"]["b"].tail(1).item())


def test_multiobjective_smac_smoke_runs_and_returns_front():
    space = SMACParameterSpace()
    space.add_parameter("x", param_type="float", bounds=(0.0, 3.0), default=1.0)

    mo = MultiObjectiveSMAC(
        model_type=_TinyModel,
        param_space=space,
        objectives={"a": _obj_a, "b": _obj_b},
        n_trials=3,
        seed=0,
    )
    out = mo.optimize()

    assert "n_evaluations" in out
    assert out["n_evaluations"] >= 1
    assert "pareto_front" in out
    assert "single_objective_results" in out
    assert set(out["single_objective_results"]) == {"a", "b"}
    front = out["pareto_front"]
    assert front.height >= 1
    for col in ("a", "b", "x"):
        assert col in front.columns
