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
    # n_trials is per objective (2 objectives × 3 trials).
    assert out["n_evaluations"] == 6


def test_multiobjective_strategy_is_forwarded_not_ignored():
    space = SMACParameterSpace()
    space.add_parameter("x", param_type="float", bounds=(0.0, 3.0), default=1.0)

    with pytest.raises(ValueError, match="not a search strategy"):
        MultiObjectiveSMAC(
            model_type=_TinyModel,
            param_space=space,
            objectives={"a": _obj_a, "b": _obj_b},
            n_trials=1,
            strategy="pareto",
        )

    with pytest.raises(ValueError, match="Unknown strategy"):
        MultiObjectiveSMAC(
            model_type=_TinyModel,
            param_space=space,
            objectives={"a": _obj_a, "b": _obj_b},
            n_trials=1,
            strategy="not_a_real_strategy",
        )

    from smac import RandomFacade

    mo = MultiObjectiveSMAC(
        model_type=_TinyModel,
        param_space=space,
        objectives={"a": _obj_a, "b": _obj_b},
        n_trials=2,
        seed=0,
        strategy="random",
        fixed_params={"steps": 2, "show_progress": False},
    )
    opts = mo._ensure_optimizers()
    try:
        assert mo.strategy == "random"
        assert all(isinstance(opt.smac, RandomFacade) for opt in opts.values())
        assert all(opt.fixed_params.get("steps") == 2 for opt in opts.values())
    finally:
        for opt in opts.values():
            opt._cleanup_output_dir()
            opt._shutdown_parallel_resources()


def test_multiobjective_fixed_params_used_when_rescoring_incumbents():
    space = SMACParameterSpace()
    space.add_parameter("x", param_type="float", bounds=(0.0, 3.0), default=1.0)
    mo = MultiObjectiveSMAC(
        model_type=_TinyModel,
        param_space=space,
        objectives={"a": _obj_a},
        n_trials=2,
        seed=0,
        strategy="random",
        fixed_params={"steps": 2, "show_progress": False},
    )
    out = mo.optimize()
    assert out["n_evaluations"] == 2
    assert out["pareto_front"].height >= 1


def test_multiobjective_rescore_uses_optimizer_seed():
    """Pareto values must match a rerun with the same model seed."""

    class _Stochastic(am.Model):
        params = {"steps": (int, 3), "show_progress": (bool, False)}

        def setup(self):
            n = 8
            self.add_agents(n, x=self.rng.random(n))

        def step(self):
            self.agents.x = self.agents.x + float(self.p.get("x", 0.0)) + self.rng.random(8) * 0.01

        def update(self):
            if self.t > 0:
                self.record_model("a", float(self.agents.x.mean()))

    def obj(model: am.Model) -> float:
        return float(model.results["model"]["a"].tail(1).item())

    space = SMACParameterSpace()
    space.add_parameter("x", param_type="float", bounds=(0.0, 1.0), default=0.2)
    mo = MultiObjectiveSMAC(
        model_type=_Stochastic,
        param_space=space,
        objectives={"a": obj},
        n_trials=2,
        seed=7,
        strategy="random",
        fixed_params={"steps": 3, "show_progress": False},
    )
    assert mo.fixed_params.get("seed") == 7
    out = mo.optimize()
    row = out["pareto_front"].row(0, named=True)
    replay = _Stochastic(
        {
            "x": row["x"],
            "steps": 3,
            "seed": 7,
            "show_progress": False,
        }
    )
    replay.results = replay.run()
    assert abs(obj(replay) - float(row["a"])) < 1e-12


def test_multiobjective_pins_deterministic_when_model_seed_is_fixed():
    space = SMACParameterSpace()
    space.add_parameter("x", param_type="float", bounds=(0.0, 3.0), default=1.0)
    mo = MultiObjectiveSMAC(
        model_type=_TinyModel,
        param_space=space,
        objectives={"a": _obj_a, "b": _obj_b},
        n_trials=3,
        seed=0,
        strategy="random",
        fixed_params={"steps": 2, "show_progress": False},
    )
    assert mo.deterministic is True
    opts = mo._ensure_optimizers()
    try:
        assert all(opt.deterministic for opt in opts.values())
        assert all(bool(opt.scenario.deterministic) for opt in opts.values())
    finally:
        for opt in opts.values():
            opt._cleanup_output_dir()
            opt._shutdown_parallel_resources()


def test_multiobjective_seed_none_is_not_deterministic():
    space = SMACParameterSpace()
    space.add_parameter("x", param_type="float", bounds=(0.0, 3.0), default=1.0)
    mo = MultiObjectiveSMAC(
        model_type=_TinyModel,
        param_space=space,
        objectives={"a": _obj_a},
        n_trials=1,
        seed=0,
        strategy="random",
        fixed_params={"steps": 2, "seed": None, "show_progress": False},
    )
    assert mo.fixed_params.get("seed") is None
    assert mo.deterministic is False
    opts = mo._ensure_optimizers()
    try:
        assert all(opt.deterministic is False for opt in opts.values())
        assert all(not bool(opt.scenario.deterministic) for opt in opts.values())
    finally:
        for opt in opts.values():
            opt._cleanup_output_dir()
            opt._shutdown_parallel_resources()
