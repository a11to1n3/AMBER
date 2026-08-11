"""Run provenance fields on results.info."""

from __future__ import annotations

import ambr as am
import pytest


class _Tiny(am.Model):
    model_reporters = {"total": lambda m: int(m.agents.wealth.sum())}

    def setup(self):
        self.add_agents(10, wealth=1)

    def step_vectorized(self):
        donors = self.agents.where(self.agents.wealth > 0)
        if len(donors) == 0:
            return
        donors.wealth -= 1
        rec = self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
        self.agents.at[rec].scatter_add(wealth=1)


@pytest.mark.unit
def test_results_info_has_provenance_fields():
    res = _Tiny({"steps": 3, "seed": 7, "show_progress": False}).run()
    info = res.info
    required = [
        "ambr_version",
        "python_version",
        "model_class",
        "parameters",
        "seed",
        "start_time",
        "end_time",
        "status",
        "completion_status",
        "run_uuid",
        "config_hash",
        "polars_version",
        "numpy_version",
        "device",
        "mode",
        "execution_lane",
        "steps",
        "run_time",
    ]
    for key in required:
        assert key in info, f"missing results.info[{key!r}]"

    assert info["seed"] == 7
    assert info["steps"] == 3
    assert info["status"] == "completed"
    assert "Tiny" in info["model_class"]
    assert info["parameters"].get("seed") == 7
    assert info["device"] in {"cpu", "gpu"}
    assert info["mode"] in {"vectorized", "oop"}
    assert len(info["run_uuid"]) >= 8
    assert len(info["config_hash"]) == 16


@pytest.mark.unit
def test_failed_run_propagates_exception():
    class Boom(am.Model):
        def setup(self):
            pass

        def step(self):
            raise ValueError("visible failure")

    with pytest.raises(ValueError, match="visible failure"):
        Boom({"steps": 1, "show_progress": False}).run()
