"""RunResults save/load and Experiment constructor polish."""

from __future__ import annotations

import warnings

import ambr as am
import pytest


class _Tiny(am.Model):
    model_reporters = {"total": lambda m: int(m.agents.wealth.sum())}

    def setup(self):
        self.add_agents(15, wealth=1)

    def step_vectorized(self):
        donors = self.agents.where(self.agents.wealth > 0)
        donors.wealth -= 1
        rec = self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
        self.agents.at[rec].scatter_add(wealth=1)


@pytest.mark.unit
def test_runresults_save_load(tmp_path):
    r = _Tiny({"steps": 5, "seed": 0, "show_progress": False}).run()
    assert isinstance(r, am.RunResults)
    overview = r.keys_overview()
    assert "model" in overview and "agents" in overview

    dest = tmp_path / "run0"
    r.save(dest)
    assert (dest / "model.parquet").is_file()
    assert (dest / "agents.parquet").is_file()
    assert (dest / "info.json").is_file()

    loaded = am.RunResults.load(dest)
    assert loaded.model.height == r.model.height
    assert loaded.agents.height == r.agents.height
    assert loaded.info.get("steps") == r.info.get("steps")


@pytest.mark.unit
def test_experiment_canonical_and_legacy_kwargs():
    sample = am.Sample(
        {"steps": 3, "seed": [0, 1], "show_progress": False},
        n=2,
    )
    exp = am.Experiment(model_type=_Tiny, sample=sample, iterations=1)
    out = exp.run()
    assert "model" in out and out["model"].height >= 2

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exp2 = am.Experiment(
            model_class=_Tiny,
            parameters=sample,
            iterations=1,
        )
        assert any("model_class" in str(x.message) or "deprecated" in str(x.message).lower() for x in w)
    out2 = exp2.run()
    assert out2["parameters"].height == 2


@pytest.mark.unit
def test_experiment_requires_sample_instance():
    with pytest.raises(TypeError, match="model_type"):
        am.Experiment(sample=am.Sample({"steps": 1}, n=1))
    with pytest.raises(TypeError, match="sample"):
        am.Experiment(model_type=_Tiny)
