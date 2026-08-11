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


@pytest.mark.unit
def test_git_revision_not_from_cwd(monkeypatch, tmp_path):
    """git_revision must not pick up an unrelated checkout's HEAD."""
    import ambr.provenance as prov

    # Clear env so only build-info / None is used
    monkeypatch.delenv("AMBER_GIT_REVISION", raising=False)
    monkeypatch.delenv("AMBER_APP_REVISION", raising=False)

    # Simulate a different repo in CWD with its own HEAD
    fake = tmp_path / "other_repo"
    fake.mkdir()
    (fake / ".git").mkdir()
    monkeypatch.chdir(fake)

    # Without env / build stamp, revision is None (not CWD git)
    assert prov._ambr_git_revision() is None

    monkeypatch.setenv("AMBER_GIT_REVISION", "deadbeefcafebabe")
    assert prov._ambr_git_revision() == "deadbeefcafebabe"

    monkeypatch.setenv("AMBER_APP_REVISION", "app-sha-1")
    assert prov._application_revision() == "app-sha-1"

    res = _Tiny({"steps": 1, "seed": 0, "show_progress": False}).run()
    assert res.info["git_revision"] == "deadbeefcafebabe"
    assert res.info["application_revision"] == "app-sha-1"
