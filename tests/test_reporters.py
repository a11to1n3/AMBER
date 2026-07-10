"""Phase 4: pure-hook update(), declarative reporters, and record_initial.

The step counter and step-data lifecycle move into run_step, so update() is a
pure hook (no super().update() footgun) while staying backward compatible.
"""

import numpy as np

import ambr as am


def test_update_is_pure_hook_no_super_needed():
    class M(am.Model):
        def setup(self):
            self.add_agents(3, x=np.zeros(3))

        def step(self):
            self.agents.x = self.agents.x + 1

        def update(self):  # deliberately no super().update()
            self.record_model("total", float(self.agents.frame["x"].sum()))

    res = M({"show_progress": False}).run(5)
    md = res["model"]
    assert md["t"].to_list() == [1, 2, 3, 4, 5]
    assert md["total"].to_list() == [3, 6, 9, 12, 15]


def test_super_update_still_legal_no_double_count():
    class M(am.Model):
        def setup(self):
            self.add_agents(2, x=np.zeros(2))

        def step(self):
            self.agents.x = self.agents.x + 1

        def update(self):
            super().update()  # legacy style must not double-advance t
            self.record_model("sum", float(self.agents.frame["x"].sum()))

    m = M({"show_progress": False})
    res = m.run(4)
    assert m.t == 4
    assert res["model"]["t"].to_list() == [1, 2, 3, 4]


def test_model_reporters_declarative():
    class M(am.Model):
        model_reporters = {
            "n": lambda m: len(m.agents),
            "mean_x": lambda m: float(m.agents.frame["x"].mean()),
        }

        def setup(self):
            self.add_agents(4, x=np.arange(4, dtype=float))

        def step(self):
            self.agents.x = self.agents.x + 1

    md = M({"show_progress": False}).run(3)["model"]
    assert md["n"].to_list() == [4, 4, 4]
    assert md["mean_x"].to_list() == [2.5, 3.5, 4.5]


def test_record_initial_adds_t0_row():
    class M(am.Model):
        record_initial = True
        model_reporters = {"mean_x": lambda m: float(m.agents.frame["x"].mean())}

        def setup(self):
            self.add_agents(4, x=np.arange(4, dtype=float))

        def step(self):
            self.agents.x = self.agents.x + 1

    md = M({"show_progress": False}).run(2)["model"]
    assert md["t"].to_list() == [0, 1, 2]
    assert md["mean_x"].to_list() == [1.5, 2.5, 3.5]


def test_model_reporters_default_is_not_shared_mutable():
    """Base Model.model_reporters must not be a shared {} (classic class-attr leak)."""
    assert am.Model.model_reporters is None
    assert am.Model.agent_reporters is None
    assert am.Model.params is None

    class A(am.Model):
        model_reporters = {"n": lambda m: 0}

    class B(am.Model):
        pass

    A.model_reporters["leaked"] = lambda m: 1
    assert "leaked" not in (B.model_reporters or {})
    assert am.Model.model_reporters is None
    del A.model_reporters["leaked"]


def test_agent_reporters_long_frame():
    class M(am.Model):
        agent_reporters = ["x"]

        def setup(self):
            self.add_agents(3, x=np.zeros(3))

        def step(self):
            self.agents.x = self.agents.x + 1

    res = M({"show_progress": False}).run(2)
    assert "agent_vars" in res
    av = res["agent_vars"]
    assert set(av.columns) == {"id", "x", "t"}
    assert av.height == 6  # 3 agents x 2 steps
    assert sorted(av["t"].unique().to_list()) == [1, 2]


def test_no_agent_vars_key_when_unused():
    class M(am.Model):
        def setup(self):
            self.add_agents(2, x=np.zeros(2))

        def step(self):
            pass

    assert "agent_vars" not in M({"show_progress": False}).run(2)
