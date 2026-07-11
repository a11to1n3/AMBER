"""UX helpers: RunResults attr access, agents.random, AgentPy-shaped runs."""

import ambr as am
from ambr.results import RunResults


class _Walker(am.Agent):
    def setup(self):
        self.wealth = 1

    def give(self):
        if self.wealth > 0:
            self.wealth -= 1


class _OOPModel(am.Model):
    def setup(self):
        # AgentPy shape: AgentList(model, n, AgentType)
        self.agents = am.AgentList(self, 8, _Walker)

    def step(self):
        self.agents.give()  # method broadcast
        other = self.agents.random()
        self.agents.by_id(other).wealth += 1


def test_run_results_attr_and_dict_access():
    m = _OOPModel({"steps": 3, "seed": 0})
    res = m.run()
    assert isinstance(res, RunResults)
    assert res["agents"] is res.agents
    assert res["model"] is res.model
    assert "steps" in res.info
    assert res.agents.height == 8


def test_agents_random_samples_ids():
    m = _OOPModel({"steps": 1, "seed": 1})
    m.setup()
    one = m.agents.random()
    assert one in m.agents.ids.to_list()
    many = m.agents.random(n=3, replace=False)
    assert len(many) == 3
    assert len(set(many)) == 3


def test_show_progress_defaults_off():
    m = am.Model({"steps": 1})
    assert m._show_progress is False
