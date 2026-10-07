"""Activation-race witness: no replay while a stage is open."""

import numpy as np

import ambr as am


def _params():
    return {"steps": 1, "show_progress": False}


class _Writer(am.Agent):
    def step(self):
        self.model.agents[0].wealth = self.id


class _SameValue(am.Agent):
    def step(self):
        self.model.agents[0].wealth = 1


class _OwnWealth(am.Agent):
    def setup(self):
        # Column values from add_agents live on the frame. The Python object
        # only sees an attribute after it is assigned.
        self.wealth = 0

    def step(self):
        self.wealth = self.wealth + 1


class _StagedRace(am.Model):
    def setup(self):
        self.add_agents(2, agent_class=_Writer, wealth=[0, 0])

    def step(self):
        for agent in self.stage_agents("write"):
            agent.step()


class _TwoStage(am.Model):
    """First stage conflicts; second stage writes a column the first has not."""

    def setup(self):
        self.add_agents(2, agent_class=_Writer, wealth=[0, 0], extra=[0, 0])

    def step(self):
        for agent in self.stage_agents("first"):
            self.agents[0].wealth = 1
        for agent in self.stage_agents("second"):
            agent.extra = agent.id


class _UnstagedRace(am.Model):
    def setup(self):
        self.add_agents(2, agent_class=_Writer, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


class _UnstagedSame(am.Model):
    def setup(self):
        self.add_agents(2, agent_class=_SameValue, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


class _CleanStaged(am.Model):
    def setup(self):
        self.add_agents(3, agent_class=_OwnWealth, wealth=[0, 0, 0])

    def step(self):
        for agent in self.stage_agents():
            agent.step()


def _races(cert):
    return [v for v in cert.violations if v.kind == "activation_race"]


def test_staged_conflict_does_not_recurse_and_witness_is_inconclusive():
    model = _StagedRace(_params())
    res = model.run(contract="check")
    cert = res["contract"][0]
    races = _races(cert)
    assert races
    assert all(v.divergence_witness is None for v in races)
    assert res["agents"].sort("id")["wealth"].to_list() == [1, 0]
    assert model.agents[0].wealth == 1
    assert model._contract.mode == "check"
    assert model._contract.active is False
    assert model._contract._in_replay is False


def test_two_stage_conflict_is_not_a_mid_step_divergence():
    model = _TwoStage(_params())
    res = model.run(contract="check")
    cert = res["contract"][0]
    races = _races(cert)
    assert races
    assert all(v.divergence_witness is None for v in races)
    frame = res["agents"].sort("id")
    assert frame["wealth"].to_list() == [1, 0]
    assert frame["extra"].to_list() == [0, 1]


def test_unstaged_conflict_replay_reports_a_bool_and_restores_frame():
    model = _UnstagedRace(_params())
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert len(races) == 1
    assert races[0].divergence_witness is True
    assert res["agents"].sort("id")["wealth"].to_list() == [1, 0]
    assert model.agents[0].wealth == 1
    assert model._contract.mode == "check"
    assert model._activation_override_order is None


def test_unstaged_identical_writes_do_not_diverge():
    model = _UnstagedSame(_params())
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert len(races) == 1
    assert races[0].divergence_witness is False
    assert res["agents"].sort("id")["wealth"].to_list() == [1, 0]


def test_disjoint_staged_writes_stay_clean():
    res = _CleanStaged(_params()).run(contract="check")
    cert = res["contract"][0]
    assert cert.ok and cert.clean
    assert res["agents"]["wealth"].to_list() == [1, 1, 1]


class _ScratchAgent(am.Agent):
    def step(self):
        self.model.scratch.append(self.id)
        self.model.agents[0].cell = len(self.model.scratch)


class _ScratchModel(am.Model):
    def setup(self):
        self.scratch = []
        self.add_agents(2, agent_class=_ScratchAgent, cell=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


class _Box:
    def __init__(self):
        self.n = 0


class _BoxAgent(am.Agent):
    def step(self):
        self.model.box.n += 1
        self.model.agents[0].wealth = self.id


class _BoxModel(am.Model):
    def setup(self):
        self.box = _Box()
        self.add_agents(2, agent_class=_BoxAgent, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


class _BufferAgent(am.Agent):
    def step(self):
        self.model.buf = np.append(self.model.buf, self.id)
        self.model.agents[0].wealth = self.id


class _BufferModel(am.Model):
    def setup(self):
        self.buf = np.array([], dtype=int)
        self.add_agents(2, agent_class=_BufferAgent, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


def test_mutated_list_makes_the_witness_inconclusive_and_is_restored():
    model = _ScratchModel(_params())
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert races
    assert all(v.divergence_witness is None for v in races)
    assert model.scratch == [0, 1]
    assert res["agents"].sort("id")["cell"].to_list() == [2, 0]


def test_custom_model_object_skips_the_swap():
    model = _BoxModel(_params())
    model._ensure_setup()
    box = model.box
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert len(races) == 1
    assert races[0].divergence_witness is None
    assert model.box is box
    assert model.box.n == 2
    assert res["agents"].sort("id")["wealth"].to_list() == [1, 0]


def test_mutated_ndarray_is_restored_and_witness_is_inconclusive():
    model = _BufferModel(_params())
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert races
    assert all(v.divergence_witness is None for v in races)
    assert model.buf.tolist() == [0, 1]
    assert res["agents"].sort("id")["wealth"].to_list() == [1, 0]
