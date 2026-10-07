"""Activation-race witness: no replay while a stage is open."""

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
