"""Activation-race witness: no replay while a stage is open."""

import random

from ambr.contract import ContractCertificate, ContractViolation
from ambr.results import RunResults

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


class _ConstDraw(am.Agent):
    def setup(self):
        self.wealth = 0
        self.noise = 0.0

    def step(self):
        self.model.agents[0].wealth = 1


class _ConstDrawModel(am.Model):
    def setup(self):
        self.add_agents(2, agent_class=_ConstDraw, wealth=[0, 0], noise=[0.0, 0.0])

    def step(self):
        self.activate_agents(mode="sequential")
        self.agents[0].noise = float(self.rng.random())
        self.agents[1].noise = float(self.random.random())


def test_same_constant_write_plus_later_draw_is_not_a_false_divergence():
    params = {"steps": 1, "seed": 5, "show_progress": False}
    checked = _ConstDrawModel(params)
    res = checked.run(contract="check")
    races = _races(res["contract"][0])
    assert len(races) == 1
    assert races[0].divergence_witness is False
    plain = _ConstDrawModel(params).run(contract="off")
    assert res["agents"].sort("id")["noise"].to_list() == (
        plain["agents"].sort("id")["noise"].to_list()
    )


def test_monitoring_does_not_change_later_draws():
    params = {"steps": 2, "seed": 5, "show_progress": False}
    checked = _ConstDrawModel(params)
    plain = _ConstDrawModel(params)
    checked.run(contract="check")
    plain.run(contract="off")
    assert checked.rng.random() == plain.rng.random()
    assert checked.random.random() == plain.random.random()


class _CopyNeighbor(am.Agent):
    def setup(self):
        self.wealth = int(self.id)

    def step(self):
        other = self.model.agents[1 - int(self.id)]
        self.wealth = other.wealth + 1


class _ReadWriteModel(am.Model):
    def setup(self):
        self.add_agents(2, agent_class=_CopyNeighbor, wealth=[0, 1])

    def step(self):
        self.activate_agents(mode="sequential")


class _StagedReadWriteModel(am.Model):
    def setup(self):
        self.add_agents(2, agent_class=_CopyNeighbor, wealth=[0, 1])

    def step(self):
        for agent in self.stage_agents("copy"):
            agent.step()


def test_read_write_dependency_is_not_clean_and_can_diverge():
    res = _ReadWriteModel(_params()).run(contract="check")
    cert = res["contract"][0]
    assert not cert.clean
    races = _races(cert)
    assert races
    assert all(v.divergence_witness is True for v in races)
    assert res["agents"].sort("id")["wealth"].to_list() == [2, 3]


def test_staged_read_write_dependency_skips_replay():
    res = _StagedReadWriteModel(_params()).run(contract="check")
    cert = res["contract"][0]
    assert not cert.clean
    races = _races(cert)
    assert races
    assert all(v.divergence_witness is None for v in races)


def test_divergence_witness_round_trips_all_three_values(tmp_path):
    cert = ContractCertificate(step=1)
    for value in (True, False, None):
        cert.add(ContractViolation(
            "activation_race",
            f"witness {value}",
            severity="error",
            columns=["wealth"],
            ids=[0, 1],
            divergence_witness=value,
        ))
    dest = tmp_path / "witness"
    RunResults({"contract": [cert], "info": {"steps": 1}}).save(dest)
    loaded = RunResults.load(dest)["contract"][0]["violations"]
    assert [item["divergence_witness"] for item in loaded] == [True, False, None]
    assert [item["divergence_reason"] for item in loaded] == [None, None, None]


class _SwapReplacesRandom(am.Agent):
    def setup(self):
        self.wealth = 0

    def step(self):
        previous = int(self.model.agents[0].wealth)
        self.model.agents[0].wealth = 1
        if self.id == 0 and previous == 1:
            self.model.random = random.Random(0)


class _SwapReplacesRandomModel(am.Model):
    def setup(self):
        self.add_agents(2, agent_class=_SwapReplacesRandom, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


class _AlternateRaises(am.Agent):
    def setup(self):
        self.wealth = 0

    def step(self):
        previous = int(self.model.agents[0].wealth)
        self.model.agents[0].wealth = self.id
        if self.id == 0 and previous == 1:
            raise ValueError("alternate order")


class _AlternateRaisesModel(am.Model):
    def setup(self):
        self.add_agents(2, agent_class=_AlternateRaises, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


def test_rng_restore_rebinds_a_replaced_stream():
    model = _UnstagedSame({"steps": 1, "seed": 1, "show_progress": False})
    model._ensure_setup()
    snapshot = model._capture_rng_state()
    original = model.random
    model.random = random.Random(0)
    assert model._restore_rng_state(snapshot) is True
    assert model.random is original


def test_swapped_random_replacement_is_not_left_installed():
    params = {"steps": 1, "seed": 5, "show_progress": False}
    checked = _SwapReplacesRandomModel(params)
    checked._ensure_setup()
    original = checked.random
    res = checked.run(contract="check")
    plain = _SwapReplacesRandomModel(params)
    plain_res = plain.run(contract="off")
    races = _races(res["contract"][0])
    assert races
    assert all(v.divergence_witness is False for v in races)
    assert checked.random is original
    assert checked.random.getstate() == plain.random.getstate()
    assert res["agents"].sort("id")["wealth"].to_list() == (
        plain_res["agents"].sort("id")["wealth"].to_list()
    )


def test_swapped_execution_exception_does_not_abort_the_forward_run(tmp_path):
    params = {"steps": 1, "seed": 3, "show_progress": False}
    plain = _AlternateRaisesModel(params).run(contract="off")
    checked = _AlternateRaisesModel(params)
    res = checked.run(contract="check")
    races = _races(res["contract"][0])
    assert races
    assert all(v.divergence_witness is None for v in races)
    assert all(v.divergence_reason and "ValueError" in v.divergence_reason for v in races)
    assert all("alternate order" in v.divergence_reason for v in races)
    assert all("ValueError" in v.detail for v in races)
    assert res["agents"].sort("id")["wealth"].to_list() == (
        plain["agents"].sort("id")["wealth"].to_list()
    )
    assert checked._contract.mode == "check"
    assert checked._contract._in_replay is False
    dest = tmp_path / "replay-error"
    res.save(dest)
    loaded = RunResults.load(dest)["contract"][0]["violations"]
    race = next(item for item in loaded if item["kind"] == "activation_race")
    assert race["divergence_witness"] is None
    assert "ValueError" in race["divergence_reason"]
