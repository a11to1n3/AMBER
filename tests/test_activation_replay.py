"""Activation-race witness: no replay while a stage is open."""

import numpy as np

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


class _CounterAgent(am.Agent):
    def step(self):
        self.model.count += 1
        self.model.agents[0].wealth = 1


class _CounterModel(am.Model):
    def setup(self):
        self.count = 0
        self.add_agents(2, agent_class=_CounterAgent, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


class _TupleAgent(am.Agent):
    def step(self):
        self.model.inner.append(self.id)
        self.model.agents[0].wealth = 1


class _TupleModel(am.Model):
    def setup(self):
        self.inner = []
        self.holder = (self.inner,)
        self.add_agents(2, agent_class=_TupleAgent, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


class _AliasAgent(am.Agent):
    def step(self):
        self.model.items.append(self.id)
        self.model.agents[0].wealth = 1


class _AliasModel(am.Model):
    def setup(self):
        self.items = []
        self.alias = self.items
        self.add_agents(2, agent_class=_AliasAgent, wealth=[0, 0])

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
        self.model.held = self.model.buf
        self.model.agents[0].wealth = self.id


class _BufferModel(am.Model):
    def setup(self):
        self.buf = np.array([], dtype=int)
        self.held = self.buf
        self.add_agents(2, agent_class=_BufferAgent, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


class _NestedAgent(am.Agent):
    def step(self):
        self.model.payload["values"][0] = self.id
        self.model.agents[0].wealth = 1


class _NestedModel(am.Model):
    def setup(self):
        self.payload = {"values": np.array([0, 0])}
        self.add_agents(2, agent_class=_NestedAgent, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


class _NestedSameAgent(am.Agent):
    def step(self):
        self.model.payload["values"][0] = 7
        self.model.agents[0].wealth = 1


class _NestedSameModel(am.Model):
    def setup(self):
        self.payload = {"values": np.array([0, 0])}
        self.add_agents(2, agent_class=_NestedSameAgent, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


def test_list_order_is_a_divergence_and_aliases_stay_restored():
    model = _ScratchModel(_params())
    model._ensure_setup()
    external = model.scratch
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert races
    assert all(v.divergence_witness is True for v in races)
    assert external is model.scratch
    assert external == [0, 1]
    assert res["agents"].sort("id")["cell"].to_list() == [2, 0]


def test_rebound_counter_is_restored_and_is_not_a_divergence():
    model = _CounterModel(_params())
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert len(races) == 1
    assert races[0].divergence_witness is False
    assert model.count == 2
    assert res["agents"].sort("id")["wealth"].to_list() == [1, 0]


def test_list_inside_tuple_keeps_identity_and_forward_contents():
    model = _TupleModel(_params())
    model._ensure_setup()
    external = model.holder[0]
    holder = model.holder
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert len(races) == 1
    assert races[0].divergence_witness is True
    assert external is model.holder[0] is model.inner
    assert holder is model.holder
    assert external == [0, 1]
    assert res["agents"].sort("id")["wealth"].to_list() == [1, 0]


def test_external_list_alias_keeps_identity_and_forward_contents():
    model = _AliasModel(_params())
    model._ensure_setup()
    external = model.items
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert len(races) == 1
    assert races[0].divergence_witness is True
    assert external is model.items is model.alias
    assert external == [0, 1]
    assert res["agents"].sort("id")["wealth"].to_list() == [1, 0]


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


def test_rebound_ndarray_keeps_post_step_identity():
    model = _BufferModel(_params())
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert races
    assert all(v.divergence_witness is True for v in races)
    assert model.held is model.buf
    assert model.buf.tolist() == [0, 1]
    assert res["agents"].sort("id")["wealth"].to_list() == [1, 0]


def test_nested_ndarray_comparison_does_not_raise_and_preserves_identity():
    params = _params()
    model = _NestedModel(params)
    model._ensure_setup()
    array = model.payload["values"]
    res = model.run(contract="check")
    plain = _NestedModel(params).run(contract="off")
    races = _races(res["contract"][0])
    assert len(races) == 1
    assert races[0].divergence_witness is True
    assert model.payload["values"] is array
    assert array.tolist() == [1, 0]
    assert res["agents"].sort("id")["wealth"].to_list() == (
        plain["agents"].sort("id")["wealth"].to_list()
    )


def test_equal_nested_ndarray_is_not_a_divergence():
    model = _NestedSameModel(_params())
    model._ensure_setup()
    array = model.payload["values"]
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert len(races) == 1
    assert races[0].divergence_witness is False
    assert model.payload["values"] is array
    assert array.tolist() == [7, 0]
