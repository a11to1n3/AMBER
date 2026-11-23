"""Activation-race witness: no replay while a stage is open."""

import random

import numpy as np
import polars as pl
import pytest

from ambr.contract import ContractCertificate, ContractViolation, ContractViolationError
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


def test_replay_restores_parameter_graph_and_roster_after_a_replay_only_birth():
    class Agent(am.Agent):
        def step(self):
            self.model.agents.by_id(0).x = self.id

    class Model(am.Model):
        model_reporters = {"counter": lambda m: m.p.counter, "count": lambda m: len(m.agents)}

        def setup(self):
            self.add_agents(2, agent_class=Agent, x=0)

        def step(self):
            self.activate_agents()
            self.p.counter += 1
            if self.agents.by_id(0).x == 0:
                self.add_agents(1, agent_class=Agent, x=0)

    params = {**_params(), "counter": 0}
    baseline = Model(params).run()
    model = Model(params)
    result = model.run(contract="check")
    race, = _races(result["contract"][0])
    assert race.divergence_witness is None
    assert "roster" in race.divergence_reason
    assert result.model.equals(baseline.model)
    assert result.agents.equals(baseline.agents)
    assert [agent.id for agent in model.agents] == [0, 1]
    assert model.p.counter == 1


def test_whole_column_read_is_attributed_to_the_active_agent():
    class Agent(am.Agent):
        def step(self):
            if self.id == 0:
                self.x = 1
            else:
                self.y = self.model.agents.x.sum()

    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=Agent, x=0, y=0)

        def step(self):
            self.activate_agents()

    result = Model(_params()).run(contract="check")
    race, = _races(result["contract"][0])
    assert race.columns == ["x"]
    assert race.ids == [0, 1]
    assert race.divergence_witness is True
    assert result.agents["y"].to_list() == [0, 1]


_FRAME_READS = {
    "model.agents_df": lambda agent: int(agent.model.agents_df["x"].sum()),
    "Agent.get_data": lambda agent: int(
        agent.model.agents.by_id(0).get_data()["x"][0]
    ),
    "Agent.get_neighbors": lambda agent: len(agent.get_neighbors(pl.col("x") > 0)),
}


@pytest.mark.parametrize("sweep", ["activate_agents", "stage_agents"])
@pytest.mark.parametrize("path", sorted(_FRAME_READS))
def test_frame_level_reads_are_attributed_to_the_active_agent(path, sweep):
    read = _FRAME_READS[path]

    class Agent(am.Agent):
        def step(self):
            if self.id == 0:
                self.x = 1
            else:
                self.y = read(self)

    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=Agent, x=0, y=0)
            if self.p.reverse:
                self.agents.reverse()

        def step_oop(self):
            if sweep == "activate_agents":
                self.activate_agents("sequential")
            else:
                for agent in self.stage_agents("read"):
                    agent.step()

    # The value agent 1 reads depends on the activation order.
    for reverse, expected_y in ((False, 1), (True, 0)):
        params = {**_params(), "reverse": reverse}
        result = Model(params).cpu(mode="oop").run(contract="check")
        cert = result["contract"][0]
        assert result.agents.sort("id")["y"].to_list() == [0, expected_y]
        assert not cert.clean
        races = [v for v in _races(cert) if v.columns == ["x"]]
        assert races and all(set(v.ids) == {0, 1} for v in races)
        if sweep == "activate_agents":
            assert all(v.divergence_witness is True for v in races)


_PHASE_READS = {
    "model.agents_df": lambda agent: agent.model.agents_df.filter(pl.col("id") == agent.id)["flag"][0],
    "Agent.get_data": lambda agent: agent.get_data()["flag"][0],
    "column view": lambda agent: agent.model.agents.flag.to_list()[agent.id],
}


@pytest.mark.parametrize("contract", ["check", "raise"])
@pytest.mark.parametrize("path", sorted(_PHASE_READS))
def test_activation_reads_of_an_earlier_phase_commit_are_not_hazards(path, contract):
    read = _PHASE_READS[path]

    class Agent(am.Agent):
        def step(self):
            self.model.seen.append(int(read(self)))

    class Model(am.Model):
        def setup(self):
            self.seen = []
            self.add_agents(2, agent_class=Agent, flag=0)

        def step_oop(self):
            self.agents.flag = [1, 1]
            self.activate_agents("sequential")

    model = Model(_params()).cpu(mode="oop")
    result = model.run(contract=contract)
    cert = result["contract"][0]
    assert model.seen == [1, 1]
    # A whole-frame read inside the sweep sees the earlier commit as input.
    assert cert.clean


@pytest.mark.parametrize("write", ["buffered", "commit"])
@pytest.mark.parametrize("path", sorted(_PHASE_READS))
def test_activation_reads_of_a_same_sweep_write_are_still_flagged(path, write):
    read = _PHASE_READS[path]

    class Agent(am.Agent):
        def step(self):
            if self.id == 0:
                if write == "buffered":
                    self.flag = 1
                else:
                    self.model.agents.at[[0]].set(flag=1)
            else:
                self.seen = int(read(self.model.agents.by_id(0)))

    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=Agent, flag=0, seen=-1)

        def step_oop(self):
            self.agents.flag = [0, 0]
            self.activate_agents("sequential")

    result = Model(_params()).cpu(mode="oop").run(contract="check")
    cert = result["contract"][0]
    assert result.agents.sort("id")["seen"].to_list() == [-1, 1]
    assert not cert.ok
    races = [v for v in _races(cert) if v.columns == ["flag"]]
    assert races and all(set(v.ids) == {0, 1} for v in races)
    with pytest.raises(ContractViolationError):
        Model(_params()).cpu(mode="oop").run(contract="raise")


def test_mixed_reduction_operators_on_one_cell_report_a_divergent_witness():
    class Agent(am.Agent):
        def step(self):
            if self.id == 0:
                self.scatter_add(x=1)
            else:
                self.model.agents.by_id(0).scatter_max(x=5)

    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=Agent, x=0)

        def step(self):
            self.activate_agents()

    result = Model(_params()).run(contract="check")
    race, = _races(result["contract"][0])
    assert race.columns == ["x"]
    assert race.ids == [0, 1]
    assert race.divergence_witness is True
    assert result.agents["x"].to_list() == [5, 0]


def _write_race_model(values, n=None):
    class Agent(am.Agent):
        def step(self):
            self.model.agents.by_id(0).x = values[self.id]

    class Model(am.Model):
        def setup(self):
            self.add_agents(n or len(values), agent_class=Agent, x=-1)

        def step(self):
            self.activate_agents("sequential")

    return Model


@pytest.mark.parametrize(
    "values, pairs, expected",
    [
        # Swapping 0 and 1 changes nothing; putting a 1 last does.
        ((1, 1, 2), [[1, 2]], [True]),
        # One swap per value that differs from the last writer's.
        ((1, 2, 3, 3), [[0, 3], [1, 3]], [True, True]),
        # All writers agree: the pair reported before stays, with no divergence.
        ((4, 4, 4), [[0, 1]], [False]),
    ],
)
def test_three_or_more_writers_replay_one_swap_per_written_value(values, pairs, expected):
    result = _write_race_model(values)(_params()).run(contract="check")
    races = _races(result["contract"][0])
    assert [race.ids for race in races] == pairs
    assert [race.divergence_witness for race in races] == expected
    if len(set(values)) > 1:
        assert f"{len(values)} activations wrote this cell" in races[0].detail
    assert result.agents.sort("id")["x"].to_list()[0] == values[-1]


def test_two_writer_race_report_is_unchanged():
    for values, witness in (((1, 2), True), ((3, 3), False)):
        result = _write_race_model(values)(_params()).run(contract="check")
        race, = _races(result["contract"][0])
        assert race.ids == [0, 1]
        assert race.divergence_witness is witness
        assert race.detail.startswith("Distinct activations 0 and 1 wrote the same cell ('x', 0).")
        assert "activations wrote this cell" not in race.detail


def test_equal_mixed_reduction_operator_sets_get_a_cross_activation_replay():
    class Agent(am.Agent):
        def step(self):
            add, top = (1, 10) if self.id == 0 else (5, 2)
            self.model.agents.at[[0]].scatter_add(x=add)
            self.model.agents.at[[0]].scatter_max(x=top)

    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=Agent, x=0)

        def step(self):
            self.activate_agents("sequential")

    result = Model(_params()).run(contract="check")
    races = _races(result["contract"][0])
    cross = [race for race in races if race.ids == [0, 1]]
    assert len(cross) == 1 and cross[0].divergence_witness is True
    assert {tuple(race.ids) for race in races} == {(0,), (1,), (0, 1)}
    # Order (0, 1) gives max(1 + 0, 10) + 5 = 15, then max(15, 2) = 15.
    assert result.agents.sort("id")["x"].to_list()[0] == 15


def test_report_pairs_are_replayed_in_report_order_before_coverage_pairs():
    n = 9

    class Agent(am.Agent):
        def step(self):
            _ = self.model.agents.by_id((self.id + 1) % n).x
            self.x = self.x
            if self.id >= n - 3:
                self.model.agents.by_id(0).y = 1 if self.id < n - 1 else 2

    class Model(am.Model):
        def setup(self):
            self.add_agents(n, agent_class=Agent, x=0, y=0)

        def step(self):
            self.activate_agents("sequential")

    result = Model(_params()).run(contract="check")
    races = _races(result["contract"][0])
    # The ring of reads is reported before the write to y, so its pairs use
    # the five replays first, in report order, as they did before coverage
    # pairs existed.
    pairs = []
    for race in races:
        if tuple(race.ids) not in pairs:
            pairs.append(tuple(race.ids))
    replayed = [pair for pair in pairs if any(
        tuple(race.ids) == pair and race.divergence_witness is not None for race in races
    )]
    assert replayed == pairs[:5]
    write, = [race for race in races if race.columns == ["y"]]
    assert write.ids == [n - 2, n - 1]
    assert write.divergence_witness is None
    assert "used by earlier reports" in write.divergence_reason
    skipped = [race for race in races if race.divergence_witness is None]
    assert skipped
    assert all("not replayed" in race.divergence_reason for race in skipped)


def _starved_pair_model(multi_first):
    """Two writers race on x[0]; seven writers write distinct values to y[1]."""
    n = 9
    pair = (7, 8) if multi_first else (0, 1)

    class Agent(am.Agent):
        def step(self):
            if self.id in pair:
                self.model.agents.by_id(0).x = self.id
            else:
                self.model.agents.by_id(1).y = self.id

    class Model(am.Model):
        def setup(self):
            self.add_agents(n, agent_class=Agent, x=0, y=0)

        def step(self):
            self.activate_agents("sequential")

    return Model, list(pair)


@pytest.mark.parametrize("multi_first", [False, True])
def test_a_two_writer_report_keeps_its_replay_next_to_a_many_writer_cell(multi_first):
    # The seven-writer cell offers five coverage pairs. They used to take the
    # whole budget before the two-writer report; now each report's own pair
    # is replayed first and the coverage pairs use the rest.
    Model, pair = _starved_pair_model(multi_first)
    result = Model(_params()).run(contract="check")
    races = _races(result["contract"][0])
    two, = [race for race in races if race.columns == ["x"]]
    assert two.ids == pair
    assert two.divergence_witness is True
    assert two.divergence_reason is None
    many = [race for race in races if race.columns == ["y"]]
    assert len(many) == 5
    assert sum(race.divergence_witness is True for race in many) == 4
    left, = [race for race in many if race.divergence_witness is None]
    assert "not replayed" in left.divergence_reason
    assert "earlier coverage pairs" in left.divergence_reason


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
    assert all(v.divergence_witness is None for v in races)
    assert all("random stream state could not be verified" in v.divergence_reason for v in races)
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


class _CollapseAliasAgent(am.Agent):
    def setup(self):
        self.wealth = 0

    def step(self):
        previous = int(self.model.agents[0].wealth)
        self.model.agents[0].wealth = 1
        if self.id == 0 and previous == 1:
            self.model.right = self.model.left


class _CollapseAliasModel(am.Model):
    def setup(self):
        self.left = []
        self.right = []
        self.add_agents(2, agent_class=_CollapseAliasAgent, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


class _DistinctAppendAgent(am.Agent):
    def setup(self):
        self.wealth = 0

    def step(self):
        self.model.left.append(1)
        self.model.right.append(1)
        self.model.agents[0].wealth = 1


class _DistinctAppendModel(am.Model):
    def setup(self):
        self.left = []
        self.right = []
        self.add_agents(2, agent_class=_DistinctAppendAgent, wealth=[0, 0])

    def step(self):
        self.activate_agents(mode="sequential")


def test_alias_collapse_is_inconclusive_and_forward_lists_stay_distinct():
    params = _params()
    model = _CollapseAliasModel(params)
    model._ensure_setup()
    left = model.left
    right = model.right
    assert left is not right
    res = model.run(contract="check")
    plain = _CollapseAliasModel(params).run(contract="off")
    races = _races(res["contract"][0])
    assert races
    assert all(v.divergence_witness is None for v in races)
    assert all(
        v.divergence_reason
        and "alias relationships are not one-to-one" in v.divergence_reason
        for v in races
    )
    assert all("alias relationships are not one-to-one" in v.detail for v in races)
    assert model.left is left and model.right is right
    assert left is not right
    assert left == [] and right == []
    assert res["agents"].sort("id")["wealth"].to_list() == (
        plain["agents"].sort("id")["wealth"].to_list()
    )


def test_equal_distinct_lists_are_not_an_alias_divergence():
    model = _DistinctAppendModel(_params())
    model._ensure_setup()
    left = model.left
    right = model.right
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert len(races) == 1
    assert races[0].divergence_witness is False
    assert model.left is left and model.right is right
    assert left is not right
    assert left == [1, 1] and right == [1, 1]
    assert res["agents"].sort("id")["wealth"].to_list() == [1, 0]


class _DropAdapterAgent(am.Agent):
    def step(self):
        self.model.events.append(self.id)
        self.model.agents[0].wealth = 1


class _DropAdapterModel(am.Model):
    def setup(self):
        self.events = []
        self.add_agents(2, agent_class=_DropAdapterAgent, wealth=[0, 0])

    def step(self):
        self.activate_agents("sequential")
        if self.events == [1, 0]:
            self.nprandom = None


class _FailPostRngRestore(_DropAdapterModel):
    """The cleanup restore reports failure without touching the streams."""

    # Counted on the class: private model attributes are replay state and
    # would be rewound with the step-entry snapshot.
    rng_restores = 0

    def __init__(self, parameters):
        super().__init__(parameters)
        type(self).rng_restores = 0

    def _restore_rng_state(self, saved):
        type(self).rng_restores += 1
        if type(self).rng_restores >= 2:
            return False
        return super()._restore_rng_state(saved)


def _drop_adapter_model(cls=_DropAdapterModel):
    model = cls({"steps": 1, "seed": 42, "show_progress": False}).cpu(mode="oop")
    model._ensure_setup()
    return model


def test_replay_restores_legacy_rng_adapter_and_forward_list():
    model = _drop_adapter_model()
    original_adapter = model.nprandom
    events = model.events
    res = model.run(contract="check")
    plain = _drop_adapter_model().run(contract="off")
    races = _races(res["contract"][0])
    assert races
    assert all(v.divergence_witness is True for v in races)
    assert model.nprandom is original_adapter
    assert model.nprandom._rng is original_adapter._rng
    assert model.events is events
    assert events == [0, 1]
    assert res["agents"].sort("id")["wealth"].to_list() == (
        plain["agents"].sort("id")["wealth"].to_list()
    )


def test_failed_rng_restore_still_restores_forward_model_state():
    model = _drop_adapter_model(_FailPostRngRestore)
    events = model.events
    res = model.run(contract="check")
    races = _races(res["contract"][0])
    assert races
    assert all(v.divergence_witness is None for v in races)
    assert all(
        v.divergence_reason
        and "forward random streams could not be restored" in v.divergence_reason
        for v in races
    )
    assert type(model).rng_restores >= 2
    assert model.nprandom is None
    assert model.events is events
    assert events == [0, 1]


@pytest.mark.parametrize("schedule", ["sequential", "broadcast"])
@pytest.mark.parametrize("stream", ["random", "rng"])
def test_replay_withholds_witness_for_a_cached_shared_draw(schedule, stream):
    class Person(am.Agent):
        def step(self):
            if self.id == 0:
                self.observed = self.model.agents.by_id(1).wealth
            else:
                self.wealth = 1
            self.sample = self.model._cached_draw()

    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=Person, wealth=1, observed=0, sample=0.0)
            self._cached_draw = getattr(self, stream).random

        def step(self):
            if schedule == "broadcast":
                self.agents.step()
            else:
                self.activate_agents()

    params = dict(_params(), seed=7)
    unchecked_model = Model(params)
    unchecked = unchecked_model.run()
    checked_model = Model(params)
    checked = checked_model.run(contract="check")
    assert checked.agents.equals(unchecked.agents)
    assert checked_model.random.getstate() == unchecked_model.random.getstate()
    assert checked_model.rng.bit_generator.state == unchecked_model.rng.bit_generator.state
    cert = checked["contract"][0]
    assert any(v.kind == "premise_p3_violation" for v in cert.violations)
    races = _races(cert)
    assert races and all(v.divergence_witness is None for v in races)
    assert all("shared random stream state changed" in v.divergence_reason for v in races)


def test_replay_withholds_witness_when_only_the_swapped_branch_uses_a_cached_draw():
    class Person(am.Agent):
        def step(self):
            if self.id == 0:
                if self.model.agents.by_id(1).wealth == 2:
                    self.sample = self.model._cached_draw()
            else:
                self.wealth = 2

    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=Person, wealth=1, sample=0.0)
            self._cached_draw = self.random.random

        def step(self):
            self.activate_agents()

    params = dict(_params(), seed=7)
    model = Model(params)
    result = model.run(contract="check")
    assert result.agents["sample"].to_list() == [0.0, 0.0]
    assert model.random.getstate() == random.Random(7).getstate()
    cert = result["contract"][0]
    assert all(v.kind != "premise_p3_violation" for v in cert.violations)
    races = _races(cert)
    assert races and all(v.divergence_witness is None for v in races)
    assert all("shared random stream state changed" in v.divergence_reason for v in races)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_legacy_model_write_inside_an_activation_is_attributed():
    class Agent(am.Agent):
        def step(self):
            if self.id == 0:
                self.model.update_agent_data(0, {"x": 1})
            else:
                self.seen = int(self.model.agents_df["x"][0])

    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=Agent, x=0, seen=-1)

        def step(self):
            self.activate_agents("sequential")

    result = Model(_params()).run(contract="check")
    assert result.agents.sort("id")["seen"].to_list() == [-1, 1]
    race, = _races(result["contract"][0])
    assert race.ids == [0, 1] and race.columns == ["x"]
    assert race.divergence_witness is True


def _raise_on_compare(*_args):
    raise RuntimeError("the monitor must not compare user values")


class _HashRaises(int):
    __hash__ = _raise_on_compare


class _EqRaises(int):
    __eq__ = _raise_on_compare
    __hash__ = int.__hash__


@pytest.mark.parametrize("value_type", [_HashRaises, _EqRaises])
@pytest.mark.parametrize("target", ["own", "shared"])
def test_tracing_never_hashes_or_compares_written_user_values(value_type, target):
    class Agent(am.Agent):
        def step(self):
            row = self if target == "own" else self.model.agents.by_id(0)
            row.x = value_type(self.id + 1)

    class Model(am.Model):
        def setup(self):
            self.add_agents(3, agent_class=Agent, x=0)

        def step(self):
            self.activate_agents("sequential")

    forward = Model(_params()).run(contract="off")
    result = Model(_params()).run(contract="check")
    assert result.agents.sort("id")["x"].to_list() == forward.agents.sort("id")["x"].to_list()
    races = _races(result["contract"][0])
    if target == "own":
        assert not races
    else:
        assert races and all(set(race.ids) <= {0, 1, 2} for race in races)


def test_replay_pair_selection_is_linear_in_the_writers_of_a_cell():
    from time import perf_counter

    from ambr.contract import ContractMonitor

    monitor = ContractMonitor()
    cell = ("x", 0)
    n = 20_000
    monitor._activation_write_values = {act: {cell: (int, act)} for act in range(n)}
    start = perf_counter()
    chosen, distinct, *_ = monitor._last_writer_swaps(list(range(n)), cell, 5)
    elapsed = perf_counter() - start
    assert distinct == n
    assert chosen == list(range(n - 6, n - 1))
    # A pairwise scan takes about 20 s here; one pass takes milliseconds.
    assert elapsed < 1.0


def _two_sweep_model(first_sweep, second_sweep, writers=(0,), readers=(1,)):
    """``first_sweep``/``second_sweep``: None for a sweep over ``model.agents``,
    or the IDs of a partial ``agents.at[ids].call(...)`` sweep."""

    class Agent(am.Agent):
        def write(self):
            if self.id in writers:
                self.model.agents.at[[0]].set(x=self.id + 1)

        def read(self):
            if self.id in readers:
                self.seen = int(self.model.agents.x[0])

    class Model(am.Model):
        def setup(self):
            self.add_agents(3, agent_class=Agent, x=0, seen=-1)

        def step(self):
            for method, ids in (("write", first_sweep), ("read", second_sweep)):
                if ids is None:
                    self.agents.call(method)
                else:
                    self.agents.at[ids].call(method)

    return Model


def test_consecutive_whole_population_sweeps_have_their_own_activation_traces():
    # Agent 0 writes in a sweep over all agents and agent 1 reads in the next
    # one: the read sees an earlier phase's output, which is not a race.
    result = _two_sweep_model(None, None)(_params()).run(contract="check")
    cert = result["contract"][0]
    assert result.agents.sort("id")["seen"].to_list() == [-1, 1, -1]
    assert cert.clean


@pytest.mark.parametrize("first, second", [([0], [1]), (None, [1]), ([0], None)])
def test_partial_sweeps_stay_in_one_activation_trace(first, second):
    # The model, not the framework, orders partial sweeps, so a read in one
    # of them of a cell written in another is checked as a race.
    result = _two_sweep_model(first, second)(_params()).run(contract="check")
    cert = result["contract"][0]
    races = _races(cert)
    assert any(race.ids == [0, 1] for race in races)
    assert all(race.divergence_witness is None for race in races)
    assert all(
        race.divergence_reason == "replay requires a single recorded activation sweep"
        for race in races
    )


def test_a_user_shuffled_one_agent_sweep_order_reports_a_race():
    # One-agent sweeps in an order the model draws: the final x[0] depends
    # on that order, so the dependence must be reported.
    class Grab(am.Agent):
        def step(self):
            if self.model.agents.x[0] == 0:
                self.model.agents.at[[0]].set(x=self.id + 1)

    class UserShuffle(am.Model):
        def setup(self):
            self.add_agents(3, agent_class=Grab, x=0)

        def step(self):
            for aid in self.rng.permutation(3).tolist():
                self.agents.at[[aid]].call("step")

    finals = set()
    for seed in (1, 2, 3, 4):
        result = UserShuffle({**_params(), "seed": seed}).run(contract="check")
        finals.add(result.agents.sort("id")["x"].to_list()[0])
        cert = result["contract"][0]
        assert not cert.ok
        assert _races(cert)
    assert len(finals) > 1


def test_a_race_inside_an_earlier_sweep_is_still_reported():
    result = _two_sweep_model(None, None, writers=(0, 1), readers=(2,))(_params()).run(
        contract="check"
    )
    race, = _races(result["contract"][0])
    assert race.ids == [0, 1] and race.columns == ["x"]
    assert race.divergence_witness is None
    assert race.divergence_reason == "replay requires a single recorded activation sweep"


def _lane_writer_model(values, wrap):
    class Agent(am.Agent):
        def step(self):
            self.model.agents.at[[0]].set(x=wrap(values[self.id]))

    class Model(am.Model):
        def setup(self):
            self.add_agents(len(values), agent_class=Agent, x=0)

        def step(self):
            self.activate_agents("sequential")

    return Model


@pytest.mark.parametrize("recorded", [True, False])
def test_lane_writes_cover_an_early_writer_whose_value_differs(recorded):
    # Only swapping the first writer with the last changes the final x.
    values = [2, 1, 1, 1, 1, 1, 1]
    wrap = (lambda v: v) if recorded else (lambda v: np.array([v]))
    result = _lane_writer_model(values, wrap)(_params()).run(contract="check")
    races = _races(result["contract"][0])
    assert result.agents.sort("id")["x"].to_list()[0] == 1
    by_pair = {tuple(race.ids): race for race in races}
    assert by_pair[(0, 6)].divergence_witness is True
    if recorded:
        assert set(by_pair) == {(0, 6)}
        assert "with 2 distinct value(s)" in by_pair[(0, 6)].detail
    else:
        # Six earlier writers, five replays: the report says so.
        assert len(by_pair) == 5
        assert "7 unrecorded value(s)" in by_pair[(0, 6)].detail
        assert "swaps 5 of the 6 earlier writers" in by_pair[(0, 6)].detail
        unconfirmed = [race for race in races if race.divergence_witness is False]
        assert unconfirmed and all(
            "Only 5 of the 6 candidate pairs" in race.divergence_reason for race in unconfirmed
        )


def test_equal_mixed_reducers_are_paired_by_their_operands():
    class Agent(am.Agent):
        def step(self):
            add, top = (1, 10) if self.id < 2 else (5, 2)
            self.model.agents.at[[0]].scatter_add(x=add)
            self.model.agents.at[[0]].scatter_max(x=top)

    class Model(am.Model):
        def setup(self):
            self.add_agents(3, agent_class=Agent, x=0)

        def step(self):
            self.activate_agents("sequential")

    result = Model(_params()).run(contract="check")
    cross = {tuple(race.ids): race.divergence_witness for race in _races(result["contract"][0])
             if len(race.ids) == 2}
    # Agents 0 and 1 reduce identically; agent 2 does not.
    assert cross == {(0, 1): False, (0, 2): True}


def test_all_equal_writes_say_which_pair_is_replayed():
    result = _write_race_model((4, 4, 4))(_params()).run(contract="check")
    race, = _races(result["contract"][0])
    assert race.ids == [0, 1]
    assert "all with the same value; the replay swaps the first two writers" in race.detail
