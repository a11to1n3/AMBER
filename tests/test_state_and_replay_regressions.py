import pickle

import numpy as np
import polars as pl
import pytest

import ambr as am


PARAMETERS = {"steps": 1, "seed": 43, "show_progress": False}


class Writer(am.Agent):
    def setup(self):
        self.x = 0

    def step(self):
        target = self.model.agents.by_id(0)
        if self.id == 0 and target.x == 1:
            self.replay_only = 7
            self.model.record_model("replay_only", 7)
        target.x = self.id
        self.model.record_model("last_writer", self.id)


class RaceModel(am.Model):
    agent_class = Writer

    def setup(self):
        self.add_agents(2, agent_class=self.agent_class)

    def step_oop(self):
        self.activate_agents("sequential")


def witnesses(result):
    return [
        violation.divergence_witness
        for violation in result["contract"][0].violations
        if violation.kind == "activation_race"
    ]


def test_agent_reads_follow_initial_columns_pending_writes_and_bulk_updates():
    model = am.Model(PARAMETERS)
    model.add_agents(2, agent_class=Writer, x=[1, 2], y=[3, 4])
    first, second = model.agents
    assert (first.x, second.x, first.y, second.y) == (1, 2, 3, 4)
    assert model.agents.x.to_list() == [1, 2]

    first.x = 5
    assert first.x == 5
    model.agents.x = [10, 20]
    model.agents.at[[first.id, first.id]].scatter_add(x=2)
    assert (first.x, second.x) == (14, 20)
    model.agents_df = model.agents_df.sort("id", descending=True)
    assert (first.x, second.x) == (14, 20)


def test_agent_reads_nullable_values_after_an_in_place_frame_edit():
    model = am.Model(PARAMETERS)
    model.add_agents(2, agent_class=am.Agent, value=[None, 3])
    first, second = model.agents
    assert (first.value, second.value) == (None, 3)

    frame = model.agents_df
    frame.replace_column(frame.columns.index("value"), pl.Series("value", [7, None]))
    assert (first.value, second.value) == (7, None)

    first.value = 5
    assert (first.value, second.value) == (5, None)
    assert model.agents_df["value"].to_list() == [5, None]


def test_reading_a_frame_preserves_model_serialization():
    model = am.Model(PARAMETERS)
    model.nprandom = None
    model.add_agents(2, value=[1, 2])
    expected = model.agents_df
    restored = pickle.loads(pickle.dumps(model))
    assert restored.agents_df.equals(expected)


def test_agent_owned_containers_remain_mutable_after_frame_reads():
    class Recorder(am.Agent):
        def setup(self):
            self.history = [self.id]
            self.stats = {"visits": 0}

        def step(self):
            self.history.append(self.id)
            self.stats["visits"] += 1

    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=Recorder)

        def step(self):
            assert self.agents_df.height == 2
            self.activate_agents()

    model = Model({**PARAMETERS, "steps": 2})
    model.run(auto_batch=False)
    assert [agent.history for agent in model.agents] == [[0, 0, 0], [1, 1, 1]]
    assert [agent.stats for agent in model.agents] == [{"visits": 2}, {"visits": 2}]
    model.agents.history = [[9], [8]]
    assert [agent.history for agent in model.agents] == [[9], [8]]


def test_replay_restores_array_contents_writeability_and_identity():
    class Model(RaceModel):
        def setup(self):
            super().setup()
            self.ledger = np.zeros(1, dtype=np.int64)
            self._original = self.ledger

        def step_oop(self):
            super().step_oop()
            if self.agents.by_id(0).x == 0:
                self.ledger[0] = 9
                self.ledger.setflags(write=False)

    model = Model(PARAMETERS).cpu(mode="oop")
    result = model.run(contract="check")
    assert witnesses(result) and all(witnesses(result))
    assert model.ledger is model._original
    assert model.ledger.flags.writeable
    assert model.ledger.tolist() == [0]


def test_replay_restores_agent_identity_after_rejecting_roster_change():
    class RenamingWriter(Writer):
        def step(self):
            target = self.model.agents.by_id(0)
            if self.id == 0 and target.x == 1:
                self.id = 99
            else:
                target.x = self.id

    class Model(RaceModel):
        agent_class = RenamingWriter

    model = Model(PARAMETERS).cpu(mode="oop")
    result = model.run(contract="check")
    assert witnesses(result) and all(value is None for value in witnesses(result))
    assert model.agents.by_id(0).id == 0
    assert model.agents.ids.to_list() == result.agents["id"].to_list() == [0, 1]


def test_replay_keeps_forward_metrics_and_agent_bindings():
    plain = RaceModel(PARAMETERS).cpu(mode="oop").run(contract="off")
    model = RaceModel(PARAMETERS).cpu(mode="oop")
    checked = model.run(contract="check")

    assert witnesses(checked) and all(witnesses(checked))
    assert checked["agents"].equals(plain["agents"])
    assert checked["model"].equals(plain["model"])
    assert checked["model"]["last_writer"].to_list() == [1]
    assert "replay_only" not in vars(model.agents.by_id(0))


class PrivateStateWriter(am.Agent):
    def step(self):
        self._calls = getattr(self, "_calls", 0) + 1
        target = self.model.agents.by_id(0)
        if self.id == 1:
            if getattr(self, "_ran_first", False):
                self.y = 100
            # Agent 1 runs first only in the swapped replay of step 0.
            if target.x == 0 and self.model.t == 0:
                self._ran_first = True
        target.x = self.id + 1


def test_replay_does_not_leak_private_agent_attributes_into_the_forward_run():
    class Model(RaceModel):
        agent_class = PrivateStateWriter

        def setup(self):
            self.add_agents(2, agent_class=self.agent_class, x=0, y=0)

    params = {**PARAMETERS, "steps": 2}
    plain = Model(params).cpu(mode="oop")
    checked = Model(params).cpu(mode="oop")
    forward = plain.run(contract="off")
    result = checked.run(contract="check")

    def private(model):
        return [
            {key: value for key, value in vars(agent).items() if key.startswith("_")}
            for agent in model.agents
        ]

    assert witnesses(result) and all(witnesses(result))
    assert result["agents"].equals(forward["agents"])
    assert private(checked) == private(plain) == [{"_calls": 2}, {"_calls": 2}]


class SameValueWriter(Writer):
    def step(self):
        self.model.agents.by_id(0).x = 1


@pytest.mark.parametrize("output", ["dict_order", "recorded_metric"])
def test_replay_witness_detects_differences_in_observable_step_outputs(output):
    class OutputWriter(SameValueWriter):
        def step(self):
            super().step()
            if output == "dict_order":
                self.model.ledger[self.id] = True
            else:
                self.model.record_model("last_writer", self.id)

    class Model(RaceModel):
        agent_class = OutputWriter

        def setup(self):
            super().setup()
            self.ledger = {}

    model = Model(PARAMETERS).cpu(mode="oop")
    result = model.run(contract="check")
    assert witnesses(result) == [True]
    if output == "dict_order":
        assert list(model.ledger) == [0, 1]
    else:
        assert result["model"]["last_writer"].to_list() == [1]


class DrawAfterSweep(RaceModel):
    agent_class = SameValueWriter

    def step_oop(self):
        self.activate_agents(self.p.activation)
        self.agents.draw = int(self.rng.integers(2**31))


@pytest.mark.parametrize("activation", ["sequential", "random"])
def test_replay_preserves_the_shuffle_draws_and_forward_rng(activation):
    params = {**PARAMETERS, "activation": activation}
    plain = DrawAfterSweep(params).cpu(mode="oop")
    checked = DrawAfterSweep(params).cpu(mode="oop")
    forward = plain.run(contract="off")
    result = checked.run(contract="check")

    assert witnesses(result) and all(witness is False for witness in witnesses(result))
    assert result["agents"].equals(forward["agents"])
    assert checked.rng.random() == plain.rng.random()


class SharedDrawWriter(SameValueWriter):
    def step(self):
        super().step()
        self.draw = int(self.model.rng.integers(2**31))


def test_replay_is_inconclusive_for_shared_draws_inside_an_activation():
    class SharedDrawModel(RaceModel):
        agent_class = SharedDrawWriter

    model = SharedDrawModel(PARAMETERS).cpu(mode="oop")
    result = model.run(contract="check")
    assert witnesses(result) and all(witness is None for witness in witnesses(result))


class DrawOnlyOnSwap(Writer):
    def step(self):
        if self.id == 0 and self.model.agents.by_id(0).x == 1:
            self.draw = float(self.model.rng.random())
        super().step()


def test_replay_is_inconclusive_when_only_the_swap_uses_shared_draws():
    class SwapDrawModel(RaceModel):
        agent_class = DrawOnlyOnSwap

    plain = SwapDrawModel(PARAMETERS).cpu(mode="oop")
    checked = SwapDrawModel(PARAMETERS).cpu(mode="oop")
    forward = plain.run(contract="off")
    result = checked.run(contract="check")
    assert witnesses(result) and all(witness is None for witness in witnesses(result))
    assert result["agents"].equals(forward["agents"])
    assert result["model"].equals(forward["model"])
    assert checked.rng.random() == plain.rng.random()


class PrivateContainerWriter(am.Agent):
    def setup(self):
        self._seen = []
        self._counts = {}
        self._values = set()
        self._total = np.zeros(1, dtype=np.int64)

    def step(self):
        target = self.model.agents.by_id(0)
        value = int(target.x)
        self._seen.append(value)
        self._counts[value] = self._counts.get(value, 0) + 1
        self._values.add(value)
        self._total += value
        self.y = len(self._seen) + len(self._counts) + len(self._values) + int(self._total[0])
        target.x = self.id + 1


def test_replay_restores_private_container_contents_in_place():
    class Model(RaceModel):
        agent_class = PrivateContainerWriter

        def setup(self):
            self.add_agents(2, agent_class=self.agent_class, x=0, y=0)

    params = {**PARAMETERS, "steps": 2}
    plain = Model(params).cpu(mode="oop")
    checked = Model(params).cpu(mode="oop")
    forward = plain.run(contract="off")
    result = checked.run(contract="check")

    def private(model):
        return [
            (agent._seen, agent._counts, agent._values, agent._total.tolist())
            for agent in model.agents
        ]

    assert witnesses(result) and all(witnesses(result))
    assert result["agents"].equals(forward["agents"])
    assert private(checked) == private(plain)
