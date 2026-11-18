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


def test_replay_keeps_forward_metrics_and_agent_bindings():
    plain = RaceModel(PARAMETERS).cpu(mode="oop").run(contract="off")
    model = RaceModel(PARAMETERS).cpu(mode="oop")
    checked = model.run(contract="check")

    assert witnesses(checked) and all(witnesses(checked))
    assert checked["agents"].equals(plain["agents"])
    assert checked["model"].equals(plain["model"])
    assert checked["model"]["last_writer"].to_list() == [1]
    assert "replay_only" not in vars(model.agents.by_id(0))


class SameValueWriter(Writer):
    def step(self):
        self.model.agents.by_id(0).x = 1


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
