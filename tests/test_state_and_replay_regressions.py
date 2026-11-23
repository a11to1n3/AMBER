import pickle
from collections import Counter, OrderedDict, defaultdict, deque

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


class CountingWriter(am.Agent):
    """Both orders write the same value; it depends only on private state."""

    def step(self):
        if self.model.p.state == "counter":
            self._n = getattr(self, "_n", 0) + 1
            value = self._n
        else:
            self._log.append(self.id)
            value = len(self._log)
        self.model.agents.by_id(0).x = value


def test_replay_starts_from_step_entry_private_agent_state():
    class Model(RaceModel):
        agent_class = CountingWriter

        def setup(self):
            self.add_agents(2, agent_class=self.agent_class, x=0)
            for agent in self.agents:
                agent._log = []

    for state in ("counter", "list"):
        params = {**PARAMETERS, "steps": 2, "state": state}
        plain = Model(params).cpu(mode="oop")
        checked = Model(params).cpu(mode="oop")
        forward = plain.run(contract="off")
        result = checked.run(contract="check")

        for cert in result["contract"]:
            races = [v for v in cert.violations if v.kind == "activation_race"]
            assert races and all(v.divergence_witness is False for v in races), state
        assert result["agents"].equals(forward["agents"])
        assert [vars(a).get("_n") for a in checked.agents] == [
            vars(a).get("_n") for a in plain.agents
        ]
        assert [a._log for a in checked.agents] == [a._log for a in plain.agents]


class RingAgent(am.Agent):
    """Reads its ring neighbour's x and rewrites its own x.

    ``rule="copy"`` keeps x, so every order gives the same state.
    ``rule="chain"`` writes the neighbour's value plus one, which depends on
    the order. Each private state variant feeds a public column, so a replay
    that leaked into the forward run would change the agent frame.
    """

    def step(self):
        model = self.model
        seen = int(model.agents.by_id((self.id + 1) % 3).x)
        self.x = seen + 1 if model.p.rule == "chain" else self.x
        state = model.p.state
        if state == "agent_counter":
            self._k = getattr(self, "_k", 0) + 1
            self.y = self._k
        elif state == "nested_list":
            self._nested[0].append(seen)
            self.y = len(self._nested[0])
        elif state == "defaultdict":
            self._seen[seen] += 1
            self.y = sum(self._seen.values())
        elif state == "containers":
            self._recent.append(seen)
            self._counts[seen] += 1
            self._order[seen] = len(self._order)
            self._values.add(seen)
            self.y = sum(self._recent) + sum(self._counts.values()) + len(self._order) + len(self._values)


class RingModel(am.Model):
    def setup(self):
        self.add_agents(3, agent_class=RingAgent, x=[0, 10, 20], y=0, calls=0)
        self._steps = 0
        for agent in self.agents:
            agent._nested = [[]]
            agent._seen = defaultdict(int)
            agent._recent = deque(maxlen=2)
            agent._counts = Counter()
            agent._order = OrderedDict()
            agent._values = set()

    def step_oop(self):
        self.activate_agents("sequential")
        if self.p.state == "model_scalar":
            self._steps += 1
            self.agents.calls = [self._steps, 0, 0]


def _private_state(model):
    agents = [
        {key: value for key, value in vars(agent).items()
         if key.startswith("_") and key != "_ambr_container_cells"}
        for agent in model.agents
    ]
    return model._steps, agents


def _race_witnesses(result):
    return [
        violation.divergence_witness
        for cert in result["contract"]
        for violation in cert.violations
        if violation.kind == "activation_race"
    ]


@pytest.mark.parametrize("contract", ["check", "warn"])
@pytest.mark.parametrize("rule", ["copy", "chain"])
@pytest.mark.parametrize(
    "state", ["agent_counter", "model_scalar", "nested_list", "defaultdict", "containers"]
)
def test_monitored_run_matches_the_unmonitored_run_with_private_state(state, rule, contract):
    params = {**PARAMETERS, "steps": 3, "state": state, "rule": rule}
    plain = RingModel(params).cpu(mode="oop")
    checked = RingModel(params).cpu(mode="oop")
    forward = plain.run(contract="off")
    if contract == "warn":
        with pytest.warns(UserWarning):
            result = checked.run(contract=contract)
    else:
        result = checked.run(contract=contract)

    assert result["agents"].equals(forward["agents"])
    assert _private_state(checked) == _private_state(plain)
    found = _race_witnesses(result)
    assert found
    if rule == "copy":
        assert all(witness is False for witness in found)
    else:
        assert all(witness is True for witness in found)


class OpaqueCounter:
    def __init__(self):
        self.n = 0


class OpaqueStateAgent(am.Agent):
    def step(self):
        target = self.model.agents.by_id(0)
        self._box.n += 1
        self.y = self._box.n
        target.x = self.id + 1


def test_replay_is_skipped_when_private_state_cannot_be_restored():
    class Model(RaceModel):
        agent_class = OpaqueStateAgent

        def setup(self):
            self.add_agents(2, agent_class=self.agent_class, x=0, y=0)
            for agent in self.agents:
                agent._box = OpaqueCounter()

    params = {**PARAMETERS, "steps": 2}
    plain = Model(params).cpu(mode="oop")
    checked = Model(params).cpu(mode="oop")
    forward = plain.run(contract="off")
    result = checked.run(contract="check")

    races = [
        violation
        for cert in result["contract"]
        for violation in cert.violations
        if violation.kind == "activation_race"
    ]
    assert races and all(violation.divergence_witness is None for violation in races)
    assert all(
        "agent 0 attribute '_box' holds a OpaqueCounter object" in violation.divergence_reason
        and "the swap was not run" in violation.divergence_reason
        for violation in races
    )
    assert result["agents"].equals(forward["agents"])
    assert [agent._box.n for agent in checked.agents] == [2, 2]


def test_framework_private_model_attributes_are_not_replay_state():
    class Reporting(RaceModel):
        model_reporters = {"agents": lambda model: len(model.agents)}
        agent_reporters = ["x"]
        record_initial = True

    models = [
        Reporting(PARAMETERS).cpu(mode="oop"),
        Reporting(PARAMETERS).cpu(mode="vectorized"),
        RingModel({**PARAMETERS, "state": "none", "rule": "copy"}).cpu(mode="oop"),
    ]
    for model in models:
        model.run(contract="check")
        model.run(steps=1, auto_batch=True)
    for model in models:
        user = {
            name for name in vars(model)
            if name.startswith("_") and model._is_replay_model_attr(name)
        }
        assert user <= {"_steps"}


class _CountingList(list):
    """A list subclass whose override changes its own state."""

    def __init__(self):
        super().__init__([1])
        self.clears = 0

    def clear(self):
        self.clears += 1
        super().clear()


class _TaggedArray(np.ndarray):
    pass


class _SameValueRace(am.Agent):
    """Both activations write 1 to agent 0's x and touch model state."""

    def step(self):
        self.model.agents.by_id(0).x = 1
        state = self.model.p.state
        if state == "factory":
            self.model._lookup.default_factory = list if self.id == 0 else dict
        elif state in ("view", "subclass_array"):
            self.model._array.base[1] += 1


class _SameValueModel(am.Model):
    def setup(self):
        state = self.p.state
        if state == "factory":
            self._lookup = defaultdict(None)
        elif state == "subclass_list":
            self._held = _CountingList()
        elif state == "view":
            self._array = np.zeros(2, dtype=np.int64)[:1]
        elif state == "subclass_array":
            self._array = np.zeros(2, dtype=np.int64).view(_TaggedArray)[:1]
        self.add_agents(2, agent_class=_SameValueRace, x=0)

    def step_oop(self):
        self.activate_agents("sequential")

    def observed(self):
        state = self.p.state
        if state == "factory":
            return self._lookup.default_factory
        if state == "subclass_list":
            return self._held.clears
        return self._array.base.tolist()


@pytest.mark.parametrize("state, witness, reason", [
    # The factory is part of a defaultdict's state: the swapped order
    # leaves the other factory installed, so the forward one must return.
    ("factory", True, None),
    # A list subclass can change its own state when the restore calls its
    # methods, so it is not snapshotted and the swap is not run.
    ("subclass_list", None, "model attribute '_held' holds a _CountingList object"),
    # A view is restored through the array that owns its memory.
    ("view", False, None),
    ("subclass_array", None, "model attribute '_array' holds a _TaggedArray array subclass"),
])
def test_replay_restores_the_full_state_of_supported_containers(state, witness, reason):
    params = {**PARAMETERS, "state": state}
    plain = _SameValueModel(params).cpu(mode="oop")
    plain.run(contract="off")
    checked = _SameValueModel(params).cpu(mode="oop")
    result = checked.run(contract="check")

    assert checked.observed() == plain.observed()
    race, = [v for v in result["contract"][0].violations if v.kind == "activation_race"]
    assert race.divergence_witness is witness
    if reason is None:
        assert race.divergence_reason is None
    else:
        assert reason in race.divergence_reason
        assert "the swap was not run" in race.divergence_reason


def _make_default_hook():
    # A fresh function per model, with no closure: its mutable default is
    # the only state it holds.
    def hook(calls=[]):  # noqa: B006 - the shared default is the state under test
        calls.append(1)
        return len(calls)

    return hook


def _make_closure_counter():
    count = 0

    def tick():
        nonlocal count
        count += 1
        return count

    return tick


class _CodeStateRace(am.Agent):
    """Both activations write 1 to agent 0's x and bump state held in code."""

    def step(self):
        self.model.agents.by_id(0).x = 1
        state = self.model.p.state
        model = self.model
        if state in ("function_default", "closure"):
            model.hook()
        elif state in ("class_counter", "module_counter"):
            model.shared.count += 1
        elif state == "model_slot":
            model._slot_count += 1
        elif state == "agent_slot":
            agent = model.agents.by_id(0)
            agent._marks += 1


class _SlottedAgent(_CodeStateRace):
    __slots__ = ("_marks",)


class _CodeStateModel(am.Model):
    __slots__ = ("_slot_count",)

    def setup(self):
        state = self.p.state
        if state == "function_default":
            self.hook = _make_default_hook()
        elif state == "closure":
            self.hook = _make_closure_counter()
        elif state == "class_counter":
            self.shared = type("Shared", (), {"count": 0})
        elif state == "module_counter":
            import types

            self.shared = types.ModuleType("shared_counter")
            self.shared.count = 0
        self._slot_count = 0
        self.add_agents(2, agent_class=_SlottedAgent, x=0)
        for agent in self.agents:
            agent._marks = 0

    def step_oop(self):
        self.activate_agents("sequential")

    def observed(self):
        state = self.p.state
        if state == "function_default":
            return len(self.hook.__defaults__[0])
        if state == "closure":
            return self.hook.__closure__[0].cell_contents
        if state in ("class_counter", "module_counter"):
            return self.shared.count
        if state == "model_slot":
            return self._slot_count
        return [agent._marks for agent in self.agents]


@pytest.mark.parametrize("state, witness, reason", [
    # A function's defaults, attributes, and closure cells are its own state.
    ("function_default", False, None),
    ("closure", False, None),
    # A class bound in model state is restored with its namespace.
    ("class_counter", False, None),
    # A module's globals are not snapshotted, so the swap is not run.
    ("module_counter", None, "model attribute 'shared' holds a module"),
    # Slot values are not in the instance __dict__.
    ("model_slot", False, None),
    ("agent_slot", False, None),
])
def test_replay_restores_state_held_in_code_and_slots_or_skips(state, witness, reason):
    params = {**PARAMETERS, "state": state}
    plain = _CodeStateModel(params).cpu(mode="oop")
    plain.run(contract="off")
    checked = _CodeStateModel(params).cpu(mode="oop")
    result = checked.run(contract="check")

    assert checked.observed() == plain.observed()
    race, = [v for v in result["contract"][0].violations if v.kind == "activation_race"]
    assert race.divergence_witness is witness
    if reason is None:
        assert race.divergence_reason is None
    else:
        assert reason in race.divergence_reason
        assert "the swap was not run" in race.divergence_reason


def test_entry_side_skip_reason_names_the_attribute():
    import threading

    class Model(_CodeStateModel):
        def setup(self):
            super().setup()
            self._lock = threading.Lock()

        def step_oop(self):
            del self._lock
            super().step_oop()

    result = Model({**PARAMETERS, "state": "model_slot"}).cpu(mode="oop").run(contract="check")
    race, = [v for v in result["contract"][0].violations if v.kind == "activation_race"]
    assert race.divergence_witness is None
    assert "model attribute '_lock' holds a lock object" in race.divergence_reason
