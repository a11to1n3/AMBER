import sys

import numpy as np
import pytest

import ambr as am


N = 96
STEPS = 3


@pytest.fixture(autouse=True)
def disable_debugger_detection(monkeypatch):
    # Coverage uses the same tracing API as debuggers. Keep the coverage hook
    # installed while testing the path a model takes outside a debugger.
    monkeypatch.setattr("ambr._auto_batch.gettrace", lambda: None)


class Counter(am.Agent):
    def setup(self):
        self.age = self.id
        self.energy = 100

    def step(self):
        self.age += 1
        self.energy = self.energy - 2

    def advance_age(self):
        self.age += 1


class Counters(am.Model):
    agent_reporters = ["age", "energy"]
    record_initial = True

    def setup(self):
        self.agents = am.AgentList(self, self.p.n, Counter)

    def step(self):
        if self.p.dispatch == "activate":
            self.activate_agents()
        elif self.p.dispatch == "call":
            assert np.array_equal(self.agents.call("step"), np.full(self.p.n, None))
        else:
            assert np.array_equal(self.agents.advance_age(), np.full(self.p.n, None))


@pytest.mark.parametrize("dispatch", ["activate", "call", "broadcast"])
def test_default_run_batches_unmodified_methods_and_preserves_recorded_state(dispatch):
    params = {"n": N, "steps": STEPS, "dispatch": dispatch}
    optimized = Counters(params)
    scalar = Counters(params)
    result = optimized.run()
    reference = scalar.run(auto_batch=False)
    assert result.info["auto_batch"] == {"enabled": True, "batches": STEPS, "agent_calls": N * STEPS}
    assert reference.info["auto_batch"]["batches"] == 0
    assert result.agent_vars.equals(reference.agent_vars)
    assert result.agents.equals(reference.agents)
    assert [vars(a)["age"] for a in optimized.agents] == list(range(STEPS, N + STEPS))


class WithPending(Counters):
    record_initial = False
    agent_reporters = None

    def step(self):
        self.activate_agents()
        assert self.agents.by_id(0).age == self.t + 1
        assert self.population.data.is_empty()


def test_batch_preserves_pending_writes_and_flush_boundary():
    result = WithPending({"n": N, "steps": STEPS}).run()
    assert result.info["auto_batch"]["batches"] == STEPS
    assert result.agents["age"].to_list() == list(range(STEPS, N + STEPS))
    assert result.agents["energy"].to_list() == [94] * N


class Donor(am.Agent):
    def setup(self):
        self.wealth = int(self.id == 0)

    def step(self):
        if self.id < 2 and self.wealth:
            self.wealth -= 1
            self.model.agents.by_id(1 - self.id).wealth += 1


class Transfer(am.Model):
    def setup(self):
        self.add_agents(N, agent_class=Donor)

    def step(self):
        self.activate_agents()


def test_order_sensitive_transfers_stay_sequential():
    result = Transfer({"steps": 1}).run()
    assert result.info["auto_batch"]["batches"] == 0
    assert result.agents["wealth"].to_list() == [1] + [0] * (N - 1)


def test_active_monitor_keeps_the_scalar_trace():
    class Monitored(Counters):
        def step(self):
            self.activate_agents()
            self.activate_agents()

    result = Monitored({"n": N, "steps": 1}).run(contract="check")
    assert result.info["auto_batch"]["batches"] == 0
    assert any(v.kind == "duplicate_write" for v in result["contract"][0].violations)


def test_active_debugger_keeps_python_calls(monkeypatch):
    monkeypatch.setattr("ambr._auto_batch.gettrace", lambda: object())
    result = Counters({"n": N, "steps": 1, "dispatch": "broadcast"}).run()
    assert result.info["auto_batch"]["batches"] == 0
    assert result.agents["age"].to_list() == list(range(1, N + 1))


def test_profiler_can_observe_agent_method_calls():
    calls = []

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is Counter.advance_age.__code__:
            calls.append(frame.f_locals["self"].id)

    saved = sys.getprofile()
    try:
        sys.setprofile(observe)
        result = Counters({"n": N, "steps": 1, "dispatch": "broadcast"}).run()
    finally:
        sys.setprofile(saved)
    assert result.info["auto_batch"]["batches"] == 0
    assert calls == list(range(N))


def test_integer_batch_respects_an_agent_id_property():
    class Agent(am.Agent):
        @property
        def id(self):
            self.model.reads += 1
            return self.__dict__["id"]

        @id.setter
        def id(self, value):
            self.__dict__["id"] = value

        def step(self):
            self.x += 1

    class Model(am.Model):
        def setup(self):
            self.reads = 0
            self.add_agents(N, agent_class=Agent, x=0)

        def step(self):
            self.reads = 0
            self.agents.call("step")
            self.record_model("reads", self.reads)

    result = Model({"steps": 1}).run()
    assert result.info["auto_batch"]["batches"] == 0
    assert result["model"]["reads"].to_list() == [N]


class InstanceOverride(Counters):
    def setup(self):
        super().setup()
        first = self.agents.by_id(0)
        object.__setattr__(first, "advance_age", lambda: setattr(first, "age", 9))


def test_per_instance_method_override_keeps_python_dispatch():
    result = InstanceOverride({"n": N, "steps": 1, "dispatch": "broadcast"}).run()
    assert result.info["auto_batch"]["batches"] == 0
    assert result.agents["age"].to_list() == [9] + list(range(2, N + 1))


class OutsideInt64(Counters):
    def setup(self):
        self.add_agents(self.p.n, agent_class=Counter, age=(1 << 63) - 1)


def test_integer_overflow_falls_back_without_wrapping():
    result = OutsideInt64({"n": N, "steps": 1, "dispatch": "broadcast"}).run()
    assert result.info["auto_batch"]["batches"] == 0
    assert result.agents["age"].to_list() == [1 << 63] * N


class ConditionalAgent(Counter):
    def setup(self):
        super().setup()
        self.active = self.id % 2 == 0

    def active_only(self):
        if self.active:
            self.age += 1

    def inactive_only(self):
        if not self.active:
            self.age -= 1

    def once_below_limit(self):
        if self.age < 2:
            self.age += 1
            self.energy -= 2

    def every_sixteenth(self):
        if self.id % 16 == 0:
            self.age += 1

    def either_branch(self):
        if self.active:
            self.age += 1
        else:
            self.age -= 1


class ConditionalModel(Counters):
    def setup(self):
        self.agents = am.AgentList(self, self.p.n, ConditionalAgent)

    def step(self):
        self.activate_agents(method=self.p.method)


@pytest.mark.parametrize("method", ["active_only", "inactive_only", "once_below_limit", "every_sixteenth"])
def test_simple_conditions_batch_and_match_each_recorded_step(method):
    params = {"n": N, "steps": STEPS, "method": method}
    result = ConditionalModel(params).run()
    reference = ConditionalModel(params).run(auto_batch=False)
    assert result.info["auto_batch"]["agent_calls"] == N * STEPS
    assert result.agent_vars.equals(reference.agent_vars)
    assert result.agents.equals(reference.agents)


def test_two_nonempty_branches_use_the_python_loop():
    result = ConditionalModel({"n": N, "steps": 1, "method": "either_branch"}).run()
    assert result.info["auto_batch"]["batches"] == 0
    assert result.agents["age"].to_list() == [i + (1 if i % 2 == 0 else -1) for i in range(N)]


class PartialPending(am.Model):
    def setup(self):
        self.add_agents(N, agent_class=ConditionalAgent, age=0, active=False)

    def step(self):
        self.agents.by_id(self.t).active = True
        self.activate_agents(method="active_only")
        assert self.agents.by_id(0).age == self.t + 1
        assert self.population.data["age"].sum() == 0


def test_conditions_read_pending_values_and_only_stage_selected_writes():
    model = PartialPending({"steps": STEPS})
    result = model.run()
    assert result.info["auto_batch"]["batches"] == STEPS
    assert result.agents["age"].to_list() == [3, 2, 1] + [0] * (N - 3)


class NullCondition(am.Model):
    def setup(self):
        self.add_agents(N, agent_class=ConditionalAgent, active=[None, True] + [False] * (N - 2))

    def step(self):
        self.activate_agents(method="active_only")


def test_null_in_condition_keeps_python_truthiness():
    result = NullCondition({"steps": 1}).run()
    assert result.info["auto_batch"]["batches"] == 0
    assert result.agents["age"].to_list() == [0, 2] + list(range(2, N))
