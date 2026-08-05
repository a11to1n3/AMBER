"""Tests for OOP activation helpers (ambr.scheduling)."""

from __future__ import annotations

import ambr as am
import numpy as np
import pytest


class CountingAgent(am.Agent):
    def setup(self):
        self.n_steps = 0
        self.n_advance = 0
        self.order_token = None

    def step(self):
        self.n_steps += 1
        # stamp order using a shared list on the model
        self.model.order.append(int(self.id))

    def advance(self):
        self.n_advance += 1


class ActivationModel(am.Model):
    def setup(self):
        self.order = []
        self.agents = am.AgentList(self, int(self.p.n), CountingAgent)

    def step_oop(self):
        mode = self.p.get("activation_mode", "sequential")
        self.activate_agents(mode=mode)


@pytest.mark.unit
def test_sequential_activation_order():
    m = ActivationModel(
        {"n": 5, "steps": 1, "seed": 0, "show_progress": False, "activation_mode": "sequential"}
    )
    m.cpu(mode="oop").run()
    assert m.order == [0, 1, 2, 3, 4]
    for a in m.agents:
        assert a.n_steps == 1


@pytest.mark.unit
def test_random_activation_is_permutation():
    m = ActivationModel(
        {"n": 8, "steps": 1, "seed": 42, "show_progress": False, "activation_mode": "random"}
    )
    m.cpu(mode="oop").run()
    assert sorted(m.order) == list(range(8))
    # With a fixed seed, order should be stable across re-runs
    m2 = ActivationModel(
        {"n": 8, "steps": 1, "seed": 42, "show_progress": False, "activation_mode": "random"}
    )
    m2.cpu(mode="oop").run()
    assert m.order == m2.order


@pytest.mark.unit
def test_simultaneous_calls_advance():
    m = ActivationModel(
        {
            "n": 4,
            "steps": 1,
            "seed": 0,
            "show_progress": False,
            "activation_mode": "simultaneous",
        }
    )
    m.cpu(mode="oop").run()
    for a in m.agents:
        assert a.n_steps == 1
        assert a.n_advance == 1


@pytest.mark.unit
def test_activate_function_and_aliases():
    m = ActivationModel({"n": 3, "steps": 1, "seed": 0, "show_progress": False})
    m.setup()
    m._setup_done = True
    am.activate(m.agents, mode="sequential")
    assert [a.n_steps for a in m.agents] == [1, 1, 1]
    act = am.RandomActivation()
    before = [a.n_steps for a in m.agents]
    act(m.agents, rng=m.rng)
    assert [a.n_steps for a in m.agents] == [b + 1 for b in before]


@pytest.mark.unit
def test_shuffled_ids_is_permutation():
    m = ActivationModel({"n": 10, "steps": 1, "seed": 1, "show_progress": False})
    m.setup()
    ids = am.shuffled_ids(m.agents, m.rng)
    assert sorted(ids.tolist()) == list(range(10))
    assert isinstance(ids, np.ndarray)


@pytest.mark.unit
def test_invalid_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        am.normalize_activation("bogus")
    m = ActivationModel({"n": 2, "steps": 1, "seed": 0, "show_progress": False})
    m.setup()
    with pytest.raises(ValueError):
        m.activate_agents(mode="nope")
