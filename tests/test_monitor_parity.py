"""Monitoring must not change a run, and must not lose coverage for common state.

Each test here pairs a monitored run with ``contract="off"`` or checks the
certificate and the swapped-order witness for state that ordinary models hold.
"""

import random

import numpy as np
import pytest

import ambr as am
from ambr.contract import _value_key


PARAMS = {"steps": 1, "seed": 11, "show_progress": False}


def _races(result, step=0):
    return [v for v in result["contract"][step].violations if v.kind == "activation_race"]


class _Hostile:
    """A value whose comparison, hash, and repr all raise."""

    def __eq__(self, other):
        raise RuntimeError("eq")

    def __hash__(self):
        raise RuntimeError("hash")

    def __repr__(self):
        raise RuntimeError("repr")

    def __lt__(self, other):
        raise RuntimeError("lt")


_WRITTEN = {
    "longdouble": lambda: np.longdouble(1.5),
    "complex": lambda: complex(1.0, 2.0),
    "structured scalar": lambda: np.zeros(1, dtype=[("a", "i4")])[0],
    "hostile object": _Hostile,
}


def _fingerprint(values):
    return [
        (type(value).__name__, value if type(value) in (int, float, complex) else None)
        for value in values
    ]


def _outcome(model_class, mode):
    try:
        result = model_class(PARAMS).run(contract=mode)
    except Exception as exc:  # the same failure must occur in every mode
        return ("raised", type(exc).__name__)
    return ("ok", _fingerprint(result.agents.sort("id")["v"].to_list()))


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("race", [False, True])
@pytest.mark.parametrize("path", ["attribute", "lane"])
@pytest.mark.parametrize("name", list(_WRITTEN))
def test_monitoring_never_raises_for_an_odd_written_value(name, path, race):
    make = _WRITTEN[name]

    class Agent(am.Agent):
        def step(self):
            target = self.model.agents.by_id(0) if race else self
            if path == "attribute":
                target.v = make()
            else:
                self.model.agents.at[[target.id]].set(v=make())

    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=Agent, v=0.0)
            self._hostile = _Hostile()
            self.hostile_map = {"k": [_Hostile()]}

        def step(self):
            self.agents.step()

    outcomes = {mode: _outcome(Model, mode) for mode in ("off", "check", "warn")}
    assert outcomes["check"] == outcomes["off"]
    assert outcomes["warn"] == outcomes["off"]


def test_value_keys_never_raise_and_unbuildable_keys_are_unknown():
    for value in (
        np.longdouble(1.5),
        np.clongdouble(1 + 2j),
        np.zeros(1, dtype=[("a", "i4")])[0],
        _Hostile(),
        np.datetime64("NaT"),
    ):
        _value_key(value)
    # np.longdouble.item() is another np.longdouble, so its key cannot be
    # built from a built-in scalar: it is unknown and matches nothing.
    assert _value_key(np.longdouble(1.5)) != _value_key(np.longdouble(1.5))
    assert _value_key(np.float64("nan")) == _value_key(np.float64("nan"))
    assert _value_key(np.float64(0.0)) != _value_key(np.float64(-0.0))
    assert _value_key(np.timedelta64(5, "ns")) != _value_key(np.timedelta64(5, "ps"))


@pytest.mark.parametrize("draw", ["random", "numpy.random"])
def test_check_matches_off_when_agents_draw_from_the_global_streams(draw):
    class Agent(am.Agent):
        def step(self):
            value = random.random() if draw == "random" else float(np.random.random())
            self.y = value
            self.model.agents.by_id(0).x = self.id

    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=Agent, x=0, y=0.0)

        def step(self):
            self.agents.step()

    def run(mode):
        random.seed(5)
        np.random.seed(5)
        result = Model({**PARAMS, "steps": 3}).run(contract=mode)
        after = (random.random(), float(np.random.random()))
        return result, result.agents.sort("id")["y"].to_list(), after

    plain, plain_y, plain_after = run("off")
    checked, checked_y, checked_after = run("check")
    assert checked_y == plain_y
    assert checked_after == plain_after
    # The race was replayed, so the comparison above covers the replay.
    witnesses = [race.divergence_witness for step in range(3) for race in _races(checked, step)]
    assert witnesses and all(witness is not None for witness in witnesses)
