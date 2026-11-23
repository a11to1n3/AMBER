"""Monitoring must not change a run, and must not lose coverage for common state.

Each test here pairs a monitored run with ``contract="off"`` or checks the
certificate and the swapped-order witness for state that ordinary models hold.
"""

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
