"""Monitoring must not change a run, and must not lose coverage for common state.

Each test here pairs a monitored run with ``contract="off"`` or checks the
certificate and the swapped-order witness for state that ordinary models hold.
"""

import dataclasses
import datetime
import decimal
import enum
import fractions
import logging
import pathlib
import random
import types

import numpy as np
import polars as pl
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


# --- NaN, NaT, -0.0, complex parts, and sets -----------------------------------


class _Inc(am.Agent):
    def step(self):
        self.x += 1


class _SameValue(am.Agent):
    def step(self):
        self.model.agents.by_id(0).x = 1


def _state_model(agent_class, value, where, change=None):
    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=agent_class, x=0)
            held = value(self) if callable(value) else value
            if where == "model":
                self.thing = held
            elif where == "model private":
                self._thing = held
            else:
                self.agents.by_id(0)._thing = held

        def step(self):
            self.agents.step()
            if change is not None:
                change(self)

    return Model


_UNTOUCHED = {
    "datetime64 NaT": lambda m: np.datetime64("NaT"),
    "timedelta64 NaT": lambda m: np.timedelta64("NaT"),
    "datetime64 array with NaT": lambda m: np.array(["NaT", "2020-01-01"], dtype="datetime64[D]"),
    "complex NaN": lambda m: complex(float("nan"), 1.0),
    "numpy complex NaN": lambda m: np.complex128(complex(float("nan"), 1.0)),
    "list with NaN and NaT": lambda m: [float("nan"), np.datetime64("NaT")],
}


@pytest.mark.parametrize("where", ["model", "model private", "agent private"])
@pytest.mark.parametrize("name", list(_UNTOUCHED))
def test_an_untouched_nan_or_nat_is_not_a_change_or_a_divergence(name, where, monkeypatch):
    monkeypatch.setattr("ambr._certify._interpreter_instrumented", lambda: False)
    value = _UNTOUCHED[name]
    cert = _state_model(_Inc, value, where)(PARAMS).cpu(mode="oop").run(contract="check")["contract"][0]
    assert cert.conditional_guarantee == "established", cert.certification_reason
    race, = _races(_state_model(_SameValue, value, where)(PARAMS).run(contract="check"))
    assert race.divergence_witness is False


def _set_attr(name, value):
    return lambda model: setattr(model, name, value(model) if callable(value) else value)


@pytest.mark.parametrize("start, change", [
    ({1}, lambda m: (m.thing.clear(), m.thing.add(True))),
    (complex(float("nan"), 1.0), _set_attr("thing", complex(float("nan"), 2.0))),
    (0.0, _set_attr("thing", -0.0)),
    (np.float64(0.0), _set_attr("thing", np.float64(-0.0))),
    (np.datetime64("NaT"), _set_attr("thing", np.datetime64(0, "D"))),
])
def test_a_changed_special_value_is_a_change(start, change, monkeypatch):
    monkeypatch.setattr("ambr._certify._interpreter_instrumented", lambda: False)
    model = _state_model(_Inc, lambda m: type(start)(start) if isinstance(start, set) else start,
                         "model", change)
    cert = model(PARAMS).cpu(mode="oop").run(contract="check")["contract"][0]
    assert cert.conditional_guarantee == "not_established"
    assert "(model attribute 'thing')" in cert.certification_reason


def test_a_reshaped_view_is_a_change_and_the_replay_restores_its_shape(monkeypatch):
    monkeypatch.setattr("ambr._certify._interpreter_instrumented", lambda: False)

    def reshape(model):
        model._view.shape = (4,)

    def build(m):
        m._owner = np.arange(4)
        return m._owner.reshape(2, 2)

    class ViewModel(_state_model(_Inc, build, "model private", reshape)):
        def setup(self):
            super().setup()
            self._view = self._thing

    cert = ViewModel(PARAMS).cpu(mode="oop").run(contract="check")["contract"][0]
    assert cert.conditional_guarantee == "not_established"
    # ``_thing`` and ``_view`` are the same view; the first binding is named.
    assert "(model attribute '_thing')" in cert.certification_reason

    class Reshaper(am.Agent):
        def step(self):
            view = self.model._view
            if view.shape == (2, 2):
                view.shape = (4,)
            self.model.agents.by_id(0).x = self.id

    class RaceModel(am.Model):
        def setup(self):
            self._owner = np.arange(4)
            self._view = self._owner.reshape(2, 2)
            self.add_agents(2, agent_class=Reshaper, x=0)

        def step(self):
            self.agents.step()

    plain = RaceModel(PARAMS)
    plain.run(contract="off")
    checked = RaceModel(PARAMS)
    result = checked.run(contract="check")
    assert checked._view.shape == plain._view.shape == (4,)
    assert _races(result)[0].divergence_witness is True


# --- Common state objects --------------------------------------------------------


class _Phase(enum.Enum):
    GROW = 1


class _Level(enum.IntEnum):
    LOW = 1


@dataclasses.dataclass
class _Config:
    rate: int = 1
    tags: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class _Frozen:
    rate: int = 1


_OBJECTS = {
    "Enum member": lambda m: _Phase.GROW,
    "IntEnum member": lambda m: _Level.LOW,
    "datetime": lambda m: datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
    "date": lambda m: datetime.date(2020, 1, 1),
    "time": lambda m: datetime.time(12, 30),
    "timedelta": lambda m: datetime.timedelta(days=1),
    "Decimal": lambda m: decimal.Decimal("1.5"),
    "Decimal NaN": lambda m: decimal.Decimal("NaN"),
    "Fraction": lambda m: fractions.Fraction(1, 3),
    "Path": lambda m: pathlib.Path("/tmp"),
    "dataclass": lambda m: _Config(),
    "frozen dataclass": lambda m: _Frozen(),
    "SimpleNamespace": lambda m: types.SimpleNamespace(a=1),
    "polars DataFrame": lambda m: pl.DataFrame({"a": [1.0, float("nan")], "b": [None, 2]}),
    "polars Series": lambda m: pl.Series("a", [1, None]),
    "Logger": lambda m: logging.getLogger("ambr.tests.monitor_parity"),
    "GridEnvironment": lambda m: am.GridEnvironment(m, size=3),
    "numpy Generator": lambda m: np.random.default_rng(1),
    "random.Random": lambda m: random.Random(1),
}


@pytest.mark.parametrize("where", ["model", "model private", "agent private"])
@pytest.mark.parametrize("name", list(_OBJECTS))
def test_common_state_objects_keep_the_certificate_and_the_replay(name, where, monkeypatch):
    monkeypatch.setattr("ambr._certify._interpreter_instrumented", lambda: False)
    value = _OBJECTS[name]
    cert = _state_model(_Inc, value, where)(PARAMS).cpu(mode="oop").run(contract="check")["contract"][0]
    assert cert.conditional_guarantee == "established", cert.certification_reason

    class Racer(am.Agent):
        def step(self):
            self.model.agents.by_id(0).x = self.id + 1

    race, = _races(_state_model(Racer, value, where)(PARAMS).run(contract="check"))
    assert race.divergence_witness is True, race.divergence_reason


@pytest.mark.parametrize("name, change, where", [
    ("dataclass", lambda m: setattr(m.thing, "rate", 2), "model attribute 'thing'"),
    ("dataclass", lambda m: m.thing.tags.append(1), "model attribute 'thing'"),
    ("frozen dataclass", lambda m: object.__setattr__(m.thing, "rate", 2), "model attribute 'thing'"),
    ("polars DataFrame", lambda m: m.thing.extend(pl.DataFrame({"a": [3.0], "b": [4]})),
     "model attribute 'thing'"),
    ("numpy Generator", lambda m: m.thing.random(), "model attribute 'thing'"),
    ("GridEnvironment", lambda m: setattr(m.thing, "torus", True), "model attribute 'thing'"),
    ("Enum member", lambda m: setattr(m, "thing", _Level.LOW), "model attribute 'thing'"),
])
def test_a_changed_state_object_withholds_the_certificate_and_names_it(name, change, where, monkeypatch):
    monkeypatch.setattr("ambr._certify._interpreter_instrumented", lambda: False)
    model = _state_model(_Inc, _OBJECTS[name], "model", change)
    cert = model(PARAMS).cpu(mode="oop").run(contract="check")["contract"][0]
    assert cert.conditional_guarantee == "not_established"
    assert f"({where})" in cert.certification_reason


def test_identity_kept_objects_are_named_in_the_certificate_assumptions(monkeypatch):
    monkeypatch.setattr("ambr._certify._interpreter_instrumented", lambda: False)
    cert = _state_model(_Inc, _OBJECTS["Logger"], "model")(PARAMS).cpu(mode="oop").run(
        contract="check")["contract"][0]
    assert cert.conditional_guarantee == "established"
    assert any(
        "Model attribute 'thing' holds a Logger kept by identity" in note
        for note in cert.assumptions
    )
    cert = _state_model(_Inc, _OBJECTS["GridEnvironment"], "model")(PARAMS).cpu(mode="oop").run(
        contract="check")["contract"][0]
    assert any(
        "Model attribute 'thing' holds a GridEnvironment" in note and "own attributes" in note
        for note in cert.assumptions
    )


class _StateUser(am.Agent):
    """Changes model-held objects during its activation, and races on x[0]."""

    def step(self):
        model = self.model
        model.cfg.rate += 1
        model.cfg.tags.append(self.id)
        model.table.extend(pl.DataFrame({"a": [self.id]}))
        model.series.append(pl.Series("s", [self.id]))
        self.y = float(model.gen.random()) + model.stream.random()
        model.ns.count += 1
        model.agents.by_id(0).x = self.id


class _StateModel(am.Model):
    def setup(self):
        self.cfg = _Config()
        self.table = pl.DataFrame({"a": [-1]})
        self.series = pl.Series("s", [-1])
        self.gen = np.random.default_rng(3)
        self.stream = random.Random(3)
        self.ns = types.SimpleNamespace(count=0)
        self.add_agents(3, agent_class=_StateUser, x=0, y=0.0)

    def step(self):
        self.agents.step()

    def observed(self):
        return (
            self.cfg.rate, list(self.cfg.tags), self.table["a"].to_list(),
            self.series.to_list(), float(self.gen.random()), self.stream.random(),
            self.ns.count,
        )


def test_replay_restores_dataclass_frame_namespace_and_stream_state():
    params = {**PARAMS, "steps": 2}
    plain = _StateModel(params)
    plain_result = plain.run(contract="off")
    checked = _StateModel(params)
    checked._ensure_setup()
    held = (checked.cfg, checked.table, checked.series, checked.gen, checked.ns)
    result = checked.run(contract="check")
    witnesses = [race.divergence_witness for step in range(2) for race in _races(result, step)]
    assert witnesses and all(witness is True for witness in witnesses)
    assert (checked.cfg, checked.table, checked.series, checked.gen, checked.ns) == held
    assert result.agents.equals(plain_result.agents)
    assert checked.observed() == plain.observed()


def test_an_unsupported_container_names_the_attribute_in_the_certificate(monkeypatch):
    monkeypatch.setattr("ambr._certify._interpreter_instrumented", lambda: False)

    class Tagged(list):
        pass

    cert = _state_model(_Inc, lambda m: Tagged([1]), "model private")(PARAMS).cpu(mode="oop").run(
        contract="check")["contract"][0]
    assert cert.conditional_guarantee == "not_established"
    assert "could not be compared with step entry (model attribute '_thing' holds a Tagged object" in (
        cert.certification_reason
    )


def test_the_snapshot_keeps_no_class_alive():
    import gc
    import weakref

    def run_once():
        class Temporary(am.Agent):
            __slots__ = ("_n",)

            def step(self):
                self.x += 1

        class Model(am.Model):
            def setup(self):
                self.add_agents(2, agent_class=Temporary, x=0)

            def step(self):
                self.agents.step()

        Model(PARAMS).run(contract="check")
        return weakref.ref(Temporary)

    ref = run_once()
    gc.collect()
    assert ref() is None
