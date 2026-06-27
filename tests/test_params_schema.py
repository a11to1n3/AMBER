"""Phase 5: typed parameter accessors + optional class-level params schema."""

import ambr as am


def test_attrdict_typed_getters():
    m = am.Model({"n": "5", "rate": "0.25", "flag": "false"})
    assert m.p.get_int("n") == 5
    assert m.p.get_float("rate") == 0.25
    assert m.p.get_bool("flag") is False           # string-aware: "false" -> False
    assert m.p.get_bool("missing", True) is True
    assert m.p.get_int("missing", 7) == 7


def test_params_schema_pre_coerces():
    class M(am.Model):
        params = {"n": (int, 200), "speed": (float, 1.0), "wrap": (bool, False)}

    m = M({"n": "50", "speed": "2.5"})             # strings as from CLI/JSON
    assert m.p.n == 50 and isinstance(m.p.n, int)
    assert m.p.speed == 2.5 and isinstance(m.p.speed, float)
    assert m.p.wrap is False                        # default applied


def test_params_schema_defaults_when_absent():
    class M(am.Model):
        params = {"n": (int, 200)}

    assert M({}).p.n == 200


def test_params_schema_bool_strings():
    class M(am.Model):
        params = {"a": (bool, False), "b": (bool, True)}

    m = M({"a": "true", "b": "0"})
    assert m.p.a is True
    assert m.p.b is False


def test_plain_model_without_schema_unaffected():
    m = am.Model({"foo": 1})
    assert m.p.foo == 1                              # no schema -> values pass through
