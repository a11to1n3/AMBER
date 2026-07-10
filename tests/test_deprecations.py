"""Phase 7: legacy verbs emit DeprecationWarning but still return canonical results.

Each alias keeps working (deprecate-and-keep) and is silenced by the
AMBER_SUPPRESS_DEPRECATIONS env var for benchmark / reproducibility runs.
"""

import warnings

import numpy as np
import polars as pl

import ambr as am


def _warns_and(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fn()
        assert any(issubclass(x.category, DeprecationWarning) for x in w), \
            "expected a DeprecationWarning"
    return result


def test_model_record_deprecated():
    m = am.Model({"show_progress": False})
    _warns_and(lambda: m.record("x", 5))
    assert m._current_step_data.get("x") == 5            # still records


def test_agentlist_select_deprecated_matches_where():
    m = am.Model({"show_progress": False})
    m.add_agents(5, x=np.arange(5, dtype=float))
    sel = _warns_and(lambda: m.agents.select(pl.col("x") >= 3))
    assert set(sel.ids.to_list()) == set(m.agents.where(pl.col("x") >= 3).ids.to_list())


def test_indexing_with_bool_mask_does_not_warn():
    # __getitem__ uses the private _select_impl, so indexing must stay quiet.
    m = am.Model({"show_progress": False})
    m.add_agents(4, agent_class=am.Agent)
    m.agents.set(x=np.arange(4, dtype=float))
    mask = np.array([True, False, True, False])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        view = m.agents[mask]
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)
    assert set(view.ids.to_list()) == {0, 2}


def test_agentlist_record_and_update_data_deprecated():
    m = am.Model({"show_progress": False})
    m.add_agents(3, x=np.zeros(3))
    _warns_and(lambda: m.agents.record("x", 7.0))
    assert m.agents.frame["x"].to_list() == [7, 7, 7]
    _warns_and(lambda: m.agents.update_data({"x": 9.0}))
    assert m.agents.frame["x"].to_list() == [9, 9, 9]


def test_agentlist_agents_and_agent_ids_deprecated():
    m = am.Model({"show_progress": False})
    m.add_agents(3, agent_class=am.Agent)
    objs = _warns_and(lambda: m.agents.agents)
    assert len(objs) == 3
    ids = _warns_and(lambda: m.agents.agent_ids)
    assert sorted(ids) == [0, 1, 2]


def test_agent_record_and_update_data_deprecated():
    m = am.Model({"show_progress": False})
    m.add_agents(1, agent_class=am.Agent)
    a = next(iter(m.agents))
    _warns_and(lambda: a.record("x", 3.0))
    _warns_and(lambda: a.update_data({"x": 4.0}))
    assert m.agents.frame["x"].to_list() == [4]


def test_model_update_agent_data_and_batch_update_deprecated():
    m = am.Model({"show_progress": False})
    m.add_agents(3, wealth=np.zeros(3, dtype=int))
    _warns_and(lambda: m.update_agent_data(0, {"wealth": 5}))
    assert m.get_agent_data(0)["wealth"].item() == 5
    _warns_and(lambda: m.batch_update_agents([1, 2], {"wealth": [7, 8]}))
    assert m.agents.frame["wealth"].to_list() == [5, 7, 8]


def test_population_mutators_deprecated():
    m = am.Model({"show_progress": False})
    m.add_agents(2, wealth=np.array([1, 2]))
    pop = m.population
    _warns_and(lambda: pop.set_agent_value(0, "wealth", 9))
    assert pop.data.filter(pl.col("id") == 0)["wealth"].item() == 9
    _warns_and(lambda: pop.batch_update({"wealth": [3, 4]}))
    assert pop.data["wealth"].to_list() == [3, 4]
    _warns_and(lambda: pop.batch_update_by_ids([0], {"wealth": [1]}))
    assert pop.data["wealth"].to_list() == [1, 4]


def test_suppress_env_silences(monkeypatch):
    monkeypatch.setenv("AMBER_SUPPRESS_DEPRECATIONS", "1")
    m = am.Model({"show_progress": False})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.record("x", 1)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)
