"""Coverage-focused tests for AgentList / view list-operations and dispatch.

Exercises the list-like mutation methods, the __getitem__/__setitem__/__add__
index variants on AgentList and _SubView, call/apply method dispatch, group_by,
by_id, get_data, and the deprecated agent_ids accessor.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import ambr as am


class _A(am.Agent):
    def setup(self):
        self.v = 0

    def bump(self, by=1):
        self.v += by
        return self.v


def _pop(n=6):
    m = am.Model({"show_progress": False})
    m.add_agents(n, agent_class=_A)
    m.agents.set(w=np.arange(n, dtype=float), team=np.arange(n) % 2)
    return m


def test_agentlist_list_mutations():
    al = _pop(6).agents
    a0 = al[0]
    assert a0 in al                       # __contains__
    assert al.index(a0) == 0
    assert al.count(a0) == 1
    assert "agents" in repr(al)           # __repr__
    popped = al.pop()                     # pop last
    assert popped not in al and len(al) == 5
    al.insert(0, popped)                  # insert
    assert al[0] is popped
    al.reverse()                          # reverse
    al.sort(key=lambda a: a.id)           # sort
    assert [a.id for a in al] == [0, 1, 2, 3, 4, 5]
    c = al.copy()                         # copy
    assert len(c) == 6 and c is not al
    extra = _A(c.model, 99)
    c.append(extra)                       # append
    c.extend([_A(c.model, 100)])          # extend
    assert len(c) == 8
    c.remove(extra)                       # remove
    assert extra not in c
    c.clear()                             # clear (empties the tracked-object list)
    assert len(c._agent_objects) == 0


def test_agentlist_getitem_variants():
    al = _pop(5).agents
    assert al[0].id == 0                  # int
    assert al[1:3] == [al[1], al[2]]      # slice
    assert set(al[[0, 2]].ids.to_list()) == {0, 2}          # list of positions
    mask = np.array([True, False, True, False, True])
    assert set(al[mask].ids.to_list()) == {0, 2, 4}         # bool mask
    assert set(al[pl.col("w") >= 3].ids.to_list()) == {3, 4}  # Expr
    assert set(al[pl.Series([1, 4])].ids.to_list()) == {1, 4}  # id Series
    with pytest.raises(ValueError):
        _ = al[np.array([True, False])]   # wrong-length bool mask
    with pytest.raises(TypeError):
        _ = al["nope"]                    # invalid index type


def test_agentlist_setitem_and_add():
    al = _pop(4).agents
    other = al[2]
    al[0] = other                         # __setitem__
    assert al[0] is other
    combined = al + al.copy()             # __add__ (AgentList)
    assert len(combined) == 8
    combined2 = al + [_A(al.model, 50)]   # __add__ (list)
    assert len(combined2) == 5
    with pytest.raises(TypeError):
        _ = al + 3


def test_call_and_apply():
    al = _pop(4).agents
    results = al.call("bump", 2)          # bump(2) on each -> returns new v
    assert list(results) == [2, 2, 2, 2]
    ids = al.apply(lambda a: a.id)
    assert sorted(ids.to_list()) == [0, 1, 2, 3]


def test_group_by_get_data_by_id_agentids():
    m = _pop(6)
    groups = m.agents.group_by("team")
    assert set(groups.keys()) == {0, 1}
    assert len(groups[0].ids) == 3
    assert m.agents.get_data().height == 6
    assert m.agents.by_id(2).id == 2
    with pytest.raises(KeyError):
        m.agents.by_id(999)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sorted(m.agents.agent_ids) == [0, 1, 2, 3, 4, 5]   # deprecated


def _pop_cols(n=6):
    m = am.Model({"show_progress": False})
    m.add_agents(n, agent_class=_A)
    m.agents.set(wealth=np.arange(n), x=np.zeros(n))
    return m


def test_scatter_add_value_shapes():
    m = _pop_cols(5)
    m.agents.at[[0, 1, 2]].scatter_add(wealth=1)              # scalar broadcast
    assert m.agents.wealth.to_list()[:3] == [1, 2, 3]        # was 0,1,2
    m.agents.at[[0, 1]].scatter_add(x=np.array([1.5, 2.5]))  # ndarray delta
    m.agents.at[[3, 4]].scatter_add(x=[1.0, 2.0])            # list delta
    m.agents.at[[0]].scatter_add(x=pl.Series([0.5]))         # pl.Series delta
    assert m.agents.x.to_list()[0] == 2.0                    # 1.5 + 0.5


def test_scatter_add_duplicate_ids_and_new_column():
    m = _pop_cols(4)
    m.agents.at[[1, 1, 3]].scatter_add(wealth=1)             # agent 1 += 2, agent 3 += 1
    w = m.agents.wealth.to_list()
    assert w[1] == 1 + 2 and w[3] == 3 + 1
    m.agents.at[[0, 2]].scatter_add(gold=5)                  # brand-new column from zeros
    g = m.agents.gold.to_list()
    assert g[0] == 5 and g[2] == 5 and g[1] == 0


def test_scatter_add_float_on_int_upcasts():
    m = _pop_cols(3)
    m.agents.at[[0, 1, 2]].scatter_add(wealth=0.5)           # float delta on int column
    assert m.agents.wealth.to_list() == [0.5, 1.5, 2.5]


def test_scatter_add_length_mismatch_and_empty_view():
    m = _pop_cols(3)
    with pytest.raises(ValueError):
        m.agents.at[[0, 1]].scatter_add(x=np.array([1.0]))   # len 1 != view length 2
    empty = m.agents.where(pl.col("wealth") > 999)           # empty view
    empty.scatter_add(newcol=1)                              # creates column, no rows touched
    assert "newcol" in m.agents.get_data().columns


def test_write_column_expr_path():
    m = _pop_cols(5)
    infected = m.agents.where(pl.col("wealth") >= 3)         # ids 3, 4
    infected.x = pl.col("wealth") * 10.0                     # Expr over existing column
    assert m.agents.where(pl.col("id") == 3).x.to_list() == [30.0]
    infected.boost = pl.col("wealth") + 1                    # Expr creating a new column
    assert m.agents.where(pl.col("id") == 4).boost.to_list() == [5]


def test_view_method_dispatch_and_alignment():
    m = _pop_cols(4)
    res = m.agents.bump(3)                                   # not a column -> per-agent dispatch
    assert list(res) == [3, 3, 3, 3]
    sub = m.agents.at[[2, 2, 0]]                             # duplicate-id scatter view
    assert sub.wealth.to_list() == [2, 2, 0]                # __getattr__ join-aligns to view order


def test_subview_getitem_variants():
    al = _pop(6).agents
    sub = al.where(pl.col("w") >= 2)      # ids 2,3,4,5
    assert sub[0].id == 2                 # int
    assert [a.id for a in sub[0:2]] == [2, 3]                 # slice
    assert set(sub[[0, 1]].ids.to_list()) == {2, 3}          # list positions
    assert set(sub[np.array([True, False, True, False])].ids.to_list()) == {2, 4}  # bool
    assert set(sub[pl.col("w") >= 4].ids.to_list()) == {4, 5}   # Expr
    assert set(sub[pl.Series([3, 5])].ids.to_list()) == {3, 5}  # id Series
    assert "agents" in repr(sub)
    with pytest.raises(TypeError):
        _ = sub["nope"]
