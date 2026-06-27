"""Phase 3: unified collection (OOP + vectorized) and state read/write façades.

`model.agents` is one collection: `add_agents(n, agent_class=...)` tracks the
objects, you can iterate / `random.choice` / `by_id` them (no hand-rolled
`agent_objects_list`), and `numpy`/`set`/`borrow`/`commit`/`frame` cover bulk
state I/O without the repeated `.to_numpy()` + per-column-assign boilerplate.
"""

import random

import numpy as np
import polars as pl

import ambr as am


class _Walker(am.Agent):
    def setup(self):
        self.x = 0


def test_add_agents_with_agent_class_tracks_objects():
    m = am.Model({"show_progress": False})
    m.add_agents(5, agent_class=_Walker)
    assert len(m.agents) == 5
    assert all(isinstance(a, am.Agent) for a in m.agents)
    a = random.Random(0).choice(m.agents)          # iterate/choose without a hand-rolled list
    assert a in m.agents
    assert m.agents.by_id(a.id) is a               # id -> object lookup


def test_numpy_and_set_roundtrip():
    m = am.Model({"show_progress": False})
    m.add_agents(4, x=np.arange(4, dtype=float), y=np.zeros(4))
    x, y = m.agents.numpy("x", "y")
    assert x.tolist() == [0, 1, 2, 3]
    m.agents.set(x=x + 1.0, y=y + 10.0)            # one call, two columns
    assert m.agents.frame["x"].to_list() == [1, 2, 3, 4]
    assert m.agents.frame["y"].to_list() == [10, 10, 10, 10]


def test_numpy_single_column_returns_array():
    m = am.Model({"show_progress": False})
    m.add_agents(3, w=np.array([5, 6, 7]))
    w = m.agents.numpy("w")
    assert isinstance(w, np.ndarray) and w.tolist() == [5, 6, 7]


def test_borrow_commit_facade_matches_tensor_lane():
    m = am.Model({"show_progress": False})
    m.add_agents(4, x=np.ones(4), y=np.zeros(4))
    x, is_view = m.agents.borrow("x")
    assert x.tolist() == [1, 1, 1, 1]
    m.agents.commit(x=x + 2.0)
    assert m.agents.frame["x"].to_list() == [3, 3, 3, 3]


def test_set_on_filtered_view():
    m = am.Model({"show_progress": False})
    m.add_agents(5, x=np.arange(5, dtype=float))
    m.agents.where(pl.col("x") >= 3).set(x=99.0)
    assert m.agents.frame["x"].to_list() == [0, 1, 2, 99, 99]
