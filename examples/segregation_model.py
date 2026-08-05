#!/usr/bin/env python3
"""Schelling segregation on a grid — AMBER smoke demo.

Per-agent happiness / move logic (OOP lane) on ``GridEnvironment``. For a
vectorized Schelling variant see ``schelling_vectorized.py``. Optional plots
need matplotlib.
"""

from __future__ import annotations

from typing import Optional, Tuple

import ambr as am
import numpy as np


def _as_tuple(pos) -> Optional[Tuple]:
    if pos is None:
        return None
    return tuple(pos) if isinstance(pos, list) else tuple(pos)


class Person(am.Agent):
    def setup(self):
        self.group = int(
            self.model.rng.integers(0, int(self.model.p.get("n_groups", 2)))
        )
        self.happy = False
        self.share_similar = 0.0

    def update_happiness(self):
        env = self.model.env
        row = self.model.agents_df.filter(
            self.model.agents_df["id"] == self.id
        )
        if row.is_empty() or "grid_position" not in row.columns:
            self.happy = True
            self.share_similar = 1.0
            return
        pos = _as_tuple(row["grid_position"].item())
        if pos is None:
            self.happy = True
            self.share_similar = 1.0
            return
        neighbors = env.get_neighbors(pos, include_diagonal=True, distance=1)
        similar = 0
        total = 0
        for npos in neighbors:
            oid = env.get_agent_at_pos(npos)
            if oid is None:
                continue
            total += 1
            other = self.model.agents.by_id(int(oid))
            if int(other.group) == int(self.group):
                similar += 1
        share = (similar / total) if total else 1.0
        self.share_similar = float(share)
        self.happy = share >= float(self.model.p.get("want_similar", 0.3))

    def find_new_home(self):
        env = self.model.env
        empty = env.empty_positions()
        if not empty:
            return
        row = self.model.agents_df.filter(
            self.model.agents_df["id"] == self.id
        )
        if row.is_empty() or "grid_position" not in row.columns:
            return
        old = _as_tuple(row["grid_position"].item())
        idx = int(self.model.rng.integers(0, len(empty)))
        new_pos = empty[idx]
        if old is not None:
            env.remove_agent_from_pos(old)
        env.add_agent_from_id(self.id, new_pos)


class SegregationModel(am.Model):
    def setup(self):
        size = int(self.p.get("size", 20))
        density = float(self.p.get("density", 0.8))
        self.env = am.GridEnvironment(self, size=(size, size))
        n = int(size * size * density)
        self.agents = am.AgentList(self, n, Person)
        for agent in self.agents:
            pos = self.env.get_random_empty_cell()
            if pos is None:
                break
            self.env.add_agent_from_id(agent.id, pos)
        for agent in self.agents:
            agent.update_happiness()

    def step(self):
        unhappy = [a for a in self.agents if not a.happy]
        order = list(unhappy)
        self.rng.shuffle(order)
        for agent in order:
            agent.find_new_home()
        for agent in self.agents:
            agent.update_happiness()

    def update(self):
        happy = sum(1 for a in self.agents if a.happy)
        n = max(len(self.agents), 1)
        mean_share = float(
            np.mean([float(a.share_similar) for a in self.agents])
        )
        self.record_model("happy_frac", happy / n)
        self.record_model("mean_share_similar", mean_share)


def _optional_plot(results) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            "Skipping plot — install a NumPy-compatible matplotlib:\n"
            "  pip install -U 'matplotlib>=3.8'\n"
            f"Underlying error: {exc!r}"
        )
        return
    agents = results["agents"]
    if "grid_position" not in agents.columns:
        print("No grid_position column to plot.")
        return
    xs, ys, groups = [], [], []
    for row in agents.iter_rows(named=True):
        pos = _as_tuple(row.get("grid_position"))
        if pos is None:
            continue
        xs.append(pos[0])
        ys.append(pos[1])
        groups.append(int(row.get("group", 0)))
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys, c=groups, cmap="coolwarm", s=20)
    plt.title("Segregation final layout")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("segregation_final.png", dpi=120)
    print("Wrote segregation_final.png")
    plt.close()


if __name__ == "__main__":
    model = SegregationModel(
        {
            "size": 15,
            "density": 0.8,
            "n_groups": 2,
            "want_similar": 0.3,
            "steps": 20,
            "seed": 0,
            "show_progress": False,
        }
    )
    results = model.run()
    print("Smoke run OK:", results["info"])
    print("metrics:", results["model"].columns)
    print(results["model"].tail(5))
    _optional_plot(results)
