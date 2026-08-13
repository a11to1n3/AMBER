#!/usr/bin/env python3
"""Canonical Schelling segregation on AMBER's grid + agent table.

Recommended spatial-model pattern:

* **Columns** — group / happy live on ``model.agents`` (Polars-backed).
* **Occupancy** — ``GridEnvironment`` owns ``grid_position`` via
  ``add_agent_from_id`` / ``remove_agent_from_pos`` / ``get_random_empty_cell``.
* **Neighbourhood** — agent methods use ``env.get_neighbors(..., radius=1)``
  and ``get_agent_at_pos`` (Moore ring). This part is per-agent by nature.
* **Filter + act** — ``agents.where(~agents.happy)`` selects who moves.

For multi-objective SMAC calibration of Schelling, see
``smac_calibration_advanced.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import ambr as am
from ambr.environments import GridEnvironment


def _as_tuple(pos) -> Optional[Tuple[int, ...]]:
    if pos is None:
        return None
    return tuple(pos) if isinstance(pos, list) else tuple(pos)


class SchellingAgent(am.Agent):
    def setup(self):
        if not hasattr(self, "group"):
            self.group = 0
        self.happy = True

    def _position(self) -> Optional[Tuple[int, ...]]:
        env: GridEnvironment = self.model.env
        for aid, p in zip(env.df["id"].to_list(), env.df["grid_position"].to_list()):
            if aid == self.id:
                return _as_tuple(p)
        return None

    def update_happiness(self, want_similar: float) -> None:
        env: GridEnvironment = self.model.env
        pos = self._position()
        if pos is None:
            self.happy = True
            return
        neighbors = env.get_neighbors(pos, radius=1)
        similar = 0
        occupied = 0
        for npos in neighbors:
            oid = env.get_agent_at_pos(npos)
            if oid is None:
                continue
            occupied += 1
            if self.model.agents.by_id(oid).group == self.group:
                similar += 1
        share = similar / occupied if occupied else 1.0
        self.happy = share >= want_similar

    def relocate(self) -> None:
        env: GridEnvironment = self.model.env
        cur = self._position()
        empty = env.get_random_empty_cell()
        if empty is None or cur is None or empty == cur:
            return
        env.remove_agent_from_pos(cur)
        env.add_agent_from_id(self.id, empty)


class SchellingModel(am.Model):
    """Grid Schelling using AgentList + GridEnvironment occupancy helpers."""

    params = {
        "grid_size": (int, 20),
        "n": (int, 300),
        "want_similar": (float, 0.5),
        "steps": (int, 30),
        "seed": (int, 0),
    }
    model_reporters = {
        "happy_frac": lambda m: float(m.agents.happy.sum()) / max(len(m.agents), 1),
    }

    def setup(self):
        g = int(self.p.grid_size)
        n = int(self.p.n)
        max_cells = g * g
        if n > max_cells:
            raise ValueError(f"n={n} exceeds grid capacity {max_cells}")

        self.env = GridEnvironment(self, size=g, torus=False)
        # OOP lane so by_id / methods work; bulk-set group after create.
        self.agents = am.AgentList(self, n, SchellingAgent)
        groups = self.rng.integers(0, 2, size=n)
        self.agents.set(group=groups)

        cells = list(self.env.positions)
        self.rng.shuffle(cells)
        for agent, pos in zip(self.agents, cells[:n]):
            self.env.add_agent_from_id(agent.id, pos)

    def step(self):
        want = float(self.p.want_similar)
        for a in self.agents:
            a.update_happiness(want)
        # Vectorized select of unhappy agents; relocate each via grid helpers.
        for aid in self.agents.where(~self.agents.happy).ids.to_list():
            self.agents.by_id(aid).relocate()


def main():
    m = SchellingModel(
        {
            "grid_size": 15,
            "n": 180,
            "want_similar": 0.4,
            "steps": 25,
            "seed": 1,
            "show_progress": False,
        }
    )
    res = m.run()
    print(res.model.tail().to_dicts())
    print("final happy_frac:", float(res.model["happy_frac"][-1]))


if __name__ == "__main__":
    main()
