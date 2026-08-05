#!/usr/bin/env python3
"""Forest fire on a grid (NetLogo FireSimple-style) using AMBER.

Runnable smoke demo (CPU). Optional animation needs a NumPy-compatible
matplotlib (``pip install -U 'matplotlib>=3.8'``). Full notebook narrative:
``forest_fire_simulation.ipynb``.
"""

from __future__ import annotations

import ambr as am
import numpy as np


class ForestFireModel(am.Model):
    """Trees on a lattice; fire spreads to orthogonal neighbours."""

    def setup(self):
        size = int(self.p.get("size", 30))
        density = float(self.p.get("tree_density", 0.6))
        if density > 1.0:
            density = density / 100.0
        self.size = size

        occ = self.rng.random((size, size)) < density
        ys, xs = np.nonzero(occ)
        n = int(xs.size)
        cond = np.zeros(n, dtype=np.int64)  # 0 alive, 1 burning, 2 burned
        for i, (x, y) in enumerate(zip(xs, ys)):
            if x == 0:  # ignite left edge
                cond[i] = 1

        self.add_agents(
            n,
            x=xs.astype(np.int64),
            y=ys.astype(np.int64),
            condition=cond,
        )
        self._index = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(xs, ys))}

    def step(self):
        cond = self.agents.numpy("condition").copy()
        xs = self.agents.numpy("x")
        ys = self.agents.numpy("y")
        burning_ids = np.flatnonzero(cond == 1)
        if burning_ids.size == 0:
            return

        to_ignite = set()
        for i in burning_ids:
            x, y = int(xs[i]), int(ys[i])
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                j = self._index.get((x + dx, y + dy))
                if j is not None and cond[j] == 0:
                    to_ignite.add(j)

        cond[burning_ids] = 2
        if to_ignite:
            cond[list(to_ignite)] = 1
        self.agents.set(condition=cond)

    def update(self):
        cond = self.agents.numpy("condition")
        self.record_model("alive", int((cond == 0).sum()))
        self.record_model("burning", int((cond == 1).sum()))
        self.record_model("burned", int((cond == 2).sum()))


def _optional_plot(results, size: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # ImportError or NumPy ABI errors
        print(
            "Skipping plot — install a NumPy-compatible matplotlib:\n"
            "  pip install -U 'matplotlib>=3.8'\n"
            f"Underlying error: {exc!r}"
        )
        return
    agents = results["agents"]
    if "step" in agents.columns:
        final = agents.filter(agents["step"] == agents["step"].max())
    else:
        final = agents
    grid = np.full((size, size), -1, dtype=int)
    for x, y, c in zip(
        final["x"].to_list(), final["y"].to_list(), final["condition"].to_list()
    ):
        grid[int(y), int(x)] = int(c)
    plt.figure(figsize=(6, 5))
    plt.imshow(grid, cmap="viridis", vmin=-1, vmax=2)
    plt.title("Forest fire final (-1 empty, 0 alive, 1 burning, 2 burned)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("forest_fire_final.png", dpi=120)
    print("Wrote forest_fire_final.png")
    plt.close()


if __name__ == "__main__":
    params = {
        "tree_density": 0.55,
        "size": 25,
        "steps": 40,
        "seed": 0,
        "show_progress": False,
    }
    model = ForestFireModel(params)
    results = model.run()
    print("Smoke run OK:", results["info"])
    print("metrics columns:", results["model"].columns)
    print(results["model"].tail(3))
    _optional_plot(results, params["size"])
