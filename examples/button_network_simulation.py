#!/usr/bin/env python3
"""Button / random-graph percolation demo (Kauffman buttons + threads).

Each step ties a random pair of buttons with a thread and tracks the size of
the largest connected component. Runnable smoke on CPU; optional plots need
matplotlib. Full notebook: ``button_network_simulation.ipynb``.
"""

from __future__ import annotations

import ambr as am
import networkx as nx
import numpy as np


class ButtonModel(am.Model):
    """Randomly connect n buttons; record giant-component fraction."""

    def setup(self):
        n = int(self.p.get("n", 200))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(n))
        self.add_agents(n, degree=0, cluster_size=1)

    def step(self):
        n = len(self.graph)
        a, b = self.rng.integers(0, n, size=2)
        if a != b:
            self.graph.add_edge(int(a), int(b))

        degrees = np.array([d for _, d in self.graph.degree()], dtype=np.int64)
        clusters = list(nx.connected_components(self.graph))
        size_of = {node: len(c) for c in clusters for node in c}
        cluster_sizes = np.array([size_of[i] for i in range(n)], dtype=np.int64)
        self.agents.set(degree=degrees, cluster_size=cluster_sizes)

    def update(self):
        n = max(len(self.graph), 1)
        clusters = list(nx.connected_components(self.graph))
        giant = max((len(c) for c in clusters), default=1)
        self.record_model("threads", self.graph.number_of_edges())
        self.record_model("giant_frac", giant / n)


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
    m = results["model"]
    plt.figure(figsize=(7, 4))
    plt.plot(m["t"].to_list(), m["giant_frac"].to_list(), lw=2)
    plt.xlabel("threads (steps)")
    plt.ylabel("giant component fraction")
    plt.title("Button network phase transition")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("button_network_giant.png", dpi=120)
    print("Wrote button_network_giant.png")
    plt.close()


if __name__ == "__main__":
    model = ButtonModel(
        {"n": 300, "steps": 250, "seed": 0, "show_progress": False}
    )
    results = model.run()
    print("Smoke run OK:", results["info"])
    print("metrics:", results["model"].columns)
    print(results["model"].tail(3))
    print("final giant_frac:", results["model"]["giant_frac"][-1])
    _optional_plot(results)
