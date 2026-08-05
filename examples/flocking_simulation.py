#!/usr/bin/env python3
"""Boids flocking (Reynolds) on continuous 2D space — AMBER smoke demo.

Uses per-agent OOP for clarity (neighbour search is O(N²)). For a dense
NumPy kernel version see ``flocking_tensor.py``. Optional plots need
matplotlib. Full notebook: ``flocking_simulation.ipynb``.
"""

from __future__ import annotations

import ambr as am
import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


class Boid(am.Agent):
    def setup(self):
        size = float(self.model.p.get("size", 40))
        self.x = float(self.model.rng.random() * size)
        self.y = float(self.model.rng.random() * size)
        ang = float(self.model.rng.random() * 2 * np.pi)
        self.vx = float(np.cos(ang))
        self.vy = float(np.sin(ang))

    def flock(self):
        m = self.model
        p = m.p
        positions = np.column_stack([m.agents.numpy("x"), m.agents.numpy("y")])
        velocities = np.column_stack(
            [m.agents.numpy("vx"), m.agents.numpy("vy")]
        )
        me = np.array([self.x, self.y], dtype=float)
        d = np.linalg.norm(positions - me, axis=1)
        d[int(self.id)] = np.inf

        outer = d <= float(p.get("outer_radius", 8.0))
        inner = d <= float(p.get("inner_radius", 2.5))

        v = np.zeros(2, dtype=float)
        if outer.any():
            center = positions[outer].mean(axis=0)
            v += (center - me) * float(p.get("cohesion_strength", 0.01))
            avg_v = velocities[outer].mean(axis=0)
            v += (avg_v - np.array([self.vx, self.vy])) * float(
                p.get("alignment_strength", 0.2)
            )
        if inner.any():
            away = me - positions[inner]
            v += away.sum(axis=0) * float(p.get("separation_strength", 0.05))

        size = float(p.get("size", 40))
        border = float(p.get("border_distance", 5.0))
        strength = float(p.get("border_strength", 0.3))
        if self.x < border:
            v[0] += strength
        elif self.x > size - border:
            v[0] -= strength
        if self.y < border:
            v[1] += strength
        elif self.y > size - border:
            v[1] -= strength

        vel = _normalize(np.array([self.vx, self.vy]) + v)
        self.vx, self.vy = float(vel[0]), float(vel[1])
        self.x = float(np.clip(self.x + self.vx, 0.0, size))
        self.y = float(np.clip(self.y + self.vy, 0.0, size))


class BoidsModel(am.Model):
    def setup(self):
        n = int(self.p.get("population", 40))
        self.agents = am.AgentList(self, n, Boid)

    def step(self):
        self.agents.flock()

    def update(self):
        vx = self.agents.numpy("vx")
        vy = self.agents.numpy("vy")
        speed = np.sqrt(vx * vx + vy * vy).mean()
        self.record_model("mean_speed", float(speed))


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
    plt.figure(figsize=(5, 5))
    plt.quiver(
        agents["x"].to_numpy(),
        agents["y"].to_numpy(),
        agents["vx"].to_numpy(),
        agents["vy"].to_numpy(),
        angles="xy",
        scale_units="xy",
        scale=0.5,
    )
    plt.title("Boids final positions / headings")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("flocking_final.png", dpi=120)
    print("Wrote flocking_final.png")
    plt.close()


if __name__ == "__main__":
    model = BoidsModel(
        {
            "population": 40,
            "size": 40,
            "steps": 30,
            "seed": 0,
            "show_progress": False,
            "inner_radius": 2.5,
            "outer_radius": 8.0,
            "cohesion_strength": 0.01,
            "separation_strength": 0.05,
            "alignment_strength": 0.2,
            "border_distance": 5.0,
            "border_strength": 0.3,
        }
    )
    results = model.run()
    print("Smoke run OK:", results["info"])
    print("metrics:", results["model"].columns)
    print(results["model"].tail(3))
    _optional_plot(results)
