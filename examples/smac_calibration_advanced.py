#!/usr/bin/env python3
"""
SMAC Calibration Example - Advanced Multi-Objective Optimization
===============================================================

Multi-objective SMAC calibration of a Schelling-style segregation model
using AMBER's :class:`~ambr.optimization.MultiObjectiveSMAC`.

Key features:
- Multi-objective optimization with Pareto frontiers
- ``SMACParameterSpace`` parameter ranges
- Grid helpers on :class:`~ambr.environments.GridEnvironment`
- Canonical agent writes (``agents.at[id].set(...)``)

Requirements:
    pip install smac ConfigSpace

For a lighter single-objective vectorized wealth example, see
``smac_calibration_basic.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import ambr as am
import numpy as np


def _as_tuple(pos) -> Optional[Tuple]:
    if pos is None:
        return None
    return tuple(pos) if isinstance(pos, list) else tuple(pos)


class SegregationModel(am.Model):
    """Schelling-style multi-type segregation on a discrete grid."""

    def setup(self):
        g = int(self.p["grid_size"])
        self.env = am.GridEnvironment(self, size=g)

        agent_types = self.get_agent_types()
        type_names = list(agent_types.keys())
        type_probs = np.array([agent_types[k] for k in type_names], dtype=float)
        type_probs = type_probs / type_probs.sum()

        n_agents = int(g * g * float(self.p["density"]))
        n_agents = min(n_agents, g * g)

        for i in range(n_agents):
            agent = am.Agent(self, i)
            agent.agent_type = str(
                self.rng.choice(type_names, p=type_probs)
            )
            agent.tolerance = float(self.get_agent_tolerance(agent.agent_type))
            agent.mobility = float(self.get_agent_mobility(agent.agent_type))
            agent.satisfaction = 0.0
            agent.moves = 0
            # Population row first so the grid can write grid_position.
            self.add_agent(agent)
            pos = self.env.get_random_empty_cell()
            if pos is None:
                break
            self.env.add_agent_from_id(i, pos)

    def get_agent_types(self) -> Dict[str, float]:
        dist = self.p.get("agent_type_distribution", "binary")
        if dist == "binary":
            a = float(self.p.get("type_A_fraction", 0.5))
            return {"type_A": a, "type_B": 1.0 - a}
        if dist == "three_types":
            a = float(self.p.get("type_A_fraction", 0.5))
            remaining = 1.0 - a
            return {
                "type_A": a,
                "type_B": remaining * 0.6,
                "type_C": remaining * 0.4,
            }
        return {"type_A": 0.33, "type_B": 0.33, "type_C": 0.34}

    def get_agent_tolerance(self, agent_type: str) -> float:
        base = float(self.p.get("base_tolerance", 0.3))
        key = f"tolerance_multiplier_{agent_type[-1]}"
        return base * float(self.p.get(key, 1.0))

    def get_agent_mobility(self, agent_type: str) -> float:
        base = float(self.p.get("base_mobility", 0.1))
        key = f"mobility_multiplier_{agent_type[-1]}"
        return base * float(self.p.get(key, 1.0))

    def step(self):
        radius = int(self.p.get("neighborhood_radius", 1))
        agent_ids = self.agents.ids.to_list()
        order = np.array(agent_ids, dtype=int)
        self.rng.shuffle(order)

        for agent_id in order.tolist():
            row = self.get_agent_data(agent_id)
            if "grid_position" not in row.columns:
                continue
            pos = _as_tuple(row["grid_position"].item())
            if pos is None:
                continue

            satisfaction = self.calculate_satisfaction(agent_id, pos, radius)
            self.agents.at[agent_id].set(satisfaction=float(satisfaction))

            tolerance = float(row["tolerance"].item())
            mobility = float(row["mobility"].item())
            if satisfaction < tolerance and self.rng.random() < mobility:
                new_pos = self.find_better_location(agent_id, pos, satisfaction)
                if new_pos is not None and new_pos != pos:
                    self.env.remove_agent_from_pos(pos)
                    self.env.add_agent_from_id(agent_id, new_pos)
                    moves = int(row["moves"].item())
                    self.agents.at[agent_id].set(moves=moves + 1)

    def calculate_satisfaction(
        self, agent_id: int, pos: Tuple, radius: Optional[int] = None
    ) -> float:
        radius = int(self.p.get("neighborhood_radius", 1) if radius is None else radius)
        agent_type = self.get_agent_data(agent_id)["agent_type"].item()
        neighbor_cells = self.env.get_neighbors(pos, radius=radius)
        if not neighbor_cells:
            return 0.0

        similar = 0
        total = 0
        for cell in neighbor_cells:
            nid = self.env.get_agent_at_pos(cell)
            if nid is None:
                continue
            total += 1
            ntype = self.get_agent_data(nid)["agent_type"].item()
            if ntype == agent_type:
                similar += 1
        return similar / total if total > 0 else 0.0

    def find_better_location(
        self, agent_id: int, current_pos: Tuple, current_satisfaction: float
    ) -> Optional[Tuple]:
        search_radius = int(self.p.get("search_radius", 4))
        empty_cells = self.env.get_empty_cells_in_radius(current_pos, search_radius)
        if not empty_cells:
            return None

        max_eval = min(len(empty_cells), int(self.p.get("max_location_evaluations", 10)))
        idx = self.rng.choice(len(empty_cells), size=max_eval, replace=False)
        sampled = [empty_cells[int(i)] for i in np.atleast_1d(idx)]

        best_pos = None
        best_sat = current_satisfaction
        for pos in sampled:
            sat = self.calculate_satisfaction(agent_id, pos)
            if sat > best_sat:
                best_sat = sat
                best_pos = pos
        return best_pos

    def update(self):
        if self.t <= 0:
            return
        self.record_model("segregation_index", self.calculate_segregation_index())
        self.record_model("clustering_coefficient", self.calculate_clustering_coefficient())
        self.record_model("mobility_rate", self.calculate_mobility_rate())
        sat = self.agents_df["satisfaction"]
        self.record_model("satisfaction_mean", float(sat.mean()))
        self.record_model("satisfaction_std", float(sat.std() or 0.0))

    def _iter_placed(self):
        df = self.agents_df
        if "grid_position" not in df.columns:
            return
        for row in df.iter_rows(named=True):
            pos = _as_tuple(row.get("grid_position"))
            if pos is None:
                continue
            yield row, pos

    def calculate_segregation_index(self) -> float:
        """Simplified type-separation index (between- vs within-type distances)."""
        type_positions: Dict[Any, List[Tuple]] = {}
        for row, pos in self._iter_placed():
            type_positions.setdefault(row["agent_type"], []).append(pos)

        if len(type_positions) < 2:
            return 0.0

        within: List[float] = []
        between: List[float] = []
        for type_a, positions_a in type_positions.items():
            sample_a = positions_a[: min(50, len(positions_a))]
            for pos_a in sample_a:
                for pos_b in positions_a:
                    if pos_a != pos_b:
                        within.append(
                            float(np.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]))
                        )
                for type_b, positions_b in type_positions.items():
                    if type_a == type_b:
                        continue
                    sample_b = positions_b[: min(50, len(positions_b))]
                    for pos_b in sample_b:
                        between.append(
                            float(np.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]))
                        )

        if not within or not between:
            return 0.0
        avg_w, avg_b = float(np.mean(within)), float(np.mean(between))
        return (avg_b - avg_w) / (avg_b + avg_w + 1e-12)

    def calculate_clustering_coefficient(self) -> float:
        values: List[float] = []
        for _, pos in self._iter_placed():
            neighbors = self.env.get_neighbors(pos, radius=1)
            occupied = [c for c in neighbors if self.env.get_agent_at_pos(c) is not None]
            if len(occupied) < 2:
                continue
            connections = 0
            possible = len(occupied) * (len(occupied) - 1) / 2
            for i, p1 in enumerate(occupied):
                for p2 in occupied[i + 1 :]:
                    if abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1:
                        connections += 1
            values.append(connections / possible if possible else 0.0)
        return float(np.mean(values)) if values else 0.0

    def calculate_mobility_rate(self) -> float:
        if self.t <= 0:
            return 0.0
        total_moves = float(self.agents_df["moves"].sum())
        n = max(len(self.agents_df), 1)
        return total_moves / (n * self.t)


def create_advanced_parameter_space():
    """Parameter space for multi-objective Schelling calibration."""
    param_space = am.SMACParameterSpace()
    param_space.add_parameter("grid_size", param_type="int", bounds=(10, 30), default=15)
    param_space.add_parameter("density", param_type="float", bounds=(0.6, 0.9), default=0.75)
    param_space.add_parameter(
        "agent_type_distribution",
        param_type="categorical",
        choices=["binary", "three_types", "uniform"],
        default="binary",
    )
    param_space.add_parameter(
        "type_A_fraction", param_type="float", bounds=(0.3, 0.7), default=0.5
    )
    param_space.add_parameter(
        "base_tolerance", param_type="float", bounds=(0.1, 0.8), default=0.3
    )
    param_space.add_parameter(
        "tolerance_multiplier_A", param_type="float", bounds=(0.5, 2.0), default=1.0
    )
    param_space.add_parameter(
        "tolerance_multiplier_B", param_type="float", bounds=(0.5, 2.0), default=1.0
    )
    param_space.add_parameter(
        "base_mobility", param_type="float", bounds=(0.01, 0.3), default=0.1
    )
    param_space.add_parameter(
        "mobility_multiplier_A", param_type="float", bounds=(0.5, 2.0), default=1.0
    )
    param_space.add_parameter(
        "mobility_multiplier_B", param_type="float", bounds=(0.5, 2.0), default=1.0
    )
    param_space.add_parameter(
        "neighborhood_radius", param_type="int", bounds=(1, 2), default=1
    )
    param_space.add_parameter(
        "search_radius", param_type="int", bounds=(2, 6), default=3
    )
    param_space.add_parameter(
        "max_location_evaluations", param_type="int", bounds=(5, 15), default=8
    )
    return param_space


def _final_metric(model: SegregationModel, name: str) -> float:
    results = getattr(model, "results", None)
    if results is None:
        raise RuntimeError("model.results is not set; run via SMACOptimizer path")
    return float(results["model"][name].tail(1).item())


def segregation_objective(model: SegregationModel) -> float:
    return abs(_final_metric(model, "segregation_index") - 0.4)


def clustering_objective(model: SegregationModel) -> float:
    return abs(_final_metric(model, "clustering_coefficient") - 0.6)


def mobility_objective(model: SegregationModel) -> float:
    return abs(_final_metric(model, "mobility_rate") - 0.05)


def satisfaction_objective(model: SegregationModel) -> float:
    return abs(_final_metric(model, "satisfaction_mean") - 0.7)


def run_multi_objective_optimization(n_trials: int = 40, seed: int = 42):
    """Run multi-objective SMAC (requires smac + ConfigSpace)."""
    print("Starting Multi-Objective SMAC Calibration with AMBER")
    print("=" * 55)

    param_space = create_advanced_parameter_space()
    objectives = {
        "segregation": segregation_objective,
        "clustering": clustering_objective,
        "mobility": mobility_objective,
        "satisfaction": satisfaction_objective,
    }
    optimizer = am.MultiObjectiveSMAC(
        model_type=SegregationModel,
        param_space=param_space,
        objectives=objectives,
        n_trials=n_trials,
        seed=seed,
        strategy="pareto",
    )
    print("Starting multi-objective optimization...")
    results = optimizer.optimize()

    print(f"\nMulti-Objective Optimization Results:")
    print("=" * 45)
    print(f"Total trials: {results['n_evaluations']}")
    print(f"Pareto front size: {len(results['pareto_front'])}")
    return optimizer, results


def analyze_pareto_frontier(optimizer, results):
    """Analyze and visualize the Pareto frontier."""
    # Plotting is optional and should not prevent importing or smoke-testing
    # the model when Matplotlib is absent (or installed in another ABI env).
    import matplotlib.pyplot as plt

    print("\nAnalyzing Pareto Frontier...")
    pareto_front = results["pareto_front"]
    history = results["history"]
    objective_names = ["segregation", "clustering", "mobility", "satisfaction"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    pairs = [
        ("segregation", "clustering"),
        ("segregation", "mobility"),
        ("clustering", "satisfaction"),
        ("mobility", "satisfaction"),
    ]
    for ax, (obj1, obj2) in zip(axes.ravel(), pairs):
        ax.scatter(history[obj1].to_list(), history[obj2].to_list(), alpha=0.35, s=20)
        ax.scatter(
            pareto_front[obj1].to_list(),
            pareto_front[obj2].to_list(),
            c="red",
            s=60,
            alpha=0.85,
            label="Pareto",
        )
        ax.set_xlabel(obj1)
        ax.set_ylabel(obj2)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("amber_multi_objective_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved amber_multi_objective_analysis.png")

    ideal = np.min(pareto_front[objective_names].to_numpy(), axis=0)
    distances = np.sum((pareto_front[objective_names].to_numpy() - ideal) ** 2, axis=1)
    best_idx = int(np.argmin(distances))
    best_solution = pareto_front.row(best_idx, named=True)
    print("Best compromise objectives:")
    for obj in objective_names:
        print(f"  {obj}: {best_solution[obj]:.4f}")
    return best_solution


if __name__ == "__main__":
    # Smoke: model alone (no SMAC) so the example is importable/runnable without SMAC.
    smoke = SegregationModel(
        {
            "grid_size": 12,
            "density": 0.7,
            "agent_type_distribution": "binary",
            "type_A_fraction": 0.5,
            "base_tolerance": 0.35,
            "base_mobility": 0.15,
            "neighborhood_radius": 1,
            "search_radius": 3,
            "max_location_evaluations": 8,
            "steps": 5,
            "seed": 0,
            "show_progress": False,
        }
    )
    smoke_res = smoke.run()
    print(
        "Smoke run OK:",
        smoke_res["info"],
        "cols=",
        smoke_res["agents"].columns,
        "metrics=",
        smoke_res["model"].columns,
    )

    try:
        import ConfigSpace  # noqa: F401
        import smac  # noqa: F401
    except ImportError:
        print("smac/ConfigSpace not installed — skipping multi-objective section")
        print("Install with: pip install smac ConfigSpace")
        raise SystemExit(0)

    optimizer, results = run_multi_objective_optimization(n_trials=20, seed=42)
    analyze_pareto_frontier(optimizer, results)
    print("\nAdvanced AMBER SMAC calibration example completed.")
