"""GridEnvironment occupancy helpers used by the Schelling / SMAC advanced example."""

import ambr as am
from ambr.environments import GridEnvironment


def _model_with_grid(size=5):
    class M(am.Model):
        def setup(self):
            pass

    m = M({"show_progress": False, "seed": 0})
    env = GridEnvironment(m, size=size)
    return m, env


def test_place_and_query_agents_on_grid():
    m, env = _model_with_grid(4)
    m.add_agents(2, agent_type=["A", "B"])
    env.add_agent_from_id(0, (1, 1))
    env.add_agent_from_id(1, (2, 2))

    assert env.get_agent_at_pos((1, 1)) == 0
    assert env.get_agent_at_pos((0, 0)) is None
    assert (1, 1) not in env.empty_positions()

    empty = env.get_random_empty_cell()
    assert empty is not None
    assert env.get_agent_at_pos(empty) is None


def test_move_via_remove_and_add():
    m, env = _model_with_grid(5)
    m.add_agents(1)
    env.add_agent_from_id(0, (0, 0))
    env.remove_agent_from_pos((0, 0))
    assert env.get_agent_at_pos((0, 0)) is None
    env.add_agent_from_id(0, (3, 3))
    assert env.get_agent_at_pos((3, 3)) == 0


def test_empty_cells_in_radius_and_neighbors_radius_alias():
    m, env = _model_with_grid(6)
    m.add_agents(1)
    env.add_agent_from_id(0, (2, 2))
    empties = env.get_empty_cells_in_radius((2, 2), radius=1)
    # Moore ring of radius 1 has 8 cells; all empty except we only placed center
    assert len(empties) == 8
    assert (2, 2) not in empties

    n = env.get_neighbors((2, 2), radius=1)
    assert len(n) == 8
    assert (2, 2) not in n


def test_schelling_vectorized_example_smoke():
    """Canonical grid Schelling example (docs/environments_schelling.rst)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "schelling_vectorized.py"
    spec = importlib.util.spec_from_file_location("schelling_vec", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    res = mod.SchellingModel(
        {
            "grid_size": 8,
            "n": 40,
            "want_similar": 0.3,
            "steps": 4,
            "seed": 2,
            "show_progress": False,
        }
    ).run()
    assert res["info"]["steps"] == 4
    assert "happy_frac" in res.model.columns
    assert "grid_position" in res.agents.columns


def test_segregation_model_smoke_runs():
    """Import path used by examples/smac_calibration_advanced.py."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "smac_calibration_advanced.py"
    spec = importlib.util.spec_from_file_location("smac_adv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.SegregationModel(
        {
            "grid_size": 8,
            "density": 0.6,
            "agent_type_distribution": "binary",
            "type_A_fraction": 0.5,
            "base_tolerance": 0.3,
            "base_mobility": 0.2,
            "neighborhood_radius": 1,
            "search_radius": 2,
            "max_location_evaluations": 5,
            "steps": 3,
            "seed": 1,
            "show_progress": False,
        }
    )
    res = model.run()
    assert res["info"]["steps"] == 3
    assert "grid_position" in res["agents"].columns
    assert "segregation_index" in res["model"].columns
    assert "satisfaction_mean" in res["model"].columns
