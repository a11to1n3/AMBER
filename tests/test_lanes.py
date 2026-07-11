"""Progressive speed-lane helpers and ArrayKernelModel."""

import numpy as np
import ambr as am
from ambr.lanes import ArrayKernelModel, status, recommend, print_status


def test_status_and_recommend():
    s = status()
    assert "gpu_available" in s
    assert "numba_available" in s
    assert "lanes" in s
    assert "vectorized" in s["lanes"]
    assert "cpu_jit" in s["lanes"]
    tip = recommend(10_000)
    assert "vectorized" in tip.lower() or "oop" in tip.lower() or "numba" in tip.lower()
    tip_big = recommend(2_000_000)
    assert isinstance(tip_big, str) and len(tip_big) > 10
    tip_ens = recommend(1000, ensemble=True)
    assert "Ensemble" in tip_ens or "Experiment" in tip_ens or "GPU" in tip_ens or "numba" in tip_ens


def test_print_status_smoke(capsys):
    print_status()
    out = capsys.readouterr().out
    assert "GPU:" in out
    assert "Lanes:" in out


class _Drift(ArrayKernelModel):
    def init_state(self, xp, n, rng, p):
        return {"x": xp.zeros(n, dtype=xp.float32)}

    def step_state(self, xp, state, rng, p):
        state["x"] = state["x"] + 1.0
        return state

    def metrics(self, xp, state):
        return {"mean_x": float(am.to_host(state["x"].mean()))}


def test_array_kernel_model_runs_on_numpy():
    res = _Drift({"n": 100, "steps": 5, "seed": 0, "prefer_gpu": False}).run()
    assert res.info["steps"] == 5
    assert res.info["array_module"] == "numpy"
    assert res.agents.height == 100
    assert res.model.height == 5
    # after 5 steps of +1, mean is 5
    assert abs(res.model["mean_x"][-1] - 5.0) < 1e-5


def test_update_where_sugar():
    class M(am.Model):
        def setup(self):
            self.add_agents(10, wealth=np.arange(10))

        def step(self):
            self.agents.update_where(self.agents.wealth >= 5, wealth=0)

    res = M({"steps": 1, "show_progress": False}).run()
    assert res.agents["wealth"].to_list() == [0, 1, 2, 3, 4, 0, 0, 0, 0, 0]
