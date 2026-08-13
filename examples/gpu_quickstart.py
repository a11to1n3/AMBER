#!/usr/bin/env python3
"""GPU / CPU placement quickstart (AMBER 0.4.4+).

Run::

    python examples/gpu_quickstart.py

Two paths:

1. **Native** — vectorized view-API ``Model`` + ``step_vectorized`` under
   ``.gpu().run()`` (device-resident columns). Falls back to CPU if CuPy is
   unavailable.
2. **Array kernel** — :class:`ambr.ArrayKernelModel` for pure array state
   (CuPy when available, else NumPy).
"""

import ambr as am


class WealthModel(am.Model):
    """Canonical view-API wealth transfer (works under cpu() and gpu())."""

    def setup(self):
        n = int(self.p.get("n", 10_000))
        self.add_agents(n, wealth=self.rng.integers(1, 10, size=n))

    def step_vectorized(self):
        donors = self.agents.where(self.agents.wealth > 0)
        if len(donors) == 0:
            return
        donors.wealth -= 1
        ids = self.agents.ids.to_numpy()
        recipients = self.rng.choice(ids, size=len(donors))
        self.agents.at[recipients].scatter_add(wealth=1)


class Drift(am.ArrayKernelModel):
    """Every agent has a scalar x that drifts upward each step."""

    def init_state(self, xp, n, rng, p):
        return {"x": rng.random(n, dtype=xp.float32)}

    def step_state(self, xp, state, rng, p):
        state["x"] = state["x"] + float(p.get("dx", 0.01))
        return state

    def metrics(self, xp, state):
        return {"mean_x": float(am.to_host(state["x"].mean()))}


if __name__ == "__main__":
    am.print_status()
    print("recommend(1_000_000):", am.recommend(1_000_000))

    cfg = {"n": 50_000, "steps": 20, "seed": 0, "show_progress": False}

    # --- Native placement: same Model on CPU or GPU ---
    cpu_res = WealthModel(cfg).cpu(mode="vectorized").run()
    print("native CPU  info:", cpu_res.info)

    if am.GPU_AVAILABLE:
        gpu_res = WealthModel(cfg).gpu().run()
        print("native GPU  info:", gpu_res.info)
    else:
        print("native GPU  skipped (CuPy / NVIDIA not available)")

    # --- Array-kernel lane ---
    drift = Drift({"n": 100_000, "steps": 20, "seed": 0, "dx": 0.01})
    res = drift.run()
    print("ArrayKernelModel info:", res.info)
    # ASCII-only: Polars' box-drawing tail() is not encodable on Windows CP1252.
    print(res.model.tail(3).to_dicts())
