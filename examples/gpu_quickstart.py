#!/usr/bin/env python3
"""Minimal GPU/CPU array model — the easy on-ramp for lane 4.

Run::

    python examples/gpu_quickstart.py

Uses CuPy when available, otherwise NumPy (same code).
"""

import ambr as am


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

    res = Drift({"n": 100_000, "steps": 20, "seed": 0, "dx": 0.01}).run()
    print("info:", res.info)
    print(res.model.tail(3))
