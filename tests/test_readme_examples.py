"""Durable smokes for the public README wealth examples.

These snippets must stay copy-paste runnable on CPU. Failures here mean the
README (or the matching quickstart vectorized path) has regressed.
"""

from __future__ import annotations

import ambr as am


def test_readme_oop_wealth():
    """README Lane A / OOP wealth transfer (AgentList + per-agent methods)."""

    class WealthAgent(am.Agent):
        def setup(self):
            self.wealth = 1

        def transfer(self):
            if self.wealth > 0:
                other = self.model.agents.by_id(self.model.agents.random())
                other.wealth += 1
                self.wealth -= 1

    class WealthModel(am.Model):
        def setup(self):
            self.agents = am.AgentList(self, self.p.n, WealthAgent)

        def step(self):
            self.agents.transfer()

        def update(self):
            self.record_model("total", int(self.agents.wealth.sum()))

    results = WealthModel({"n": 50, "steps": 20, "seed": 1, "show_progress": False}).run()
    model_df = results.model
    assert "total" in model_df.columns
    assert model_df.height == 20
    # Wealth is conserved: every agent starts with 1.
    assert int(model_df["total"][-1]) == 50
    assert results.agents.height == 50


def test_readme_vectorized_wealth_view_api():
    """README vectorized wealth using the view API (where / assign / scatter_add).

    ``agents.array(...)`` is read-only on the CPU Polars path; the public sample
    must use the view API so ``.cpu(mode="vectorized").run()`` succeeds.
    """

    class WealthModel(am.Model):
        model_reporters = {"total_wealth": lambda m: int(m.agents.wealth.sum())}

        def setup(self):
            self.add_agents(100, wealth=self.rng.integers(1, 10, size=100))

        def step_vectorized(self):
            donors = self.agents.where(self.agents.wealth > 0)
            donors.wealth -= 1
            recipients = self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
            self.agents.at[recipients].scatter_add(wealth=1)

    results = (
        WealthModel({"steps": 100, "seed": 42, "show_progress": False})
        .cpu(mode="vectorized")
        .run()
    )
    model_df = results.model
    assert "total_wealth" in model_df.columns
    assert model_df.height == 100
    # Conservation: total wealth is constant across steps.
    totals = model_df["total_wealth"].to_list()
    assert len(set(totals)) == 1
    assert results.agents.height == 100
    assert "wealth" in results.agents.columns


def test_quickstart_analytical_record_model_in_update():
    """record_model must be called from update() (or model_reporters), not only step()."""
    import numpy as np

    class AnalyticalWealthModel(am.Model):
        def setup(self):
            self.add_agents(100, wealth=self.rng.integers(1, 10, size=100))

        def step(self):
            donors = self.agents.where(self.agents.wealth > 0)
            donors.wealth -= 1
            ids = self.agents.ids.to_numpy()
            recipients = self.rng.choice(ids, size=len(donors))
            self.agents.at[recipients].scatter_add(wealth=1)

        def update(self):
            wealth = self.agents.wealth
            self.record_model("mean_wealth", float(wealth.mean()))
            self.record_model("wealth_std", float(wealth.std() or 0.0))
            self.record_model("gini", self._gini(wealth.to_numpy()))

        @staticmethod
        def _gini(values):
            if values.size == 0 or values.sum() == 0:
                return 0.0
            sorted_vals = np.sort(values)
            n = len(sorted_vals)
            cum = np.cumsum(sorted_vals)
            return (n + 1 - 2 * cum.sum() / cum[-1]) / n

    results = AnalyticalWealthModel(
        {"steps": 10, "seed": 42, "show_progress": False}
    ).run()
    cols = set(results.model.columns)
    assert {"mean_wealth", "wealth_std", "gini", "t"} <= cols
    assert results.model.height == 10
