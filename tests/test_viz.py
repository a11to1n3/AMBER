"""Tests for ambr.viz plotting helpers."""

from __future__ import annotations

import ambr as am
import pytest


class TinyModel(am.Model):
    model_reporters = {"total": lambda m: int(m.agents.wealth.sum())}

    def setup(self):
        n = 20
        self.add_agents(
            n,
            wealth=1,
            x=self.rng.integers(0, 5, size=n),
            y=self.rng.integers(0, 5, size=n),
        )

    def step_vectorized(self):
        donors = self.agents.where(self.agents.wealth > 0)
        donors.wealth -= 1
        rec = self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
        self.agents.at[rec].scatter_add(wealth=1)


@pytest.mark.unit
def test_plot_helpers_require_matplotlib_or_run():
    """When matplotlib works, plot; when broken/missing, raise ImportError cleanly."""
    r = TinyModel({"steps": 3, "seed": 0, "show_progress": False}).run()
    if not am.HAS_MATPLOTLIB:
        with pytest.raises(ImportError, match="matplotlib"):
            am.plot_timeseries(r, columns=["total"])
        with pytest.raises(ImportError, match="matplotlib"):
            am.plot_grid(r)
        return

    # Prefer MPLBACKEND=Agg from the environment (CI); do not force use() here
    # beyond what the process already has.
    ax = am.plot_timeseries(r, columns=["total"], title="wealth")
    assert ax is not None
    assert ax.get_title() == "wealth"
    ax2 = am.plot_grid(r, color="wealth", title="grid")
    assert ax2 is not None
    assert ax2.get_title() == "grid"
    with pytest.raises(KeyError):
        am.plot_grid(r.agents, x="nope", y="y")
