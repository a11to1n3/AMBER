"""Fused wealth-transfer variants for throughput experiments only.

The canonical AMBER idiom (``where`` → ``donors.wealth -= 1`` →
``at[recipients].scatter_add(wealth=1)``) lives in
:class:`~models.amber_models.AMBERVectorizedWealthTransfer` and the docs.
These classes fuse donor debits and recipient credits into one ``scatter_add``
for benchmark sweeps where Polars/view overhead dominates.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import ambr as am
import numpy as np


class AMBERFusedWealthTransfer(am.Model):
    """Single fused scatter_add wealth transfer (not the documented view idiom)."""

    def setup(self):
        n = self.p.get("n", 100)
        self.add_agents(
            n,
            wealth=np.full(n, self.p.get("initial_wealth", 1), dtype=np.int64),
        )

    def step(self):
        xp = self.xp
        ids, wealth = self.agents.array("id", "wealth")
        donor_mask = wealth > 0
        n_active = int(xp.asarray(donor_mask).sum())
        if n_active == 0:
            return
        donor_ids = ids[donor_mask]
        recipient_ids = self.rng.choice(ids, size=n_active)
        all_ids = xp.concatenate([donor_ids, recipient_ids])
        deltas = xp.concatenate([
            xp.full(n_active, -1, dtype=xp.int64),
            xp.full(n_active, 1, dtype=xp.int64),
        ])
        self.agents.at[all_ids].scatter_add(wealth=deltas)


AMBER_FUSED_MODELS = {
    "wealth_transfer": AMBERFusedWealthTransfer,
}