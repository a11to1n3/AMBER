"""mesa-frames benchmark models.

Implements the three benchmark models (wealth_transfer, random_walk,
sir_epidemic) on mesa-frames' Polars-backed ``AgentSetPolars``. Each uses the
*same vectorized algorithm* as AMBER's vectorized models, so the comparison
isolates the storage / dispatch backend rather than the algorithm.

mesa-frames API used (0.1.x alpha): ``ModelDF(seed=...)``,
``AgentSetPolars.add(df)`` / ``set(col, values)`` / ``.agents`` (the backing
``pl.DataFrame``); ``model.agents += set``; ``model.agents.do("step")``. The
identity column is ``unique_id``.
"""

import os
import sys

import numpy as np
import polars as pl
from mesa_frames import AgentSetPolars, ModelDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _schelling_core import schelling_setup, schelling_step

SEED = 42


# --------------------------------------------------------------------------- #
# Agent sets
# --------------------------------------------------------------------------- #

class _WealthSet(AgentSetPolars):
    def __init__(self, model, n, initial_wealth):
        super().__init__(model)
        self.add(pl.DataFrame({
            "unique_id": np.arange(n, dtype=np.int64),
            "wealth": np.full(n, initial_wealth, dtype=np.int64),
        }))

    def step(self):
        df = self.agents
        ids = df["unique_id"].to_numpy()
        wealth = df["wealth"].to_numpy()
        donor_mask = wealth > 0
        k = int(donor_mask.sum())
        if k == 0:
            return
        recipient_ids = self.model.rng.choice(ids, size=k)
        new = wealth.copy()
        pos = {int(u): i for i, u in enumerate(ids)}
        np.add.at(new, np.nonzero(donor_mask)[0], -1)
        np.add.at(
            new,
            np.fromiter((pos[int(r)] for r in recipient_ids), dtype=np.int64, count=k),
            1,
        )
        self.set("wealth", new)


class _WalkSet(AgentSetPolars):
    def __init__(self, model, n, world_size):
        super().__init__(model)
        rng = model.rng
        self.add(pl.DataFrame({
            "unique_id": np.arange(n, dtype=np.int64),
            "x": rng.random(n) * world_size,
            "y": rng.random(n) * world_size,
        }))

    def step(self):
        m = self.model
        df = self.agents
        n = df.height
        xs = df["x"].to_numpy() + m.rng.uniform(-m.speed, m.speed, n)
        ys = df["y"].to_numpy() + m.rng.uniform(-m.speed, m.speed, n)
        np.clip(xs, 0, m.world_size, out=xs)
        np.clip(ys, 0, m.world_size, out=ys)
        self.set({"x": xs, "y": ys})


class _SIRSet(AgentSetPolars):
    S, I, R = 0, 1, 2

    def __init__(self, model, n, world_size, initial_infected):
        super().__init__(model)
        rng = model.rng
        status = np.full(n, self.S, dtype=np.int64)
        status[:initial_infected] = self.I
        self.add(pl.DataFrame({
            "unique_id": np.arange(n, dtype=np.int64),
            "status": status,
            "infection_time": np.zeros(n, dtype=np.int64),
            "x": rng.random(n) * world_size,
            "y": rng.random(n) * world_size,
        }))

    def step(self):
        m = self.model
        rng = m.rng
        df = self.agents
        n = df.height

        # --- movement ---
        xs = df["x"].to_numpy() + rng.uniform(-m.movement_speed, m.movement_speed, n)
        ys = df["y"].to_numpy() + rng.uniform(-m.movement_speed, m.movement_speed, n)
        np.clip(xs, 0, m.world_size, out=xs)
        np.clip(ys, 0, m.world_size, out=ys)
        self.set({"x": xs, "y": ys})

        df = self.agents
        status = df["status"].to_numpy().copy()
        inf_time = df["infection_time"].to_numpy().copy()
        uid = df["unique_id"].to_numpy()

        # --- infection: all-pairs susceptible x infected within radius ---
        infected_df = df.filter(pl.col("status") == self.I).select(
            pl.col("x").alias("ix"), pl.col("y").alias("iy")
        )
        susceptible_df = df.filter(pl.col("status") == self.S)
        if infected_df.height and susceptible_df.height:
            pairs = susceptible_df.join(infected_df, how="cross").with_columns(
                ((pl.col("x") - pl.col("ix")) ** 2
                 + (pl.col("y") - pl.col("iy")) ** 2).alias("dist_sq")
            ).filter(pl.col("dist_sq") <= m.infection_radius ** 2)
            if pairs.height:
                draws = rng.random(pairs.height)
                hits = pairs.with_columns(pl.Series("draw", draws)).filter(
                    pl.col("draw") < m.transmission_rate
                )
                if hits.height:
                    newly = hits["unique_id"].unique().to_numpy()
                    mask = np.isin(uid, newly)
                    status[mask] = self.I
                    inf_time[mask] = 0

        # --- recovery ---
        infected_mask = status == self.I
        inf_time = np.where(infected_mask, inf_time + 1, inf_time)
        status = np.where(
            infected_mask & (inf_time >= m.recovery_time), self.R, status
        )
        self.set({"status": status, "infection_time": inf_time})


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class _BaseModel(ModelDF):
    def __init__(self, n, steps, cfg):
        super().__init__(seed=SEED)
        self.rng = np.random.default_rng(SEED)
        self.n = n
        self.steps = steps
        self.cfg = cfg
        self._build()

    def run(self):
        for _ in range(self.steps):
            self.agents.do("step")


class WealthModel(_BaseModel):
    def _build(self):
        self.agents += _WealthSet(self, self.n, self.cfg.get("initial_wealth", 1))


class WalkModel(_BaseModel):
    def _build(self):
        self.world_size = self.cfg.get("world_size", 100)
        self.speed = self.cfg.get("speed", 1.0)
        self.agents += _WalkSet(self, self.n, self.world_size)


class SIRModel(_BaseModel):
    def _build(self):
        self.world_size = self.cfg.get("world_size", 100)
        self.movement_speed = self.cfg.get("movement_speed", 2.0)
        self.infection_radius = self.cfg.get("infection_radius", 5.0)
        self.transmission_rate = self.cfg.get("transmission_rate", 0.1)
        self.recovery_time = self.cfg.get("recovery_time", 14)
        self.agents += _SIRSet(
            self, self.n, self.world_size, self.cfg.get("initial_infected", 5)
        )


class _SchellingSet(AgentSetPolars):
    def __init__(self, model, n, G, x, y, t, tol):
        super().__init__(model)
        self.G, self.tol, self.types = G, tol, t
        self.add(pl.DataFrame({
            "unique_id": np.arange(n, dtype=np.int64),
            "x": x.astype(np.int32),
            "y": y.astype(np.int32),
        }))

    def step(self):
        df = self.agents
        x = df["x"].to_numpy().astype(np.int32)
        y = df["y"].to_numpy().astype(np.int32)
        nx, ny = schelling_step(x, y, self.types, self.G, self.tol, self.model.rng, np)
        self.set({"x": nx, "y": ny})


class SchellingModel(_BaseModel):
    def _build(self):
        self.tolerance = float(self.cfg.get("tolerance", 0.3))
        x, y, t, G = schelling_setup(
            self.n, self.cfg.get("density", 0.8), self.cfg.get("fraction_a", 0.5), self.rng, np)
        self.agents += _SchellingSet(self, self.n, G, x, y, t, self.tolerance)


MESA_FRAMES_MODELS = {
    "wealth_transfer": WealthModel,
    "random_walk": WalkModel,
    "sir_epidemic": SIRModel,
    "schelling": SchellingModel,
}
