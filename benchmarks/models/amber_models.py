"""
AMBER Model Implementations for Benchmarking

These models are designed for performance comparison against AgentPy and Mesa.
Each model implements the same logic to ensure fair comparison.
"""

import sys
import os

# Add parent directory to path for amber import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import ambr as am
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _schelling_core import schelling_setup, schelling_step


# =============================================================================
# Wealth Transfer Model
# =============================================================================

class WealthAgent(am.Agent):
    """Agent that can transfer wealth to other agents."""

    def setup(self):
        self.wealth = self.model.p.get('initial_wealth', 1)

    def step(self):
        if self.wealth > 0:
            # Give 1 unit to a random other agent
            other = self.model.random.choice(self.model.agent_objects_list)
            if other.id != self.id:
                self.wealth -= 1
                other.wealth += 1


class AMBERWealthTransfer(am.Model):
    """
    Boltzmann Wealth Distribution Model (AMBER Implementation).

    Agents randomly transfer wealth to each other, leading to
    an exponential wealth distribution over time.
    """

    def setup(self):
        n = self.p.get('n', 100)
        self.agent_objects = {}
        self.agent_objects_list = []

        for i in range(n):
            agent = WealthAgent(self, i)
            agent.setup()
            self.agent_objects[i] = agent
            self.agent_objects_list.append(agent)

        self._record_state()

    def _record_state(self):
        """Record current wealth distribution."""
        total_wealth = sum(a.wealth for a in self.agent_objects_list)
        self.record_model('total_wealth', total_wealth)
        self.record_model('gini', self._calculate_gini())

    def _calculate_gini(self):
        """Calculate Gini coefficient of wealth distribution."""
        wealths = sorted([a.wealth for a in self.agent_objects_list])
        n = len(wealths)
        if n == 0 or sum(wealths) == 0:
            return 0
        cumulative = sum((i + 1) * w for i, w in enumerate(wealths))
        return (2 * cumulative) / (n * sum(wealths)) - (n + 1) / n

    def step(self):
        # Shuffle and step all agents
        agents = list(self.agent_objects_list)
        self.random.shuffle(agents)
        for agent in agents:
            agent.step()

    def update(self):
        super().update()
        self._record_state()


# =============================================================================
# SIR Epidemic Model
# =============================================================================

class SIRAgent(am.Agent):
    """Agent with SIR (Susceptible-Infected-Recovered) health states."""

    STATUS_S = 0  # Susceptible
    STATUS_I = 1  # Infected
    STATUS_R = 2  # Recovered

    def setup(self):
        self.status = self.STATUS_S
        self.infection_time = 0
        world_size = self.model.p.get('world_size', 100)
        self.x = self.model.random.uniform(0, world_size)
        self.y = self.model.random.uniform(0, world_size)

        # Initial infections
        if self.id < self.model.p.get('initial_infected', 5):
            self.status = self.STATUS_I

    def move(self):
        """Random walk movement."""
        speed = self.model.p.get('movement_speed', 2.0)
        world_size = self.model.p.get('world_size', 100)

        self.x += self.model.random.uniform(-speed, speed)
        self.y += self.model.random.uniform(-speed, speed)

        # Boundary wrap
        self.x = max(0, min(world_size, self.x))
        self.y = max(0, min(world_size, self.y))

    def infect_neighbors(self):
        """Try to infect nearby susceptible agents."""
        if self.status != self.STATUS_I:
            return

        radius = self.model.p.get('infection_radius', 5.0)
        transmission = self.model.p.get('transmission_rate', 0.1)

        for other in self.model.agent_objects_list:
            if other.id == self.id or other.status != self.STATUS_S:
                continue

            dist_sq = (self.x - other.x)**2 + (self.y - other.y)**2
            if dist_sq <= radius**2:
                if self.model.random.random() < transmission:
                    other.status = self.STATUS_I
                    other.infection_time = 0

    def update_health(self):
        """Update health status based on infection duration."""
        if self.status == self.STATUS_I:
            self.infection_time += 1
            if self.infection_time >= self.model.p.get('recovery_time', 14):
                self.status = self.STATUS_R


class AMBERSIRModel(am.Model):
    """
    SIR Epidemic Model (AMBER Implementation).

    Spatial disease spread with Susceptible-Infected-Recovered dynamics.
    """

    def setup(self):
        n = self.p.get('n', 100)
        self.agent_objects = {}
        self.agent_objects_list = []

        for i in range(n):
            agent = SIRAgent(self, i)
            agent.setup()
            self.agent_objects[i] = agent
            self.agent_objects_list.append(agent)

        self._record_state()

    def _record_state(self):
        """Record SIR counts."""
        s = sum(1 for a in self.agent_objects_list if a.status == SIRAgent.STATUS_S)
        i = sum(1 for a in self.agent_objects_list if a.status == SIRAgent.STATUS_I)
        r = sum(1 for a in self.agent_objects_list if a.status == SIRAgent.STATUS_R)
        self.record_model('susceptible', s)
        self.record_model('infected', i)
        self.record_model('recovered', r)

    def step(self):
        for agent in self.agent_objects_list:
            agent.move()
        for agent in self.agent_objects_list:
            agent.infect_neighbors()
        for agent in self.agent_objects_list:
            agent.update_health()

    def update(self):
        super().update()
        self._record_state()


# =============================================================================
# Random Walk Model
# =============================================================================

class WalkAgent(am.Agent):
    """Agent that performs random walk in 2D space."""

    def setup(self):
        world_size = self.model.p.get('world_size', 100)
        self.x = self.model.random.uniform(0, world_size)
        self.y = self.model.random.uniform(0, world_size)

    def step(self):
        speed = self.model.p.get('speed', 1.0)
        world_size = self.model.p.get('world_size', 100)

        self.x += self.model.random.uniform(-speed, speed)
        self.y += self.model.random.uniform(-speed, speed)

        self.x = max(0, min(world_size, self.x))
        self.y = max(0, min(world_size, self.y))


class AMBERRandomWalk(am.Model):
    """
    Random Walk Model (AMBER Implementation).

    Basic 2D random walk for benchmarking high-frequency updates.
    """

    def setup(self):
        n = self.p.get('n', 100)
        self.agent_objects = {}
        self.agent_objects_list = []

        for i in range(n):
            agent = WalkAgent(self, i)
            agent.setup()
            self.agent_objects[i] = agent
            self.agent_objects_list.append(agent)

        self._record_state()

    def _record_state(self):
        """Record average position."""
        avg_x = sum(a.x for a in self.agent_objects_list) / len(self.agent_objects_list)
        avg_y = sum(a.y for a in self.agent_objects_list) / len(self.agent_objects_list)
        self.record_model('avg_x', avg_x)
        self.record_model('avg_y', avg_y)

    def step(self):
        for agent in self.agent_objects_list:
            agent.step()

    def update(self):
        super().update()
        self._record_state()


# =============================================================================
# Vectorized variants — use AMBER's view API (the idiom the docs now teach).
#
# The three classes above (AMBERWealthTransfer, AMBERSIRModel, AMBERRandomWalk)
# hand-roll ``self.agent_objects_list`` and loop over it in pure Python so the
# cross-framework comparison stays apples-to-apples at the OOP abstraction
# level. The classes below implement the same models using the vectorized
# ``model.agents.where(...) / .at[ids] / .scatter_add(...)`` surface that
# AMBER actually ships — this is where the columnar backend pays off.
# =============================================================================

class AMBERVectorizedWealthTransfer(am.Model):
    """Wealth transfer using the view API with fused donor/recipient scatter."""

    def setup(self):
        n = self.p.get('n', 100)
        self.add_agents(
            n,
            wealth=np.full(n, self.p.get('initial_wealth', 1), dtype=np.int64),
        )

    def step(self):
        ids = self.agents.ids.to_numpy()
        wealth = self.agents.wealth.to_numpy()
        donor_mask = wealth > 0
        n_active = int(donor_mask.sum())
        if n_active == 0:
            return
        donor_ids = ids[donor_mask]
        recipient_ids = self.rng.choice(ids, size=n_active)
        all_ids = np.concatenate([donor_ids, recipient_ids])
        deltas = np.concatenate([np.full(n_active, -1, dtype=np.int64),
                                 np.full(n_active, 1, dtype=np.int64)])
        self.agents.at[all_ids].scatter_add(wealth=deltas)

    def update(self):
        super().update()
        # Aggregate metrics — one Polars expression each.
        wealth = self.agents.wealth
        self.record_model('total_wealth', int(wealth.sum()))
        self.record_model('gini', self._gini(wealth.to_numpy()))

    @staticmethod
    def _gini(values):
        if values.size == 0 or values.sum() == 0:
            return 0.0
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        weighted_sum = np.sum(np.arange(1, n + 1) * sorted_vals)
        return (2 * weighted_sum) / (n * sorted_vals.sum()) - (n + 1) / n


class AMBERVectorizedSIRModel(am.Model):
    """SIR epidemic on a continuous 2D world using columnar updates.

    All three phases of the step — movement, infection, recovery — are
    expressed as DataFrame operations. The infection phase is quadratic in
    population (all-pairs distance test) but runs as a single Polars join
    rather than a Python double loop.
    """

    STATUS_S = 0
    STATUS_I = 1
    STATUS_R = 2

    def setup(self):
        n = self.p.get('n', 100)
        world_size = self.p.get('world_size', 100)
        initial_infected = self.p.get('initial_infected', 5)

        status = np.full(n, self.STATUS_S, dtype=np.int64)
        status[:initial_infected] = self.STATUS_I

        self.add_agents(
            n,
            status=status,
            infection_time=np.zeros(n, dtype=np.int64),
            x=self.rng.random(size=n) * world_size,
            y=self.rng.random(size=n) * world_size,
        )

    def step(self):
        n = len(self.agents)
        speed = self.p.get('movement_speed', 2.0)
        world_size = self.p.get('world_size', 100)

        # --- movement ---
        xs = self.agents.x.to_numpy() + self.rng.uniform(-speed, speed, n)
        ys = self.agents.y.to_numpy() + self.rng.uniform(-speed, speed, n)
        np.clip(xs, 0, world_size, out=xs)
        np.clip(ys, 0, world_size, out=ys)
        self.agents.x = xs
        self.agents.y = ys

        # --- infection ---
        radius = self.p.get('infection_radius', 5.0)
        transmission = self.p.get('transmission_rate', 0.1)
        df = self.agents_df

        infected_df = df.filter(pl.col('status') == self.STATUS_I).select(
            pl.col('x').alias('ix'), pl.col('y').alias('iy'),
            pl.col('id').alias('iid'),
        )
        susceptible_df = df.filter(pl.col('status') == self.STATUS_S)

        if infected_df.height and susceptible_df.height:
            # Cross join susceptibles × infected, keep pairs within radius,
            # then sample one uniform per candidate pair for transmission.
            pairs = susceptible_df.join(infected_df, how='cross').with_columns(
                ((pl.col('x') - pl.col('ix')) ** 2
                 + (pl.col('y') - pl.col('iy')) ** 2).alias('dist_sq')
            ).filter(pl.col('dist_sq') <= radius ** 2)
            if pairs.height:
                draws = self.rng.random(size=pairs.height)
                hits = pairs.with_columns(pl.Series('draw', draws)).filter(
                    pl.col('draw') < transmission
                )
                if hits.height:
                    newly_infected_ids = hits['id'].unique().to_list()
                    self.agents.at[newly_infected_ids].status = self.STATUS_I
                    self.agents.at[newly_infected_ids].infection_time = 0

        # --- recovery ---
        recovery_time = self.p.get('recovery_time', 14)
        active = self.agents.where(pl.col('status') == self.STATUS_I)
        active.infection_time = active.infection_time + 1
        recovered = self.agents.where(
            (pl.col('status') == self.STATUS_I)
            & (pl.col('infection_time') >= recovery_time)
        )
        if len(recovered) > 0:
            recovered.status = self.STATUS_R

    def update(self):
        super().update()
        status = self.agents.status.to_numpy()
        self.record_model('susceptible', int((status == self.STATUS_S).sum()))
        self.record_model('infected', int((status == self.STATUS_I).sum()))
        self.record_model('recovered', int((status == self.STATUS_R).sum()))


class AMBERVectorizedRandomWalk(am.Model):
    """Random walk expressed as two numpy updates and one columnar write."""

    def setup(self):
        n = self.p.get('n', 100)
        world_size = self.p.get('world_size', 100)
        self.add_agents(
            n,
            x=self.rng.random(size=n) * world_size,
            y=self.rng.random(size=n) * world_size,
        )

    def step(self):
        n = len(self.agents)
        speed = self.p.get('speed', 1.0)
        world_size = self.p.get('world_size', 100)
        xs = self.agents.x.to_numpy() + self.rng.uniform(-speed, speed, n)
        ys = self.agents.y.to_numpy() + self.rng.uniform(-speed, speed, n)
        np.clip(xs, 0, world_size, out=xs)
        np.clip(ys, 0, world_size, out=ys)
        self.agents.x = xs
        self.agents.y = ys

    def update(self):
        super().update()
        self.record_model('avg_x', float(self.agents.x.mean()))
        self.record_model('avg_y', float(self.agents.y.mean()))


# =============================================================================
# Schelling Segregation
# =============================================================================

class _SchellingAgent(am.Agent):
    def setup(self):
        pass


class AMBERSchelling(am.Model):
    """Agent-based Schelling segregation (per-agent loop over a toroidal grid)."""

    def setup(self):
        n = self.p.get('n', 100)
        x, y, t, self.G = schelling_setup(
            n, self.p.get('density', 0.8), self.p.get('fraction_a', 0.5), self.rng, np)
        self.tolerance = float(self.p.get('tolerance', 0.3))
        self.agent_objects = {}
        self.agent_objects_list = []
        self.occ = {}
        for i in range(n):
            a = _SchellingAgent(self, i)
            a.x, a.y, a.type = int(x[i]), int(y[i]), int(t[i])
            self.agent_objects[i] = a
            self.agent_objects_list.append(a)
            self.occ[(a.x, a.y)] = a.type

    def step(self):
        G, occ = self.G, self.occ
        agents = list(self.agent_objects_list)
        self.random.shuffle(agents)
        for a in agents:
            same = total = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    v = occ.get(((a.x + dx) % G, (a.y + dy) % G))
                    if v is not None:
                        total += 1
                        if v == a.type:
                            same += 1
            if total > 0 and same < self.tolerance * total:   # unhappy -> relocate
                for _ in range(20):
                    cx, cy = self.random.randrange(G), self.random.randrange(G)
                    if (cx, cy) not in occ:
                        del occ[(a.x, a.y)]
                        a.x, a.y = cx, cy
                        occ[(cx, cy)] = a.type
                        break


class AMBERVectorizedSchelling(am.Model):
    """Schelling segregation via the grid-vectorized core (one columnar update)."""

    def setup(self):
        n = self.p.get('n', 100)
        x, y, t, self.G = schelling_setup(
            n, self.p.get('density', 0.8), self.p.get('fraction_a', 0.5), self.rng, np)
        self.tolerance = float(self.p.get('tolerance', 0.3))
        self._types = t
        self.add_agents(n, x=x.astype(np.int32), y=y.astype(np.int32))

    def step(self):
        x = self.agents.x.to_numpy().astype(np.int32)
        y = self.agents.y.to_numpy().astype(np.int32)
        nx, ny = schelling_step(x, y, self._types, self.G, self.tolerance, self.rng, np)
        self.agents.x = nx
        self.agents.y = ny


# Model registry for benchmark runner
AMBER_MODELS = {
    'wealth_transfer': AMBERWealthTransfer,
    'sir_epidemic': AMBERSIRModel,
    'random_walk': AMBERRandomWalk,
    'schelling': AMBERSchelling,
}

# Separate registry for the vectorized variants — loaded by runner.py under
# the "AMBER (vectorized)" framework label so we can chart both alongside
# Mesa / AgentPy.
AMBER_VECTORIZED_MODELS = {
    'wealth_transfer': AMBERVectorizedWealthTransfer,
    'sir_epidemic': AMBERVectorizedSIRModel,
    'random_walk': AMBERVectorizedRandomWalk,
    'schelling': AMBERVectorizedSchelling,
}

if __name__ == '__main__':
    # Quick test
    model = AMBERWealthTransfer({'n': 100, 'steps': 10, 'initial_wealth': 1})
    results = model.run()
    print(f"Wealth Transfer - Final Gini: {results['model_data']['gini'][-1]:.3f}")

    model = AMBERSIRModel({'n': 100, 'steps': 10, 'initial_infected': 5})
    results = model.run()
    print(f"SIR - Final Infected: {results['model_data']['infected'][-1]}")

    model = AMBERRandomWalk({'n': 100, 'steps': 10, 'speed': 1.0})
    results = model.run()
    print(f"Random Walk - Final Avg X: {results['model_data']['avg_x'][-1]:.2f}")
