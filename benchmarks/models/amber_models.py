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
    """Wealth transfer using the canonical AMBER view API (quickstart idiom)."""

    def setup(self):
        n = self.p.get('n', 100)
        self.add_agents(
            n,
            wealth=np.full(n, self.p.get('initial_wealth', 1), dtype=np.int64),
        )

    def step(self):
        donors = self.agents.where(self.agents.wealth > 0)
        if len(donors) == 0:
            return
        donors.wealth -= 1
        ids = self.agents.ids.to_numpy()
        recipients = self.rng.choice(ids, size=len(donors))
        self.agents.at[recipients].scatter_add(wealth=1)

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


def _sir_infect_cell_list(
    xp,
    x,
    y,
    status,
    status_s,
    status_i,
    world_size,
    radius,
    transmission,
    rng,
    max_per_cell=64,
    mem_budget_bytes=128 * 1024 * 1024,
):
    """Fixed-radius infection via a uniform-grid cell list (O(N · K)).

    Cell side equals ``radius``, so every agent within the infection radius of
    a susceptible lives in that susceptible's 3×3 neighbour block. Agents are
    scattered into a ``(n_cells, K)`` table with ``K = min(max_occupancy,
    max_per_cell)`` so peak work/memory stays linear in N even when the
    benchmark keeps a fixed world (density grows with N). Under extreme
    overcrowding some co-cell agents are skipped — same practical trade-off as
    capped spatial messaging.

    Same high-level semantics as the old all-pairs path: synchronous snapshot;
    a susceptible becomes infected if *any* retained infected neighbour is
    within radius and passes an independent Bernoulli(transmission) draw.
    """
    n = int(x.shape[0])
    if n == 0:
        return status, None

    is_i = status == status_i
    is_s = status == status_s
    if int(xp.count_nonzero(is_i)) == 0 or int(xp.count_nonzero(is_s)) == 0:
        return status, None

    r2 = float(radius) * float(radius)
    ncell = max(1, int(float(world_size) // float(radius)))
    cs = float(world_size) / float(ncell)
    ncell2 = ncell * ncell
    K = max(1, int(max_per_cell))

    cx = xp.clip((x / cs).astype(xp.int64), 0, ncell - 1)
    cy = xp.clip((y / cs).astype(xp.int64), 0, ncell - 1)

    # Occupancy table holds *infected only* so a K-cap never drops the seeds
    # that drive the epidemic (a full-population table under high density can
    # exclude the few I agents from every cell's retained slots).
    inf_idx = xp.nonzero(is_i)[0]
    cell_i = cx[inf_idx] * ncell + cy[inf_idx]
    order_i = xp.argsort(cell_i)
    sorted_cell_i = cell_i[order_i]
    sorted_inf = inf_idx[order_i]
    counts_i = xp.bincount(cell_i, minlength=ncell2)
    if int(counts_i.max()) <= 0:
        return status, None

    starts_i = xp.zeros(ncell2, dtype=xp.int64)
    if ncell2 > 1:
        starts_i[1:] = xp.cumsum(counts_i)[:-1]
    n_inf = int(inf_idx.shape[0])
    slot_i = xp.arange(n_inf, dtype=xp.int64) - starts_i[sorted_cell_i]
    keep_i = slot_i < K
    table = xp.full((ncell2, K), -1, dtype=xp.int64)
    table[sorted_cell_i[keep_i], slot_i[keep_i]] = sorted_inf[keep_i]

    # Susceptible cell coords for the 3×3 gather.
    sus_idx = xp.nonzero(is_s)[0]
    n_sus = int(sus_idx.shape[0])
    cx_s = cx[sus_idx]
    cy_s = cy[sus_idx]
    ncols = 9 * K
    # ~32 B per (agent, candidate) for ids + coords + masks + draws.
    chunk = max(1, int(mem_budget_bytes // max(ncols * 32, 1)))
    newly_flags = xp.zeros(n, dtype=bool)

    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    for start in range(0, n_sus, chunk):
        end = min(start + chunk, n_sus)
        local = slice(start, end)
        idx = sus_idx[local]
        m = int(idx.shape[0])
        cxi = cx_s[local]
        cyi = cy_s[local]
        xi = x[idx]
        yi = y[idx]

        cand = xp.empty((m, ncols), dtype=xp.int64)
        col = 0
        for dxc, dyc in offsets:
            ncx = xp.clip(cxi + dxc, 0, ncell - 1)
            ncy = xp.clip(cyi + dyc, 0, ncell - 1)
            cand[:, col:col + K] = table[ncx * ncell + ncy]
            col += K

        valid = cand >= 0
        safe = xp.where(valid, cand, 0)
        dx = x[safe] - xi[:, None]
        dy = y[safe] - yi[:, None]
        within = ((dx * dx + dy * dy) <= r2) & valid
        draws = rng.random((m, ncols)) < transmission
        hit = (within & draws).any(axis=1)
        newly_flags[idx] = hit

    if not bool(newly_flags.any()):
        return status, None

    status = xp.asarray(status).copy()
    status[newly_flags] = status_i
    return status, newly_flags


class AMBERVectorizedSIRModel(am.Model):
    """SIR epidemic on a continuous 2D world using columnar updates.

    Movement, infection, and recovery use the view API (``agents.array`` /
    column assignment). Infection uses a **uniform-grid cell list** (O(N · k)
    fixed-radius query) on NumPy or CuPy via ``self.xp``, so the same step
    body works under ``model.cpu()`` and ``model.gpu()`` and scales past the
    all-pairs O(N²) memory wall.
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
        xp = self.xp
        n = len(self.agents)
        speed = self.p.get('movement_speed', 2.0)
        world_size = self.p.get('world_size', 100)
        radius = float(self.p.get('infection_radius', 5.0))
        transmission = float(self.p.get('transmission_rate', 0.1))
        recovery_time = int(self.p.get('recovery_time', 14))

        x, y = self.agents.array('x', 'y')
        status = self.agents.array('status')
        infection_time = self.agents.array('infection_time')

        # --- movement ---
        x = xp.clip(x + self.rng.uniform(-speed, speed, n), 0.0, world_size)
        y = xp.clip(y + self.rng.uniform(-speed, speed, n), 0.0, world_size)

        # --- infection (synchronous cell-list; scales with N) ---
        status, newly = _sir_infect_cell_list(
            xp, x, y, status,
            self.STATUS_S, self.STATUS_I,
            world_size, radius, transmission, self.rng,
            max_per_cell=int(self.p.get("max_per_cell", 64)),
        )
        if newly is not None:
            infection_time = xp.asarray(infection_time).copy()
            infection_time[newly] = 0

        # --- recovery ---
        infection_time = xp.where(
            status == self.STATUS_I, infection_time + 1, infection_time
        )
        status = xp.where(
            (status == self.STATUS_I) & (infection_time >= recovery_time),
            self.STATUS_R,
            status,
        )

        self.agents.x = x
        self.agents.y = y
        self.agents.status = status
        self.agents.infection_time = infection_time

    def update(self):
        super().update()
        status = self.agents.array('status')
        from ambr.gpu import to_host
        status = to_host(status)
        self.record_model('susceptible', int((status == self.STATUS_S).sum()))
        self.record_model('infected', int((status == self.STATUS_I).sum()))
        self.record_model('recovered', int((status == self.STATUS_R).sum()))


class AMBERVectorizedRandomWalk(am.Model):
    """Random walk via the view API; same step on ``cpu()`` and ``gpu()``."""

    def setup(self):
        n = self.p.get('n', 100)
        world_size = self.p.get('world_size', 100)
        self.add_agents(
            n,
            x=self.rng.random(size=n) * world_size,
            y=self.rng.random(size=n) * world_size,
        )

    def step(self):
        xp = self.xp
        n = len(self.agents)
        speed = self.p.get('speed', 1.0)
        world_size = self.p.get('world_size', 100)
        x, y = self.agents.array('x', 'y')
        x = xp.clip(x + self.rng.uniform(-speed, speed, n), 0.0, world_size)
        y = xp.clip(y + self.rng.uniform(-speed, speed, n), 0.0, world_size)
        self.agents.x = x
        self.agents.y = y

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
    """Schelling via the shared grid core + view API; works under cpu()/gpu()."""

    def setup(self):
        n = self.p.get('n', 100)
        x, y, t, self.G = schelling_setup(
            n, self.p.get('density', 0.8), self.p.get('fraction_a', 0.5), self.rng, np)
        self.tolerance = float(self.p.get('tolerance', 0.3))
        self._types = np.asarray(t)
        self.add_agents(n, x=np.asarray(x, dtype=np.int32), y=np.asarray(y, dtype=np.int32))

    def step(self):
        xp = self.xp
        x = xp.asarray(self.agents.array('x'), dtype=xp.int32)
        y = xp.asarray(self.agents.array('y'), dtype=xp.int32)
        t = xp.asarray(self._types)
        nx, ny = schelling_step(
            x, y, t, self.G, self.tolerance, self.rng, xp
        )
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
