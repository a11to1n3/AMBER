"""
Vectorized AMBER Models for Benchmarking

These models use the performance utilities (KD-Tree, vectorized ops)
to demonstrate the maximum speed achievable with AMBER.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import ambr as am
import numpy as np

# Import performance utilities
from ambr.performance import (
    SpatialIndex, 
    vectorized_move, 
    vectorized_wealth_transfer,
    vectorized_random_velocities,
    HAS_SCIPY,
    HAS_NUMBA,
)

if HAS_NUMBA:
    from ambr.performance import fast_neighbors_within_radius


# =============================================================================
# Vectorized Wealth Transfer Model
# =============================================================================

class VectorizedWealthTransfer(am.Model):
    """
    Vectorized Wealth Transfer using numpy operations.
    """
    
    def setup(self):
        n = self.p.get('n', 100)
        initial_wealth = self.p.get('initial_wealth', 1)
        
        # Use numpy random generator for vectorized operations
        seed = self.p.get('seed', None)
        self.rng = np.random.default_rng(seed)
        
        # Store state as numpy arrays for vectorized operations
        self.n_agents = n
        self.wealths = np.full(n, initial_wealth, dtype=np.float64)
        self._record_state()
    
    def _record_state(self):
        total_wealth = self.wealths.sum()
        gini = self._calculate_gini()
        self.record_model('total_wealth', total_wealth)
        self.record_model('gini', gini)
    
    def _calculate_gini(self):
        sorted_wealths = np.sort(self.wealths)
        n = len(sorted_wealths)
        if n == 0 or sorted_wealths.sum() == 0:
            return 0
        cumulative = np.sum((np.arange(1, n + 1)) * sorted_wealths)
        return (2 * cumulative) / (n * sorted_wealths.sum()) - (n + 1) / n
    
    def step(self):
        # Vectorized wealth transfer
        # 1. Find agents with wealth > 0
        can_give = self.wealths > 0
        givers = np.where(can_give)[0]
        
        if len(givers) == 0:
            return
            
        # 2. Random receivers for each giver (avoid self)
        receivers = self.rng.integers(0, self.n_agents, size=len(givers))
        
        # 3. Swap any self-transfers
        self_transfer = givers == receivers
        receivers[self_transfer] = (receivers[self_transfer] + 1) % self.n_agents
        
        # 4. Vectorized transfer
        self.wealths = vectorized_wealth_transfer(
            self.wealths,
            np.ones(len(givers)),  # Transfer 1 unit each
            givers,
            receivers
        )
    
    def update(self):
        super().update()
        self._record_state()


# =============================================================================
# Vectorized SIR Model (KD-Tree + Vectorized)
# =============================================================================

class VectorizedSIRModel(am.Model):
    """
    Vectorized SIR Model using KD-Tree for O(log n) neighbor queries.
    """
    
    STATUS_S = 0
    STATUS_I = 1
    STATUS_R = 2
    
    def setup(self):
        n = self.p.get('n', 100)
        initial_infected = self.p.get('initial_infected', 5)
        world_size = self.p.get('world_size', 100)
        
        # Use numpy random generator
        seed = self.p.get('seed', None)
        self.rng = np.random.default_rng(seed)
        
        self.n_agents = n
        self.world_size = world_size
        self.infection_radius = self.p.get('infection_radius', 5.0)
        self.transmission_rate = self.p.get('transmission_rate', 0.1)
        self.recovery_time = self.p.get('recovery_time', 14)
        self.movement_speed = self.p.get('movement_speed', 2.0)
        
        # Store state as numpy arrays
        self.positions = self.rng.uniform(0, world_size, (n, 2))
        self.statuses = np.zeros(n, dtype=np.int8)
        self.statuses[:initial_infected] = self.STATUS_I
        self.infection_times = np.zeros(n, dtype=np.int32)
        
        # Spatial index for fast neighbor queries
        if HAS_SCIPY:
            self.spatial_index = SpatialIndex()
        else:
            self.spatial_index = None
            
        self._record_state()
    
    def _record_state(self):
        s = np.sum(self.statuses == self.STATUS_S)
        i = np.sum(self.statuses == self.STATUS_I)
        r = np.sum(self.statuses == self.STATUS_R)
        self.record_model('susceptible', int(s))
        self.record_model('infected', int(i))
        self.record_model('recovered', int(r))
    
    def step(self):
        # 1. Movement (vectorized)
        velocities = vectorized_random_velocities(
            self.n_agents, 
            self.movement_speed,
            self.rng
        )
        self.positions = vectorized_move(
            self.positions,
            velocities,
            bounds=(0, self.world_size),
            wrap=False
        )
        
        # 2. Rebuild spatial index
        if self.spatial_index is not None:
            self.spatial_index.build(self.positions)
        
        # 3. Infection (using KD-Tree)
        infected_indices = np.where(self.statuses == self.STATUS_I)[0]
        
        for inf_idx in infected_indices:
            if self.spatial_index is not None:
                # O(log n) neighbor query
                neighbors = self.spatial_index.query_radius(
                    self.positions[inf_idx], 
                    self.infection_radius
                )
            else:
                # Fallback to O(n) search
                dists = np.linalg.norm(
                    self.positions - self.positions[inf_idx], 
                    axis=1
                )
                neighbors = np.where(dists <= self.infection_radius)[0]
            
            for neighbor_idx in neighbors:
                if neighbor_idx == inf_idx:
                    continue
                if self.statuses[neighbor_idx] == self.STATUS_S:
                    if self.rng.random() < self.transmission_rate:
                        self.statuses[neighbor_idx] = self.STATUS_I
                        self.infection_times[neighbor_idx] = 0
        
        # 4. Recovery
        self.infection_times[self.statuses == self.STATUS_I] += 1
        recovered = (self.statuses == self.STATUS_I) & \
                   (self.infection_times >= self.recovery_time)
        self.statuses[recovered] = self.STATUS_R
    
    def update(self):
        super().update()
        self._record_state()


# =============================================================================
# Vectorized Random Walk (Fully Vectorized)
# =============================================================================

class VectorizedRandomWalk(am.Model):
    """
    Vectorized Random Walk using fully vectorized numpy operations.
    """
    
    def setup(self):
        n = self.p.get('n', 100)
        world_size = self.p.get('world_size', 100)
        self.speed = self.p.get('speed', 1.0)
        
        # Use numpy random generator
        seed = self.p.get('seed', None)
        self.rng = np.random.default_rng(seed)
        
        self.n_agents = n
        self.world_size = world_size
        self.positions = self.rng.uniform(0, world_size, (n, 2))
        self._record_state()
    
    def _record_state(self):
        avg_x = self.positions[:, 0].mean()
        avg_y = self.positions[:, 1].mean()
        self.record_model('avg_x', float(avg_x))
        self.record_model('avg_y', float(avg_y))
    
    def step(self):
        # Fully vectorized movement
        velocities = vectorized_random_velocities(
            self.n_agents,
            self.speed,
            self.rng
        )
        self.positions = vectorized_move(
            self.positions,
            velocities,
            bounds=(0, self.world_size),
            wrap=False
        )
    
    def update(self):
        super().update()
        self._record_state()


# Model registry
VECTORIZED_MODELS = {
    'wealth_transfer': VectorizedWealthTransfer,
    'sir_epidemic': VectorizedSIRModel,
    'random_walk': VectorizedRandomWalk,
}


if __name__ == '__main__':
    import time
    
    print("Testing Vectorized Models")
    print("=" * 50)
    
    # Test Wealth Transfer
    print("\n1. Vectorized Wealth Transfer")
    model = VectorizedWealthTransfer({'n': 1000, 'steps': 100, 'initial_wealth': 1})
    start = time.perf_counter()
    model.run()
    elapsed = time.perf_counter() - start
    print(f"   1000 agents, 100 steps: {elapsed:.3f}s")
    
    # Test SIR
    print("\n2. Vectorized SIR Epidemic")
    model = VectorizedSIRModel({
        'n': 1000, 'steps': 100, 'initial_infected': 10,
        'world_size': 100, 'infection_radius': 5.0
    })
    start = time.perf_counter()
    model.run()
    elapsed = time.perf_counter() - start
    print(f"   1000 agents, 100 steps: {elapsed:.3f}s")
    print(f"   Using KD-Tree: {HAS_SCIPY}")
    
    # Test Random Walk
    print("\n3. Vectorized Random Walk")
    model = VectorizedRandomWalk({'n': 10000, 'steps': 100, 'speed': 1.0})
    start = time.perf_counter()
    model.run()
    elapsed = time.perf_counter() - start
    print(f"   10000 agents, 100 steps: {elapsed:.3f}s")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
