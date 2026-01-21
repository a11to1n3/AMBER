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
import random


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
        self.x = random.uniform(0, world_size)
        self.y = random.uniform(0, world_size)
        
        # Initial infections
        if self.id < self.model.p.get('initial_infected', 5):
            self.status = self.STATUS_I
    
    def move(self):
        """Random walk movement."""
        speed = self.model.p.get('movement_speed', 2.0)
        world_size = self.model.p.get('world_size', 100)
        
        self.x += random.uniform(-speed, speed)
        self.y += random.uniform(-speed, speed)
        
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
                if random.random() < transmission:
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
        self.x = random.uniform(0, world_size)
        self.y = random.uniform(0, world_size)
    
    def step(self):
        speed = self.model.p.get('speed', 1.0)
        world_size = self.model.p.get('world_size', 100)
        
        self.x += random.uniform(-speed, speed)
        self.y += random.uniform(-speed, speed)
        
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


# Model registry for benchmark runner
AMBER_MODELS = {
    'wealth_transfer': AMBERWealthTransfer,
    'sir_epidemic': AMBERSIRModel,
    'random_walk': AMBERRandomWalk,
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
