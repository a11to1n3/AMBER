"""
AgentPy Model Implementations for Benchmarking

These models implement the same logic as AMBER models for fair comparison.
AgentPy uses self.p for parameter access.
"""

import agentpy as ap
import random


# =============================================================================
# Wealth Transfer Model
# =============================================================================

class APWealthAgent(ap.Agent):
    """Agent that can transfer wealth to other agents."""
    
    def setup(self):
        self.wealth = self.model.p.get('initial_wealth', 1)
    
    def step(self):
        if self.wealth > 0:
            # Give 1 unit to a random other agent
            other = self.model.agents.random()
            if other != self:
                self.wealth -= 1
                other.wealth += 1


class AgentPyWealthTransfer(ap.Model):
    """
    Boltzmann Wealth Distribution Model (AgentPy Implementation).
    """
    
    def setup(self):
        n = self.p.get('n', 100)
        self.agents = ap.AgentList(self, n, APWealthAgent)
    
    def step(self):
        self.agents.step()
    
    def update(self):
        total_wealth = sum(a.wealth for a in self.agents)
        self.record('total_wealth', total_wealth)
        self.record('gini', self._calculate_gini())
    
    def _calculate_gini(self):
        wealths = sorted([a.wealth for a in self.agents])
        n = len(wealths)
        if n == 0 or sum(wealths) == 0:
            return 0
        cumulative = sum((i + 1) * w for i, w in enumerate(wealths))
        return (2 * cumulative) / (n * sum(wealths)) - (n + 1) / n
    
    def end(self):
        pass


# =============================================================================
# SIR Epidemic Model
# =============================================================================

class APSIRAgent(ap.Agent):
    """Agent with SIR health states."""
    
    STATUS_S = 0
    STATUS_I = 1
    STATUS_R = 2
    
    def setup(self):
        self.status = self.STATUS_S
        self.infection_time = 0
        world_size = self.model.p.get('world_size', 100)
        self.x = random.uniform(0, world_size)
        self.y = random.uniform(0, world_size)
        
        # Initial infections - use agent index from model
        if self.id < self.model.p.get('initial_infected', 5):
            self.status = self.STATUS_I
    
    def move(self):
        speed = self.model.p.get('movement_speed', 2.0)
        world_size = self.model.p.get('world_size', 100)
        
        self.x += random.uniform(-speed, speed)
        self.y += random.uniform(-speed, speed)
        
        self.x = max(0, min(world_size, self.x))
        self.y = max(0, min(world_size, self.y))
    
    def infect_neighbors(self):
        if self.status != self.STATUS_I:
            return
            
        radius = self.model.p.get('infection_radius', 5.0)
        transmission = self.model.p.get('transmission_rate', 0.1)
        
        for other in self.model.agents:
            if other == self or other.status != self.STATUS_S:
                continue
            
            dist_sq = (self.x - other.x)**2 + (self.y - other.y)**2
            if dist_sq <= radius**2:
                if random.random() < transmission:
                    other.status = self.STATUS_I
                    other.infection_time = 0
    
    def update_health(self):
        if self.status == self.STATUS_I:
            self.infection_time += 1
            if self.infection_time >= self.model.p.get('recovery_time', 14):
                self.status = self.STATUS_R


class AgentPySIRModel(ap.Model):
    """
    SIR Epidemic Model (AgentPy Implementation).
    """
    
    def setup(self):
        n = self.p.get('n', 100)
        self.agents = ap.AgentList(self, n, APSIRAgent)
    
    def step(self):
        self.agents.move()
        self.agents.infect_neighbors()
        self.agents.update_health()
    
    def update(self):
        s = sum(1 for a in self.agents if a.status == APSIRAgent.STATUS_S)
        i = sum(1 for a in self.agents if a.status == APSIRAgent.STATUS_I)
        r = sum(1 for a in self.agents if a.status == APSIRAgent.STATUS_R)
        self.record('susceptible', s)
        self.record('infected', i)
        self.record('recovered', r)
    
    def end(self):
        pass


# =============================================================================
# Random Walk Model
# =============================================================================

class APWalkAgent(ap.Agent):
    """Agent that performs random walk."""
    
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


class AgentPyRandomWalk(ap.Model):
    """
    Random Walk Model (AgentPy Implementation).
    """
    
    def setup(self):
        n = self.p.get('n', 100)
        self.agents = ap.AgentList(self, n, APWalkAgent)
    
    def step(self):
        self.agents.step()
    
    def update(self):
        avg_x = sum(a.x for a in self.agents) / len(self.agents)
        avg_y = sum(a.y for a in self.agents) / len(self.agents)
        self.record('avg_x', avg_x)
        self.record('avg_y', avg_y)
    
    def end(self):
        pass


# Model registry for benchmark runner
AGENTPY_MODELS = {
    'wealth_transfer': AgentPyWealthTransfer,
    'sir_epidemic': AgentPySIRModel,
    'random_walk': AgentPyRandomWalk,
}

if __name__ == '__main__':
    # Quick test
    model = AgentPyWealthTransfer({'n': 100, 'steps': 10, 'initial_wealth': 1})
    results = model.run()
    print(f"AgentPy Wealth Transfer - Complete")
    
    model = AgentPySIRModel({'n': 100, 'steps': 10, 'initial_infected': 5})
    results = model.run()
    print(f"AgentPy SIR - Complete")
    
    model = AgentPyRandomWalk({'n': 100, 'steps': 10, 'speed': 1.0, 'world_size': 100})
    results = model.run()
    print(f"AgentPy Random Walk - Complete")
