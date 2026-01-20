"""
Mesa Model Implementations for Benchmarking

These models implement the same logic as AMBER models for fair comparison.
Compatible with Mesa 3.x (no schedulers, uses AgentSet).
"""

import mesa
import random


# =============================================================================
# Wealth Transfer Model
# =============================================================================

class MesaWealthAgent(mesa.Agent):
    """Agent that can transfer wealth to other agents."""
    
    def __init__(self, model, initial_wealth=1):
        super().__init__(model)
        self.wealth = initial_wealth
    
    def step(self):
        if self.wealth > 0:
            # Give 1 unit to a random other agent
            all_agents = list(self.model.agents)
            other = random.choice(all_agents)
            if other != self:
                self.wealth -= 1
                other.wealth += 1


class MesaWealthTransfer(mesa.Model):
    """
    Boltzmann Wealth Distribution Model (Mesa Implementation).
    """
    
    def __init__(self, n=100, initial_wealth=1, steps=100, **kwargs):
        super().__init__()
        self.n = n
        self.max_steps = steps
        self._initial_wealth = initial_wealth
        
        # Data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "total_wealth": lambda m: sum(a.wealth for a in m.agents),
                "gini": self._compute_gini
            }
        )
        
        # Create agents (Mesa 3.x style)
        for i in range(n):
            MesaWealthAgent(self, initial_wealth)
        
        self.datacollector.collect(self)
    
    def _compute_gini(self):
        wealths = sorted([a.wealth for a in self.agents])
        n = len(wealths)
        if n == 0 or sum(wealths) == 0:
            return 0
        cumulative = sum((i + 1) * w for i, w in enumerate(wealths))
        return (2 * cumulative) / (n * sum(wealths)) - (n + 1) / n
    
    def step(self):
        # Mesa 3.x uses shuffle_do for random agent activation
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
    
    def run(self):
        for _ in range(self.max_steps):
            self.step()
        return {'model_data': self.datacollector.get_model_vars_dataframe()}


# =============================================================================
# SIR Epidemic Model
# =============================================================================

class MesaSIRAgent(mesa.Agent):
    """Agent with SIR health states."""
    
    STATUS_S = 0
    STATUS_I = 1
    STATUS_R = 2
    
    def __init__(self, model, world_size=100, is_infected=False):
        super().__init__(model)
        self.status = self.STATUS_I if is_infected else self.STATUS_S
        self.infection_time = 0
        self.x = random.uniform(0, world_size)
        self.y = random.uniform(0, world_size)
        self.world_size = world_size
    
    def move(self):
        speed = self.model.movement_speed
        
        self.x += random.uniform(-speed, speed)
        self.y += random.uniform(-speed, speed)
        
        self.x = max(0, min(self.world_size, self.x))
        self.y = max(0, min(self.world_size, self.y))
    
    def infect_neighbors(self):
        if self.status != self.STATUS_I:
            return
            
        radius = self.model.infection_radius
        transmission = self.model.transmission_rate
        
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
            if self.infection_time >= self.model.recovery_time:
                self.status = self.STATUS_R
    
    def step(self):
        self.move()
        self.infect_neighbors()
        self.update_health()


class MesaSIRModel(mesa.Model):
    """
    SIR Epidemic Model (Mesa Implementation).
    """
    
    def __init__(self, n=100, initial_infected=5, world_size=100,
                 movement_speed=2.0, infection_radius=5.0, 
                 transmission_rate=0.1, recovery_time=14, steps=100, **kwargs):
        super().__init__()
        self.n = n
        self.max_steps = steps
        self.movement_speed = movement_speed
        self.infection_radius = infection_radius
        self.transmission_rate = transmission_rate
        self.recovery_time = recovery_time
        
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "susceptible": lambda m: sum(1 for a in m.agents if a.status == MesaSIRAgent.STATUS_S),
                "infected": lambda m: sum(1 for a in m.agents if a.status == MesaSIRAgent.STATUS_I),
                "recovered": lambda m: sum(1 for a in m.agents if a.status == MesaSIRAgent.STATUS_R),
            }
        )
        
        for i in range(n):
            MesaSIRAgent(self, world_size, is_infected=(i < initial_infected))
        
        self.datacollector.collect(self)
    
    def step(self):
        self.agents.do("step")
        self.datacollector.collect(self)
    
    def run(self):
        for _ in range(self.max_steps):
            self.step()
        return {'model_data': self.datacollector.get_model_vars_dataframe()}


# =============================================================================
# Random Walk Model
# =============================================================================

class MesaWalkAgent(mesa.Agent):
    """Agent that performs random walk."""
    
    def __init__(self, model, world_size=100):
        super().__init__(model)
        self.x = random.uniform(0, world_size)
        self.y = random.uniform(0, world_size)
        self.world_size = world_size
    
    def step(self):
        speed = self.model.speed
        
        self.x += random.uniform(-speed, speed)
        self.y += random.uniform(-speed, speed)
        
        self.x = max(0, min(self.world_size, self.x))
        self.y = max(0, min(self.world_size, self.y))


class MesaRandomWalk(mesa.Model):
    """
    Random Walk Model (Mesa Implementation).
    """
    
    def __init__(self, n=100, world_size=100, speed=1.0, steps=100, **kwargs):
        super().__init__()
        self.n = n
        self.max_steps = steps
        self.speed = speed
        
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "avg_x": lambda m: sum(a.x for a in m.agents) / len(m.agents),
                "avg_y": lambda m: sum(a.y for a in m.agents) / len(m.agents),
            }
        )
        
        for i in range(n):
            MesaWalkAgent(self, world_size)
        
        self.datacollector.collect(self)
    
    def step(self):
        self.agents.do("step")
        self.datacollector.collect(self)
    
    def run(self):
        for _ in range(self.max_steps):
            self.step()
        return {'model_data': self.datacollector.get_model_vars_dataframe()}


# Model registry for benchmark runner
MESA_MODELS = {
    'wealth_transfer': MesaWealthTransfer,
    'sir_epidemic': MesaSIRModel,
    'random_walk': MesaRandomWalk,
}

if __name__ == '__main__':
    # Quick test
    model = MesaWealthTransfer(n=100, steps=10, initial_wealth=1)
    results = model.run()
    print(f"Mesa Wealth Transfer - Complete")
    
    model = MesaSIRModel(n=100, steps=10, initial_infected=5)
    results = model.run()
    print(f"Mesa SIR - Complete")
    
    model = MesaRandomWalk(n=100, steps=10, speed=1.0)
    results = model.run()
    print(f"Mesa Random Walk - Complete")
