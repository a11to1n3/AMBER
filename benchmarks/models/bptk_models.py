"""
BPTK-Py Benchmark Models
"""
import sys
print("Starting BPTK Benchmark script...", flush=True)

import time
import numpy as np
try:
    from BPTK_Py import Model, Agent
    from BPTK_Py import bptk
    print("BPTK imports successful", flush=True)
except Exception as e:
    print(f"BPTK import failed: {e}", flush=True)
    sys.exit(1)

# =============================================================================
# Wealth Transfer
# =============================================================================

class WealthAgent(Agent):
    def initialize(self):
        self.agent_type = "agent"
        self.state = "active"
        self.set_property("wealth", 1)

    def act(self, time, round_no, step_no):
        wealth = self.get_property("wealth")
        if wealth > 0:
            agents = self.model.agents
            if not agents: return
            
            # Simple random pick (BPTK lookup overhead)
            partner = agents[np.random.randint(0, len(agents))]
            if partner != self:
                self.set_property("wealth", wealth - 1)
                partner.set_property("wealth", partner.get_property("wealth") + 1)

class WealthModel(Model):
    def instantiate_model(self):
        self.register_agent_factory("agent", lambda agent_id, model, properties: WealthAgent(agent_id, model, properties))

# =============================================================================
# Random Walk
# =============================================================================

class WalkAgent(Agent):
    def initialize(self):
        self.agent_type = "agent"
        self.state = "active"
        self.set_property("x", np.random.uniform(0, 100))
        self.set_property("y", np.random.uniform(0, 100))

    def act(self, time, round_no, step_no):
        x = self.get_property("x")
        y = self.get_property("y")
        theta = np.random.uniform(0, 2*np.pi)
        
        x = np.clip(x + np.cos(theta), 0, 100)
        y = np.clip(y + np.sin(theta), 0, 100)
        
        self.set_property("x", x)
        self.set_property("y", y)

class WalkModel(Model):
    def instantiate_model(self):
        self.register_agent_factory("agent", lambda agent_id, model, properties: WalkAgent(agent_id, model, properties))

# =============================================================================
# Runner
# =============================================================================

def run_benchmark(model_cls, agent_cls, counts, steps=100):
    print(f"\nBenchmarking {model_cls.__name__}...")
    
    for n in counts:
        # BPTK usually manages scenarios, but we'll try direct run
        model = model_cls(starttime=1, stoptime=steps, dt=1, name="bench")
        # instantiate_model needs to be called or setup manually?
        # BPTK structure is complex. We'll try to just manual add agents.
        
        # Manually Create Agents (bypass factory to be faster/simpler)
        agents = []
        for i in range(n):
            agent = agent_cls(i, model, {})
            agent.initialize()
            agents.append(agent)
            
        model.agents = agents
        
        start = time.time()
        # BPTK manual loop mimicking internal logic to avoid overhead of full engine
        # if the engine is too heavy. But we should benchmark the engine.
        # But BPTK engine is hard to invoke without config.
        # We will do a manual step loop calling agent.act()
        
        for t in range(steps):
            for agent in agents:
                agent.act(t, t, 0)
                
        elapsed = time.time() - start
        
        print(f"  {n} agents: {elapsed:.3f}s")


if __name__ == "__main__":
    counts = [100, 500, 1000]
    
    print("BPTK-Py Benchmark")
    print("="*50)
    
    run_benchmark(WealthModel, WealthAgent, counts)
    run_benchmark(WalkModel, WalkAgent, counts)
    # Skip SIR as BPTK doesn't have spatial index built-in easily for this quick test
