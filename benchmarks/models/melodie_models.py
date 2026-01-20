"""
Melodie Benchmark Models
"""
import time
import numpy as np
import Melodie
import os
import shutil

# =============================================================================
# Wealth Transfer
# =============================================================================

class WealthAgent(Melodie.Agent):
    def setup(self):
        self.wealth = 0.0

class WealthEnvironment(Melodie.Environment):
    def setup(self):
        pass

class WealthModel(Melodie.Model):
    def setup(self):
        self.agent_list = self.create_agent_list(WealthAgent)
        self.environment = self.create_environment(WealthEnvironment)

    def run(self):
        for _ in range(self.scenario.periods):
            self.step()
            
    def step(self):
        agents = self.agent_list
        # Vectorized-style logic isn't default in Melodie loops, 
        # so we use standard Python loops as per Melodie examples
        for agent in agents:
            if agent.wealth > 0:
                receiver = np.random.randint(0, len(agents))
                if receiver != agent.id:
                    agent.wealth -= 1
                    agents[receiver].wealth += 1

class WealthScenario(Melodie.Scenario):
    def setup(self):
        self.periods = 0
        self.agent_num = 0
        self.initial_wealth = 1

    def load_data(self):
        # We'll set parameters dynamically
        pass

# =============================================================================
# Random Walk
# =============================================================================

class WalkAgent(Melodie.Agent):
    def setup(self):
        self.x = 0.0
        self.y = 0.0
        self.speed = 1.0

class WalkModel(Melodie.Model):
    def setup(self):
        self.agent_list = self.create_agent_list(WalkAgent)
        self.environment = self.create_environment(WealthEnvironment)

    def run(self):
        for _ in range(self.scenario.periods):
            self.step()

    def step(self):
        for agent in self.agent_list:
            theta = np.random.uniform(0, 2*np.pi)
            agent.x += agent.speed * np.cos(theta)
            agent.y += agent.speed * np.sin(theta)
            # Clip
            agent.x = np.clip(agent.x, 0, 100)
            agent.y = np.clip(agent.y, 0, 100)

class WalkScenario(Melodie.Scenario):
    def setup(self):
        self.periods = 0
        self.agent_num = 0

# =============================================================================
# SIR Model
# =============================================================================

class SIRAgent(Melodie.Agent):
    def setup(self):
        self.x = 0.0
        self.y = 0.0
        self.status = 0 # 0:S, 1:I, 2:R
        self.infection_time = 0

class SIRModel(Melodie.Model):
    def setup(self):
        self.agent_list = self.create_agent_list(SIRAgent)
        self.environment = self.create_environment(WealthEnvironment)
        self.infection_radius = 5.0
        self.transmission_rate = 0.1
        self.recovery_time = 14

    def run(self):
        for _ in range(self.scenario.periods):
            self.step()

    def step(self):
        # Movement
        for agent in self.agent_list:
            theta = np.random.uniform(0, 2*np.pi)
            agent.x = np.clip(agent.x + 2.0 * np.cos(theta), 0, 100)
            agent.y = np.clip(agent.y + 2.0 * np.sin(theta), 0, 100)
            
        # Infection (O(N^2) naive)
        infected = [a for a in self.agent_list if a.status == 1]
        susceptible = [a for a in self.agent_list if a.status == 0]
        
        for inf in infected:
            for sus in susceptible:
                dist = np.sqrt((inf.x - sus.x)**2 + (inf.y - sus.y)**2)
                if dist <= self.infection_radius:
                    if np.random.random() < self.transmission_rate:
                        sus.status = 1
                        sus.infection_time = 0
                        
            inf.infection_time += 1
            if inf.infection_time >= self.recovery_time:
                inf.status = 2

class SIRScenario(Melodie.Scenario):
    def setup(self):
        self.periods = 0
        self.agent_num = 0

# =============================================================================
# Runner
# =============================================================================

def run_benchmark(model_cls, scenario_cls, agent_counts, steps=100):
    results = {}
    print(f"\nBenchmarking {model_cls.__name__}...")
    
    # Melodie requires config
    config = Melodie.Config(
        project_name='MelodieBenchmark',
        project_root='.',
        sqlite_folder='.',
        output_folder='.',
        input_folder='.',
    )
    
    for n in agent_counts:
        # Cleanup
        if os.path.exists('MelodieBenchmark.sqlite'):
            os.remove('MelodieBenchmark.sqlite')
            
        # Init model
        scenario = scenario_cls()
        scenario.manager = None # Hack to bypass manager check if needed
        scenario.periods = steps
        scenario.agent_num = n
        scenario.id = 0
        
        model = model_cls(config, scenario)
        model.setup()
        
        # Manually add agents
        for i in range(n):
            agent = model.agent_list.add()
            agent.id = i
            agent.setup()
            
        # Init agent properties
        for i in range(n):
            agent = model.agent_list[i]
            agent.id = i
            if hasattr(agent, 'wealth'): agent.wealth = 1
            if hasattr(agent, 'x'): 
                agent.x = np.random.uniform(0, 100)
                agent.y = np.random.uniform(0, 100)
            if hasattr(agent, 'status'):
                agent.status = 1 if i < 5 else 0
        
        # Run
        start_time = time.time()
        model.run()
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"  {n} agents: {elapsed:.3f}s")
        results[n] = elapsed
        
    return results

if __name__ == "__main__":
    counts = [100, 500, 1000, 5000]
    
    print("Melodie Benchmark")
    print("="*50)
    
    # 1. Wealth
    run_benchmark(WealthModel, WealthScenario, counts)
    
    # 2. Random Walk
    run_benchmark(WalkModel, WalkScenario, counts)
    
    # 3. SIR
    run_benchmark(SIRModel, SIRScenario, counts)
