"""
SimPy Benchmark Models (Process-Based ABM)
SimPy is a Discrete Event Simulation (DES) library.
To use it for ABM, we treat each agent as a process that waits for 1 tick.
"""

import time
import simpy
import random
import math
import sys

# =============================================================================
# Wealth Transfer
# =============================================================================

def wealth_agent(env, agent_id, agents_list):
    """Wealth Transfer Agent Process"""
    wealth = 1
    while True:
        if wealth > 0:
            # Pick random partner
            partner_id = random.randrange(len(agents_list))
            if partner_id != agent_id:
                # Direct interaction (process-based)
                # In SimPy, we usually use resources/stores, but for speed 
                # we'll use a shared list like standard ABMs
                wealth -= 1
                # We can't easily modify other process variables directly without shared state
                # So we use the shared list 'agents_data'
                agents_list[partner_id]['wealth'] += 1
                agents_list[agent_id]['wealth'] = wealth
                
        yield env.timeout(1)

def run_wealth_benchmark(n=100, steps=100):
    env = simpy.Environment()
    # Shared state
    agents_data = [{'id': i, 'wealth': 1} for i in range(n)]
    
    # Start processes
    for i in range(n):
        env.process(wealth_agent(env, i, agents_data))
        
    env.run(until=steps)

# =============================================================================
# Random Walk
# =============================================================================

def walk_agent(env, agent_id, agents_list):
    """Random Walk Agent Process"""
    x = random.uniform(0, 100)
    y = random.uniform(0, 100)
    speed = 1.0
    
    while True:
        theta = random.uniform(0, 2*math.pi)
        x += speed * math.cos(theta)
        y += speed * math.sin(theta)
        
        # Output to shared state (mimicking update)
        agents_list[agent_id]['x'] = x
        agents_list[agent_id]['y'] = y
        
        yield env.timeout(1)

def run_walk_benchmark(n=100, steps=100):
    env = simpy.Environment()
    agents_data = [{'id': i, 'x': 0, 'y': 0} for i in range(n)]
    
    for i in range(n):
        env.process(walk_agent(env, i, agents_data))
        
    env.run(until=steps)

# =============================================================================
# SIR Model
# =============================================================================

def sir_agent(env, agent_id, agents_list, params):
    """SIR Agent Process"""
    # Init state
    x = random.uniform(0, 100)
    y = random.uniform(0, 100)
    # Init state
    x = random.uniform(0, 100)
    y = random.uniform(0, 100)
    status = 1 # Force 100% infected for Dense Benchmark (User Request)
    # if agent_id < 5: 
    #     status = 1 # I
    infection_time = 0
    
    agents_list[agent_id]['status'] = status
    agents_list[agent_id]['x'] = x
    agents_list[agent_id]['y'] = y
    
    while True:
        # Move
        theta = random.uniform(0, 2*math.pi)
        x += 2.0 * math.cos(theta)
        y += 2.0 * math.sin(theta)
        agents_list[agent_id]['x'] = x
        agents_list[agent_id]['y'] = y
        
        # Infection Logic
        # This is tricky in DES. Agents usually react to events.
        # Active polling (neighbor search) every step is very inefficient in SimPy
        # but is the only way to replicate the ABM logic exactly.
        
        if status == 1: # Infected
            # Look for S neighbors
            for other in agents_list:
                if other['status'] == 0:
                    dist = math.sqrt((x - other['x'])**2 + (y - other['y'])**2)
                    if dist <= 5.0 and random.random() < 0.1:
                        other['status'] = 1 # Infect them
                        other['infection_time'] = 0 # Need to track this
            
            infection_time += 1
            if infection_time >= 14:
                status = 2 # R
                agents_list[agent_id]['status'] = 2
                
        yield env.timeout(1)

def run_sir_benchmark(n=100, steps=100):
    env = simpy.Environment()
    agents_data = [{'id': i, 'status': 0, 'x': 0, 'y': 0, 'infection_time': 0} for i in range(n)]
    params = {}
    
    for i in range(n):
        env.process(sir_agent(env, i, agents_data, params))
        
    env.run(until=steps)
    
    infected_count = sum(1 for a in agents_data if a['status'] == 1 or a['status'] == 2)
    print(f"  Final Infected: {infected_count}/{n}")


# =============================================================================
# Runner
# =============================================================================

if __name__ == "__main__":
    counts = [100, 500, 1000, 5000]
    
    print("SimPy Benchmark")
    print("="*50)
    
    # Wealth
    print("\nWealth Transfer:")
    for n in counts:
        start = time.time()
        run_wealth_benchmark(n, 100)
        print(f"  {n} agents: {time.time() - start:.3f}s")
        
    # Walk
    print("\nRandom Walk:")
    for n in counts:
        start = time.time()
        run_walk_benchmark(n, 100)
        print(f"  {n} agents: {time.time() - start:.3f}s")
        
    # SIR
    # SimPy overhead + O(N^2) loop inside a generator = EXTREMELY SLOW
    # We will limit to small counts
    print("\nSIR Epidemic:")
    for n in [100, 500, 1000]: 
        start = time.time()
        run_sir_benchmark(n, 100)
        print(f"  {n} agents: {time.time() - start:.3f}s")
