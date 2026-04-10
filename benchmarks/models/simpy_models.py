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
    """Wealth Transfer Agent Process."""
    while True:
        current = agents_list[agent_id]['wealth']
        if current > 0:
            partner_id = random.randrange(len(agents_list))
            if partner_id != agent_id:
                agents_list[agent_id]['wealth'] = current - 1
                agents_list[partner_id]['wealth'] += 1

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
    """Random Walk Agent Process."""
    world_size = 100.0
    x = random.uniform(0, world_size)
    y = random.uniform(0, world_size)
    speed = 1.0

    while True:
        theta = random.uniform(0, 2 * math.pi)
        x += speed * math.cos(theta)
        y += speed * math.sin(theta)
        # Clamp to world bounds (parity with AMBER/AgentPy/Mesa/Melodie).
        x = max(0.0, min(world_size, x))
        y = max(0.0, min(world_size, y))

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
    """SIR Agent Process."""
    initial_infected = params.get('initial_infected', 5)
    world_size = params.get('world_size', 100.0)
    radius = params.get('infection_radius', 5.0)
    transmission = params.get('transmission_rate', 0.1)
    recovery_time = params.get('recovery_time', 14)
    speed = params.get('movement_speed', 2.0)

    x = random.uniform(0, world_size)
    y = random.uniform(0, world_size)
    status = 1 if agent_id < initial_infected else 0  # 1 = I, 0 = S
    infection_time = 0

    agents_list[agent_id]['status'] = status
    agents_list[agent_id]['x'] = x
    agents_list[agent_id]['y'] = y

    while True:
        # Move + clamp (parity with the other frameworks)
        theta = random.uniform(0, 2 * math.pi)
        x = max(0.0, min(world_size, x + speed * math.cos(theta)))
        y = max(0.0, min(world_size, y + speed * math.sin(theta)))
        agents_list[agent_id]['x'] = x
        agents_list[agent_id]['y'] = y

        # Read current status in case we were just infected by another agent.
        status = agents_list[agent_id]['status']

        if status == 1:  # Infected — try to spread to susceptible neighbours
            for other in agents_list:
                if other['status'] == 0:
                    dx = x - other['x']
                    dy = y - other['y']
                    if dx * dx + dy * dy <= radius * radius and random.random() < transmission:
                        other['status'] = 1
                        other['infection_time'] = 0

            infection_time = agents_list[agent_id].get('infection_time', 0) + 1
            agents_list[agent_id]['infection_time'] = infection_time
            if infection_time >= recovery_time:
                status = 2
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
