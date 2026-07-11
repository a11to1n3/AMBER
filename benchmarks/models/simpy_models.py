"""
SimPy Benchmark Models (Process-Based ABM)
SimPy is a Discrete Event Simulation (DES) library.
To use it for ABM, we treat each agent as a process that waits for 1 tick.
"""

import time
import simpy
import random
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

def run_wealth_benchmark(n=100, steps=100, initial_wealth=1):
    env = simpy.Environment()
    # Shared state
    agents_data = [{'id': i, 'wealth': initial_wealth} for i in range(n)]

    # Start processes
    for i in range(n):
        env.process(wealth_agent(env, i, agents_data))

    env.run(until=steps)

# =============================================================================
# Random Walk
# =============================================================================

def walk_agent(env, agent_id, agents_list, params=None):
    """Random Walk Agent Process."""
    params = params or {}
    world_size = params.get('world_size', 100.0)
    x = random.uniform(0, world_size)
    y = random.uniform(0, world_size)
    speed = params.get('speed', 1.0)

    while True:
        x += random.uniform(-speed, speed)
        y += random.uniform(-speed, speed)
        # Clamp to world bounds (parity with AMBER/AgentPy/Mesa/Melodie).
        x = max(0.0, min(world_size, x))
        y = max(0.0, min(world_size, y))

        agents_list[agent_id]['x'] = x
        agents_list[agent_id]['y'] = y

        yield env.timeout(1)

def run_walk_benchmark(n=100, steps=100, world_size=100.0, speed=1.0):
    env = simpy.Environment()
    agents_data = [{'id': i, 'x': 0, 'y': 0} for i in range(n)]
    params = {'world_size': world_size, 'speed': speed}

    for i in range(n):
        env.process(walk_agent(env, i, agents_data, params))

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
        x = max(0.0, min(world_size, x + random.uniform(-speed, speed)))
        y = max(0.0, min(world_size, y + random.uniform(-speed, speed)))
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

def run_sir_benchmark(
    n=100,
    steps=100,
    initial_infected=5,
    world_size=100.0,
    movement_speed=2.0,
    infection_radius=5.0,
    transmission_rate=0.1,
    recovery_time=14,
):
    env = simpy.Environment()
    agents_data = [{'id': i, 'status': 0, 'x': 0, 'y': 0, 'infection_time': 0} for i in range(n)]
    params = {
        'initial_infected': initial_infected,
        'world_size': world_size,
        'movement_speed': movement_speed,
        'infection_radius': infection_radius,
        'transmission_rate': transmission_rate,
        'recovery_time': recovery_time,
    }

    for i in range(n):
        env.process(sir_agent(env, i, agents_data, params))

    env.run(until=steps)

    infected_count = sum(1 for a in agents_data if a['status'] == 1 or a['status'] == 2)
    print(f"  Final Infected: {infected_count}/{n}")


# =============================================================================
# Schelling Segregation
# =============================================================================

def schelling_agent(env, agent_id, agents_data, occ, params):
    G, tol = params['G'], params['tolerance']
    while True:
        yield env.timeout(1)
        a = agents_data[agent_id]
        x, y, atype = a['x'], a['y'], a['atype']
        same = total = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                v = occ.get(((x + dx) % G, (y + dy) % G))
                if v is not None:
                    total += 1
                    if v == atype:
                        same += 1
        if total > 0 and same < tol * total:
            for _ in range(20):
                cx, cy = random.randrange(G), random.randrange(G)
                if (cx, cy) not in occ:
                    del occ[(x, y)]
                    a['x'], a['y'] = cx, cy
                    occ[(cx, cy)] = atype
                    break


def run_schelling_benchmark(n=100, steps=100, density=0.8, fraction_a=0.5, tolerance=0.3):
    G = int((n / density) ** 0.5) + 1
    cells = list(range(G * G))
    random.shuffle(cells)
    cells = cells[:n]
    n_a = int(n * fraction_a)
    agents_data, occ = [], {}
    for i in range(n):
        x, y, atype = cells[i] % G, cells[i] // G, (1 if i < n_a else 2)
        agents_data.append({'id': i, 'x': x, 'y': y, 'atype': atype})
        occ[(x, y)] = atype
    params = {'G': G, 'tolerance': tolerance}
    env = simpy.Environment()
    for i in range(n):
        env.process(schelling_agent(env, i, agents_data, occ, params))
    env.run(until=steps)


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
