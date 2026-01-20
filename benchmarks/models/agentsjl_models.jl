# Agents.jl Benchmark Models (Fixed API)
using Agents
using Random

# =============================================================================
# Wealth Transfer Model (WORKS!)
# =============================================================================

@agent struct WealthAgent(NoSpaceAgent)
    wealth::Float64
end

function wealth_transfer_step!(agent, model)
    if agent.wealth > 0
        partner = random_agent(model)
        if partner !== nothing && partner.id != agent.id
            agent.wealth -= 1
            partner.wealth += 1
        end
    end
end

function run_wealth_benchmark(; n=100, steps=100)
    model = StandardABM(WealthAgent; agent_step! = wealth_transfer_step!)
    for _ in 1:n
        add_agent!(model, 1.0)
    end
    step!(model, steps)
    return model
end

# =============================================================================
# Random Walk Model (Simplified - no space needed for timing)
# =============================================================================

@agent struct WalkAgent2(NoSpaceAgent)
    x::Float64
    y::Float64
    speed::Float64
end

function random_walk_step2!(agent, model)
    θ = rand() * 2π
    agent.x += agent.speed * cos(θ)
    agent.y += agent.speed * sin(θ)
    # Clamp to bounds
    agent.x = clamp(agent.x, 0.0, 100.0)
    agent.y = clamp(agent.y, 0.0, 100.0)
end

function run_walk_benchmark(; n=100, steps=100)
    model = StandardABM(WalkAgent2; agent_step! = random_walk_step2!)
    for _ in 1:n
        add_agent!(model, rand() * 100, rand() * 100, 1.0)
    end
    step!(model, steps)
    return model
end

# =============================================================================
# SIR Model (Simplified - O(n^2) like Python version)
# =============================================================================

@agent struct SIRAgent2(NoSpaceAgent)
    x::Float64
    y::Float64
    status::Symbol
    infection_time::Int
end

function sir_step2!(agent, model)
    speed = 2.0
    radius = 5.0
    trans_rate = 0.1
    recovery = 14
    
    # Move
    θ = rand() * 2π
    agent.x = clamp(agent.x + speed * cos(θ), 0.0, 100.0)
    agent.y = clamp(agent.y + speed * sin(θ), 0.0, 100.0)
    
    # Infection spread
    if agent.status == :I
        for other in allagents(model)
            if other.status == :S
                dx = agent.x - other.x
                dy = agent.y - other.y
                dist = sqrt(dx*dx + dy*dy)
                if dist <= radius && rand() < trans_rate
                    other.status = :I
                    other.infection_time = 0
                end
            end
        end
        agent.infection_time += 1
        if agent.infection_time >= recovery
            agent.status = :R
        end
    end
end

function run_sir_benchmark(; n=100, steps=100)
    model = StandardABM(SIRAgent2; agent_step! = sir_step2!)
    for i in 1:n
        status = i <= 5 ? :I : :S
        add_agent!(model, rand()*100, rand()*100, status, 0)
    end
    step!(model, steps)
    return model
end

# =============================================================================
# Benchmark Runner
# =============================================================================

function run_benchmarks(; agent_counts=[100, 500, 1000], steps=100)
    println("Agents.jl Benchmark")
    println("="^50)
    
    for (name, runner) in [
        ("wealth_transfer", run_wealth_benchmark),
        ("random_walk", run_walk_benchmark),
        ("sir_epidemic", run_sir_benchmark)
    ]
        println("\n$name:")
        for n in agent_counts
            # Warmup
            runner(; n=min(n, 100), steps=10)
            # Timed
            t = @elapsed runner(; n=n, steps=steps)
            println("  $n agents: $(round(t, digits=3))s")
        end
    end
end

# Run benchmarks
run_benchmarks(; agent_counts=[100, 500, 1000, 5000], steps=100)
