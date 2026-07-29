# Agents.jl Benchmark Models (Fixed API)
using Agents
using Random
using Printf

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
    agent.x += agent.speed * (2rand() - 1)
    agent.y += agent.speed * (2rand() - 1)
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
    agent.x = clamp(agent.x + speed * (2rand() - 1), 0.0, 100.0)
    agent.y = clamp(agent.y + speed * (2rand() - 1), 0.0, 100.0)

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
# Schelling Segregation (GridSpaceSingle)
# =============================================================================

@agent struct SchellingAgent(GridAgent{2})
    group::Int
end

function schelling_step!(agent, model)
    same = 0
    total = 0
    for neighbor in nearby_agents(agent, model)
        total += 1
        if agent.group == neighbor.group
            same += 1
        end
    end
    if total > 0 && same < model.tolerance * total
        move_agent_single!(agent, model)
    end
    return
end

function run_schelling_benchmark(; n=100, steps=100)
    density = 0.8
    G = ceil(Int, sqrt(n / density))
    space = GridSpaceSingle((G, G); periodic=true)
    model = StandardABM(SchellingAgent, space;
                        properties=Dict(:tolerance => 0.3),
                        agent_step! = schelling_step!)
    half = n ÷ 2
    for i in 1:n
        add_agent_single!(model; group = (i <= half ? 1 : 2))
    end
    step!(model, steps)
    return model
end

# =============================================================================
# Benchmark Runner
# =============================================================================

function _trimmed_mean(times)
    sorted_times = sort(times)
    if length(sorted_times) >= 3
        sorted_times = sorted_times[1:end-1]
    end
    return sum(sorted_times) / length(sorted_times)
end

function _timing_summary(runner; n, steps, runs)
    Random.seed!(42)
    runner(; n=min(n, 100), steps=min(steps, 10))
    times = Float64[]
    for _ in 1:runs
        Random.seed!(42)
        push!(times, @elapsed runner(; n=n, steps=steps))
    end
    return (
        mean = _trimmed_mean(times),
        samples = times,
    )
end

function _sample_list(times)
    return join((@sprintf("%.9f", t) for t in times), ",")
end

const MODEL_EXPONENT = Dict(
    "wealth_transfer" => 1.0, "random_walk" => 1.0, "sir_epidemic" => 2.0,
)

function _predict_next(history, next_n, exponent)
    n1, t1 = history[end]
    if length(history) >= 2 && history[end-1][2] > 0 && t1 > 0 && history[end-1][1] < n1
        n0, t0 = history[end-1]
        k = clamp(log(t1 / t0) / log(n1 / n0), 0.0, 3.0)
    else
        k = exponent
    end
    return t1 * (next_n / n1) ^ k
end

function run_benchmarks(; agent_counts=[100, 500, 1000, 5000], steps=50, runs=10, budget=15.0)
    println("Agents.jl Benchmark")
    println("="^50)

    counts = sort(agent_counts)
    for (name, runner) in [
        ("wealth_transfer", run_wealth_benchmark),
        ("random_walk", run_walk_benchmark),
        ("sir_epidemic", run_sir_benchmark),
        ("schelling", run_schelling_benchmark)
    ]
        println("\n$name:")
        exponent = get(MODEL_EXPONENT, name, 1.0)
        history = Tuple{Int,Float64}[]
        retired = false
        for n in counts
            if retired
                continue  # printing nothing -> Python records this cell as N/A
            end
            if !isempty(history) && _predict_next(history, n, exponent) > budget
                retired = true
                continue
            end
            summary = _timing_summary(runner; n=n, steps=steps, runs=runs)
            @printf(
                "  %d agents: %.9fs samples=[%s]\n",
                n,
                summary.mean,
                _sample_list(summary.samples),
            )
            push!(history, (n, summary.mean))
            if summary.mean > budget
                retired = true
            end
        end
    end
end

# Parse optional --steps and --agents arguments so the outer Python runner
# can hold every framework to the same (agent_counts, steps) configuration.
function _parse_args(args)
    agent_counts = [100, 500, 1000, 5000]
    steps = 50
    runs = 10
    budget = 15.0
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--steps" && i + 1 <= length(args)
            steps = parse(Int, args[i + 1])
            i += 2
        elseif a == "--agents" && i + 1 <= length(args)
            agent_counts = parse.(Int, split(args[i + 1], ','))
            i += 2
        elseif a == "--runs" && i + 1 <= length(args)
            runs = parse(Int, args[i + 1])
            i += 2
        elseif a == "--budget" && i + 1 <= length(args)
            budget = parse(Float64, args[i + 1])
            i += 2
        else
            i += 1
        end
    end
    return (agent_counts, steps, runs, budget)
end

let (agent_counts, steps, runs, budget) = _parse_args(ARGS)
    run_benchmarks(; agent_counts=agent_counts, steps=steps, runs=runs, budget=budget)
end
