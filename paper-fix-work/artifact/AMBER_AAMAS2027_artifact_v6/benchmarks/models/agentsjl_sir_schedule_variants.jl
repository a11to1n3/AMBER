# Deterministic Agents.jl split-SIR runner for benchmark evidence.
#
# This keeps the SIRAgent2 NoSpaceAgent container shape used by
# agentsjl_models.jl, but replaces host RNG calls with deterministic shared
# inputs and exposes explicit synchronous and sequential schedules.

using Agents

const STATUS_S = :S
const STATUS_I = :I
const STATUS_R = :R

const GOLDEN = UInt64(0x9e3779b97f4a7c15)
const C1 = UInt64(0xbf58476d1ce4e5b9)
const C2 = UInt64(0x94d049bb133111eb)
const C3 = UInt64(0xd6e8feb86659fd93)
const C4 = UInt64(0xa5a3564e27f886d3)

const TAG_INIT = 11
const TAG_MOVE_X = 23
const TAG_MOVE_Y = 29
const TAG_TRANSMIT = 37

@agent struct SIRAgent2(NoSpaceAgent)
    x::Float64
    y::Float64
    status::Symbol
    infection_time::Int
end

noop_agent_step!(agent, model) = nothing

Base.@kwdef struct SirConfig
    n::Int
    steps::Int
    seed::Int = 42
    initial_infected::Int = 5
    world_size::Float64 = 100.0
    movement_speed::Float64 = 2.0
    infection_radius::Float64 = 5.0
    transmission_rate::Float64 = 0.1
    recovery_time::Int = 14
end

function splitmix64_int(x::UInt64)::UInt64
    z = x + GOLDEN
    z = xor(z, z >> 30) * C1
    z = xor(z, z >> 27) * C2
    return xor(z, z >> 31)
end

function hash_int(seed::Int, tag::Int, step::Int, a::Int, b::Int=0)::UInt64
    x = UInt64(seed)
    x = xor(x, UInt64(tag) * C1)
    x = xor(x, UInt64(step + 1_000_003) * C2)
    x = xor(x, UInt64(a) * C3)
    x = xor(x, UInt64(b) * C4)
    return splitmix64_int(x)
end

function uniform01_int(seed::Int, tag::Int, step::Int, a::Int, b::Int=0)::Float64
    return Float64(hash_int(seed, tag, step, a, b) >> 11) * (1.0 / Float64(1 << 53))
end

function semantic_id(agent)::Int
    return Int(agent.id) - 1
end

function build_model(cfg::SirConfig)
    model = StandardABM(SIRAgent2; agent_step! = noop_agent_step!)
    for idx in 0:(cfg.n - 1)
        x = uniform01_int(cfg.seed, TAG_INIT, -1, idx, 0) * cfg.world_size
        y = uniform01_int(cfg.seed, TAG_INIT, -1, idx, 1) * cfg.world_size
        status = idx < cfg.initial_infected ? STATUS_I : STATUS_S
        add_agent!(model, x, y, status, 0)
    end
    return model
end

function ordered_agents(model)
    return sort!(collect(allagents(model)); by = agent -> semantic_id(agent))
end

function status_code(status::Symbol)::Int
    if status == STATUS_S
        return 0
    elseif status == STATUS_I
        return 1
    elseif status == STATUS_R
        return 2
    end
    error("unknown status: $status")
end

function counts(agents)::Tuple{Int, Int, Int}
    s = count(agent -> agent.status == STATUS_S, agents)
    i = count(agent -> agent.status == STATUS_I, agents)
    r = count(agent -> agent.status == STATUS_R, agents)
    return (s, i, r)
end

function snapshot(agents, cfg::SirConfig)
    x = Vector{Float64}(undef, cfg.n)
    y = Vector{Float64}(undef, cfg.n)
    status = Vector{Symbol}(undef, cfg.n)
    infection_time = Vector{Int}(undef, cfg.n)
    for agent in agents
        pos = semantic_id(agent) + 1
        x[pos] = agent.x
        y[pos] = agent.y
        status[pos] = agent.status
        infection_time[pos] = agent.infection_time
    end
    return x, y, status, infection_time
end

function assign_snapshot!(agents, x, y, status, infection_time)
    for agent in agents
        pos = semantic_id(agent) + 1
        agent.x = x[pos]
        agent.y = y[pos]
        agent.status = status[pos]
        agent.infection_time = infection_time[pos]
    end
end

function move_arrays!(cfg::SirConfig, step::Int, x::Vector{Float64}, y::Vector{Float64})
    for idx in 0:(cfg.n - 1)
        dx = (uniform01_int(cfg.seed, TAG_MOVE_X, step, idx) * 2.0 - 1.0) * cfg.movement_speed
        dy = (uniform01_int(cfg.seed, TAG_MOVE_Y, step, idx) * 2.0 - 1.0) * cfg.movement_speed
        x[idx + 1] = clamp(x[idx + 1] + dx, 0.0, cfg.world_size)
        y[idx + 1] = clamp(y[idx + 1] + dy, 0.0, cfg.world_size)
    end
end

function sir_step2_sync!(agents, cfg::SirConfig, step::Int)
    x, y, status, infection_time = snapshot(agents, cfg)
    move_arrays!(cfg, step, x, y)
    radius_sq = cfg.infection_radius ^ 2
    newly_infected = falses(cfg.n)

    for source in 0:(cfg.n - 1)
        if status[source + 1] != STATUS_I
            continue
        end
        for target in 0:(cfg.n - 1)
            if status[target + 1] != STATUS_S
                continue
            end
            dx = x[source + 1] - x[target + 1]
            dy = y[source + 1] - y[target + 1]
            if dx * dx + dy * dy <= radius_sq
                draw = uniform01_int(cfg.seed, TAG_TRANSMIT, step, source, target)
                if draw < cfg.transmission_rate
                    newly_infected[target + 1] = true
                end
            end
        end
    end

    for idx in 1:cfg.n
        if newly_infected[idx]
            status[idx] = STATUS_I
            infection_time[idx] = 0
        end
    end
    for idx in 1:cfg.n
        if status[idx] == STATUS_I
            infection_time[idx] += 1
            if infection_time[idx] >= cfg.recovery_time
                status[idx] = STATUS_R
            end
        end
    end
    assign_snapshot!(agents, x, y, status, infection_time)
end

function sir_step2_async!(agents, cfg::SirConfig, step::Int)
    x, y, _, _ = snapshot(agents, cfg)
    move_arrays!(cfg, step, x, y)
    for agent in agents
        pos = semantic_id(agent) + 1
        agent.x = x[pos]
        agent.y = y[pos]
    end

    radius_sq = cfg.infection_radius ^ 2
    for source in agents
        if source.status != STATUS_I
            continue
        end
        source_id = semantic_id(source)
        for target in agents
            target_id = semantic_id(target)
            if source_id == target_id || target.status != STATUS_S
                continue
            end
            dx = source.x - target.x
            dy = source.y - target.y
            if dx * dx + dy * dy <= radius_sq
                draw = uniform01_int(cfg.seed, TAG_TRANSMIT, step, source_id, target_id)
                if draw < cfg.transmission_rate
                    target.status = STATUS_I
                    target.infection_time = 0
                end
            end
        end
    end

    for agent in agents
        if agent.status == STATUS_I
            agent.infection_time += 1
            if agent.infection_time >= cfg.recovery_time
                agent.status = STATUS_R
            end
        end
    end
end

function run_sir_benchmark_variant(; cfg::SirConfig, schedule::String)
    model = build_model(cfg)
    agents = ordered_agents(model)
    trajectory = Tuple{Int, Int, Int}[]
    for step in 0:(cfg.steps - 1)
        if schedule == "sync"
            sir_step2_sync!(agents, cfg, step)
        elseif schedule == "async"
            sir_step2_async!(agents, cfg, step)
        else
            error("unknown schedule: $schedule")
        end
        push!(trajectory, counts(agents))
    end
    return model, trajectory
end

function outcome(model, trajectory)
    agents = ordered_agents(model)
    status_checksum = 0
    infection_time_checksum = 0
    for agent in agents
        weight = semantic_id(agent) + 1
        status_checksum += status_code(agent.status) * weight
        infection_time_checksum += agent.infection_time * weight
    end
    sir_counts = counts(agents)
    return (
        counts = sir_counts,
        status_checksum = status_checksum,
        infection_time_checksum = infection_time_checksum,
        trajectory = trajectory,
    )
end

function run_schedule(cfg::SirConfig, schedule::String)
    model, trajectory = run_sir_benchmark_variant(cfg = cfg, schedule = schedule)
    return outcome(model, trajectory)
end

function parse_args(args)
    options = Dict{String, String}()
    i = 1
    while i <= length(args)
        key = args[i]
        if startswith(key, "--") && i + 1 <= length(args)
            options[key[3:end]] = args[i + 1]
            i += 2
        else
            i += 1
        end
    end
    return options
end

function json_array(values)
    return "[" * join(string.(values), ",") * "]"
end

function trajectory_json(trajectory)
    return "[" * join(["[$(point[1]),$(point[2]),$(point[3])]" for point in trajectory], ",") * "]"
end

function print_payload(samples, result)
    result_counts = result.counts
    payload = string(
        "{",
        "\"raw_samples_s\":", json_array(samples), ",",
        "\"counts\":[", result_counts[1], ",", result_counts[2], ",", result_counts[3], "],",
        "\"status_checksum\":", result.status_checksum, ",",
        "\"infection_time_checksum\":", result.infection_time_checksum, ",",
        "\"trajectory\":", trajectory_json(result.trajectory),
        "}",
    )
    println(payload)
end

function main()
    opts = parse_args(ARGS)
    cfg = SirConfig(
        n = parse(Int, get(opts, "n", "500")),
        steps = parse(Int, get(opts, "steps", "20")),
        seed = parse(Int, get(opts, "seed", "42")),
        initial_infected = parse(Int, get(opts, "initial-infected", "5")),
        world_size = parse(Float64, get(opts, "world-size", "100.0")),
        movement_speed = parse(Float64, get(opts, "movement-speed", "2.0")),
        infection_radius = parse(Float64, get(opts, "infection-radius", "5.0")),
        transmission_rate = parse(Float64, get(opts, "transmission-rate", "0.1")),
        recovery_time = parse(Int, get(opts, "recovery-time", "14")),
    )
    schedule = get(opts, "schedule", "sync")
    runs = parse(Int, get(opts, "runs", "5"))

    run_schedule(cfg, schedule)
    samples = Float64[]
    last_result = nothing
    for _ in 1:runs
        elapsed = @elapsed begin
            last_result = run_schedule(cfg, schedule)
        end
        push!(samples, elapsed)
    end
    print_payload(samples, last_result)
end

main()
