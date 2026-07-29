# Agents.jl dynamic graph coordination benchmark for AMBER.
#
# Mirrors benchmarks/run_dynamic_graph_variants.py: deterministic initial
# opinions, deterministic step-varying sparse edges, and synchronous
# bounded-confidence opinion updates.

using Agents
using Statistics

const MASK64 = typemax(UInt64)
const GOLDEN = UInt64(0x9e3779b97f4a7c15)
const C1 = UInt64(0xbf58476d1ce4e5b9)
const C2 = UInt64(0x94d049bb133111eb)
const C3 = UInt64(0xd6e8feb86659fd93)
const C4 = UInt64(0xa5a3564e27f886d3)

const TAG_INIT = 101
const TAG_RANDOM_EDGE = 109
const TAG_DYNAMIC_EDGE = 211

@agent struct DynamicGraphAgent(NoSpaceAgent)
    opinion::Float64
end

noop_agent_step!(agent, model) = nothing

Base.@kwdef struct GraphConfig
    n::Int
    steps::Int
    seed::Int = 42
    degree::Int = 8
    confidence::Float64 = 0.18
    alpha::Float64 = 0.45
end

function splitmix64_int(x::UInt64)::UInt64
    z = x + GOLDEN
    z = xor(z, z >> 30) * C1
    z = xor(z, z >> 27) * C2
    return xor(z, z >> 31)
end

function mix5(a::UInt64, b::UInt64, c::UInt64, d::UInt64, e::UInt64)::UInt64
    return xor(xor(xor(xor(a, b), c), d), e)
end

function mix6(a::UInt64, b::UInt64, c::UInt64, d::UInt64, e::UInt64, f::UInt64)::UInt64
    return xor(mix5(a, b, c, d, e), f)
end

function uniform01_int(seed::Int, tag::Int, step::Int, a::Int, b::Int = 0)::Float64
    x = mix5(
        UInt64(seed),
        UInt64(tag) * C1,
        UInt64(step + 1_000_003) * C2,
        UInt64(a) * C3,
        UInt64(b) * C4,
    )
    return Float64(splitmix64_int(x) >> 11) * (1.0 / Float64(1 << 53))
end

function initial_opinions(cfg::GraphConfig)::Vector{Float64}
    return [uniform01_int(cfg.seed, TAG_INIT, -1, idx) for idx in 0:(cfg.n - 1)]
end

function validate_degree(cfg::GraphConfig)::Int
    cfg.n >= 2 || error("n must be at least 2")
    (cfg.degree >= 2 && iseven(cfg.degree)) || error("degree must be an even integer >= 2")
    degree = min(cfg.degree, cfg.n - 1)
    if isodd(degree)
        degree -= 1
    end
    degree >= 2 || error("effective degree must be at least 2")
    return degree
end

function dynamic_neighbors(cfg::GraphConfig, step::Int)::Vector{Vector{Int}}
    degree = validate_degree(cfg)
    neighbors = [Int[] for _ in 1:cfg.n]
    step_part = UInt64(step + 1_000_003) * C4
    for src in 0:(cfg.n - 1)
        chosen = Set{Int}()
        salt = 0
        while length(chosen) < degree
            raw = splitmix64_int(
                mix6(
                    UInt64(cfg.seed),
                    UInt64(TAG_DYNAMIC_EDGE) * C1,
                    UInt64(TAG_RANDOM_EDGE) * C2,
                    UInt64(src) * C3,
                    step_part,
                    UInt64(salt) * C4,
                ),
            )
            dst = Int(raw % UInt64(cfg.n))
            salt += 1
            if dst != src
                push!(chosen, dst)
            end
        end
        for dst in sort(collect(chosen))
            push!(neighbors[src + 1], dst + 1)
        end
    end
    return neighbors
end

function semantic_id(agent)::Int
    return Int(agent.id) - 1
end

function build_model(cfg::GraphConfig)
    model = StandardABM(DynamicGraphAgent; agent_step! = noop_agent_step!)
    for opinion in initial_opinions(cfg)
        add_agent!(model, opinion)
    end
    return model
end

function ordered_agents(model)
    return sort!(collect(allagents(model)); by = agent -> semantic_id(agent))
end

function snapshot_opinions(agents, cfg::GraphConfig)::Vector{Float64}
    opinions = Vector{Float64}(undef, cfg.n)
    for agent in agents
        opinions[semantic_id(agent) + 1] = agent.opinion
    end
    return opinions
end

function assign_opinions!(agents, opinions::Vector{Float64})
    for agent in agents
        agent.opinion = opinions[semantic_id(agent) + 1]
    end
end

function sync_step!(agents, cfg::GraphConfig, step::Int)::Int
    opinions = snapshot_opinions(agents, cfg)
    next_opinions = copy(opinions)
    active_edges = 0
    neighbors = dynamic_neighbors(cfg, step)

    for i in 1:cfg.n
        accepted_sum = 0.0
        accepted_count = 0
        oi = opinions[i]
        for j in neighbors[i]
            oj = opinions[j]
            if abs(oj - oi) <= cfg.confidence
                accepted_sum += oj
                accepted_count += 1
            end
        end
        active_edges += accepted_count
        if accepted_count > 0
            neighbor_mean = accepted_sum / accepted_count
            next_opinions[i] = oi + cfg.alpha * (neighbor_mean - oi)
        end
    end

    assign_opinions!(agents, next_opinions)
    return active_edges
end

function run_graph(cfg::GraphConfig)
    model = build_model(cfg)
    agents = ordered_agents(model)
    active_edges_last = 0
    for step in 0:(cfg.steps - 1)
        active_edges_last = sync_step!(agents, cfg, step)
    end
    opinions = snapshot_opinions(agents, cfg)
    weights = collect(1:cfg.n)
    checksum = sum(opinions .* weights)
    return (
        opinions = opinions,
        mean = mean(opinions),
        std = stdm(opinions, mean(opinions); corrected = false),
        min_opinion = minimum(opinions),
        max_opinion = maximum(opinions),
        active_edges_last = active_edges_last,
        checksum = checksum,
    )
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

function print_payload(samples, result)
    payload = string(
        "{",
        "\"raw_samples_s\":", json_array(samples), ",",
        "\"opinions\":", json_array(result.opinions), ",",
        "\"mean\":", result.mean, ",",
        "\"std\":", result.std, ",",
        "\"min_opinion\":", result.min_opinion, ",",
        "\"max_opinion\":", result.max_opinion, ",",
        "\"active_edges_last\":", result.active_edges_last, ",",
        "\"checksum\":", result.checksum,
        "}",
    )
    println(payload)
end

function main()
    opts = parse_args(ARGS)
    cfg = GraphConfig(
        n = parse(Int, get(opts, "n", "500")),
        steps = parse(Int, get(opts, "steps", "20")),
        seed = parse(Int, get(opts, "seed", "42")),
        degree = parse(Int, get(opts, "degree", "8")),
        confidence = parse(Float64, get(opts, "confidence", "0.18")),
        alpha = parse(Float64, get(opts, "alpha", "0.45")),
    )
    runs = parse(Int, get(opts, "runs", "5"))

    run_graph(cfg)
    samples = Float64[]
    last_result = nothing
    for _ in 1:runs
        t0 = time_ns()
        last_result = run_graph(cfg)
        push!(samples, (time_ns() - t0) / 1.0e9)
    end
    print_payload(samples, last_result)
end

main()
