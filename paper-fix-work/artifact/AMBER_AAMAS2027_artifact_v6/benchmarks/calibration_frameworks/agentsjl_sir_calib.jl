# Agents.jl side of the cross-framework calibration benchmark.
#
# Reads a plain-text problem file (observed curve + shared candidate (beta,gamma)
# set), runs the identical well-mixed SIR for every candidate, and reports the
# recovered parameters, wall-clock, and out-of-sample validation loss. The JIT
# is warmed before timing so the measured throughput is steady-state (matching
# how the Python frameworks are timed).
#
# Protocol (argv[1] = path):
#   line 1: n steps eval_seed
#   line 2: observed curve (steps floats)
#   line 3: validation seeds (ints)
#   line 4: gt_beta gt_gamma
#   line 5: beta_lo beta_hi gamma_lo gamma_hi
#   line 6+: one "beta gamma" candidate per line
# Output (stdout, final line):
#   beta gamma best_loss n_evals wall_s val_loss recovery_error

using Agents
using Random
using Statistics

@agent SIRAgent NoSpaceAgent begin
    status::Int
end

function sir_curve(beta, gamma, n, steps, seed)
    model = ABM(SIRAgent; rng = MersenneTwister(seed))
    k = max(1, floor(Int, 0.02 * n))
    for i in 1:n
        add_agent!(model, i <= k ? 1 : 0)
    end
    rng = model.rng
    curve = zeros(Float64, steps)
    for t in 1:steps
        agents = collect(allagents(model))
        i_frac = count(a -> a.status == 1, agents) / n
        foi = beta * i_frac
        for a in agents
            if a.status == 0
                if rand(rng) < foi
                    a.status = 1
                end
            elseif a.status == 1
                if rand(rng) < gamma
                    a.status = 2
                end
            end
        end
        curve[t] = count(a -> a.status == 1, allagents(model)) / n
    end
    return curve
end

function main()
    lines = readlines(ARGS[1])
    hdr = parse.(Float64, split(lines[1]))
    n = Int(hdr[1]); steps = Int(hdr[2]); eval_seed = Int(hdr[3])
    observed = parse.(Float64, split(lines[2]))
    val_seeds = parse.(Int, split(lines[3]))
    gt = parse.(Float64, split(lines[4]))            # gt_beta gt_gamma
    bnd = parse.(Float64, split(lines[5]))           # b_lo b_hi g_lo g_hi
    candidates = [parse.(Float64, split(l)) for l in lines[6:end] if !isempty(strip(l))]

    sse(curve) = sum((curve .- observed) .^ 2)
    sir_curve(0.3, 0.1, n, steps, eval_seed)         # JIT warmup (not timed)

    t0 = time()
    best_loss = Inf; best = candidates[1]
    for c in candidates
        l = sse(sir_curve(c[1], c[2], n, steps, eval_seed))
        if l < best_loss
            best_loss = l; best = c
        end
    end
    wall = time() - t0

    val = mean([sse(sir_curve(best[1], best[2], n, steps, s)) for s in val_seeds])
    rec = sqrt(mean([((best[1] - gt[1]) / (bnd[2] - bnd[1]))^2,
                     ((best[2] - gt[2]) / (bnd[4] - bnd[3]))^2]))
    println(join([best[1], best[2], best_loss, length(candidates), wall, val, rec], " "))
end

main()
