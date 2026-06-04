# AMBER vs AgentPy vs Mesa Performance Benchmark

This directory contains performance benchmarks comparing AMBER against **AgentPy** and **Mesa**.

AMBER ships **two** benchmark implementations so you can see what the
columnar backend actually buys you:

* **`AMBER`** — per-agent Python loops over a hand-rolled `agent_objects_list`.
  This is the apples-to-apples comparison with AgentPy / Mesa: all three
  frameworks run the same OOP-style code, and the numbers show how their
  *naive* code paths compare.
* **`AMBER (vectorized)`** — the same models written with AMBER's view API
  (`add_agents(n, **cols)`, `agents.where(...)`, `agents.at[ids].scatter_add(...)`).
  This is the idiom the library now ships and the docs teach. It pays
  the Polars overhead per step but eliminates per-agent Python object
  work, so the step cost is nearly independent of population size.

## Quick Start

```bash
# Install benchmark dependencies
pip install -r requirements.txt

# Quick run (small scale)
python runner.py --quick

# Full run across 100 / 500 / 1k / 5k / 10k agents
python runner.py --full

# Compare just AMBER variants vs AgentPy
python runner.py --frameworks AMBER "AMBER (vectorized)" AgentPy \
    --agents 500 1000 5000 --steps 50 --runs 10
```

## Metrics Measured

| Metric | Description |
|--------|-------------|
| **Execution Time** | Wall-clock time for complete simulation |
| **Memory Usage** | Peak memory consumption (MB) |
| **Time per Step** | Average time per simulation step |
| **Scaling Factor** | Performance change ratio vs. agent count |

## Models Compared

1. **Wealth Transfer** — Boltzmann wealth distribution model
2. **SIR Epidemic** — Spatial disease spread model
3. **Random Walk** — Basic 2D random walk

## Correctness audit

Runtime numbers are meaningless if two frameworks are computing different
things. Before reporting any timings, every framework in this repo is run
through [`correctness_check.py`](correctness_check.py), which executes each
model at a fixed `(n=500, steps=50, initial_infected=5)` configuration and
checks observable invariants:

* **wealth_transfer** — total wealth must equal `n × initial_wealth` (strict
  Boltzmann conservation). All seven frameworks report `total=500`, `mean=1.00`.
* **random_walk** — all agent positions must be inside `[0, world_size]`.
  All seven frameworks report `out_of_bounds = 0`.
* **sir_epidemic** — `S + I + R` must equal `n`, and initial `I` must equal
  `initial_infected`. All seven frameworks report `total=500`.

Three correctness bugs in the checked-in **SimPy** benchmarks were found and
fixed during this audit before timing was repeated:

1. **`wealth_transfer` race condition.** The original SimPy `wealth_agent`
   kept a local `wealth` counter and only ever decremented it, silently
   discarding all gifts it received from other agents. Over 50 steps, 47%
   of the population's wealth simply disappeared — total went from 500 to
   265. Fixed by reading the shared dict each iteration.
2. **`random_walk` missing boundary clamp.** The SimPy walker never clamped
   its position, so agents drifted outside `[0, 100]`. 47/500 agents were
   out of bounds after 50 steps. Fixed by adding the clamp every other
   framework uses.
3. **`sir_epidemic` 100 % initially infected.** A hardcoded
   `status = 1 # Force 100% infected for Dense Benchmark` line meant every
   SimPy agent started infected — the SIR dynamics were reduced to "wait 14
   steps for everyone to recover" with no actual infection spread. Fixed
   to match the other frameworks' `initial_infected = 5` semantics.

A fourth issue was a **step-count mismatch for Agents.jl**: the Julia
script had `steps=100` hardcoded into its `run_benchmarks` call, meaning
every Agents.jl timing reported in older runs was for twice the work the
Python frameworks were doing. The script now accepts `--steps` and
`--agents` on the command line and the master runner passes them through.

A later audit also aligned the **random-walk and SIR movement kernel**:
SimPy, Melodie, and Agents.jl now use the same independent x/y displacement
as AMBER, AgentPy, and Mesa. Agents.jl now also uses the same run count and
slowest-sample trim protocol as the Python frameworks.

After these fixes, every framework passes the structural invariants used for
timing admission. The SIR benchmark still has a documented update-ordering
caveat: AMBER (vectorized) and Melodie use a synchronous infection phase,
while the other implementations use sequential/asynchronous infection. Treat
the SIR timings as a comparison of the same spatial epidemic workload class,
not as proof of identical stochastic trajectories.

## Latest verified-correct results — all seven frameworks

Run on 2026-06-04, Python 3.12.7, Julia 1.12.3, 50 executed steps per
simulation, seeded runs, 10 runs averaged (slowest trimmed). Apple Silicon.

**Execution time — Wealth Transfer**

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| Agents.jl | **0.5 ms** | **1.3 ms** | **7.2 ms** |
| AMBER (vectorized) | 4.2 ms | 6.0 ms | 20 ms |
| AMBER (loop) | 16 ms | 32 ms | 169 ms |
| SimPy | 18 ms | 37 ms | 216 ms |
| Melodie | 18 ms | 36 ms | 177 ms |
| AgentPy | 26 ms | 51 ms | 266 ms |
| Mesa | 254 ms | 969 ms | 22.61 s |

**Execution time — Random Walk**

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| Agents.jl | **0.2 ms** | **0.3 ms** | **1.6 ms** |
| AMBER (vectorized) | 2.4 ms | 2.7 ms | 4.8 ms |
| Mesa | 13 ms | 25 ms | 131 ms |
| AgentPy | 14 ms | 28 ms | 141 ms |
| SimPy | 22 ms | 45 ms | 254 ms |
| AMBER (loop) | 33 ms | 66 ms | 332 ms |
| Melodie | 101 ms | 205 ms | 1.03 s |

**Execution time — SIR Epidemic**

| Framework | 500 | 1000 | 5000 |
|---|---|---|---|
| Agents.jl | **4.2 ms** | **37 ms** | 813 ms |
| AMBER (vectorized) | 88 ms | 112 ms | **497 ms** |
| SimPy | 107 ms | 411 ms | 4.67 s |
| AgentPy | 197 ms | 826 ms | 10.98 s |
| AMBER (loop) | 140 ms | 799 ms | 9.53 s |
| Mesa | 265 ms | 1.07 s | 16.63 s |
| Melodie | 595 ms | 1.98 s | 20.09 s |

**Ratio of each framework's time to `AMBER (vectorized)`, averaged across
all three agent counts above (lower means that framework is closer to
AMBER vectorized; values < 1 mean it beats AMBER vectorized):**

| Framework | Wealth Transfer | Random Walk | SIR Epidemic |
|---|---|---|---|
| Agents.jl | 0.2× (faster) | 0.2× (faster) | 0.7× (faster on average) |
| AMBER (loop) | 5.9× | 35.7× | 9.3× |
| SimPy | 7.1× | 26.2× | 4.8× |
| Melodie | 6.3× | 110.6× | 21.6× |
| AgentPy | 9.3× | 15.3× | 10.6× |
| Mesa | 450.9× | 14.0× | 15.4× |

See [`results/scaling_chart_all.png`](results/scaling_chart_all.png) for the
log-log scaling plot (the wider the gap at the right edge of each subplot,
the better AMBER scales).

![All-framework scaling](results/scaling_chart_all.png)

## Interpreting the numbers

**Who wins each model at the 5000-agent point (the realistic ABM scale):**

* **Wealth transfer**: Agents.jl wins (7.2 ms), AMBER (vectorized) is the
  fastest Python-hosted implementation (20 ms).
  Julia's JIT compiler wins the microbenchmark because the per-step work
  is so small (two array updates) that Polars' per-expression overhead
  dominates. Every other Python-hosted implementation is roughly 9× to
  1130× slower.
* **Random walk**: Agents.jl wins (1.6 ms), AMBER (vectorized) is the
  fastest Python-hosted implementation (4.8 ms).
* **SIR epidemic**: AMBER (vectorized) is the fastest Python-hosted row in the
  mixed headline grid.  The SIR rows do not all use the same update schedule,
  so use this row as workload-class timing only; use the split sync/async SIR
  runner below for semantics-aligned evidence.

**Overall**: AMBER (vectorized) is the fastest Python-hosted framework on
every headline model at every tested scale. Agents.jl wins the cheaper wealth
and random-walk microbenchmarks. The SIR headline remains schedule-mixed, so it
should not be read as an equivalent-trajectory AMBER-over-Julia claim.

**Sync vs async SIR update semantics.** AMBER (vectorized) uses a
**synchronous** update step for the infection phase — it snapshots the
infected/susceptible sets at the start of the step, then applies all
infection events simultaneously. The other frameworks (including AMBER
loop) use **asynchronous / sequential** update — a newly infected agent
can already infect its own neighbours later in the same step. Both are
valid SIR discretizations; the convention difference is why the correctness
audit checks population conservation and parameter routing rather than
claiming identical trajectories.

## Reproducing these numbers

**One-command master run** (produces JSON, Markdown, and the log-log chart):

```bash
python benchmarks/run_all_frameworks.py --agents 500 1000 5000 --steps 50 --runs 10
```

**Outputs:**

- `results/benchmark_results_all.json` — raw per-(framework, model, n) timings
- `results/summary_table_all.md` — the table above
- `results/scaling_chart_all.png` — the log-log chart above

**Dependencies:**

```bash
# Python (use a 3.10+ interpreter so Mesa 3.x is available)
pip install polars numpy agentpy "mesa>=3.0" simpy Melodie matplotlib tabulate

# Julia (for the Agents.jl row only)
brew install julia                   # or your OS's equivalent
julia -e 'using Pkg; Pkg.add("Agents")'
```

If Julia isn't on `PATH`, `run_all_frameworks.py` will skip the Agents.jl
row and still produce a six-framework comparison.

## Semantics-aligned SIR and dynamic graph runners

The headline SIR benchmark intentionally preserves the checked-in framework
implementations, so it mixes synchronous, sequential, and event-scheduled SIR
conventions. The split runner below fixes initial positions, movement deltas,
and pair-level transmission draws, then reports sync and async rows separately:

```bash
python benchmarks/run_sir_schedule_variants.py --agents 500 1000 --steps 50 --runs 10 \
    --budget-skip-modes async_simpy_refactored \
    --budget-skip-reason "declared run-budget exclusion"
python benchmarks/run_sir_schedule_variants.py --agents 500 1000 5000 --steps 50 --runs 10 \
    --modes agentsjl_actual_source_sync agentsjl_actual_source_async --resume-existing
```

Outputs land in `results/sir_schedule_results.json` and
`results/sir_schedule_results.md`. The current artifact records 42 measured
CI-grade rows, three explicit async SimPy budget skips, and zero reference
mismatches.

The dynamic graph runner adds a bounded-confidence opinion workload with a
deterministic sparse edge relation regenerated every step:

```bash
python benchmarks/run_dynamic_graph_variants.py --agents 500 1000 5000 \
    --seeds 42 77 123 --steps 20 --runs 5
```

Outputs land in `results/dynamic_graph_results.json` and
`results/dynamic_graph_results.md`. The current artifact records 63 measured
rows across NumPy, Polars, a Python object loop, AMBER, Mesa, AgentPy, and
Agents.jl, with zero reference mismatches.

## Three-framework registry runner

The legacy `runner.py` registers only AMBER / AgentPy / Mesa and is still
useful when you want to iterate on those three specifically:

```bash
python benchmarks/runner.py --frameworks AMBER "AMBER (vectorized)" AgentPy Mesa \
    --agents 500 1000 5000 --steps 50 --runs 10
```

Outputs land in `results/benchmark_results.json` / `summary_table.md` /
`scaling_chart.png` — the older three-framework filenames — so they don't
overwrite the seven-framework artefacts produced by `run_all_frameworks.py`.

## Architecture

```
benchmarks/
├── models/
│   ├── amber_models.py     # AMBER implementations (per-agent + vectorized)
│   ├── agentpy_models.py   # AgentPy implementations
│   └── mesa_models.py      # Mesa implementations
├── runner.py               # Benchmark runner
├── requirements.txt        # Dependencies
└── results/                # Output folder
```
