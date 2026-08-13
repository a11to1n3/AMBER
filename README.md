# AMBER (Agent-based Modeling with Blazingly Efficient Records)

[![CI](https://github.com/a11to1n3/AMBER/actions/workflows/ci.yml/badge.svg)](https://github.com/a11to1n3/AMBER/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/a11to1n3/AMBER/graph/badge.svg)](https://codecov.io/gh/a11to1n3/AMBER)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/ambr.svg)](https://pypi.org/project/ambr/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3_Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

AMBER is a Python framework for agent-based modeling that uses Polars for
efficient columnar population state. It provides an AgentPy-shaped OOP lane and
a high-performance vectorized lane (optional GPU on NVIDIA+CuPy). Multi-run
parallelism is **opt-in** (`Experiment` / `ParallelRunner` / `GPUEnsembleRunner`),
not automatic from a single `.run()`.

## 🚀 Performance

AMBER stores the entire population as a columnar Polars DataFrame and
exposes a vectorized view API (`agents.where(...)`, `agents.at[ids]`,
`scatter_add`) that compiles per-step updates down to a handful of
columnar operations. Models can provide `step_vectorized()` and `step_oop()`
for explicit native lanes; legacy models with only `step()` keep the fallback.
The vectorized lane runs on GPU via `model.gpu().run()`.

### GPU requirements (read this first)

| Claim | Requires |
|-------|----------|
| CPU vectorized (default fast path) | `pip install ambr` (+ `ambr[perf]` / Numba recommended) |
| Headline **AMBER (GPU) vs FLAME** table below | **NVIDIA GPU + CuPy** (`pip install 'ambr[gpu]'` or a CUDA-matched wheel) |
| Apple Metal / MPS | **Not supported** — AMBER does not use Mac GPU |

Check the machine before quoting GPU numbers:

```python
import ambr as am
am.print_status()          # must show GPU: yes for FLAME-class claims
print(am.recommend(1_000_000))
```

Default GitHub CI has **no CUDA**. Re-verify GPU claim paths on a CUDA host with:

```bash
python scripts/run_gpu_claims.py          # full (incl. large-N)
python scripts/run_gpu_claims.py --quick  # fast smoke
```

### Headline comparison (committed evidence)

**Single source of truth for the README table (do not mix other tables):**  
[`benchmarks/results/benchmark_results_snapshot_correct_10run_10m.json`](https://github.com/a11to1n3/AMBER/blob/main/benchmarks/results/benchmark_results_snapshot_correct_10run_10m.json)  
(summary:
[`summary_table_snapshot_correct_10run_10m.md`](https://github.com/a11to1n3/AMBER/blob/main/benchmarks/results/summary_table_snapshot_correct_10run_10m.md)).

Other files under [`benchmarks/results/`](https://github.com/a11to1n3/AMBER/blob/main/benchmarks/results/) (including
`summary_table.md`) are **exploratory / historical** and may use different
protocols — never combine them with the headline table without an explicit
caption.

**Protocol:** NVIDIA RTX 5090; 10M agents; 50 steps; **10 runs** retained
(no outlier deletion); one untimed warm-up per cell; timed scope =
construct + setup + step loop + result assembly. AMBER GPU synchronizes
after the run; FLAME GPU 2 returns at a simulation-complete boundary.
This is an **implementation comparison** under that protocol — not a claim
of byte-identical dynamics across frameworks.

**AMBER (GPU) vs FLAME GPU 2 at 10M agents** (mean wall-clock):

| Model | AMBER (GPU) | FLAME GPU 2 | Speedup (FLAME / AMBER) |
|---|---:|---:|---:|
| Wealth | **94 ms** | 194 ms | ~2.05× |
| Random walk | **80 ms** | 161 ms | ~2.00× |
| SIR (cell-list) | **2.08 s** | 3.68 s | ~1.77× |
| Schelling | **295 ms** | 18.7 s | ~63× (setup-inclusive; **exploratory**) |

- **Wealth / walk / SIR** are the comparable headline class (~1.8–2.1×).
- **Schelling** includes heavy Python-side setup inside the timed region for
  the FLAME harness; do not treat ~63× as a pure step-kernel speedup.
- Multi-framework scale-out charts (Mesa, mesa-frames, Agents.jl, …) are
  **exploratory** and live under
  [`benchmarks/results/`](https://github.com/a11to1n3/AMBER/blob/main/benchmarks/results/); some cells OOM or hit
  budgets — missing cells are not zeros.
- Reproducer: [`benchmarks/run_all_frameworks.py`](https://github.com/a11to1n3/AMBER/blob/main/benchmarks/run_all_frameworks.py).  
  Correctness gates: [`benchmarks/correctness_check.py`](https://github.com/a11to1n3/AMBER/blob/main/benchmarks/correctness_check.py).  
  Details: [`benchmarks/README.md`](https://github.com/a11to1n3/AMBER/blob/main/benchmarks/README.md).

![Framework scaling chart](https://raw.githubusercontent.com/a11to1n3/AMBER/main/benchmarks/results/scaling_chart.png)

**API:** implement `step_vectorized()` (or legacy `step()`); place with
`.cpu(mode="vectorized")` or `.gpu()`. GPU is vectorized-only. Private
optimized GPU loops require an explicit
`approve_fast_path(evidence)` label (caller-attested provenance; AMBER
checks presence of the label, not the evidence content) and
`contract="off"` — see [`docs/going_faster.rst`](https://ambr.readthedocs.io/en/latest/going_faster.html).

Default CI has no CUDA. Optional **GPU claims** workflow
(`.github/workflows/gpu-nightly.yml`) **hard-fails as NOT VERIFIED** without
a GPU (never soft-green) and runs `scripts/run_gpu_claims.py --quick` on
self-hosted CUDA runners (`GPU_RUNNER` repository variable).

## 🚀 Quick Start

AMBER supports an **AgentPy-shaped OOP lane** and a **vectorized lane** on the
same model. Start with whichever feels natural.

**AgentPy-shaped (method broadcast, `AgentList`):**

```python
import ambr as am

class WealthAgent(am.Agent):
    def setup(self):
        self.wealth = 1
    def transfer(self):
        if self.wealth > 0:
            other = self.model.agents.by_id(self.model.agents.random())
            other.wealth += 1
            self.wealth -= 1

class WealthModel(am.Model):
    def setup(self):
        self.agents = am.AgentList(self, self.p.n, WealthAgent)
    def step_oop(self):
        # Explicit OOP lane (default mode is vectorized)
        self.agents.transfer()
    def update(self):
        self.record_model('total', int(self.agents.wealth.sum()))

results = WealthModel({'n': 50, 'steps': 20, 'seed': 1}).cpu(mode="oop").run()
print(results.model.tail(3).to_dicts())      # also results['model']
print(results.agents.head().to_dicts())
print(results.info.get("mode"), results.info.get("execution_lane"))
```

**Vectorized (columnar; best at large N):**

```python
import ambr as am

class WealthModel(am.Model):
    model_reporters = {'total_wealth': lambda m: int(m.agents.wealth.sum())}

    def setup(self):
        self.add_agents(100, wealth=self.rng.integers(1, 10, size=100))

    def step_vectorized(self):
        # View API: where / column assign / scatter_add (not agents.array mutate).
        # agents.array(...) returns a read-only snapshot on the CPU Polars path.
        donors = self.agents.where(self.agents.wealth > 0)
        donors.wealth -= 1
        recipients = self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
        self.agents.at[recipients].scatter_add(wealth=1)

# Fluent placement: device + optional mode; run(mode=...) still overrides.
# GPU when NVIDIA+CuPy available; otherwise CPU vectorized.
_m = WealthModel({'steps': 100, 'seed': 42, 'show_progress': False})
results = _m.gpu().run() if am.GPU_AVAILABLE else _m.cpu(mode="vectorized").run()
print(results.info)
print(results.model.tail(5).to_dicts())
print(results.agents.head(10).to_dicts())
```

Coming from AgentPy? See [`docs/from_agentpy.rst`](https://ambr.readthedocs.io/en/latest/from_agentpy.html).

**Going faster / GPU** — same `Model` class; pick placement and lane hooks (see
[`docs/going_faster.rst`](https://ambr.readthedocs.io/en/latest/going_faster.html)):

```python
import ambr as am
am.print_status()                 # GPU? which lane?
print(am.recommend(1_000_000))  # one-line suggestion

# Same view-API model shape as the vectorized quickstart above
class WealthModel(am.Model):
    def setup(self):
        n = int(self.p.get("n", 100_000))
        self.add_agents(n, wealth=1)
    def step_vectorized(self):
        donors = self.agents.where(self.agents.wealth > 0)
        donors.wealth -= 1
        self.agents.at[
            self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
        ].scatter_add(wealth=1)

# Native path: step_vectorized + .gpu() when NVIDIA+CuPy are available
model = WealthModel({"n": 100_000, "steps": 50, "seed": 0, "show_progress": False})
if am.GPU_AVAILABLE:
    results = model.gpu().run()
else:
    results = model.cpu(mode="vectorized").run()  # Mac / no-CUDA fallback
print(results.info)

# Array-kernel lane (CuPy if available, else NumPy) for pure array state:
class Drift(am.ArrayKernelModel):
    def init_state(self, xp, n, rng, p):
        return {"x": rng.random(n, dtype=xp.float32)}
    def step_state(self, xp, state, rng, p):
        state["x"] = state["x"] + 0.01
        return state
    def metrics(self, xp, state):
        return {"mean_x": float(am.to_host(state["x"].mean()))}

print(Drift({"n": 100_000, "steps": 20, "show_progress": False}).run().info)
```

`self.rng` is the canonical seeded RNG (a NumPy `Generator`); `self.random` is
the stdlib one. Both are seeded from the `seed` parameter. Progress printing is
off by default (`show_progress=True` to re-enable).

> **New in 0.5.0:** production-candidate. **Breaking / install:** matplotlib
> moved to `ambr[viz]`; scikit-optimize was removed and replaced by SMAC
> under `ambr[advanced]` (SMAC + ConfigSpace + scikit-learn). Python
> **3.10–3.13** only. **API:** `record_model` inside `step()` is kept;
> missing optimization metrics raise; SMAC defaults to `on_error='raise'`;
> `RunResults.save` uses a versioned manifest. Upgrade with
> `pip install -U 'ambr>=0.5.0'`. Full list:
> [CHANGELOG.md](https://github.com/a11to1n3/AMBER/blob/main/CHANGELOG.md).
>
> **0.4.7:** GPU claim script renamed to
> [`scripts/run_gpu_claims.py`](https://github.com/a11to1n3/AMBER/blob/main/scripts/run_gpu_claims.py);
> paper-campaign machine labels removed from user-facing docs.

## ⚡ Vectorized View API

The view API compiles per-step updates to a handful of Polars expressions
— regardless of population size:

```python
def step(self):
    # Bulk columnar reads/writes over the entire population
    self.agents.x = self.agents.x + self.rng.uniform(-1, 1, len(self.agents))

    # Filtered writes: only agents matching a condition
    infected = self.agents.where(self.agents.status == 1)
    infected.infection_time += 1

    # scatter_add: flow-of-resources with duplicate-id safety
    self.agents.at[[1, 1, 3]].scatter_add(wealth=1)  # agent 1 gets +2, agent 3 gets +1
```

## Canonical API (0.5)

AMBER 0.5 settles on one obvious verb per task. The legacy spellings still work
(they emit a `DeprecationWarning` and are scheduled for removal in **1.0**); set
`AMBER_SUPPRESS_DEPRECATIONS=1` to silence them in benchmark / reproducibility runs.
Batch performance comes from these verbs (columnar writes), not from extra public
`batch_*` helpers.

| Task | Canonical | Legacy (deprecated → 1.0) |
|------|-----------|---------------------------|
| NumPy RNG | `self.rng` | `self.nprandom` |
| Device / run mode | `model.cpu(mode=...).run()` / `model.gpu().run()` | `run(backend=...)` |
| Record a model metric | `model_reporters = {...}` or `record_model(k, v)` | `record(k, v)` |
| Filter agents | `agents.where(expr)` / `agents[mask]` / `agents.at[ids]` | `agents.select(...)` |
| Per-agent write | `agent.col = v` | `agent.record(...)`, `agent.update_data(...)`, `update_agent_data` |
| Bulk / multi-column write | `agents.set(**cols)` or `view.col = …` | `agents.record` / `update_data`, `batch_update_agents`, `Population.batch_*` |
| Accumulate (duplicate ids) | `agents.at[ids].scatter_add(...)` | double ordinary writes in one step |
| Array kernels | `agents.borrow` / `agents.commit` (or `TensorLane`) | hand-maintained parallel NumPy buffers |
| Read agent objects | iterate `model.agents`, `agents.by_id(i)` | `agents.agents`, `agents.agent_ids` |
| Bulk numpy round-trip | `agents.numpy(...)` + `agents.set(...)` | `.to_numpy()` + per-column assign only |
| Typed parameters | `params = {'n': (int, 200)}`, then `self.p.n` | `int(self.p.get('n', 200))` |
| Grid wrap | `GridEnvironment(torus=True)` | `wrap=` / `.wrap` |
| Agent table assign | view / `_set_frame` | `population.data = ...` (setter warns) |

`update()` is a **pure hook** — overriding it no longer requires
`super().update()`. Declare `model_reporters` / `agent_reporters` for
declarative metrics, and set `record_initial = True` to capture a `t=0` row.

## 🔒 Snapshot-view contract

**What this is:** an **operational runtime monitor** at instrumented read/write
seams (OOP buffer vs view/lane, mutable borrows, cross-path writes).

**What this is not:** a proof that a vectorized or GPU rewrite preserves an
intended activation schedule, confluence, or bit-identical trajectories vs an
OOP loop. `cert.clean` means *no monitored hazard was observed*, not that every
possible ordering is equivalent.

Whether a vectorized refactor preserves an intended update schedule remains a
semantic question for the modeller. Run with a contract mode and inspect the
per-step records:

```python
import ambr as am

class WealthModel(am.Model):
    def setup(self):
        self.add_agents(50, wealth=1)
    def step_vectorized(self):
        donors = self.agents.where(self.agents.wealth > 0)
        donors.wealth -= 1
        self.agents.at[self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))].scatter_add(wealth=1)

model = WealthModel({"steps": 20, "seed": 0, "show_progress": False})
results = model.run(steps=20, contract="check")   # "off" | "check" | "warn" | "raise"
for cert in results["contract"]:
    if not cert.ok:
        print(cert.step, cert.violations)
```

`check` records a `ContractCertificate` per step; `warn` also emits a warning per
violation; `raise` stops on the first error. Mode `off` (default) adds no monitor
bookkeeping. `cert.clean` means that no monitored error or warning was observed,
not that every possible activation order is equivalent.

The monitor watches **two write paths** (and combinations):

* **Buffered (OOP)** — `agent.col = …` / queued cell writes
* **Lane / view** — `agents.col = …`, `agents.set(...)`, `borrow`/`commit`
* **Cross-path** — same column via both OOP and view in one step → `cross_path_write`
* **Mutable raw arrays** — `agents.array(...)` → `uncertified_mutable_borrow`

`scatter_add` is the sanctioned multi-write reducer (not treated as a conflicting
ordinary commit). Prefer those APIs over assigning `population.data` directly.

## 🎮 GPU backend & batched calibration

**Single-run (native):** place a vectorized model on device with
`model.gpu().run()` when NVIDIA+CuPy are available. Prefer `step_vectorized()`
(legacy `step()` still works).
Numeric columns stay device-resident for the run. Contract modes use the
instrumented general path; private model-specific fast loops run only with
`contract="off"` **and** an explicit per-instance
`approve_fast_path(evidence)` declaration. The evidence string is a
caller-supplied provenance label, not something AMBER verifies. Without it,
`gpu().run()` uses the general path. Private loops are not covered by the
monitor. OOP agents use `cpu(mode="oop")` — not GPU.

```python
import ambr as am

class WealthModel(am.Model):
    def setup(self):
        n = int(self.p.get("n", 1_000_000))
        self.add_agents(n, wealth=1)
    def step_vectorized(self):
        donors = self.agents.where(self.agents.wealth > 0)
        donors.wealth -= 1
        self.agents.at[
            self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
        ].scatter_add(wealth=1)

# Requires NVIDIA + CuPy. On CPU-only machines use .cpu(mode="vectorized").
model = WealthModel({"n": 1_000_000, "steps": 50, "seed": 0, "show_progress": False})
results = model.gpu().run() if am.GPU_AVAILABLE else model.cpu(mode="vectorized").run()
print(results.info)
# Private fast loop (only if the model defines one; evidence is not verified):
# model.approve_fast_path("my-bench-label").gpu().run(contract="off")
```

**Many short runs (calibration):** the *ensemble* axis (`B` simulations × `N`
agents) batches into one device pass — the natural fit when you evaluate
thousands of small replicate runs:

```python
import numpy as np
import ambr as am
from ambr.gpu_ensemble import GPUEnsembleRunner, BatchedWellMixedSIR

# Needs NVIDIA + CuPy. B parameter sets in one (B, N) GPU pass.
B = 4
betas = np.full(B, 0.3)
gammas = np.full(B, 0.1)
i0 = np.full(B, 0.01)
runner = GPUEnsembleRunner(BatchedWellMixedSIR())
if am.GPU_AVAILABLE:
    traj = runner.run(
        n_agents=10_000, steps=30,
        params={"beta": betas, "gamma": gammas, "i0_frac": i0},
    )  # -> {metric: (B, steps)}
    print({k: v.shape for k, v in traj.items()})
else:
    print("GPU ensemble requires NVIDIA + CuPy; skip on this host")
# Optional SMAC loop: smac_batch_calibrate(BatchedWellMixedSIR(), bounds, loss_fn, ...)
# (needs pip install 'ambr[advanced]')
```

`ambr.gpu` provides the array-module abstraction (`get_array_module`, `to_device`,
`to_host`) and falls back to NumPy when CuPy is unavailable. Requires **NVIDIA
GPU + CuPy** (not Apple Metal/MPS).

## 🔬 Optimization

AMBER includes powerful optimization capabilities for parameter tuning:

```python
import ambr as am
from ambr.optimization import ParameterSpace, grid_search

class MyModel(am.Model):
    def setup(self):
        n = int(self.p.get("agents", 50))
        self.add_agents(n, wealth=int(self.p.get("initial_value", 1)))
    def step_vectorized(self):
        donors = self.agents.where(self.agents.wealth > 0)
        donors.wealth -= 1
        self.agents.at[
            self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
        ].scatter_add(wealth=1)
    def update(self):
        self.record_model("total_wealth", int(self.agents.wealth.sum()))

parameter_space = ParameterSpace({
    "agents": [20, 40],
    "initial_value": [1, 5],
    "steps": 10,
    "seed": 0,
    "show_progress": False,
})
results = grid_search(MyModel, parameter_space, metric="total_wealth", minimize=False)
best_params = results[0]["parameters"]
print(best_params, results[0]["objective"])
```

Beyond `grid_search`, AMBER ships `random_search`, `bayesian_optimization`
(SMAC RandomForest surrogate; GP is unsupported), and `SMACOptimizer`
(same RandomForest stack) — plus the GPU batched ensemble above for
derivative-free calibration at scale.

## 📦 Installation

```bash
pip install ambr

# Optional extras
pip install 'ambr[perf]'              # Numba CPU scatter (recommended on Mac)
pip install 'ambr[gpu]'               # NVIDIA + CuPy (not Metal/MPS)
pip install 'ambr[viz]'               # matplotlib plot helpers
pip install 'ambr[advanced]'          # SMAC optimization
pip install 'ambr[advanced,viz]'      # SMAC + matplotlib (example plots)
```

```python
import ambr as am
print(am.__version__)   # 0.5.0+
am.print_status()
```

## 🏗️ Features

- **Simple API**: AgentPy-shaped OOP lane + vectorized columnar views on one model
- **High Performance**: Polars DataFrames; optional Numba (`ambr[perf]`) for scatters
- **Device placement**: Keras-style `model.cpu(mode=...)` / `model.gpu()` with
  `step_vectorized` / `step_oop` hooks (GPU is vectorized-only)
- **Speed lanes**: `am.print_status()` / `am.recommend(n)` / `ArrayKernelModel`
- **Snapshot-view monitor**: operational diagnostics for observed write/borrow
  conflicts and uncertified mutable arrays (not a schedule proof)
- **GPU backend**: native vectorized path + optional approved private loops +
  CuPy helpers + batched ensemble for calibration
- **Optimization**: grid / random / Bayesian (SMAC) search, plus GPU-batched calibration
- **Declarative reporting**: `model_reporters` / `agent_reporters` and a typed `params` schema
- **Environments**: Support for grid, network, and continuous space environments
- **OOP activation**: optional `activate_agents("random"|"sequential"|"simultaneous")` (not a schedule proof)
- **Plot helpers**: `plot_timeseries` / `plot_grid` from RunResults (`ambr[viz]` alias)
- **Experiments**: Run multiple simulations with parameter sampling
- **Random Number Generation**: Reproducible simulations with controlled randomness
- **RunResults**: `results.agents` and `results['agents']` both work

## 📚 Examples

Working examples are available in the `examples/` directory:

- **Schelling (grid)** — `examples/schelling_vectorized.py` (canonical occupancy helpers)
- **Wealth Transfer** — economic inequality / dual-lane quickstart
- **Virus Spread** — epidemiological SIR model
- **Flocking** — Boids + optional tensor-lane variant
- **Forest Fire** — cellular automata fire spread
- **GPU quickstart** — `model.gpu().run()` on a view-API model, or `ArrayKernelModel`
- **Ensemble / SMAC smoke** — `examples/smac_batch_sir_smoke.py` (SMAC needs `ambr[advanced]`)
- **SMAC calibration** — basic / advanced Schelling multi-objective

## 📖 Documentation

- **Docs**: https://ambr.readthedocs.io/
- **Paper**: https://arxiv.org/abs/2601.16292 — package vs paper claims: [docs/paper_and_package.rst](https://ambr.readthedocs.io/en/latest/paper_and_package.html)
- **Going faster** (lanes / Numba / GPU): [docs/going_faster.rst](https://ambr.readthedocs.io/en/latest/going_faster.html)
- **Environments & Schelling**: [docs/environments_schelling.rst](https://ambr.readthedocs.io/en/latest/environments_schelling.html)
- **From AgentPy**: [docs/from_agentpy.rst](https://ambr.readthedocs.io/en/latest/from_agentpy.html)
- **Deprecations (→ 1.0)**: [docs/deprecations.rst](https://ambr.readthedocs.io/en/latest/deprecations.html)
- **Versioning / 1.0 roadmap**: [docs/versioning.rst](https://ambr.readthedocs.io/en/latest/versioning.html), [docs/roadmap_1_0.rst](https://ambr.readthedocs.io/en/latest/roadmap_1_0.html)
- **Public API surface**: [docs/public_api.rst](https://ambr.readthedocs.io/en/latest/public_api.html)
- **Changelog**: [CHANGELOG.md](https://github.com/a11to1n3/AMBER/blob/main/CHANGELOG.md)

## 📝 How to cite?

If you use AMBER in academic work, please cite the paper:

```bibtex
@article{pham2026amber,
  title={AMBER: A Columnar Architecture for High-Performance Agent-Based Modeling in Python},
  author={Pham, Anh-Duy},
  journal={arXiv preprint arXiv:2601.16292},
  year={2026}
}
```

Paper: https://arxiv.org/abs/2601.16292

For the **software**, this repository also ships [`CITATION.cff`](https://github.com/a11to1n3/AMBER/blob/main/CITATION.cff)
(GitHub “Cite this repository”). Manuscript drafts and build artifacts are
**not** kept in the library tree — only the public paper citation and software
metadata.

## 🤝 Contributing

We welcome contributions! See [docs/contributing.rst](https://ambr.readthedocs.io/en/latest/contributing.html)
(or the Contributing page on Read the Docs).

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](https://github.com/a11to1n3/AMBER/blob/main/LICENSE) file for details.
