# AMBER (Agent-based Modeling with Blazingly Efficient Records)

[![CI](https://github.com/a11to1n3/AMBER/actions/workflows/ci.yml/badge.svg)](https://github.com/a11to1n3/AMBER/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/a11to1n3/AMBER/graph/badge.svg)](https://codecov.io/gh/a11to1n3/AMBER)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyPI version](https://img.shields.io/pypi/v/ambr.svg)](https://pypi.org/project/ambr/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3_Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

AMBER is a Python framework for agent-based modeling that uses Polars for efficient data handling and analysis. AMBER provides a clean, robust API for creating parallel, high-performance simulations in Python.

## 🚀 Performance

AMBER stores the entire population as a columnar Polars DataFrame and
exposes a vectorized view API (`agents.where(...)`, `agents.at[ids]`,
`scatter_add`) that compiles per-step updates down to a handful of
columnar operations. Models can provide `step_vectorized()` and `step_oop()`
for explicit native lanes; legacy models with only `step()` keep the fallback.
The vectorized lane runs on GPU via `model.gpu().run()`.

**Large-N multi-framework scaling (1k→10M agents, 50 steps, 10 runs trimmed
mean, NVIDIA RTX 5090).** Ten frameworks: AMBER (GPU / vectorized / loop),
mesa-frames, FLAME GPU 2, Agents.jl, SimPy, Melodie, AgentPy, Mesa.
Full tables:
[`benchmarks/results/summary_table.md`](benchmarks/results/summary_table.md).
Reproducer: [`benchmarks/run_all_frameworks.py`](benchmarks/run_all_frameworks.py).
Correctness gates:
[`benchmarks/correctness_check.py`](benchmarks/correctness_check.py).

![Framework scaling 1k→10M agents](benchmarks/results/scaling_chart.png)

**At 1M / 10M agents (where each framework still finishes):**

| Model | AMBER (GPU) | AMBER (vectorized) | FLAME GPU 2 | Next best CPU-scale peer |
|---|---:|---:|---:|---:|
| Wealth | 3.91 s / 193 s | 6.44 s / 214 s | **28 ms / 226 ms** | Agents.jl 8.53 s @ 1M |
| Random walk | 198 ms / 2.04 s | 531 ms / 6.23 s | **20 ms / 201 ms** | mesa-frames 3.55 s / 20.8 s |
| Schelling | **428 ms / 5.17 s** | 2.64 s / 59.8 s | 2.06 s / 20.8 s | mesa-frames 4.33 s / 86.9 s |
| SIR (cell-list) | 882 ms / **9.39 s** | 31.3 s / 308 s | **108 ms / 3.80 s** | — |

- **Schelling:** AMBER (GPU) is the fastest row at 1M and 10M among measured
  frameworks (beats FLAME GPU 2 and mesa-frames).
- **Wealth / random walk:** FLAME GPU 2 leads; AMBER (GPU) still beats other
  Python-hosted stacks that reach those scales.
- **SIR:** AMBER uses **cell-list** infection. GPU reaches 10M at 9.39 s
  (~33× faster than vectorized’s 308 s); FLAME still leads at 3.80 s.
- **API:** write the view-API `step` once; place with `.cpu(mode="vectorized")`
  or `.gpu()`. Details: [`benchmarks/README.md`](benchmarks/README.md).

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
    def step(self):
        self.agents.transfer()
    def update(self):
        self.record_model('total', int(self.agents.wealth.sum()))

results = WealthModel({'n': 50, 'steps': 20, 'seed': 1}).run()
print(results.model)      # also results['model']
print(results.agents.head())
```

**Vectorized (columnar; best at large N):**

```python
import ambr as am

class WealthModel(am.Model):
    model_reporters = {'total_wealth': lambda m: int(m.agents.wealth.sum())}

    def setup(self):
        self.add_agents(100, wealth=self.rng.integers(1, 10, size=100))

    def step_vectorized(self):
        xp = self.xp
        wealth = self.agents.array("wealth")
        donors = xp.nonzero(wealth > 0)[0]
        if int(donors.size) == 0:
            return
        wealth[donors] -= 1
        recipients = self.rng.choice(
            self.agents.array("id"), size=int(donors.size)
        )
        self.agents.at[recipients].scatter_add(wealth=1)

# Fluent placement (0.4.3): device + optional mode; run(mode=...) still overrides.
results = WealthModel({'steps': 100, 'seed': 42}).cpu(mode="vectorized").run()
# Same class + step on GPU (device-resident columns; needs NVIDIA + CuPy):
# results = WealthModel({'steps': 100, 'seed': 42}).gpu().run()
print(results.model.tail(5))
print(results.agents.head(10))
```

Coming from AgentPy? See [`docs/from_agentpy.rst`](docs/from_agentpy.rst).

**Going faster / GPU** — same `Model` and `step`; pick placement and lane (see
[`docs/going_faster.rst`](docs/going_faster.rst)):

```python
import ambr as am
am.print_status()                 # GPU? which lane?
print(am.recommend(1_000_000))  # one-line suggestion

# Native path: view-API step + .gpu() (or .cpu(mode="vectorized"))
model = WealthModel({"n": 100_000, "steps": 50, "seed": 0})
results = model.gpu().run()                    # or .cpu(mode="vectorized").run()

# Array-kernel lane (CuPy if available, else NumPy) for pure array state:
class Drift(am.ArrayKernelModel):
    def init_state(self, xp, n, rng, p):
        return {"x": rng.random(n, dtype=xp.float32)}
    def step_state(self, xp, state, rng, p):
        state["x"] = state["x"] + 0.01
        return state
    def metrics(self, xp, state):
        return {"mean_x": float(am.to_host(state["x"].mean()))}

print(Drift({"n": 100_000, "steps": 20}).run().info)
```

`self.rng` is the canonical seeded RNG (a NumPy `Generator`); `self.random` is
the stdlib one. Both are seeded from the `seed` parameter. Progress printing is
off by default (`show_progress=True` to re-enable).

> **Execution lanes:** Keras-style **`model.cpu(mode=...)` / `model.gpu()`**
> placement dispatches `step_vectorized()` for vectorized CPU/GPU runs and
> `step_oop()` for CPU Agent-object runs. GPU execution is vectorized-only;
> Python Agent objects use `cpu(mode="oop")`. See the
> [changelog](CHANGELOG.md).
>
> **New in 0.4.1:** [AgentPy-shaped UX](docs/from_agentpy.rst) (`RunResults`,
> `agents.random()`), [progressive speed lanes](docs/going_faster.rst)
> (`am.print_status()`, `am.recommend(n)`, `ArrayKernelModel`), optional
> **Numba** CPU path (`pip install 'ambr[perf]'` — great on Mac), contract /
> write-path hardening, SMAC install pin, and Schelling grid helpers.
>
> **New in 0.4:** a runtime [snapshot-view contract](#-snapshot-view-contract)
> checker, a [GPU backend + batched calibration](#-gpu-backend--batched-calibration),
> one [canonical verb per task](#-canonical-api-04) (legacy spellings still work),
> declarative `model_reporters`, and a typed `params` schema.
>
> **New in 0.3.0:** Setting ``agent.wealth = 5`` on a Python Agent
> automatically syncs to the DataFrame. You can freely mix OOP-style
> and vectorized access without desync.

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

## 🧭 Canonical API (0.4)

AMBER 0.4 settles on one obvious verb per task. The legacy spellings still work
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

Whether a vectorized refactor preserves an intended update schedule is a
semantic question. AMBER's runtime monitor reports selected operational hazards
at instrumented API seams; it does not prove schedule equivalence for arbitrary
NumPy, CuPy, or user-kernel code. Run with a contract mode and inspect the
per-step records:

```python
results = model.run(steps=100, contract="check")   # "off" | "check" | "warn" | "raise"
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

**Single-run (native, 0.4.3):** place a model on device through the same public
API. Models may provide `step_vectorized()` and an optional private
model-specific fast loop behind `gpu().run()`. Numeric columns stay
device-resident for the run. Contract modes use the instrumented general path;
private fast loops are selected only with `contract="off"` and an explicit
per-instance `approve_fast_path(evidence)` declaration. The evidence string is
a caller-supplied provenance label, not something AMBER verifies. Without it,
`gpu().run()` uses the instrumented-capable general path. Private loops are not
covered by the monitor.

```python
# Same WealthModel as the vectorized quickstart
results = WealthModel({"n": 1_000_000, "steps": 50, "seed": 0}).gpu().run()
# Switch back: model.cpu(mode="vectorized").run(...)
```

**Many short runs (calibration):** the *ensemble* axis (`B` simulations × `N`
agents) batches into one device pass — the natural fit when you evaluate
thousands of small replicate runs:

```python
from ambr.gpu_ensemble import GPUEnsembleRunner, BatchedWellMixedSIR, smac_batch_calibrate

# Evaluate B parameter sets in one (B, N) GPU pass
runner = GPUEnsembleRunner(BatchedWellMixedSIR())
traj = runner.run(n_agents=100_000, steps=60,
                  params={"beta": betas, "gamma": gammas, "i0_frac": i0})  # -> {metric: (B, steps)}

# SMAC ask -> one batched GPU evaluation -> tell
best, history = smac_batch_calibrate(BatchedWellMixedSIR(), bounds, loss_fn,
                                     n_agents=100_000, steps=60)
```

`ambr.gpu` provides the array-module abstraction (`get_array_module`, `to_device`,
`to_host`) and falls back to NumPy when CuPy is unavailable. Requires **NVIDIA
GPU + CuPy** (not Apple Metal/MPS).

## 🔬 Optimization

AMBER includes powerful optimization capabilities for parameter tuning:

```python
from ambr.optimization import ParameterSpace, grid_search

# Define parameter space
parameter_space = ParameterSpace({
    'agents': [10, 50, 100],
    'initial_value': [1, 5, 10],
    'steps': 100
})

# Run optimization
results = grid_search(MyModel, parameter_space, 'some_metric')
best_params = results[0]['parameters']
```

Beyond `grid_search`, AMBER ships `random_search`, `bayesian_optimization`
(SMAC Gaussian-process), and `SMACOptimizer` (random-forest surrogate) — plus the
GPU batched ensemble above for derivative-free calibration at scale.

## 📦 Installation

```bash
pip install ambr

# Optional extras
pip install 'ambr[perf]'       # Numba CPU scatter (recommended on Mac)
pip install 'ambr[advanced]'   # SMAC optimization
```

```python
import ambr as am
print(am.__version__)   # 0.4.3+
am.print_status()
```

## 🏗️ Features

- **Simple API**: AgentPy-shaped OOP lane + vectorized columnar views on one model
- **High Performance**: Polars DataFrames; optional Numba (`ambr[perf]`) for scatters
- **Device placement**: Keras-style `model.cpu(mode=...)` / `model.gpu()` with mode-specific step hooks
- **Speed lanes**: `am.print_status()` / `am.recommend(n)` / `ArrayKernelModel`
- **Snapshot-view monitor**: runtime diagnostics for observed write/borrow conflicts and untraceable mutable arrays
- **GPU backend**: native view-API path + CuPy helpers + batched ensemble for calibration
- **Optimization**: grid / random / Bayesian (SMAC) search, plus GPU-batched calibration
- **Declarative reporting**: `model_reporters` / `agent_reporters` and a typed `params` schema
- **Environments**: Support for grid, network, and continuous space environments
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
- **SMAC calibration** — basic / advanced Schelling multi-objective

## 📖 Documentation

- **Docs**: https://ambr.readthedocs.io/
- **Paper**: https://arxiv.org/abs/2601.16292
- **Going faster** (lanes / Numba / GPU): [docs/going_faster.rst](docs/going_faster.rst)
- **Environments & Schelling**: [docs/environments_schelling.rst](docs/environments_schelling.rst)
- **From AgentPy**: [docs/from_agentpy.rst](docs/from_agentpy.rst)
- **Deprecations (→ 1.0)**: [docs/deprecations.rst](docs/deprecations.rst)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

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

(Source drafts and build artifacts for the manuscript are **not** kept in this
repository — only the public citation.)

## 🤝 Contributing

We welcome contributions! See [docs/contributing.rst](docs/contributing.rst)
(or the Contributing page on Read the Docs).

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
