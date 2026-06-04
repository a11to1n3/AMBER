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
Polars expressions — regardless of population size.

**Benchmark against six representative ABM/simulation frameworks
— 5000 agents, 50 executed steps, Python 3.12, Julia 1.12.3, Apple Silicon.**
All numbers are seeded wall-clock timings, averaged over 10 runs
(slowest trimmed). Every framework is **checked against output
invariants** (wealth conservation, boundary clamping, S+I+R population
conservation) before timing — see
[`benchmarks/correctness_check.py`](benchmarks/correctness_check.py).
Reproducer: [`benchmarks/run_all_frameworks.py`](benchmarks/run_all_frameworks.py).

| Framework | Language | Arch. | Wealth Transfer | Random Walk | SIR Epidemic |
|---|---|---|---:|---:|---:|
| **AMBER (vectorized)** | Python | Columnar (Polars) | 20 ms | 4.8 ms | **497 ms** |
| Agents.jl | Julia | Object | **7.2 ms** | **1.6 ms** | 813 ms |
| AMBER (loop) | Python | Object | 169 ms | 332 ms | 9.53 s |
| Mesa | Python | Object | 22.61 s | 131 ms | 16.63 s |
| AgentPy | Python | Object | 266 ms | 141 ms | 10.98 s |
| SimPy | Python | Event loop | 216 ms | 254 ms | 4.67 s |
| Melodie | Python | Hybrid | 177 ms | 1.03 s | 20.09 s |

**AMBER (vectorized) is the fastest Python-hosted framework on every
model at 5000 agents**. Against Agents.jl, it wins the SIR benchmark
and trails the Julia implementation on wealth transfer and random walk,
where per-step work is small enough that Julia's compiled dispatch has
less fixed overhead.

![Seven-framework scaling chart](benchmarks/results/scaling_chart_all.png)

See [`benchmarks/README.md`](benchmarks/README.md) for the full table at
500 / 1000 / 5000 agents, speedup ratios, a per-model correctness audit,
and the documented SIR update-semantics caveat.

## 🚀 Quick Start

```python
import ambr as am
import numpy as np

# Define a model with the vectorized view API — no per-agent loops.
class WealthModel(am.Model):
    def setup(self):
        self.add_agents(100, wealth=np.random.randint(1, 10, size=100))

    def step(self):
        donors = self.agents.where(self.agents.wealth > 0)
        donors.wealth -= 1
        ids = self.agents.ids.to_numpy()
        recipients = self.nprandom.choice(ids, size=len(donors))
        self.agents.at[recipients].scatter_add(wealth=1)
        self.record('total_wealth', int(self.agents.wealth.sum()))

model = WealthModel({'steps': 100, 'seed': 42, 'show_progress': False})
results = model.run()
print(results['agents'].head(10))
```

> **New in 0.3.0:** Setting ``agent.wealth = 5`` on a Python Agent
> automatically syncs to the DataFrame. You can freely mix OOP-style
> and vectorized access without desync.

## ⚡ Vectorized View API

The view API compiles per-step updates to a handful of Polars expressions
— regardless of population size:

```python
def step(self):
    # Bulk columnar reads/writes over the entire population
    self.agents.x = self.agents.x + self.nprandom.uniform(-1, 1, len(self.agents))

    # Filtered writes: only agents matching a condition
    infected = self.agents.where(self.agents.status == 1)
    infected.infection_time += 1

    # scatter_add: flow-of-resources with duplicate-id safety
    self.agents.at[[1, 1, 3]].scatter_add(wealth=1)  # agent 1 gets +2, agent 3 gets +1
```

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

## 📦 Installation

```bash
pip install ambr
```

## 🏗️ Features

- **Simple API**: Intuitive interface for agent-based modeling
- **High Performance**: Efficient data handling with Polars DataFrames
- **Optimization**: Built-in parameter optimization with grid search, random search, and Bayesian optimization
- **Environments**: Support for grid, network, and continuous space environments
- **Experiments**: Run multiple simulations with parameter sampling
- **Random Number Generation**: Reproducible simulations with controlled randomness

## 📚 Examples

Working examples are available in the `examples/` directory:

- **Wealth Transfer Model**: Economic inequality simulation
- **Virus Spread Model**: Epidemiological SIR model
- **Flocking Simulation**: Boids flocking behavior
- **Forest Fire Model**: Cellular automata fire spread
- **Network Simulations**: Graph-based agent interactions

## 📖 Documentation

- **Documentation**: https://ambr.readthedocs.io/
- **Paper**: https://arxiv.org/abs/2601.16292

## 📝 How to cite?

If you use AMBER in your research, please cite our paper:

```bibtex
@article{pham2026amber,
  title={AMBER: A Columnar Architecture for High-Performance Agent-Based Modeling in Python},
  author={Pham, Anh-Duy},
  journal={arXiv preprint arXiv:2601.16292},
  year={2026}
}
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
