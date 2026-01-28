# AMBER (Agent-based Modeling with Blazingly Efficient Records)

[![CI](https://github.com/a11to1n3/AMBER/actions/workflows/ci.yml/badge.svg)](https://github.com/a11to1n3/AMBER/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/a11to1n3/AMBER/graph/badge.svg)](https://codecov.io/gh/a11to1n3/AMBER)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3_Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

AMBER is a Python framework for agent-based modeling that uses Polars for efficient data handling and analysis. AMBER provides a clean, robust API for creating parallel, high-performance simulations in Python.

## 🚀 Performance

AMBER achieves high performance through its columnar memory layout (SoA), KD-Trees, and SIMD-vectorized operations.

**Benchmark vs Other Frameworks (5,000 Agents, 100 Steps)**

| Framework | Language | Architecture | Speed Rank |
|-----------|----------|--------------|------------|
| **Agents.jl** | Julia | Vectorized | 🥇 1st (0.02s - 0.9s) |
| **AMBER** | Python | Vectorized | 🥈 **2nd (0.1s - 2.3s)** |
| **SimPy (Dense)** | Python | Process/DES | 🥉 3rd (0.3s - 4.3s) |
| **Melodie** | Python | Hybrid | 4th (0.4s - 20s) |
| **AgentPy** | Python | Object | 5th (2s - 30s) |
| **Mesa** | Python | Object | 6th (50s - 30s) |


*\*SimPy is exceptionally fast for sparse models (like SIR) due to its event-driven nature, but AMBER is faster for dense/movement-heavy models.*

AMBER is the **state-of-the-art for dense simulation** in Python, while SimPy offers an alternative for event-driven logic.

![Comparison Chart](benchmarks/results/scaling_chart.png)

## 🚀 Quick Start

```python
import ambr as am

# Define a custom agent
class MyAgent(am.Agent):
    def setup(self):
        self.value = self.p.initial_value
        
    def step(self):
        self.value += self.model.random.randint(-1, 2)
        self.record('value', self.value)

# Define a model
class MyModel(am.Model):
    def setup(self):
        self.agents = am.AgentList(self, self.p.agents, MyAgent)
        
    def step(self):
        self.agents.call('step')
        
    def update(self):
        super().update()
        self.agents.record('value')

# Run a simulation
parameters = {
    'agents': 10,
    'initial_value': 5,
    'steps': 100
}

model = MyModel(parameters)
results = model.run()
```

## ⚡ Advanced: Vectorized Updates

For maximum performance, access the underlying `Population` manager to perform SIMD-vectorized state updates, bypassing Python loop overhead:

```python
def step(self):
    # Create a batch update context
    with self.population.create_batch_context() as batch:
        # Queue updates logic (executed in Rust/Polars)
        batch.add_update(target_ids, 'wealth', 1)
        batch.add_update(source_ids, 'wealth', -1)
    
    # State is applied atomically here
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