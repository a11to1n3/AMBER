---
title: "AMBER: Rethinking Agent-Based Modeling in Python"
tags:
  - Python
  - agent-based modeling
  - simulation
  - polars
  - performance
authors:
  - name: Anh-Duy Pham
    orcid: 0000-0003-3832-9453
    affiliation: 1
affiliations:
  - name: University of Würzburg, Germany
    index: 1
date: 21 January 2026
bibliography: paper.bib
---

# Summary

AMBER (Agent-based Modeling with Blazingly Efficient Records) is an open-source Python framework for building, running, and analyzing agent-based models (ABMs). ABMs simulate systems as collections of autonomous agents whose local interactions give rise to emergent, system-level phenomena—from epidemics spreading through populations to markets self-organizing through trader behavior.

AMBER provides researchers with a familiar, intuitive agent-oriented programming interface while fundamentally changing how agent state is stored internally. Instead of scattering state across individual Python objects (the traditional approach), AMBER centralizes agent attributes in a columnar DataFrame backed by Polars—a high-performance, Rust-based data engine. This architectural choice enables vectorized batch operations, efficient filtering and aggregation, and seamless integration with Python's data science ecosystem.

Key features include: (1) a `Population` abstraction that stores agent state as columns rather than objects; (2) spatial environments (grid, continuous space, networks) for modeling spatial interactions; (3) experiment utilities for parameter sweeps and Monte Carlo repetitions; and (4) optimization tools for model calibration. AMBER is designed for computational social scientists, epidemiologists, ecologists, and researchers in any domain where large-scale agent-based simulation is required.

# Statement of Need

Agent-based modeling has become a standard methodology across disciplines—from epidemiology and ecology to economics and social science [@abar2017review]. Python's accessibility and rich ecosystem make it an attractive platform for ABM, with Mesa [@masad2015mesa] and AgentPy [@foramitti2021agentpy] emerging as the two prominent open-source frameworks.

However, researchers frequently face a trade-off: Python-based frameworks offer usability but encounter performance bottlenecks at scale, while high-performance alternatives (Julia's Agents.jl [@datseris2024agents], Java's Repast [@north2013complex], GPU-accelerated FLAME GPU 2 [@richmond2023flame]) require language switching or specialized expertise. This gap leaves researchers with simulations that either run slowly or require significant investment to port to faster platforms.

AMBER addresses this gap by targeting researchers who need **both** Python's ecosystem integration **and** scalable performance. Specifically, AMBER is designed for models that:

- Update many agents' attributes at each time step (wealth transfer, disease states)
- Frequently filter or aggregate over agent subpopulations (counting infected agents, finding neighbors)
- Require collecting large amounts of per-step data for downstream analysis

The target audience includes computational social scientists, epidemiologists, and any researcher who currently uses Mesa or AgentPy but finds performance limiting as their models scale to thousands or tens of thousands of agents.

# State of the Field

**Python ABM frameworks.** Mesa [@masad2015mesa] pioneered accessible ABM in Python and remains widely used for teaching and research. AgentPy [@foramitti2021agentpy] emphasizes integrated workflows with parameter sampling and Jupyter-based analysis. Melodie [@yu2023melodie] separates environment and scenario components while accelerating selected modules with Cython. SimPy [@zinoviev2024discrete] provides discrete-event simulation but follows a different paradigm focused on resource queuing rather than population-level agent interactions.

**Non-Python alternatives.** NetLogo [@tisue2004netlogo] offers rapid development with a visual interface. Repast [@north2013complex] and MASON [@luke2005mason] provide mature Java-based platforms. FLAME GPU 2 [@richmond2023flame] achieves high performance through GPU acceleration. Agents.jl [@datseris2024agents] offers a performant Julia-based framework.

**Why AMBER rather than contributing to existing tools?** Existing Python ABM frameworks are architecturally committed to object-oriented agent representation. Retrofitting a columnar backend would require fundamental redesign incompatible with their established APIs and user bases. AMBER's unique scholarly contribution is demonstrating that a columnar, DataFrame-backed architecture can achieve order-of-magnitude speedups for common ABM patterns while preserving an intuitive agent-oriented programming model. This architectural innovation could not be achieved as an incremental contribution to existing frameworks.

# Software Design

AMBER's central design decision is the **paradigm shift from object-oriented to columnar agent state representation** (\autoref{fig:paradigm}).

![The paradigm shift from traditional OOP-based ABM (left), where each agent is a separate Python object with scattered memory access, to AMBER's columnar approach (right), where agent attributes are stored in a centralized DataFrame enabling vectorized operations.](paradigm_shift.png){#fig:paradigm width="100%"}

**Trade-offs considered:**

1. **Memory layout vs. programming model.** Storing agent state in column-oriented DataFrames enables cache-efficient access and vectorized operations. The trade-off is that individual agent attribute access involves DataFrame lookup rather than direct object attribute access. We mitigate this by providing an `Agent` wrapper class that presents a familiar interface while delegating storage to the underlying DataFrame.

2. **Backend choice.** We evaluated Pandas, NumPy, and Polars as potential backends. Polars was selected because it is (a) multi-threaded by default, (b) built on Apache Arrow for memory efficiency [@arrow2026], and (c) provides an expression-based API well-suited to batch operations. The trade-off is an additional dependency and potential unfamiliarity for users accustomed to Pandas.

3. **Vectorization granularity.** AMBER provides explicit vectorized utilities (`vectorized_move()`, `vectorized_wealth_transfer()`) for common patterns. Users can also drop down to raw Polars expressions. This layered approach balances ease-of-use with performance access.

**Data structure and access patterns.** In AMBER, agent state is stored in a `Population` object backed by a Polars DataFrame. Each agent attribute becomes a column, and each agent becomes a row:

```python
# Internal representation (simplified)
# agent_id | x    | y    | health | wealth
# 0        | 12.5 | 34.2 | 85     | 1000
# 1        | 5.1  | 67.8 | 92     | 1200
# ...      | ...  | ...  | ...    | ...
```

Users interact with agents through familiar object-oriented syntax. When accessing `agent.wealth`, AMBER looks up the value in the underlying DataFrame. When assigning `agent.wealth = 500`, AMBER updates the corresponding cell. This abstraction layer preserves intuitive code while maintaining columnar storage.

**Vectorized operations.** AMBER provides vectorized utilities for common ABM patterns that would otherwise require slow Python loops:

```python
# Traditional per-agent loop (slow)
for agent in model.agents:
    agent.wealth += 10

# AMBER vectorized alternative (fast)
model.population.batch_update('wealth', lambda w: w + 10)

# Or using built-in utilities
from ambr import vectorized_wealth_transfer
vectorized_wealth_transfer(model.population, amount=10)
```

The `BatchUpdateContext` context manager enables efficient multi-attribute updates:

```python
with model.population.batch_update_context() as batch:
    batch.set('x', new_x_values)
    batch.set('y', new_y_values)
    batch.set('velocity', new_velocities)
```

**Architecture overview:**

- **Model cluster**: `Agent`, `Population`, `AgentList`, `Model`, and data collection components
- **Environments cluster**: `GridEnvironment`, `SpaceEnvironment`, `NetworkEnvironment`
- **Experiment cluster**: `Experiment`, `Sample`, `IntRange` for parameter exploration
- **Optimization cluster**: Grid search, random search, Bayesian optimization, SMAC integration
- **Infrastructure cluster**: Polars backend, `BatchUpdateContext`, vectorized utilities, `SpatialIndex`

**Example usage.** The following code demonstrates an SIR epidemic model using AMBER's vectorized operations:

```python
import ambr as am
import polars as pl

class SIRModel(am.Model):
    def setup(self):
        n = self.params['n_agents']
        # Vectorized agent creation with batch attribute setting
        self.create_agents(n)
        states = ['I'] + ['S'] * (n - 1)  # One infected
        self.population.set_column('state', states)
    
    def step(self):
        pop = self.population
        # Vectorized state transitions using Polars expressions
        infected_mask = pop.get_column('state') == 'I'
        
        # Get neighbors of infected agents and update susceptible ones
        for idx in pop.filter(infected_mask).get_column('agent_id'):
            neighbor_ids = self.get_neighbors(idx)
            pop.update_where(
                (pl.col('agent_id').is_in(neighbor_ids)) & 
                (pl.col('state') == 'S'),
                {'state': 'I'}
            )
        
        # Vectorized counting
        infected_count = pop.count_where(pl.col('state') == 'I')
        self.record_model('infected', infected_count)

model = SIRModel({'n_agents': 10000, 'steps': 100})
results = model.run()
```

# Research Impact Statement

**Realized impact:**

- **Benchmarks demonstrating substantial speedups.** On canonical ABM workloads (Wealth Transfer, SIR Epidemic, Random Walk), AMBER achieves up to 93× faster execution than Mesa at 10,000 agents (\autoref{fig:scaling}). These benchmarks are reproducible via the `benchmarks/` directory in the repository.

![Scaling comparison of AMBER, Mesa, and AgentPy across three benchmark models. AMBER demonstrates consistently lower execution times as agent populations scale.](../benchmarks/results/scaling_chart.png){#fig:scaling width="100%"}

**Community-readiness signals:**

- **Comprehensive documentation** at ReadTheDocs with API reference, tutorials, and examples
- **Examples covering common ABM patterns**: SIR epidemic, forest fire, flocking, wealth transfer
- **Automated testing** with CI/CD pipeline ensuring correctness across Python versions
- **pip-installable** via `pip install ambr`

**Near-term significance:**

- AMBER addresses a documented gap between usability and performance in Python ABM
- The columnar architecture is novel in the Python ABM space and provides a foundation for future optimization (e.g., GPU acceleration via cuDF)
- Target users (Mesa/AgentPy users facing performance limits) represent a substantial, identifiable community

# Acknowledgements

We acknowledge the open-source communities behind Polars and Apache Arrow, which enable high-performance columnar data processing in the Python ecosystem [@polars_github; @arrow2026].

# AI Usage Disclosure

Generative AI tools (Claude) were used to assist with manuscript editing, rephrasing, and code example formatting. All technical claims, software descriptions, benchmark results, and citations were reviewed and verified by the author. The software implementation itself was developed by the author, with AI assistance used for documentation writing and code commenting.

# References
