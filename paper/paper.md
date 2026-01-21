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

# Introduction

Agent-based models (ABMs) characterize physical, biological, and social systems as dynamic interactions among autonomous agents from a bottom-up perspective. The agents can be molecules, animals, people, or any discrete entity whose behavior can be modeled. The interactions can range from water molecules forming a vortex, to predators chasing prey, to traders executing orders in financial markets.

Agents' interactions can give rise to emergent properties—system-level phenomena not explicitly programmed but arising from local rules. This capacity to model emergence is a core reason for using ABMs. Additionally, ABMs are flexible enough to incorporate agents' heterogeneity (e.g., different preferences, decision rules, or resource endowments) and bounded rationality based on empirical observations.

Python has become a popular language for ABM due to its rapid prototyping capabilities and seamless integration with the scientific computing ecosystem. However, widely used Python ABM frameworks typically represent each agent as a separate Python object, with per-agent attribute access and update loops. This object-oriented design, while intuitive, can become a performance bottleneck when scaling to large populations or when models require frequent state updates and neighborhood queries [@masad2015mesa; @foramitti2021agentpy].

AMBER (Agent-based Modeling with Blazingly Efficient Records) is a general framework for developing agent-based models in Python. It is published and maintained on GitHub at <https://github.com/a11to1n3/AMBER>. AMBER retains a familiar, agent-oriented programming interface while fundamentally changing how agent state is stored and updated—using a columnar, DataFrame-backed architecture built on Polars [@polars_zenodo; @polars_github].

# Statement of Need

Among numerous frameworks for agent-based modeling in different programming languages, Mesa [@masad2015mesa] and AgentPy [@foramitti2021agentpy] are the two prominent open-source frameworks in Python. The object-oriented paradigm of Python fits the "agent perspective" of ABM naturally, and modelers benefit from the wealth of packages available for data analysis and visualization.

In practice, researchers frequently face a trade-off between:

1. **Usability and ecosystem integration**, exemplified by Python-based frameworks like Mesa and AgentPy, which offer accessible APIs but can encounter performance bottlenecks in large-scale simulations; and
2. **High performance**, typically achieved by adopting alternative languages (e.g., Julia with Agents.jl [@datseris2024agents], Java with Repast [@north2013complex] or MASON [@luke2005mason], or GPU-accelerated frameworks like FLAME GPU 2 [@richmond2023flame]).

AMBER is distinguished from Mesa and AgentPy by the following aspects.

**First**, AMBER stores agent state in a centralized, columnar DataFrame rather than scattered across individual Python objects. This "paradigm shift" (illustrated in \autoref{fig:paradigm}) enables vectorized operations for common patterns like "update all agents' wealth" or "filter agents where health < 50."

**Second**, AMBER's columnar backend (Polars) is a Rust-based, multi-threaded engine with native support for Apache Arrow's columnar data model [@polars_github; @arrow2026]. This reduces Python interpreter overhead for bulk operations.

**Third**, simulation outputs are already in tabular form, eliminating the need for per-agent serialization. Modelers can directly export or analyze results using standard DataFrame workflows.

**Fourth**, AMBER provides spatial abstractions (grid and continuous spaces), network environments, and utilities for experiments and parameter optimization—all integrated into a cohesive workflow.

# Overview

The modules in AMBER can be organized into six clusters: **Model**, **Environments**, **Experiment**, **Modelling Manager**, **Optimization**, and **Infrastructure**.

## Model

The modules in the Model cluster focus on describing the target system. Developed with AMBER, a model object can contain the following components:

- **Agent** — makes decisions, interacts with others, and stores micro-level variables. Each agent defines `setup()` and `step()` methods to specify initialization and per-step behavior.
- **Population** — the primary owner of agent state. Unlike traditional OOP-based frameworks that scatter state across Python objects, AMBER stores agent attributes as columns in a Polars DataFrame, enabling efficient vectorized operations.
- **AgentList** — contains a collection of agents and provides relevant functions for iteration, filtering, and batch operations.
- **Model** — coordinates the simulation lifecycle, including agent creation, step execution, and data collection. Stores model-level parameters and macro-level variables.
- **Data collection** — built-in recording of model-level and agent-level variables at each step, with results stored in tabular format for easy analysis.

![The paradigm shift from traditional OOP-based ABM (left), where each agent is a separate Python object with scattered memory access, to AMBER's columnar approach (right), where agent attributes are stored in a centralized DataFrame enabling vectorized operations.](paradigm_shift.png){#fig:paradigm width="100%"}

Taking an SIR epidemic model as an example, as shown below, the model is defined by subclassing the `Model` class and implementing `setup()` and `step()` methods:

```python
import ambr as am

class SIRModel(am.Model):
    def setup(self):
        # Create agents with initial states
        for i in range(self.params['n_agents']):
            agent = am.Agent(self, i)
            agent.state = 'S' if i > 0 else 'I'  # One infected
            self.add_agent(agent)
    
    def step(self):
        for agent in self.agents:
            if agent.state == 'I':
                # Infect susceptible neighbors
                for neighbor in agent.neighbors():
                    if neighbor.state == 'S':
                        neighbor.state = 'I'
        
        # Record macro-level statistics
        self.record_model('infected', 
            sum(1 for a in self.agents if a.state == 'I'))

model = SIRModel({'n_agents': 1000, 'steps': 100})
results = model.run()
```

Finally, by calling the `model.run()` method, the simulation starts.

## Environments

The modules in the Environments cluster provide spatial and network structures for agent interactions:

- **GridEnvironment** — constructed with cell objects, describes a discrete 2D grid that agents can occupy. Supports configurable neighborhood rules (Moore, von Neumann) and provides functions for agent placement, movement, and neighbor queries.
- **SpaceEnvironment** — describes a continuous 2D or 3D space with distance-based interactions. Agents have floating-point coordinates and can query neighbors within a specified radius.
- **NetworkEnvironment** — constructed with edge objects, describes a graph structure that links agents. Built on NetworkX, it supports various network topologies and provides functions for neighbor queries and network analysis.

Each environment coordinates agent placement and movement through a unified interface, making it easy to switch between different spatial representations.

## Experiment

Running multi-parameter experiments and Monte Carlo repetitions is a core component of the ABM workflow [@foramitti2021agentpy]. AMBER includes dedicated modules for experimental workflows:

- **Experiment** — manages parameter sweeps and repeated runs, automatically handling parallelization and result aggregation.
- **Sample** — defines a range of parameter values to explore.
- **IntRange** — specifies integer parameter ranges with step sizes.

The experiment infrastructure supports batch execution with configurable workers, making it easy to explore parameter spaces and collect statistically robust results.

## Modelling Manager

To combine everything and finally start running, the Modelling Manager cluster includes modules for different simulation objectives:

- **Model.run()** — executes the simulation logic written in the model, iterating through the specified number of steps and collecting data.
- **Experiment** — manages parameter sweeps and repeated runs for exploring parameter spaces and collecting statistically robust results.
- **Optimization utilities** — calibrate model parameters by minimizing the distance between model output and empirical evidence, or optimize agent behaviors for specific objectives.

Taking the SIR epidemic model as an example, the simulation can be run directly or through an experiment:

```python
import ambr as am

# Direct simulation
model = SIRModel({'n_agents': 1000, 'steps': 100})
results = model.run()

# Parameter sweep with Experiment
from ambr import Experiment, Sample, IntRange

exp = Experiment(
    model_class=SIRModel,
    parameters={
        'n_agents': IntRange(100, 1000, step=100),
        'infection_rate': Sample([0.1, 0.2, 0.3])
    },
    iterations=10
)
exp.run()
results = exp.results()
```

## Optimization

AMBER includes modules for model calibration and parameter optimization:

- **grid_search** and **random_search** — exhaustive and stochastic parameter exploration for finding optimal configurations.
- **bayesian_optimization** — sample-efficient optimization using Gaussian processes, ideal for expensive-to-evaluate models.
- **SMACOptimizer** — integration with SMAC (Sequential Model-based Algorithm Configuration) for state-of-the-art hyperparameter optimization.
- **MultiObjectiveSMAC** — extends SMAC for multi-objective optimization scenarios.

These tools enable modelers to calibrate model parameters against empirical data or optimize agent behaviors for specific objectives.

## Infrastructure

The Infrastructure cluster includes modules that provide support for the components above:

- **Polars backend** — provides high-performance DataFrame operations for agent state management. All agent attributes are stored in a columnar format, enabling vectorized operations and efficient memory usage.
- **Data recording** — the `record_model()` and `record_agent()` methods collect simulation data at each step, storing results in tabular format for easy export and analysis.
- **Vectorized utilities** — pre-built functions like `vectorized_move()`, `vectorized_wealth_transfer()`, and `vectorized_random_velocities()` for common ABM patterns that benefit from batch operations.
- **Performance tools** — includes `SpatialIndex` for efficient neighbor queries and `ParallelRunner` for multi-threaded execution of independent simulations.
- **BatchUpdateContext** — a context manager for efficient batch updates to agent attributes, reducing Python overhead for bulk operations.

The infrastructure is designed to be transparent to users—modelers write intuitive agent-oriented code while AMBER automatically leverages vectorized operations where possible.

# Performance

To evaluate AMBER's scaling behavior, we benchmarked three canonical ABM workloads—Wealth Transfer, SIR Epidemic, and Random Walk—against Mesa and AgentPy across population sizes from 100 to 10,000 agents (\autoref{fig:scaling}).

![Scaling comparison of AMBER, Mesa, and AgentPy across three benchmark models. AMBER demonstrates consistently lower execution times as agent populations scale.](../benchmarks/results/scaling_chart.png){#fig:scaling width="100%"}

Key observations:

- **Wealth Transfer**: AMBER achieves up to 93× faster execution than Mesa at 10,000 agents (2.1s vs. 195.5s), as this workload is dominated by batch attribute updates—precisely the pattern AMBER optimizes.
- **SIR Epidemic**: AMBER maintains a 1.2–1.7× advantage. The more modest speedup reflects higher per-agent branching logic.
- **Random Walk**: AMBER's advantage becomes pronounced at larger populations due to vectorized position updates.

Performance depends on model characteristics: highly heterogeneous, branch-heavy logic may reduce the fraction of work expressible as vectorized operations. AMBER is best suited for models with substantial population-level operations.

# Resources

On the AMBER GitHub repository (<https://github.com/a11to1n3/AMBER>), we provide:

- **Documentation** — comprehensive API reference and user guide at ReadTheDocs.
- **Tutorial** — step-by-step guides for building models, from simple to complex.
- **Examples** — reference implementations including SIR epidemic, forest fire, flocking, and wealth transfer models.
- **Benchmarks** — reproducible performance comparisons with other frameworks.

# Acknowledgements

We acknowledge the open-source communities behind Polars and Apache Arrow, which enable high-performance, columnar data processing in the Python ecosystem [@polars_github; @arrow2026].

# AI Usage Disclosure

Generative AI tools were used to assist with manuscript editing and rephrasing. All technical claims, software descriptions, and citations were reviewed and verified by the author.

# References
