---
title: "AMBER: DataFrame-Backed Agent-Based Modeling in Python with a Familiar Agent-Oriented API"
tags:
  - Python
  - agent-based modeling
  - simulation
  - polars
  - performance
authors:
  - name: Author Name
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Institution Name, Country
    index: 1
date: 20 January 2026
bibliography: paper.bib
---

## Summary

Agent-based modeling (ABM) is a powerful paradigm for studying complex systems by simulating populations of heterogeneous agents whose local interactions give rise to emergent collective behavior. Python has become a popular language for ABM due to its support for rapid prototyping and seamless integration with the scientific computing ecosystem. However, widely used Python ABM frameworks typically represent each agent as a separate Python object, with per-agent attribute access and update loops. This design can become a performance bottleneck when scaling to large populations or when models require frequent neighborhood queries and repeated attribute updates [@masad2015mesa; @foramitti2021agentpy].

AMBER is an open-source ABM framework for Python that retains a familiar, agent-oriented programming interface while fundamentally changing how agent state is stored and updated. Rather than scattering state across many Python objects, AMBER centralizes agent attributes in a columnar tabular backend built on Polars [@polars_zenodo; @polars_github]. This data-oriented layout enables efficient filtering, aggregation, and batch state updates using vectorized expressions. It also makes simulation outputs immediately available in a tabular form suitable for downstream analysis. In addition to its core state backend, AMBER provides spatial abstractions (including grid and continuous spaces), network environments, and utilities for experiments and parameter search—all within the workflow style that Python ABM users expect [@amber_github].

## Statement of Need

A diverse landscape of ABM tools exists, spanning educational environments, domain-specific platforms, and general-purpose simulation engines [@abar2017review]. In practice, researchers frequently face a trade-off between:

1. **Usability and ecosystem integration**, exemplified by Python-based frameworks such as Mesa and AgentPy, which offer accessible APIs but can encounter performance bottlenecks in large-scale or interaction-dense simulations [@masad2015mesa; @foramitti2021agentpy]; and
2. **High performance**, typically achieved by adopting alternative runtime models or languages (e.g., Julia, Java, C++, or GPU-accelerated frameworks), which may introduce adoption barriers or reduce interoperability with the Python data science ecosystem [@datseris2019agentsjl; @north2013repast; @luke2005mason; @richmond2023flamegpu2].

AMBER bridges this gap by rethinking the data layout and execution strategy for agent state updates while preserving a Pythonic, agent-oriented interface. This approach is particularly beneficial for models that (i) update the attributes of many agents at each time step, (ii) frequently filter or aggregate over agent subpopulations, and/or (iii) need to collect substantial per-step state for downstream analysis.

## Related Work and Positioning

**Python ABM and simulation frameworks.** Mesa pioneered accessible ABM in Python and remains widely used for both teaching and research [@masad2015mesa]. AgentPy emphasizes an integrated workflow for interactive simulations and experiments, including parameter sampling, repeated runs, and analysis within Jupyter-based environments [@foramitti2021agentpy]. Melodie extends this ecosystem by introducing an explicit environment component and additional infrastructure for scenario and data management; it also accelerates selected components via Cython [@yu2023melodie].

SimPy provides a process-based discrete-event simulation (DES) framework in Python, where entities are modeled as generator functions that yield events [@simpy]. While SimPy excels at queueing systems, manufacturing processes, and resource-constrained scenarios, it follows a different paradigm from population-centric ABM: entities interact through shared resources and event scheduling rather than through explicit agent-to-agent interactions in spatial or network environments. AMBER targets the latter use case—models where many agents maintain individual state attributes that are updated and queried each time step.

Other Python simulation tools occupy complementary niches. PyCX provides a minimalist collection of sample ABM implementations for educational purposes [@sayama2013pycx]. Py2NetLogo enables Python-based control of NetLogo models, bridging Python's analytical ecosystem with NetLogo's mature ABM environment [@py2netlogo]. For multi-agent systems with communication protocols, frameworks such as SPADE (Smart Python Agent Development Environment) focus on FIPA-compliant agent communication rather than population-level simulation dynamics [@gregori2006spade]. These projects demonstrate the breadth of Python's simulation ecosystem but also underscore the gap that AMBER addresses: providing a high-performance, population-centric ABM framework with a columnar state backend while retaining a simple, agent-oriented API.

**ABM tools beyond Python.** NetLogo remains a widely adopted environment for rapid ABM development and interactive exploration [@wilensky1999netlogo]. Repast and MASON provide mature ABM platforms within the Java ecosystem [@north2013repast; @luke2005mason]. GAMA targets spatially explicit models with support for participatory workflows and GIS integration [@taillandier2019gama]. FLAME GPU 2 enables high-performance ABM on GPUs by mapping agent operations onto data-parallel execution [@richmond2023flamegpu2]. In Julia, Agents.jl offers a performant ABM framework with a lightweight model description syntax [@datseris2019agentsjl]. AMBER complements these tools by targeting the Python ecosystem specifically, offering a data-oriented state representation that accelerates common ABM workloads without requiring users to switch languages.

**AMBER's principal contribution** relative to prior Python frameworks is architectural: it adopts a centralized, columnar representation of agent attributes (via Polars) while continuing to expose a conventional agent-oriented interface. Polars is a Rust-based, multi-threaded analytical engine featuring a vectorized expression API and native support for Apache Arrow's columnar data model [@polars_github; @arrow]. AMBER leverages this foundation to reduce Python interpreter overhead for common "update many agents" operations and to make filtering and aggregation over agent populations first-class operations in the modeling workflow [@amber_github].

## Software Description

### Core Abstraction: A Population with Columnar State

In AMBER, the *population* is the primary owner of agent state. Agent attributes are stored as columns in a tabular structure rather than being distributed across many Python objects. This design, described in AMBER's documentation as keeping the framework "still easy to use" while being "much faster," treats the population/state store and spatial abstractions (grid and continuous space) as central concepts [@amber_github]. Users define agent behaviors through agent types, while the underlying state is stored and updated via vectorized operations whenever possible.

This architecture yields two practical benefits:

1. **Efficient selection and aggregation:** Selecting agent subsets by attribute conditions (e.g., "all infected agents") is naturally expressed as column-based filters, and computing group statistics is straightforward.
2. **Streamlined data collection:** Because state is already tabular, model outputs can be exported or analyzed using standard data workflows without requiring per-agent object serialization.

### Spaces and Neighborhood Queries

Many ABMs depend on spatial locality—grid neighborhoods, continuous distance thresholds, or network adjacency. AMBER provides grid and continuous space abstractions and coordinates agent placement and movement through a unified space interface [@amber_github]. This separation of spatial logic from state storage allows models to retrieve neighborhoods and apply interaction rules cleanly.

### Experiments, Parameter Sweeps, and Optimization Utilities

Running multi-parameter experiments and Monte Carlo repetitions is a core component of the "ABM workflow" in Python [@foramitti2021agentpy]. AMBER includes experiment utilities and demonstrates optimization routines (e.g., evolutionary and Bayesian optimization) within its repository documentation, positioning these as integral parts of an end-to-end modeling workflow rather than external glue code [@amber_github]. Melodie similarly provides calibration and training modules with a focus on workflow infrastructure, though with a different architecture and additional data-management components [@yu2023melodie]. AMBER's approach keeps user-facing model definitions compact while delegating performance-sensitive state operations to the columnar backend.

## Performance Evaluation

To evaluate AMBER's scaling behavior, we benchmarked three canonical ABM workloads—Wealth Transfer, SIR Epidemic, and Random Walk—against Mesa and AgentPy across population sizes ranging from 100 to 10,000 agents. All benchmarks were run for 100 simulation steps on the same hardware. Figure 1 presents execution time (in seconds, log scale) as a function of agent count.

![Scaling comparison of AMBER, Mesa, and AgentPy across three benchmark models (Wealth Transfer, SIR Epidemic, Random Walk). AMBER demonstrates consistently lower execution times as agent populations scale, with the largest performance advantage in the Wealth Transfer model where batch attribute updates dominate runtime.](benchmarks/results/scaling_chart.png){#fig:scaling width="100%"}

Key observations include:

- **Wealth Transfer:** AMBER achieves up to 93× faster execution than Mesa at 10,000 agents (2.1s vs. 195.5s), as this workload is dominated by batch attribute updates—precisely the pattern AMBER optimizes.
- **SIR Epidemic:** AMBER maintains a 1.2–1.7× advantage over Mesa and AgentPy. The more modest speedup reflects the higher proportion of per-agent branching logic (state transitions).
- **Random Walk:** All frameworks show similar performance at small scales; AMBER's advantage becomes more pronounced at larger populations due to efficient vectorized position updates.

As with all ABM benchmarks, observed speedups depend on model characteristics: highly heterogeneous, branch-heavy per-agent logic may reduce the fraction of work expressible as batched column operations. AMBER is therefore best understood as an architectural option that makes *vectorizable, population-level operations* efficient in Python while retaining an agent-oriented model definition [@amber_github].

## Availability

AMBER is developed openly on GitHub [@amber_github]. Its performance-oriented backend relies on Polars and the Arrow ecosystem [@polars_zenodo; @arrow]. Project documentation includes usage examples, feature descriptions, and benchmark instructions [@amber_github].

## Acknowledgements

We acknowledge the open-source communities behind Polars and Apache Arrow, which enable high-performance, columnar data processing in the Python ecosystem [@polars_github; @arrow].

## AI Usage Disclosure

Generative AI tools were used to assist with manuscript editing and rephrasing. All technical claims, software descriptions, and citations were reviewed and verified by the authors.

## References
