AMBER Documentation
===================

**AMBER** (Agent-based Modeling with Blazingly Efficient Records) is a powerful Python framework for building and running agent-based models. It provides a comprehensive toolkit for researchers and practitioners to create complex simulations with ease.

.. image:: https://github.com/a11to1n3/AMBER/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/a11to1n3/AMBER/actions/workflows/ci.yml
   :alt: CI Status

.. image:: https://codecov.io/gh/a11to1n3/AMBER/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/a11to1n3/AMBER
   :alt: Coverage

.. image:: https://img.shields.io/pypi/v/ambr.svg
   :target: https://pypi.org/project/ambr/
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/ambr.svg
   :target: https://pypi.org/project/ambr/
   :alt: Python Versions

Features
--------

* **Vectorized view API**: update the whole population in a handful of Polars
  expressions — no per-agent loops, regardless of population size
* **Snapshot-view contract**: opt-in runtime checking that the columnar fast
  path preserves the intended update schedule (the zero-overhead default is off)
* **GPU backend**: a CuPy array backend with a NumPy fallback, plus a batched
  ensemble that runs ``B`` simulations in one device pass for calibration
* **Flexible environments**: grid, continuous space, and network topologies
* **Optimization**: grid / random / Bayesian (SMAC) search and GPU-batched calibration
* **Declarative reporting & typed params**: ``model_reporters`` / ``agent_reporters``
  and a class-level ``params`` schema
* **Reproducible**: one canonical seeded RNG (``self.rng``); deterministic runs

Quick Start
-----------

Install AMBER using pip:

.. code-block:: bash

   pip install ambr

Create your first model:

.. code-block:: python

   import ambr as am

   class WealthModel(am.Model):
       # Declarative per-step metric -> results['model'].
       model_reporters = {'total_wealth': lambda m: int(m.agents.wealth.sum())}

       def setup(self):
           # Bulk-create the population in one columnar write — no per-agent loop.
           self.add_agents(100, wealth=self.rng.integers(1, 10, size=100))

       def step(self):
           # Every agent with wealth > 0 gives $1 to a random other agent.
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1
           recipients = self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

   # Run the model
   model = WealthModel({'steps': 50, 'seed': 42})
   results = model.run()

For more examples, check the ``examples/`` directory in the repository.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorial
   benchmarks
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 