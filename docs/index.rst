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

* **AgentPy-shaped OOP + vectorized lanes** on the same model (see
  :doc:`from_agentpy` and :doc:`going_faster`)
* **Vectorized view API**: ``where`` / ``at`` / ``set`` / ``scatter_add`` —
  do not mutate ``agents.array(...)`` in place on CPU
* **Snapshot-view contract**: opt-in **operational** monitor (not a schedule
  proof; default ``off``)
* **CPU acceleration**: optional Numba (``pip install 'ambr[perf]'``) —
  recommended on Mac / no-CUDA
* **Device placement**: Keras-style ``model.cpu(mode=...).run()`` /
  ``model.gpu().run()`` (GPU is **vectorized-only**, **NVIDIA + CuPy** only —
  not Apple Metal/MPS); see :doc:`going_faster`
* **GPU ensemble**: ``GPUEnsembleRunner`` for many short runs; parallelism is
  **never** automatic from a single ``.run()``
* **OOP activation helpers**: ``activate_agents("random"|"sequential"|"simultaneous")``
* **Viz helpers**: ``plot_timeseries`` / ``plot_grid`` (``ambr[viz]``)
* **Environments**: grid, continuous space, and network topologies
* **Optimization**: grid / random / Bayesian (SMAC) and GPU-batched calibration
* **RunResults**: attribute or dict access; ``save`` / ``load``
* **Reproducible**: seeded ``self.rng``; see :doc:`reproducibility`

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

       def step_vectorized(self):
           # Every agent with wealth > 0 gives $1 to a random other agent.
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1
           recipients = self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
           self.agents.at[recipients].scatter_add(wealth=1)

   # Fluent placement: GPU when NVIDIA+CuPy available, else CPU vectorized
   model = WealthModel({'steps': 50, 'seed': 42, 'show_progress': False})
   if am.GPU_AVAILABLE:
       results = model.gpu().run()
   else:
       results = model.cpu(mode="vectorized").run()
   print(results.info)
   print(results.model.tail(3).to_dicts())       # also results['model']
   print(am.recommend(10_000))

For more examples, check the ``examples/`` directory in the repository.
See :doc:`changelog` for **0.5.0** (step-data integrity, versioned RunResults,
strict optimization, extras / first-run honesty, real release/GPU gates).
Earlier 0.4.x notes: 0.4.7 label scrub; 0.4.6 doc-fence CI / RunResults I/O /
1.0 prep; 0.4.5 ``ambr[gpu]``; 0.4.4 lanes and the operational contract.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   from_agentpy
   going_faster
   environments_schelling
   tutorial
   reproducibility
   paper_and_package
   versioning
   public_api
   roadmap_1_0
   release_gates
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
   deprecations
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
