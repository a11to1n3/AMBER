API Reference
=============

This section provides detailed documentation for all AMBER classes, functions, and modules.

Core Components
---------------

.. toctree::
   :maxdepth: 2

   model
   agent
   population
   environments
   sequences

Utilities
---------

.. toctree::
   :maxdepth: 2

   optimization
   experiment
   base
   results
   performance

Advanced (0.4 / 0.4.1)
----------------------

.. toctree::
   :maxdepth: 2

   contract
   tensor_lane
   lanes
   gpu
   gpu_ensemble

Quick Reference
---------------

**Core Classes:**

* :class:`ambr.Model` - Base class for all agent-based models
* :class:`ambr.Agent` - Individual agent with behaviors and properties
* :class:`ambr.BaseAgent` - Abstract base class for custom agents
* :class:`ambr.BaseModel` - Abstract base class for custom models
* :class:`ambr.Population` - SoA memory manager for high-performance state handling
* :class:`ambr.RunResults` - Dict-like ``model.run()`` result with attribute access

**Environments:**

* :class:`ambr.GridEnvironment` - 2D grid-based spatial environment
* :class:`ambr.SpaceEnvironment` - Continuous 2D space environment
* :class:`ambr.NetworkEnvironment` - Graph/network-based environment

**Data Structures:**

* :class:`ambr.AgentList` - List-like container for agents with additional functionality

**Optimization:**

* :func:`ambr.grid_search` - Exhaustive parameter space search
* :func:`ambr.random_search` - Random parameter sampling
* :func:`ambr.bayesian_optimization` - Bayesian optimization of model parameters

**Experiments:**

* :class:`ambr.Experiment` - Framework for running multiple model configurations
* :class:`ambr.Sample` - Parameter sampling for experiments
* :class:`ambr.IntRange` - Integer range specification for parameters

**Snapshot-view contract (0.4):**

* :class:`ambr.contract.ContractCertificate` - Per-step conformance record from ``model.run(contract=...)``

**Tensor lane (0.4):**

* :func:`ambr.tensor_lane.borrow_numeric` - Zero-copy borrow of a numeric column
* :func:`ambr.tensor_lane.commit_columns` - Atomic write-back of derived columns

**Speed lanes (0.4.1):**

* :func:`ambr.print_status` / :func:`ambr.recommend` - machine/lane status and hints
* :class:`ambr.ArrayKernelModel` - single-run CuPy/NumPy array model
* Optional ``pip install 'ambr[perf]'`` (Numba) for CPU scatter JIT

**GPU & calibration (0.4):**

* :func:`ambr.gpu.get_array_module` - Resolve the CuPy/NumPy array module
* :class:`ambr.GPUEnsembleRunner` - Batched (B × N) ensemble runner (NumPy fallback in CI)
* :class:`ambr.BatchedWellMixedSIR` - Reference well-mixed SIR batched model
* :func:`ambr.smac_batch_calibrate` - SMAC calibration over a batched ensemble
