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
   scheduling
   viz

Utilities
---------

.. toctree::
   :maxdepth: 2

   optimization
   experiment
   base
   results
   performance

Advanced
--------

.. toctree::
   :maxdepth: 2

   contract
   tensor_lane
   lanes
   gpu
   gpu_ensemble
   scheduling
   viz

Quick Reference
---------------

**Core Classes:**

* :class:`ambr.Model` - Base class for user models (``setup`` / ``step`` / ``run``)
* :class:`ambr.Agent` - User-facing agent (``setup``, DataFrame-synced attributes)
* :class:`ambr.BaseModel` / :class:`ambr.BaseAgent` - Low-level primitives (prefer ``Model`` / ``Agent``)
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
* :func:`ambr.bayesian_optimization` - SMAC RandomForest Bayesian optimization

**Experiments:**

* :class:`ambr.Experiment` - Framework for running multiple model configurations
* :class:`ambr.Sample` - Parameter sampling for experiments
* :class:`ambr.IntRange` - Integer range specification for parameters

**Snapshot-view contract:**

* :class:`ambr.contract.ContractCertificate` - Per-step **operational** monitor
  record from ``model.run(contract=...)`` (not a schedule proof)

**Tensor lane:**

* :func:`ambr.tensor_lane.borrow_numeric` - Zero-copy borrow of a numeric column
* :func:`ambr.tensor_lane.commit_columns` - Atomic write-back of derived columns

**Speed lanes:**

* :func:`ambr.print_status` / :func:`ambr.recommend` - machine/lane status and hints
* :class:`ambr.ArrayKernelModel` - single-run CuPy/NumPy array model
* Optional ``pip install 'ambr[perf]'`` (Numba) for CPU scatter JIT

**Device placement & lanes:**

* :meth:`ambr.Model.cpu` / :meth:`ambr.Model.gpu` - Keras-style placement;
  ``step_vectorized`` / ``step_oop`` (GPU is vectorized-only, NVIDIA+CuPy;
  see :doc:`gpu`, :doc:`../going_faster`)
* :meth:`ambr.Model.approve_fast_path` / :meth:`ambr.Model.revoke_fast_path_approval`
  - opt-in private GPU loop; **caller-attested** label only
* ``ambr.EXECUTION_DEVICES`` / ``EXECUTION_MODES`` / ``ExecutionConfig``

**GPU & calibration:**

* :func:`ambr.gpu.get_array_module` - CuPy/NumPy array module
* :class:`ambr.GPUEnsembleRunner` - Batched (B × N) ensemble (NumPy fallback)
* :class:`ambr.BatchedWellMixedSIR` - Reference well-mixed SIR batched model
* :func:`ambr.smac_batch_calibrate` - SMAC over a batched ensemble
  (``ambr[advanced]``)

**Activation & viz (0.4.6):**

* :func:`ambr.activate` / :meth:`ambr.Model.activate_agents` - OOP activation
  helpers (not a schedule proof); see :doc:`scheduling`
* :func:`ambr.plot_timeseries` / :func:`ambr.plot_grid` - matplotlib helpers;
  see :doc:`viz`
