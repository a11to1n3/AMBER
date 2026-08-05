Public API surface
==================

The **supported** import surface is ``ambr.__all__``. Prefer::

   import ambr as am
   # am.Model, am.Agent, am.AgentList, …

Core modelling
--------------

* ``Model``, ``Agent``, ``BaseModel``, ``BaseAgent``
* ``AgentList``, ``Population``, ``BatchUpdateContext``
* ``GridEnvironment``, ``SpaceEnvironment``, ``NetworkEnvironment``
* ``RunResults``

Execution & lanes
-----------------

* ``EXECUTION_DEVICES``, ``EXECUTION_MODES``, ``ExecutionConfig``
* ``ArrayKernelModel``, ``status``, ``print_status``, ``recommend``
* ``TensorLane``, ``borrow_numeric``, ``commit_columns``
* ``GPU_AVAILABLE``, ``get_array_module``, ``to_device``, ``to_host``,
  ``require_gpu``, ``synchronize``, ``scatter_add``
* ``GPUEnsembleRunner``, ``BatchedWellMixedSIR``, ``smac_batch_calibrate``

Experiments & optimization
--------------------------

* ``Experiment``, ``Sample``, ``IntRange``
* ``ParameterSpace``, ``grid_search``, ``random_search``,
  ``bayesian_optimization``, ``SMACOptimizer``, ``MultiObjectiveSMAC``,
  ``SMACParameterSpace``
* ``ParallelRunner``, ``SpatialIndex``, performance helpers
  (``HAS_NUMBA``, ``HAS_SCIPY``, …)

Contract
--------

* ``ContractCertificate``, ``ContractViolation``, ``ContractViolationError``,
  ``ContractMonitor``, ``CONTRACT_MODES``

Activation & viz (optional helpers)
-----------------------------------

* ``activate``, ``Activation``, ``SequentialActivation``,
  ``RandomActivation``, ``SimultaneousActivation``, ``shuffled_ids``,
  ``ACTIVATION_MODES``, ``normalize_activation``
* ``plot_timeseries``, ``plot_grid``, ``HAS_MATPLOTLIB``

Not public
----------

* ``ambr._deprecation``, ``ambr._id_index``, other ``ambr._*`` modules
* Benchmark-private GPU kernels under ``benchmarks/models/``
* Undocumented ``Model._*`` methods (except where Sphinx explicitly documents
  an extension hook)

Verify in code::

   import ambr as am
   print(sorted(am.__all__))

See :doc:`versioning` for SemVer guarantees on this list.
