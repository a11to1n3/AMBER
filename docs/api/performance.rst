Performance utilities
=====================

Optional CPU acceleration (Numba, SciPy KD-tree) and parallel experiment
helpers. Install with::

   pip install 'ambr[perf]'

Flags
-----

.. autodata:: ambr.performance.HAS_NUMBA
   :annotation:

.. autodata:: ambr.performance.HAS_SCIPY
   :annotation:

Scatter helpers (vectorized write path)
---------------------------------------

.. autofunction:: ambr.performance.apply_scatter_add

.. autofunction:: ambr.performance.apply_scatter_write

.. autofunction:: ambr.performance.scatter_add_1d

.. autofunction:: ambr.performance.scatter_write_1d

Spatial & parallel
------------------

.. autoclass:: ambr.performance.SpatialIndex
   :members:

.. autoclass:: ambr.performance.ParallelRunner
   :members:

**Parallelism is opt-in.** ``model.run()`` always executes **one** simulation
on the calling process. Use:

* :class:`~ambr.performance.ParallelRunner` — many independent CPU processes
* :class:`~ambr.experiment.Experiment` — sequential parameter sweep (one process)
* :class:`~ambr.gpu_ensemble.GPUEnsembleRunner` — many short runs as one ``(B, N)`` GPU/CPU batch

Example::

   from ambr import ParallelRunner

   runner = ParallelRunner(MyModel, n_workers=4)
   outs = runner.run([
       {"steps": 20, "seed": s, "show_progress": False} for s in range(8)
   ])

.. autofunction:: ambr.performance.check_performance_deps
