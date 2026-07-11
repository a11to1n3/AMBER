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

.. autofunction:: ambr.performance.check_performance_deps
