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

**Spawn-safe usage.** ``ParallelRunner`` always uses
``multiprocessing`` **spawn**. The model class must be **importable**
(module-level, not defined in ``__main__`` interactively without a module
path). In scripts and notebooks, guard the entry point with
``if __name__ == "__main__":``. Inspect failures via
:class:`~ambr.performance.RunOutcome` (``status``, ``error_type``,
``error_message``).

Example (module-level model + ``__main__`` guard — required for spawn)::

.. code-block:: python

   import ambr as am
   from ambr import ParallelRunner

   class MyModel(am.Model):
       def setup(self):
           self.add_agents(int(self.p.get("n_agents", 20)), wealth=1)

       def step_vectorized(self):
           pass

   def main():
       runner = ParallelRunner(MyModel, n_workers=2)
       outs = runner.run([
           {"n_agents": 20, "steps": 5, "seed": s, "show_progress": False}
           for s in range(4)
       ])
       for o in outs:
           if o.status == "success":
               print(o.index, o.result["info"].get("run_uuid"))
           else:
               print(o.index, o.status, o.error_type, o.error_message)

   if __name__ == "__main__":
       main()

.. autofunction:: ambr.performance.check_performance_deps
