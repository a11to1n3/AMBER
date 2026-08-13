Experiment
==========

.. automodule:: ambr.experiment
   :members:
   :undoc-members:
   :show-inheritance:

Tools for sequential parameter sweeps. **Parallelism is not automatic** —
see :class:`~ambr.performance.ParallelRunner` (CPU processes) and
:class:`~ambr.gpu_ensemble.GPUEnsembleRunner` (GPU batches).

Experiment class
----------------

.. autoclass:: ambr.Experiment
   :members:
   :undoc-members:

Canonical usage::

   import ambr as am
   from ambr import Experiment, Sample, IntRange

   class MyModel(am.Model):
       model_reporters = {"n": lambda m: len(m.agents)}
       def setup(self):
           self.add_agents(int(self.p.get("n_agents", 50)), wealth=1)
       def step_vectorized(self):
           pass

   sample = Sample(
       {
           "n_agents": IntRange(20, 60),  # start inclusive, end exclusive
           "steps": 10,
           "seed": [0, 1, 2],
           "show_progress": False,
       },
       n=6,  # number of combinations to draw
   )
   experiment = Experiment(
       model_type=MyModel,
       sample=sample,
       iterations=1,
   )
   results = experiment.run()
   # results is a dict:
   #   info       — Python dict (provenance / metadata), not a Polars frame
   #   parameters — Polars frame of sampled combinations
   #   agents     — Polars frame (concatenated end-of-run agent tables)
   #   model      — Polars frame (model-level time series across runs)
   print(results["info"])
   print(results["model"].head().to_dicts())

Legacy kwargs ``model_class=`` / ``parameters=`` still work with a
``DeprecationWarning`` (removed in **1.0**).

Sample class
------------

.. autoclass:: ambr.Sample
   :members:
   :undoc-members:

``Sample(parameters, n)`` requires **both** the parameter map and ``n``
(number of combinations). Combinations are available as
``sample.combinations`` (list of dicts).

**Sampling is not a full Cartesian product and not independent random
draws.** For each index ``i`` in ``0..n-1``:

* **Fixed scalars** are copied into every combination.
* **Lists** are **cycled by index** (``value_list[i % len]``). Multiple list
  parameters stay **aligned by index** — they are not crossed.
* **IntRange** values are **deterministically spread** across
  ``[start, end)`` (middle value when ``n == 1``).

For a true factorial grid use :func:`~ambr.grid_search` / your own product;
for independent random samples use :func:`~ambr.random_search`.

IntRange class
--------------

.. autoclass:: ambr.IntRange
   :members:
   :undoc-members:

Python ``range`` semantics: ``start`` inclusive, ``end`` exclusive
(``IntRange(1, 10)`` → values ``1..9``).
