Run results
===========

Containers returned by :meth:`ambr.Model.run`.

.. autoclass:: ambr.results.RunResults
   :members:
   :undoc-members:
   :show-inheritance:

Cookbook
--------

Attribute access (AgentPy-shaped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   results = model.run()
   results.agents          # same as results['agents']
   results.model           # model-level time series (Polars)
   results.info            # steps, run_time, device, mode, …
   print(results.keys_overview())

Group / aggregate with Polars
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   import polars as pl

   # last total_wealth per seed after an Experiment
   final = (
       experiment_results["model"]
       .group_by("seed")
       .agg(pl.col("total_wealth").last())
   )

Save / load
~~~~~~~~~~~

::

   results.save("artifacts/run_seed0")
   restored = am.RunResults.load("artifacts/run_seed0")
   assert restored.model.height == results.model.height

Compared to AgentPy ``DataDict``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

==========================  ===============================================
AgentPy                     AMBER
==========================  ===============================================
``results.variables.Model`` ``results.model`` (Polars DataFrame)
``results.variables.Agent`` opt-in ``agent_reporters`` → long ``agent_vars``
``results.info``            ``results.info`` (dict)
save/load helpers           :meth:`~ambr.results.RunResults.save` / ``load``
Sobol / arrange APIs        use Polars / external SALib as needed
==========================  ===============================================

See also :doc:`../from_agentpy`.
