Visualization helpers
=====================

.. automodule:: ambr.viz
   :members:
   :undoc-members:
   :show-inheritance:

Lightweight matplotlib charts from :class:`~ambr.results.RunResults`. This is
**not** a Solara/dashboard product — use these for tutorials and notebooks,
or bring your own UI.

Install matplotlib via the optional extra (not a core dependency)::

   pip install 'ambr[viz]'

In CI / headless docs builds set ``MPLBACKEND=Agg`` (the library never calls
``matplotlib.use`` itself). Example::

   # export MPLBACKEND=Agg   # CI / headless
   import ambr as am

   class M(am.Model):
       model_reporters = {"total": lambda m: int(m.agents.wealth.sum())}
       def setup(self):
           self.add_agents(30, wealth=1, x=0, y=0)
       def step_vectorized(self):
           pass

   r = M({"steps": 5, "seed": 0, "show_progress": False}).run()
   am.plot_timeseries(r, columns=["total"])
   am.plot_grid(r, x="x", y="y")
