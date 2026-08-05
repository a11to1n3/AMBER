Visualization helpers
=====================

.. automodule:: ambr.viz
   :members:
   :undoc-members:
   :show-inheritance:

Lightweight matplotlib charts from :class:`~ambr.results.RunResults`. This is
**not** a Solara/dashboard product — use these for tutorials and notebooks,
or bring your own UI.

Install (matplotlib is already a core dependency; the extra is a doc alias)::

   pip install 'ambr[viz]'

Example::

   import ambr as am
   import matplotlib
   matplotlib.use("Agg")

   class M(am.Model):
       model_reporters = {"total": lambda m: int(m.agents.wealth.sum())}
       def setup(self):
           self.add_agents(30, wealth=1, x=0, y=0)
       def step_vectorized(self):
           pass

   r = M({"steps": 5, "seed": 0, "show_progress": False}).run()
   am.plot_timeseries(r, columns=["total"])
   am.plot_grid(r, x="x", y="y")
