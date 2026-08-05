Activation helpers (OOP)
========================

.. automodule:: ambr.scheduling
   :members:
   :undoc-members:
   :show-inheritance:

Thin Mesa-inspired activation for **tracked Python agents**. Vectorized models
encode order inside ``step_vectorized`` (see :func:`~ambr.scheduling.shuffled_ids`).

These helpers do **not** prove schedule equivalence. The snapshot-view contract
remains an operational monitor only.

Example (OOP)::

   import ambr as am

   class Walker(am.Agent):
       def setup(self):
           self.x = 0
       def step(self):
           self.x += 1

   class WalkModel(am.Model):
       def setup(self):
           self.agents = am.AgentList(self, self.p.n, Walker)
       def step_oop(self):
           # sequential | random | simultaneous
           self.activate_agents(mode=self.p.get("activation", "random"))

   results = WalkModel(
       {"n": 50, "steps": 10, "seed": 0, "show_progress": False}
   ).cpu(mode="oop").run()
