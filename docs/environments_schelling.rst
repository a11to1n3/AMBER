Environments & Schelling
========================

AMBER ships three environment types:

* :class:`~ambr.GridEnvironment` — discrete lattice (Schelling, CA, grids)
* :class:`~ambr.SpaceEnvironment` — continuous 2D space
* :class:`~ambr.NetworkEnvironment` — graph / network topologies

Agent **attributes** stay on the columnar table (``model.agents`` /
``agents_df``). The environment owns **placement** (where each agent sits).

Grid occupancy helpers (0.4.1)
------------------------------

On :class:`~ambr.GridEnvironment`:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Helper
     - Role
   * - ``add_agent_from_id(id, pos)`` / ``add_agent(agent, pos)``
     - Place or move an agent onto a cell
   * - ``remove_agent_from_pos(pos)``
     - Clear a cell; returns the agent id that left
   * - ``get_agent_at_pos(pos)``
     - Occupant id or ``None``
   * - ``get_random_empty_cell()``
     - Uniform empty cell (uses ``model.rng`` when present)
   * - ``get_empty_cells_in_radius(pos, radius)``
     - Empty Moore neighbourhood
   * - ``get_neighbors(pos, radius=1)``
     - Moore ring of cells (alias of distance-based neighbours)
   * - ``empty_positions()``
     - All unoccupied cells

Positions are stored on the frame as ``grid_position`` (object column of
tuples). Prefer these helpers over hand-rolled occupancy dicts.

Canonical Schelling pattern
---------------------------

Full runnable code: ``examples/schelling_vectorized.py``.

1. **Create the grid and population**

   .. code-block:: python

      self.env = am.GridEnvironment(self, size=self.p.grid_size)
      self.agents = am.AgentList(self, self.p.n, SchellingAgent)
      self.agents.set(group=self.rng.integers(0, 2, size=self.p.n))

2. **Place agents once** (random unique cells)

   .. code-block:: python

      cells = list(self.env.positions)
      self.rng.shuffle(cells)
      for agent, pos in zip(self.agents, cells[: self.p.n]):
          self.env.add_agent_from_id(agent.id, pos)

3. **Each step: happiness (per-agent) then filter + relocate**

   Neighbourhood inspection is irregular → agent methods are fine.
   Selecting *who* moves is vectorized:

   .. code-block:: python

      for a in self.agents:
          a.update_happiness(self.p.want_similar)
      for aid in self.agents.where(~self.agents.happy).ids.to_list():
          self.agents.by_id(aid).relocate()  # uses get_random_empty_cell

4. **Report with model_reporters**

   .. code-block:: python

      model_reporters = {
          "happy_frac": lambda m: float(m.agents.happy.sum()) / max(len(m.agents), 1),
      }

When *not* to force full vectorization
--------------------------------------

Moore-neighbourhood Schelling happiness is a classic “per-agent is OK”
workload: each agent’s local view is small and sparse. AMBER still wins on
metrics, placement bookkeeping, and filtered selection (``where`` / ``set`` /
``scatter_add``). Dense continuous interactions are a better fit for the
tensor lane (``borrow`` / ``commit``) or :class:`~ambr.ArrayKernelModel` —
see :doc:`going_faster`.

Related examples
----------------

* ``examples/schelling_vectorized.py`` — short canonical grid Schelling
* ``examples/smac_calibration_advanced.py`` — multi-objective SMAC Schelling
* ``examples/segregation_model.py`` — notebook-style OOP segregation (legacy style)

Smoke test: ``tests/test_grid_schelling_helpers.py``.
