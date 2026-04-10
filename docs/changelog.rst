Changelog
=========

All notable changes to AMBER will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.2.0] - 2026-04-10
---------------------

Added
~~~~~
- Vectorized view API on ``AgentList`` as the primary vectorized interface.
  ``model.agents.where(predicate)`` / ``model.agents[mask]`` return filtered
  views; ``model.agents.at[ids]`` returns scatter views. Column attribute
  access on any view returns a Polars Series backed by ``model.agents_df``,
  and column attribute assignment queues through the batched flush path.
- ``Model.add_agents(n, agent_class=None, **columns)`` — columnar bulk-create
  that delegates to ``Population.batch_add_agents`` and, if ``agent_class``
  is supplied, also wires up lightweight Python ``Agent`` instances for
  per-agent method dispatch.
- ``view.scatter_add(**increments)`` — id-indexed accumulate for flow-of-
  resources updates. Correctly sums duplicate ids instead of last-write-wins.
- New ``Model.get_agent_data`` and ``Model.run_step`` methods, finally
  matching the signatures the tutorial and examples had been assuming.
- ``Population.batch_add_agents`` now accepts ``pl.Series`` column inputs
  alongside lists and numpy arrays.
- Full quickstart and tutorial rewrite showing the vectorized idiom as the
  default path. Docs now build strict (``sphinx-build -W``).
- ``TestVectorizedWorkflows`` integration tests covering wealth-transfer,
  SIR transitions, scatter-add invariants, and mixed per-agent/view usage.

Changed (breaking)
~~~~~~~~~~~~~~~~~~
- ``AgentList.<col>`` attribute access now reads from ``model.agents_df``
  and returns a Polars Series. Previously it aggregated Python attributes
  from each ``Agent`` instance into a numpy array, which silently desynced
  from any DataFrame-level write (the biggest footgun in the library).

  **Migration:** if you set custom state via Python attributes on Agent
  objects and read it back as a numpy array, switch to::

      # Before
      agent.custom = value
      arr = self.agents.custom  # np.ndarray (Python attrs)

      # After — column lives in the DataFrame
      self.agents.custom = value_array  # bulk write
      series = self.agents.custom       # pl.Series (agents_df)

- ``AgentList.__getattr__`` no longer forwards unknown names to per-agent
  method calls. Use the explicit ``agents.call('method_name', ...)`` entry
  point, which has been there since v0.1.4.

Deprecated
~~~~~~~~~~
- ``Population.create_batch_context()`` now emits a ``DeprecationWarning``.
  Replace with ``model.agents.at[ids].col = values`` or
  ``model.agents.at[ids].scatter_add(col=delta)``.

Fixed
~~~~~
- Batched ``Agent.record()`` path: per-call DataFrame clones are replaced
  by a hash-join flush. On a 10 000-agent wealth-transfer benchmark, step
  throughput improved by ~6300× over the previous per-call ``with_columns``
  path (0.33 s vs. ~35 min projected for 100 steps).
- ``AgentList.record()`` / ``update_data()`` on a subset view now only
  touch rows inside the subset. Previously both methods accidentally
  broadcast their value to every agent in the population.
- ``Agent.record()`` / ``update_data()`` auto-create missing columns
  instead of raising ``ColumnNotFoundError``.
- ``GridEnvironment.get_neighbors()`` with wrapping now deduplicates
  wrapped offsets and excludes the origin when ``distance >= dim``.
- ``MultiObjectiveSMAC`` target-function lambdas now capture their
  objective by value (loop-closure bug).
- ``pytest.ini`` / ``MANIFEST.in`` / ``docs/api/population.rst``: corrected
  ``src/amber`` → ``src/ambr`` path typo that had been hiding coverage
  tracking on ``population.py`` and autodoc on ``Population``.
- Sphinx docs now build under ``sphinx-build -W --keep-going`` with zero
  warnings.

[0.1.5] - 2026-01-30
---------------------

Fixed
~~~~~
- Fixed regression in ``Model.add_agent()`` making it defensive against users overwriting ``self.agents`` (critical for integration tests)

[0.1.4] - 2026-01-30
---------------------

Added
~~~~~
- Vectorized attribute access on ``AgentList`` (returns numpy arrays)
- Advanced indexing/masking support for ``AgentList`` (slices, list of indices, boolean masks)
- Concatenation support for ``AgentList`` using the ``+`` operator
- ``Model.record()`` as an alias for ``record_model()`` (AgentPy compatibility)

Changed
~~~~~~~
- ``AgentList.select()`` now supports boolean mask indexing
- ``AgentList.call()`` now returns a numpy array of results

[0.1.3] - 2026-01-28
---------------------

Fixed
~~~~~
- Fixed test suite to properly validate ``AttrDict`` wrapper behavior

[0.1.2] - 2026-01-28
---------------------

Added
~~~~~
- ``AttrDict`` class for AgentPy-compatible attribute-style parameter access (``self.p.param_name``)
- Method forwarding on ``AgentList`` for AgentPy-style syntax (``agents.method()`` calls method on all agents)

Changed
~~~~~~~
- ``AgentList.call()`` now returns list of results from each agent's method call
- ``BaseModel`` and ``BaseAgent`` now use ``AttrDict`` for parameter storage

[0.1.1] - 2026-01-21
---------------------

Fixed
~~~~~
- Renamed stale `amber` imports to `ambr` in example notebooks and documentation (`# <PR/Commit>`)
- Corrected version consistency across `setup.py`, `__init__.py`, and documentation

Changed
~~~~~~~
- Bumped version to 0.1.1

Added
~~~~~
- Comprehensive documentation with ReadTheDocs integration
- Jupyter notebook examples converted to Python scripts
- Tutorial and API reference documentation
- Contributing guidelines and development workflow

[0.1.0] - 2024-06-19
---------------------

Added
~~~~~
- Initial release of AMBER framework
- Core Model and Agent classes
- GridEnvironment, SpaceEnvironment, and NetworkEnvironment
- AgentList for managing agent collections
- Optimization module with grid search, random search, and Bayesian optimization
- Experiment framework for parameter sweeps
- Comprehensive test suite with 100% pass rate
- GitHub Actions CI/CD pipeline
- Coverage reporting with Codecov
- Example notebooks for various simulation types

Features
~~~~~~~~
- **High-Performance Data Structures**: Using Polars for fast DataFrame operations
- **Flexible Environments**: Support for grid, continuous space, and network topologies
- **Built-in Optimization**: Parameter tuning with multiple optimization algorithms
- **Comprehensive Testing**: Full test coverage across all modules
- **Professional Documentation**: Sphinx-based documentation with examples

Fixed
~~~~~
- Model execution flow and data recording issues
- Environment class compatibility with different input formats
- Random number generation compatibility across numpy versions
- DataFrame schema consistency in agent data management
- Mock object compatibility in test environments

Changed
~~~~~~~
- Moved from examples/ to docs/examples/ for better organization
- Enhanced README with professional presentation
- Improved error handling and user feedback

Technical Details
~~~~~~~~~~~~~~~~~
- **Dependencies**: Polars, NumPy, NetworkX, Matplotlib, Seaborn
- **Python Support**: 3.8+
- **Testing**: pytest with comprehensive coverage
- **Documentation**: Sphinx with ReadTheDocs theme
- **CI/CD**: GitHub Actions with multi-OS and multi-Python testing 