Changelog
=========

All notable changes to AMBER will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[0.4.3] - 2026-07-17
--------------------

Native GPU path and Keras-style device placement on one ``Model`` /
``step`` body.

Added
~~~~~
- Keras-style placement: ``model.cpu(mode=...)`` / ``model.gpu(mode=...)``
  (default mode ``vectorized``; ``run(mode=...)`` still overrides) via
  ``ambr.execution``.
- Device-resident columns for the native view API under ``Model.gpu()``
  (``device_columns``, GPU sequence writes, ``DeviceRNG``, ``scatter_add``).
- Tests: ``tests/test_execution_api.py``; expanded GPU backend tests.

Changed
~~~~~~~
- AMBER (GPU) benchmarks run the same vectorized models via
  ``model.gpu().run()`` (main harness is not a separate kernel path).
- Fused wealth throughput variant lives in ``amber_fused_models.py``.
- FLAME GPU 2 Schelling workload + GPU chart docs note.

[0.4.2] - 2026-07-11
--------------------

Hygiene, CI quality, and docs release on top of 0.4.1.

Added
~~~~~
- GPU ensemble CPU tests (``GPUEnsembleRunner`` / ``BatchedWellMixedSIR``) and
  public package exports for the ensemble helpers.
- CI installs ``ambr[perf]`` on Ubuntu + Python 3.12 (Numba scatter path).
- Canonical Schelling example + ``docs/environments_schelling.rst``.
- ``docs/deprecations.rst`` (removed in 1.0 table).
- Ruff + mypy CI job (gradual mypy module set).
- pre-commit hooks: nbstripout + ruff.

Changed
~~~~~~~
- Git history rewrite (drop paper drafts / notebook bloat from history).
- Branch protection on ``main`` / ``dev`` (required CI checks).
- Docs / README: public arXiv paper citation; manuscript drafts stay out of
  the package repository (``paper/`` ignored).
- Repo hygiene: stripped notebook outputs; no tracked benchmark JSON; tighter
  ``.gitignore`` / ``MANIFEST.in``.

[0.4.1] - 2026-07-11
--------------------

Polish release on top of 0.4.0: AgentPy-shaped UX, progressive speed lanes
(with Mac-friendly Numba), contract/write-path hardening, SMAC install
reliability, and Schelling/grid helpers.

Added
~~~~~
- UX / AgentPy lane: ``RunResults`` attribute access, ``agents.random()``,
  default ``show_progress=False``, and ``docs/from_agentpy.rst``.
- Easier speed lanes: ``am.print_status()`` / ``am.recommend(n)``,
  ``ArrayKernelModel``, ``agents.update_where``, ``docs/going_faster.rst``,
  ``examples/gpu_quickstart.py``.
- Optional Numba CPU path (``ambr[perf]``): JIT ``scatter_add`` / subset
  writes; status reports Numba (useful on Mac without CUDA).
- Shared write helpers: ``ambr._id_index`` (id→row cache) and
  ``performance.apply_scatter_add`` / ``apply_scatter_write``.
- Grid occupancy helpers on ``GridEnvironment`` for Schelling-style models
  (``get_random_empty_cell``, ``get_agent_at_pos``, ``add_agent_from_id``,
  ``remove_agent_from_pos``, ``get_empty_cells_in_radius``,
  ``get_neighbors(..., radius=)``). Restores ``examples/smac_calibration_advanced.py``.
- MultiObjectiveSMAC CI smoke; ``ambr[advanced]`` pins
  ``scikit-learn>=1.6.1,<1.9`` for SMAC 2.4 compatibility.

Changed
~~~~~~~
- Contract monitor extraction: runtime snapshot-view bookkeeping moved from
  ``Model`` into ``ambr.contract.ContractMonitor`` (``Model`` keeps
  ``contract_certificates`` / ``_contract_mode`` for callers).
- Unified write/contract seam: whole-column view writes and tensor-lane commits
  report through ``Model._set_frame(..., written_columns=...)``, so same-step
  double commits on the view path are visible under
  ``contract="check"|"warn"|"raise"``. ``scatter_add`` remains the sanctioned
  multi-write reducer (not counted as ordinary multi-write).
- Cross-path detection: same-step OOP + lane/view writes on one column raise
  ``cross_path_write``.
- Atomic ``agents.set`` (one frame update, one commit per column); safer
  ``None`` defaults for ``model_reporters`` / ``agent_reporters`` / ``params``.
- ``update_agent_data`` / ``batch_update_agents`` use the buffered and view
  write seams; ``Environment.df`` uses real ``Model._set_frame`` when available.
- Public package exports for ``ContractMonitor``, ``TensorLane``,
  ``borrow_numeric``, ``commit_columns``, GPU helpers, ``RunResults``,
  ``ArrayKernelModel``, and lane helpers (``status``, ``print_status``,
  ``recommend``).
- Canonical-verb docs: select / write / scatter_add / borrow-commit surface
  (quickstart, sequences API, README); examples prefer ``agents.at[...].set``
  over deprecated ``update_agent_data``.
- Write-path performance: shared id→row scatter for filtered assigns and OOP
  flush; contiguous ``0..N-1`` id layout cached per id-version.
- ``MultiObjectiveSMAC`` rebuilt on per-objective ``SMACOptimizer``.

Deprecated
~~~~~~~~~~
- ``Model.update_agent_data`` → ``agent.<col> = value`` / ``agents.at[id].set``
- ``Model.batch_update_agents`` → ``agents.at[ids].set``
- ``Population.set_agent_value`` / ``batch_update`` / ``batch_update_by_ids``
  → view ``set`` / column assign (until 1.0)
- Assigning ``Population.data = ...`` → view write path (setter warns;
  internal ``replace_frame`` is the quiet Model seam)

[0.4.0] - 2026-06-27
--------------------

The 0.4 release settles the public API on one canonical verb per task (legacy
spellings still work, emitting a ``DeprecationWarning`` and scheduled for
removal in 1.0; set ``AMBER_SUPPRESS_DEPRECATIONS=1`` to silence them), and adds
the snapshot-view contract checker, a resident tensor lane, a CuPy GPU backend,
and SMAC 2.x-compatible optimization.

Added
~~~~~
- Snapshot-view contract: ``model.run(contract="check"|"warn"|"raise")`` records
  a per-step ``ContractCertificate`` that columnar fast-path updates preserve the
  intended update schedule; inspect ``results["contract"]`` (mode ``off`` is the
  zero-overhead default).
- Tensor lane: zero-copy ``agents.borrow(col)`` / ``agents.commit(**cols)`` over
  the Polars frame, routed through the contract.
- GPU backend: ``ambr.gpu`` array-module abstraction with NumPy fallback, and
  ``ambr.gpu_ensemble`` (``GPUEnsembleRunner``, ``BatchedWellMixedSIR``,
  ``smac_batch_calibrate``) batching a ``(B × N)`` ensemble into one device pass.
- Declarative ``model_reporters`` / ``agent_reporters`` and ``record_initial``;
  typed ``params`` schema and ``AttrDict.get_int`` / ``get_float`` / ``get_bool``.
- ``AgentList`` façades ``by_id`` / ``numpy`` / ``set`` / ``borrow`` / ``commit`` /
  ``frame``; ``add_agents(n, agent_class=...)`` tracks Python agent objects.
- Cross-framework calibration and scaling benchmarks (Agents.jl, mesa-frames,
  FLAME GPU) and a tensor-lane flocking example.

Changed
~~~~~~~
- One canonical RNG: ``self.rng`` (NumPy ``Generator``) and ``self.random``
  (stdlib), both seeded from ``seed``; seeded runs no longer touch global
  ``np.random``.
- Encapsulated write boundary (internal ``Model._set_frame``); ``agents.frame``
  read accessor.
- ``update()`` is a pure hook (no ``super().update()`` needed); step advance and
  data finalize moved into the runner.
- Vectorized ``SpaceEnvironment.get_neighbors`` (NumPy + scipy KD-tree);
  ``wrap=`` collapses to ``torus=``. Examples migrated to ``rng``.

Fixed
~~~~~
- ``optimization``: explicit ``rng`` threading (no global seed mutation) and
  SMAC 2.x compatibility for ``SMACOptimizer`` / ``bayesian_optimization``.
- ``NetworkEnvironment.__init__`` no longer drops ``node_id`` when adding
  ``network_distance``; ``SpaceEnvironment.move_agent`` no longer raises a dtype
  error on object-typed position columns.

Deprecated
~~~~~~~~~~~
- Legacy verbs kept until 1.0 (each warns toward the canonical form):
  ``nprandom`` → ``rng``; ``model.record`` → ``record_model`` /
  ``model_reporters``; ``agents.select`` → ``agents.where``; ``agents.record`` /
  ``agents.update_data`` → ``agents.set`` / ``agent.<col> = v``; ``agents.agents``
  / ``agents.agent_ids`` → iterate / ``agents.by_id``; ``GridEnvironment(wrap=)``
  / ``.wrap`` → ``torus=``.

[0.3.13] - 2026-06-04
---------------------

Changed
~~~~~~~
- Ignored root-level LaTeX paper build outputs so local manuscript
  compilation does not dirty package release branches.
- Kept project metadata, runtime fallback version, documentation version, and
  changelogs aligned for the package release.

[0.3.12] - 2026-06-04
---------------------

Changed
~~~~~~~
- Tightened the source distribution surface so local-only directories such as
  tests, benchmark fixtures, documentation builds, GitHub metadata, assistant
  metadata, and paper drafts stay out of package archives.
- Added release-workflow package-surface validation for the wheel and source
  distribution before creating the GitHub release.
- Kept project metadata, runtime fallback version, documentation version, and
  changelogs aligned for the package release.

[0.3.11] - 2026-06-04
---------------------

Changed
~~~~~~~
- Cut a clean package release from an up-to-date ``dev`` branch into ``main``.
- Kept release metadata, runtime fallback version, documentation version, and
  changelogs aligned for the wheel and source distribution.
- Explicitly ignored local assistant metadata along with paper drafts and paper
  archives so package releases stay focused on the AMBER library.

[0.3.10] - 2026-06-04
----------------------

Changed
~~~~~~~
- Cut a main-anchored package release after synchronizing ``dev``, ``main``,
  and their remotes.
- Kept package metadata, runtime fallback version, documentation version, and
  release notes aligned for the distribution artifacts.
- Preserved local paper drafts and generated paper archives as ignored,
  non-package artifacts.

[0.3.9] - 2026-06-04
---------------------

Changed
~~~~~~~
- Added a root split-SIR benchmark runner with deterministic shared inputs,
  explicit sync/async schedules, Agents.jl fixtures, checked result artifacts,
  and declared async SimPy budget skips.
- Added a root dynamic-graph coordination benchmark runner and checked result
  artifacts across AMBER, NumPy, Polars, a Python object loop, Mesa, AgentPy,
  and Agents.jl.
- Clarified public benchmark wording so the schedule-mixed headline SIR row is
  not presented as an equivalent-trajectory AMBER-over-Julia claim.

[0.3.8] - 2026-06-04
---------------------

Changed
~~~~~~~
- Upgraded the all-framework benchmark runner to default to 10 full-run
  samples, preserve raw Agents.jl subprocess samples, and write raw sample
  counts, IQRs, and bootstrap median intervals into
  ``benchmark_results_all.json``.
- Regenerated the all-framework headline benchmark artifacts so all 63
  framework/model/size rows carry 10 raw timing samples, including Agents.jl.

[0.3.7] - 2026-06-04
---------------------

Fixed
~~~~~
- Moved GitHub Actions workflows to current Node-24-ready action versions for
  release, checkout, setup-python, upload-artifact, and Codecov steps.
- Corrected the Codecov upload input from ``file`` to ``files``.

[0.3.6] - 2026-06-04
---------------------

Changed
~~~~~~~
- Added a tag-driven GitHub release workflow that builds and validates wheel
  and source distributions before attaching them to the release.
- Extended CI coverage to the ``dev`` branch and Python 3.9, matching the
  declared package support floor.
- Added package metadata consistency checks so runtime, project, and
  documentation versions do not drift across releases.
- Tightened release-package hygiene around generated paper outputs and source
  distribution contents.

[0.3.5] - 2026-06-04
---------------------

Changed
~~~~~~~
- Cut a package-only release with synchronized project metadata, runtime
  version, and Sphinx documentation version.
- Kept local paper drafts and generated paper archives outside the tracked
  package release surface.

[0.3.4] - 2026-06-03
---------------------

Fixed
~~~~~
- Synchronized Sphinx documentation metadata with the package release version.
- Kept runtime, project metadata, and generated documentation versions aligned
  for patch releases.

[0.3.3] - 2026-06-03
---------------------

Changed
~~~~~~~
- Package metadata now lives in ``pyproject.toml`` and builds through the
  standard PEP 517/518 Python packaging flow.
- ``setup.py`` is kept as a compatibility shim for older editable-install
  tooling.
- Wheel and source distributions now include package URLs and keyword
  metadata.

Fixed
~~~~~
- Removed tracked coverage artifacts from the repository.
- Tightened source-distribution hygiene so local paper drafts, caches, and
  coverage files stay out of release artifacts.
- Corrected the ``Makefile`` coverage target to use the ``src/ambr`` package
  path.

[0.3.2] - 2026-06-03
---------------------

Fixed
~~~~~
- ``Model.run()`` and ``Model.run_step()`` now execute exactly the requested
  number of model steps after setup.
- Benchmark helpers now validate structural correctness before timing wealth
  transfer, random walk, and SIR runs.
- Agents.jl benchmark parameters are routed through the master runner instead
  of hardcoded step counts.

Changed
~~~~~~~
- Regenerated all-framework benchmark results with seeded timing,
  slowest-sample trimming, and documented SIR update-ordering caveats.
- Preserved raw per-run timing samples for Python-hosted frameworks in the
  benchmark JSON.
- Updated README benchmark tables and installation docs for the current Python
  support floor.
- Stopped tracking draft paper files in the package repository.

[0.3.0] - 2026-05-09
---------------------

Added
~~~~~
- ``Agent.__setattr__`` now routes non-internal attribute writes through
  ``model._queue_write()``, keeping Python ``Agent`` objects and the
  columnar DataFrame in sync. Setting ``agent.wealth = 5`` automatically
  updates the DataFrame — no more silent desync between OOP-style and
  view-API access.
- ``Environment.df`` is now a property that reads from and writes to
  ``model.population.data`` (or ``model.agents_df`` for mock models),
  eliminating DataFrame staleness in ``GridEnvironment``,
  ``SpaceEnvironment``, and ``NetworkEnvironment``.
- ``IntRange`` now supports ``__contains__``, ``__iter__``, and ``__len__``
  for better ergonomics with Python's standard range semantics.
- ``SpaceEnvironment`` initialisation no longer drops the ``space_position``
  column when both ``space_position`` and ``space_distance`` need to be
  created.

Changed
~~~~~~~
- ``IntRange`` semantics are now consistently **exclusive-end**, matching
  Python's ``range()``: ``IntRange(1, 10)`` yields values ``1..9``.
  Previously the semantics varied across ``ParameterSpace.sample()``
  (inclusive) and ``Sample._generate_combinations()`` (exclusive).
- ``Model.run()`` now delegates to ``Model.run_step()``, removing 30 lines
  of duplicated setup/step/update/finalize logic.
- ``Model.add_agents(n, agent_class=..., **columns)`` now forwards Python
  ``Agent`` attributes set during ``setup()`` into the batch DataFrame
  write, so columns declared via ``self.wealth = 5`` in ``setup()``
  appear in the population automatically.
- ``Population._align_and_concat()`` now emits a ``UserWarning`` instead
  of silently casting mismatched column types to ``Utf8``.
- ``__init__.py`` namespace cleanup was removed; the aggressive module-
  level symbol deletion no longer hides legitimately imported names.
- Dependencies: ``setup.py`` now declares ``python_requires=">=3.9"``
  (was ``>=3.8``, which is incompatible with ``numpy>=1.20``).
  Python 3.8 classifier removed.
- Test suite: ``test_population.py`` migrated from ``unittest.TestCase``
  to ``pytest`` style, consistent with every other test file.

Fixed
~~~~~
- ``bayesian_optimization()`` is now backed by **SMAC3** (RandomForest
  surrogate model + Expected Improvement acquisition) instead of being
  pure random search. Requires ``smac`` to be installed. The simple
  ``ParameterSpace`` is automatically converted to a SMAC3
  ``ConfigurationSpace``.
- ``Environment.df`` setter correctly handles mock models (no longer
  trips on ``hasattr`` returning ``True`` for auto-created Mock attrs).
- ``SpaceEnvironment`` constructor no longer overwrites the
  ``space_position`` column when adding ``space_distance``.

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
- **Python Support**: 3.9+
- **Testing**: pytest with comprehensive coverage
- **Documentation**: Sphinx with ReadTheDocs theme
- **CI/CD**: GitHub Actions with multi-OS and multi-Python testing
