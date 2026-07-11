# Changelog

## Unreleased

## v0.4.1 - 2026-07-11

Polish release on top of 0.4.0: clearer AgentPy-shaped UX, progressive speed
lanes (including Mac-friendly Numba), contract/write-path hardening, SMAC
install reliability, and Schelling/grid helpers.

### Added

- **UX / AgentPy lane:** `RunResults` (dict + `results.agents` attr access),
  `agents.random()`, quieter default (`show_progress=False`), and
  `docs/from_agentpy.rst` (side-by-side migration + product judgement).
- **Easier speed lanes:** `am.print_status()` / `am.recommend(n)`,
  `ArrayKernelModel` (single-run GPU/CPU arrays), `agents.update_where(...)`,
  and `docs/going_faster.rst` + `examples/gpu_quickstart.py`.
- **Numba CPU path:** optional `ambr[perf]` (`numba`); JIT `scatter_add` and
  subset column writes when installed (strong default for Mac / no-CUDA).
  Status/recommend report Numba; `am.numba_jit` re-exports the decorator.
- **Shared write helpers:** `ambr._id_index` (id→row cache) and
  `performance.apply_scatter_add` / `apply_scatter_write` (one path for the
  view API and OOP flush).
- **Grid occupancy helpers** on `GridEnvironment` for Schelling-style models:
  `get_random_empty_cell`, `get_agent_at_pos`, `add_agent` / `add_agent_from_id`,
  `remove_agent_from_pos`, `get_empty_cells_in_radius`, and `get_neighbors(..., radius=)`
  (Moore neighbourhood alias). Restores `examples/smac_calibration_advanced.py`.
- **MultiObjectiveSMAC CI smoke** (`tests/test_multiobjective_smac.py`, skipped
  without the `advanced` / smac extra).
- **SMAC install constraint:** `ambr[advanced]` pins `scikit-learn>=1.6.1,<1.9`
  so SMAC 2.4 can import (`sklearn.tree._tree.DTYPE` removed in sklearn 1.9;
  automl/SMAC3#1314). Clearer error if SMAC import fails.

### Changed

- **Contract monitor extraction.** Runtime snapshot-view bookkeeping moved from
  `Model` into `ambr.contract.ContractMonitor`. `Model` keeps a thin public
  surface (`contract_certificates`, `_contract_mode`) for callers and tests.
- **Unified write/contract seam.** Whole-column view writes
  (`agents.col = ...`) and tensor-lane commits report through
  `Model._set_frame(..., written_columns=...)`, so same-step double commits on
  the view path are visible to `contract="check"|"warn"|"raise"`. `scatter_add`
  still does not count as an ordinary multi-write (sanctioned reducer).
- **Cross-path detection.** Same-step writes that mix the buffered (OOP) path
  and the lane/view path on one column raise `cross_path_write`.
- **Atomic `agents.set`.** Multi-column `set(...)` / deprecated `update_data`
  apply in one frame update with one contract commit per column.
- **Safer class defaults.** `Model.model_reporters` / `agent_reporters` /
  `params` default to `None` (no shared mutable `{}` / `[]` on the base class).
- `update_agent_data` / `batch_update_agents` route through the buffered and
  view write seams (contract-observed) instead of raw `Population` mutators.
- `Environment.df` routes through a real `Model._set_frame` when available;
  deprecated `Agent.record` / `update_data` go through `__setattr__` so the
  instance cache and write queue stay aligned.
- Public package exports for `ContractMonitor`, `TensorLane`, `borrow_numeric`,
  `commit_columns`, GPU helpers, `RunResults`, `ArrayKernelModel`, and lane
  helpers (`status`, `print_status`, `recommend`).
- **Canonical-verb docs.** Quickstart / sequences API / README document the
  small select → write → scatter_add → borrow/commit surface; extra batch
  aliases are not the performance path. Examples prefer `agents.at[...].set`
  over deprecated `update_agent_data`.
- **Write-path performance.** Filtered `view.col = …` uses id→row scatter into
  column arrays when ids are unique; OOP flush uses the same position map;
  contiguous `0..N-1` id layout is cached per id-version. `MultiObjectiveSMAC`
  rebuilt on per-objective `SMACOptimizer` (no broken Abstract MO path).

### Deprecated

- `Model.update_agent_data` → `agent.<col> = value` or `agents.at[id].set(...)`
- `Model.batch_update_agents` → `agents.at[ids].set(...)`
- `Population.set_agent_value` / `batch_update` / `batch_update_by_ids` →
  view `set` / column assign (still functional until 1.0)
- Assigning `Population.data = ...` → view write path (setter warns; internal
  `replace_frame` is the quiet Model seam)

## v0.4.0 - 2026-06-27

The 0.4 release settles the public API on one canonical verb per task (legacy
spellings still work, emitting a `DeprecationWarning` and scheduled for removal
in 1.0; set `AMBER_SUPPRESS_DEPRECATIONS=1` to silence them in benchmark /
reproducibility runs), and adds the snapshot-view contract checker, a resident
tensor lane, a CuPy GPU backend, and SMAC 2.x-compatible optimization.

### Added

- **Snapshot-view contract.** `model.run(contract="check"|"warn"|"raise")` records
  a per-step `ContractCertificate` certifying that columnar fast-path updates
  preserve the intended update schedule (e.g. catching a same-step
  read-after-write). Inspect `results["contract"]`; mode `off` (default) adds
  zero overhead. (`ambr.contract`)
- **Tensor lane.** Zero-copy `agents.borrow(col)` / `agents.commit(**cols)` over
  the Polars frame for NumPy/array kernels, routed through the contract so
  borrow/commit stay observable. (`ambr.tensor_lane`)
- **GPU backend.** `ambr.gpu` array-module abstraction (`get_array_module`,
  `to_device`, `to_host`) with a NumPy fallback when CuPy is unavailable, plus
  `ambr.gpu_ensemble` (`GPUEnsembleRunner`, `BatchedWellMixedSIR`,
  `smac_batch_calibrate`) that batches a `(B simulations × N agents)` ensemble
  into one device pass for derivative-free calibration at scale.
- **Declarative reporting.** Class-level `model_reporters` / `agent_reporters`
  evaluated by the runner, plus `record_initial = True` to capture a `t=0` row.
- **Typed parameters.** Class-level `params = {"n": (int, 200)}` schema pre-coerces
  `self.p.n` at init; `AttrDict.get_int` / `get_float` / `get_bool`.
- **Collection & state façades** on `AgentList`: `by_id`, `numpy`, `set`,
  `borrow`, `commit`, `frame`; `add_agents(n, agent_class=...)` tracks the Python
  agent objects so OOP-style code can iterate `model.agents` / `agents.by_id(i)`.
- Cross-framework calibration and scaling benchmarks (Agents.jl, mesa-frames,
  FLAME GPU) and a tensor-lane flocking example.

### Changed

- **One canonical RNG.** `self.rng` (a NumPy `Generator`) is canonical and
  `self.random` is the stdlib one; both are seeded from `seed`. Seeded runs no
  longer leak into the process-global `np.random`.
- **Encapsulated write boundary.** All frame writes route through an internal
  `Model._set_frame`; `model.agents.frame` reads the current frame.
- `update()` is now a pure hook — overriding it no longer requires
  `super().update()`. Step advance and per-step data finalize moved into the
  runner.
- `SpaceEnvironment.get_neighbors` is vectorized (NumPy brute force with a scipy
  KD-tree fast path for large populations); `wrap=` collapses to `torus=`.
- Examples migrated to the canonical `rng`; benchmark models updated.

### Fixed

- `optimization`: explicit `rng` threading through `ParameterSpace.sample` and
  `random_search` (no global-`np.random` seed mutation), and SMAC 2.x
  compatibility for `SMACOptimizer` / `bayesian_optimization`.
- `NetworkEnvironment.__init__` no longer drops the `node_id` column when it adds
  `network_distance`.
- `SpaceEnvironment.move_agent` no longer raises a dtype error against
  object-typed position columns.

### Deprecated

- Legacy verbs kept working until 1.0, each emitting a `DeprecationWarning`
  pointing at the canonical form: `self.nprandom` → `self.rng`; `model.record` →
  `record_model` / `model_reporters`; `agents.select` → `agents.where`;
  `agents.record` / `agents.update_data` → `agents.set` / `agent.<col> = v`;
  `agents.agents` / `agents.agent_ids` → iterate `model.agents` / `agents.by_id`;
  `GridEnvironment(wrap=)` / `.wrap` → `torus=`.

## v0.3.13 - 2026-06-04

### Changed

- Ignore root-level LaTeX paper build outputs so local manuscript compilation
  does not dirty package release branches.
- Keep project metadata, runtime fallback version, documentation version, and
  changelogs aligned for the package release.

## v0.3.12 - 2026-06-04

### Changed

- Tighten the source distribution surface so local-only directories such as
  tests, benchmark fixtures, documentation builds, GitHub metadata, assistant
  metadata, and paper drafts stay out of package archives.
- Add release-workflow package-surface validation for the wheel and source
  distribution before creating the GitHub release.
- Keep project metadata, runtime fallback version, documentation version, and
  changelogs aligned for the package release.

## v0.3.11 - 2026-06-04

### Changed

- Cut a clean package release from an up-to-date `dev` branch into `main`.
- Keep release metadata, runtime fallback version, documentation version, and
  changelogs aligned for the wheel and source distribution.
- Explicitly ignore local assistant metadata along with paper drafts and paper
  archives so package releases stay focused on the AMBER library.

## v0.3.10 - 2026-06-04

### Changed

- Cut a main-anchored package release after synchronizing `dev`, `main`, and
  their remotes.
- Keep package metadata, runtime fallback version, documentation version, and
  release notes aligned for the distribution artifacts.
- Preserve local paper drafts and generated paper archives as ignored,
  non-package artifacts.

## v0.3.9 - 2026-06-04

### Changed

- Add a root split-SIR benchmark runner with deterministic shared inputs,
  explicit sync/async schedules, Agents.jl fixtures, checked result artifacts,
  and declared async SimPy budget skips.
- Add a root dynamic-graph coordination benchmark runner and checked result
  artifacts across AMBER, NumPy, Polars, a Python object loop, Mesa, AgentPy,
  and Agents.jl.
- Clarify public benchmark wording so the schedule-mixed headline SIR row is
  not presented as an equivalent-trajectory AMBER-over-Julia claim.

## v0.3.8 - 2026-06-04

### Changed

- Upgrade the all-framework benchmark runner to default to 10 full-run samples,
  preserve raw Agents.jl subprocess samples, and write raw sample counts, IQRs,
  and bootstrap median intervals into `benchmark_results_all.json`.
- Regenerate the all-framework headline benchmark artifacts so all 63
  framework/model/size rows carry 10 raw timing samples, including Agents.jl.

## v0.3.7 - 2026-06-04

### Fixed

- Move GitHub Actions workflows to current Node-24-ready action versions for
  release, checkout, setup-python, upload-artifact, and Codecov steps.
- Correct the Codecov upload input from `file` to `files`.

## v0.3.6 - 2026-06-04

### Changed

- Add a tag-driven GitHub release workflow that builds and validates wheel and
  source distributions before attaching them to the release.
- Extend CI coverage to the `dev` branch and Python 3.9, matching the declared
  package support floor.
- Add package metadata consistency checks so runtime, project, and documentation
  versions do not drift across releases.
- Tighten release-package hygiene around generated paper outputs and source
  distribution contents.

## v0.3.5 - 2026-06-04

### Changed

- Cut a package-only release with synchronized project metadata, runtime
  version, and Sphinx documentation version.
- Keep local paper drafts and generated paper archives outside the tracked
  package release surface.

## v0.3.4 - 2026-06-03

### Fixed

- Synchronize Sphinx documentation metadata with the package release version.
- Keep runtime, project metadata, and generated documentation versions aligned
  for patch releases.

## v0.3.3 - 2026-06-03

### Changed

- Move package metadata to `pyproject.toml` so AMBER builds through the
  standard PEP 517/518 Python packaging flow.
- Keep `setup.py` as a compatibility shim for older editable-install tooling.
- Add package URLs and keyword metadata for cleaner wheel and source
  distributions.

### Fixed

- Remove tracked coverage artifacts from the repository.
- Tighten source-distribution hygiene so local paper drafts, caches, and
  coverage files stay out of release artifacts.
- Correct the `Makefile` coverage target to use the `src/ambr` package path.

## v0.3.2 - 2026-06-03

### Fixed

- Execute exactly the requested number of model steps. `Model.run()` and
  `Model.run_step()` now run setup once before the first step instead of
  consuming the first update as a setup-only tick.
- Align benchmark helpers so wealth transfer, random walk, and SIR runs pass
  structural correctness checks before timing.
- Route Agents.jl benchmark parameters through the master runner instead of
  relying on hardcoded step counts.

### Changed

- Regenerate all-framework benchmark results with seeded timing, slowest-sample
  trimming, and documented SIR update-ordering caveats.
- Preserve raw per-run timing samples in the benchmark JSON for Python-hosted
  frameworks; Agents.jl rows are marked as aggregate-only subprocess timings.
- Update README benchmark tables and installation docs for the current Python
  support floor.
- Stop tracking draft paper files in the package repository.
