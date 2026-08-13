# Changelog

## Unreleased

## v0.5.0 - 2026-08-13

Production-candidate cut of the post-0.4.7 correctness, integrity, and
release-gate series. **Requires a new PyPI version** (0.4.7 already published).

### Fixed

- **MultiObjectiveSMAC seed**: optimizer ``seed`` (or ``fixed_params['seed']``)
  is applied to every trial and to incumbent re-scoring so Pareto values match
  the searched front on stochastic models.
- **Experiment concat**: ``agents`` / ``model`` frames use
  ``diagonal_relaxed`` so param-gated extra columns do not raise ShapeError.
- **Virus example**: live ``agents_df`` is the unique-id population table
  (no per-step history concat).
- **SMAC basic example**: default ``__main__`` is 10 trials × 15 steps;
  ``--full`` restores the long comparison/importance workflow.
- **README / UX**: 0.5.0 upgrade note; ``help(ambr)`` matches the vectorized
  quickstart; ParallelRunner documented as core; GPU quickstart CI no longer
  ``|| true``.
- **Windows GPU quickstart**: print ``DataFrame.to_dicts()`` (ASCII) instead
  of Polars box-drawing ``tail()`` so CP1252 stdout does not raise.
- **SMAC deterministic**: ``SMACOptimizer`` / ``MultiObjectiveSMAC`` set
  ``Scenario(deterministic=True)`` only when ``fixed_params['seed']`` is
  not ``None`` (``{"seed": None}`` stays stochastic).
- **SMAC basic example**: parse ``--help`` first; label incumbent vs history
  min; replay the best config with the same seed.
- **First-run / Windows**: example scripts print ASCII (no emoji; Polars
  ``to_dicts()`` instead of box-drawing ``tail()``). Simple SMAC defaults to
  10x15 with a pinned seed; ``--full`` restores comparison.
- **bayesian_optimization**: ``deterministic`` follows a pinned non-``None``
  model ``seed`` (same rule as ``SMACOptimizer``); SMAC trial seed is applied
  via ``setdefault``. Penalize-path failure records look up the config
  without the injected trial seed.
- **Install wording**: officially tested/supported on Python 3.10–3.13
  (``requires-python`` remains ``>=3.10``; 3.14+ is not a declared target).
- **README extras**: document standalone ``ambr[gpu]`` and ``ambr[viz]``.
  Remaining README / Sphinx DataFrame prints use ``.to_dicts()``.
  Tutorial fences (spatial inspect + experiment summary) included; CI
  scans doc fences for ``print(df)`` without ``.to_dicts()``.

- **MultiObjectiveSMAC strategy**: `strategy` is validated and forwarded to
  each scalar `SMACOptimizer` (`bayesian` / `random` /
  `algorithm_configuration`). `strategy='pareto'` now raises `ValueError`
  (the Pareto set is always assembled after the fact; it is not a search
  facade). `fixed_params` are applied to trials and incumbent re-scoring.
- **SMAC advanced example**: `n_trials` is documented as per-objective;
  default demo is 3×4 evaluations with `fixed_params` (`steps=8`,
  `grid_size=10`) instead of 20×4 implicit 100-step runs.

- **Example notebooks**: regenerated `button_network_simulation.ipynb`,
  `flocking_simulation.ipynb`, and `forest_fire_simulation.ipynb` from the
  working `.py` models (no more `pl.concat` of history into `agents_df`).
  CI executes the notebooks (`example-notebooks` job).
- **bayesian_optimization docs**: the helper uses SMAC's RandomForest, not a
  Gaussian process. README, API docs, and the function docstring now match
  the implementation.
- **SMAC examples**: plotting is optional; a clean `ambr[advanced]` install
  completes search without matplotlib. Document `ambr[advanced,viz]`.
- **Release metadata**: contact email is `anh-duy.pham@uni-wuerzburg.de`
  (no `example.com` placeholder); `CITATION.cff` `date-released` matches
  this 0.5.0 date.

- **SMACOptimizer contract (0.5.x)**: ``strategy='random'`` selects RandomFacade;
  ``fixed_params`` merge into every trial; ``optimize()`` returns
  ``n_evaluations`` and history columns ``cost`` / ``objective`` / ``time`` /
  ``trial``; multi-fidelity Scenario gets ``min_budget``/``max_budget`` from
  fidelity parameter bounds; unsupported options (``log_ei``, GP /
  ``random_forest_with_instances``) raise clear ``ValueError``.
- **SMACOptimizer isolation**: each instance uses a unique temp
  ``output_directory`` (no silent reuse of cwd ``smac3_output/``).
- **Multi-fidelity budgets**: fidelity params are budget-only (not CS samples);
  integer fidelities coerce SH rungs to ``int``; history reports evaluated budget.
- **bayesian_optimization**: fixed floats go to ``fixed_params`` (degenerate
  float HPs no longer crash ConfigSpace/SMAC).
- **NetworkEnvironment identity**: agent-first resolution for
  ``get_neighbors`` / ``get_distance`` / ``get_degree`` / ``get_clustering`` /
  ``add_edge`` / ``remove_edge``; missing ``node_id`` column no longer crashes
  (unplaced agent → empty/0); use ``as_node=True`` for graph-node ids.
- **SMAC on_error='raise'**: re-raise target/objective exceptions after SMAC
  returns (SMAC swallows crashes into CRASHED/inf trials); multi-process via
  pickle side-channel under the run directory.
- **SMACOptimizer ``n_workers>1``**: pickle-safe trial evaluator with a clean
  ``(config, seed)`` signature (no partial-arg SMAC warnings); explicitly
  close Dask client/cluster after ``optimize()`` so workers do not linger.
- **SMAC temp dirs**: create ``amber_smac_*`` only after constructor
  validation; remove after ``optimize()`` / failed construction unless
  ``AMBER_SMAC_KEEP_OUTPUT=1``.
- **Search exhaustion**: ``isinstance`` /
  ``ConfigurationSpaceExhaustedException`` only (plus message markers); no
  broad name-substring matching.
- **on_error='raise' side-channel**: if the exception is not picklable, persist
  a structured type/message/traceback payload and re-raise
  ``RemoteObjectiveError`` (never silent ``best_cost=inf``).
- **Docs / examples**: SMAC calibration scripts match the optimizer return
  shape; environment API uses ``grid_position`` / ``node_id``; ParallelRunner
  docs are spawn-safe (Sphinx fence fixed); OOP README/quickstart use
  ``step_oop`` + ``cpu(mode="oop")``; Sample/Experiment contracts document zip
  sampling and ``info`` as a dict; BaseAgent/BaseModel no longer presented as
  user bases; GPU CI described as hard NOT VERIFIED (not soft-skip).

- **Windows RunResults I/O**: exclusive payload writes use ``O_BINARY`` so LF
  is not expanded to CRLF (SHA-256 checksums stay stable on Windows CI).
- **GPU nightly**: no longer runs on every path-push (would fail red without
  a GPU runner); schedule / ``workflow_dispatch`` still hard-require CUDA.
- **ParallelRunner fail_fast**: terminates the full active-worker registry
  (no orphaned sibling processes / delayed side effects).
- **ParallelRunner worker registry**: register each process in ``active``
  immediately after ``start()``, before parent-side ``child_conn.close()``,
  so a close failure cannot leave a live worker outside cleanup.
- **Checkpoint writes**: random exclusive temp + ``O_NOFOLLOW`` / ``fsync`` /
  ``os.replace`` (no predictable ``*.tmp`` symlink escape).
- **Checkpoint dtypes**: frames stored as Arrow IPC (base64), not lossy
  record JSON.
- **max_in_flight**: never exceeds ``n_workers``; reject non-positive limits.
- **Manifest integrity**: ``sha256`` required (64 hex chars) on every entry.
- **Release wheels**: stamp ``GITHUB_SHA`` into ``_build_info.GIT_REVISION``
  and assert it in the built wheel.
- **Min-deps CI** aligned to declared floors; ``SECURITY.md`` supports 0.5.x.
- **ParallelRunner retry + fail_fast**: retries register in the live process
  registry immediately (not only a dual ``still_active`` list), so fail_fast
  cleanup terminates them.
- **Release wheel install assert**: resolve the built wheel to an absolute path
  before ``pip install`` under a temporary ``cwd`` (relative ``dist/…`` failed).
- **Checkpoint schema v2/v3/v4**: writers emit ``schema_version=4`` with
  ``polars_ipc_b64`` frames + revision-aware workload fingerprint; schemas
  1–3 still load (including historical schema-1 files that already stored
  IPC). IPC encode failures raise ``CheckpointSerializationError`` instead
  of silently storing ``repr(df)``.
- **Checkpoint resume integrity**: schema-4 fingerprints include AMBER
  version/revision, model source digest, and optional
  ``workload_revision`` / ``AMBER_APP_REVISION`` so code edits invalidate
  resume. Schema-3 fingerprints are validated with the original
  model+params algorithm so existing schema-3 files still resume.
  Identity-less schema-1/2 resume is refused by default
  (``allow_unverified_checkpoint=True`` for an explicit unsafe migration;
  ``_load_checkpoint`` remains available for inspection). Per-index params
  must still match. Never-run slots use ``status=cancelled`` (not
  ``error_type == "Cancelled"``) and are omitted from checkpoints so resume
  re-queues them; a user exception class named ``Cancelled`` remains a
  persisted ``failed`` outcome.

### Changed

- **Step-data lifecycle (breaking for silent loss)**: `run_step` now allocates
  the model-data row *before* `step()`, so values recorded via
  `record_model` inside `step()` are retained. Precedence on duplicate keys
  (later wins): `step()` → declarative `model_reporters` → `update()`.
  A failed step discards the partial row and does not append it; `t` is not
  advanced. Contract modes (`off` / `check` / `warn` / `raise`) share this
  behaviour. Docs updated to match (quickstart / tutorial no longer claim
  step recordings are discarded).
- **Optimization metrics are strict by default (breaking)**:
  `objective_function` no longer silently returns `0` for missing, empty,
  non-numeric, or non-finite metrics — it raises `KeyError` / `ValueError`.
  `iterations` must be `>= 1`.
- **SMAC error handling (breaking)**: `bayesian_optimization` and
  `SMACOptimizer` default to `on_error='raise'`. Pass `on_error='penalize'` to
  map evaluation failures to a large finite cost and keep structured failure
  records (`configuration`, `exception_type`, `message`, `traceback`). Broad
  `except Exception: pass` around `smac.optimize()` is removed; only the
  documented configuration-space-exhausted condition is treated as non-fatal.
- **RunResults persistence (breaking layout)**: `results.save` / `RunResults.load`
  now use a versioned `manifest.json` (schema v1) mapping logical keys to
  opaque files under `frames/` and `json/`. User keys never enter filesystem
  paths; checksums are verified on load; incomplete/corrupt saves fail
  clearly. Preferred format is `format="parquet"` with optional
  `allow_fallback=True`. Full contract certificates (violations, not just
  counts) are persisted. Legacy 0.4.x directories still load with a migration
  warning. Manifest commit uses exclusive random temps + `O_NOFOLLOW` /
  `fsync` / `os.replace` (no predictable `manifest.json.tmp` symlink escape).
- **Optional deps / lazy viz (breaking install surface)**: core package no
  longer depends on matplotlib, seaborn, or scikit-optimize. Plot helpers live
  under `ambr[viz]`; SMAC stays under `ambr[advanced]`. `import ambr` does not
  load matplotlib — `plot_grid` / `plot_timeseries` / `HAS_MATPLOTLIB` resolve
  via lazy `__getattr__`. `ambr.viz` no longer calls `matplotlib.use("Agg")`;
  set `MPLBACKEND=Agg` in CI and docs builds instead.
- **Virus example usability**: uses `run_step()` (not bare `step`/`update`),
  `model.rng` only, UI gated behind `__main__`, `--headless` three-step smoke,
  `anywidget` in `ambr[examples]`, and background-thread status reports
  failures instead of always showing "Completed".
- **Real release gates** (`.github/workflows/release.yml`): validate
  `vX.Y.Z == project.version` and prove the version is absent from PyPI;
  build the wheel once and test only that artifact (CPU matrix + CUDA);
  CUDA missing is **NOT VERIFIED** (never soft-green); SHA-pinned Actions;
  least-privilege permissions (`id-token: write` only on publish); protected
  `pypi` environment for maintainer approval; SBOM + provenance attestation;
  GPU hardware evidence artifact. See `docs/release_gates.rst`.
- **GPU nightly**: no longer soft-skips green without CUDA — reports
  **NOT VERIFIED** and fails; uploads hardware evidence when a GPU runner is
  configured (`GPU_RUNNER`).
- **GPU teardown**: `end_execution` always clears `model._execution` even if
  sync/synchronize fails; simulation exceptions are not masked by teardown
  errors.
- **Run provenance**: `results.info` now records AMBER/Python versions,
  fully-qualified model class, parameters/seed, start/end timestamps and
  status, run UUID, config hash, Polars/NumPy/CuPy/CUDA versions, device and
  execution lane, optional git/app revision (`AMBER_GIT_REVISION` /
  build-info only — never CWD `git rev-parse`).
- **ParallelRunner**: returns ordered `RunOutcome` records
  (`success`/`failed`/`timeout`) with error type/message/traceback; hard
  process terminate on timeout; JSON checkpoints with `trust_checkpoint`;
  `fail_fast`, `retry`, `max_in_flight`, checkpoint/resume.
- **Docs CI + maintenance**: `sphinx-build -W` in CI; fixed malformed RST
  tables; absolute GitHub/RTD URLs in the PyPI README; `SECURITY.md`,
  `CODEOWNERS`, Dependabot, `pip-audit`, issue templates; min/latest
  dependency lanes; Python **3.10–3.13** hard-gated (3.14 not advertised until
  release matrix covers it); dropped EOL **3.9**.

### Notes

- Supported Python: **3.10–3.13** (release wheel test matrix matches).
- Tag ``v0.5.0`` from ``main`` after ``dev`` merge; release workflow refuses
  re-publishing existing PyPI versions.

## v0.4.7 - 2026-08-05

Patch over 0.4.6: remove paper-campaign machine labels from user-facing docs
and rename the GPU verification script.

### Changed

- Renamed ``scripts/run_host_b_gpu_claims.py`` → ``scripts/run_gpu_claims.py``
  (0.4.6 wheels still contain the old path; upgrade to 0.4.7 for the new name).
- Docs / CI comments use generic **NVIDIA + CuPy / CUDA host** wording only
  (no internal Host A/B or provider-specific campaign labels).
- Benchmark helper scripts: neutralize ``vast.ai`` comments; generic CUDA
  rerun tags.

### Notes

- **Do not re-tag 0.4.6** — PyPI artifacts are immutable. Install with
  ``pip install -U 'ambr>=0.4.7'``.

## v0.4.6 - 2026-08-05

Claim honesty, doc/CI smokes, GPU claim verification, activation/viz helpers,
RunResults I/O, calibration docs, and 1.0 freeze prep — without removing
deprecated APIs yet (still scheduled for 1.0).

### Added

- **Doc-fence CI smoke** (`tests/test_doc_fences.py`): syntax-checks fenced
  Python in README and key docs; executes self-contained samples (large-N
  scaled down unless `AMBER_DOC_FENCE_FULL=1`). Intentional fragments are
  allowlisted.
- **GPU claim verification script** (`scripts/run_gpu_claims.py`): re-verify
  `.gpu().run()`, ArrayKernelModel (CuPy), ensemble, and GPU pytest modules on
  NVIDIA + CuPy (default CI has no CUDA).
- **README GPU requirements banner** and pointer to that script.
- **`tests/test_readme_examples.py`**: durable smokes for README OOP +
  vectorized wealth (view API) and `record_model` in `update()`.
- **OOP activation helpers** (`ambr.scheduling`): `activate`,
  `Activation` / Sequential|Random|Simultaneous aliases, `shuffled_ids`, and
  `Model.activate_agents(mode=...)` for tracked agents (not a schedule proof).
- **Viz helpers** (`ambr.viz`): `plot_timeseries`, `plot_grid`; optional
  extra `ambr[viz]` (matplotlib is already a core dependency).
- **GPU nightly / manual workflow** (`.github/workflows/gpu-nightly.yml`):
  soft-skips without CUDA; runs GPU quick claims when `nvidia-smi` works
  (self-hosted GPU runner via `vars.GPU_RUNNER`).
- **RunResults I/O**: `results.save(path)` / `RunResults.load(path)`
  (parquet + info.json); `keys_overview()` helper.
- **Docs**: `docs/reproducibility.rst` (CPU≠GPU bit-identical policy, seeds,
  contract/fast-path limits); RunResults cookbook; Experiment/ParallelRunner
  opt-in parallelism wording.
- **Example**: `examples/smac_batch_sir_smoke.py` (ensemble always; SMAC
  skipped honestly without `ambr[advanced]`).
- **1.0 freeze prep**: `ambr.deprecation_inventory` +
  `tests/test_deprecation_inventory.py`; docs
  `versioning`, `public_api`, `paper_and_package`, `roadmap_1_0`.

### Changed

- Public wealth / GPU samples use the **view API** (`where` / assign /
  `scatter_add`) and honest `GPU_AVAILABLE` branching (no read-only
  `agents.array` mutate; no commented-only `.gpu()` as the primary path).
- Tutorial / API docs: `Sample(n=...)`, `Experiment(model_type=..., sample=...)`,
  experiment results as Polars dict frames; contract wording stresses
  **operational monitor, not schedule proof**.
- Benchmarks README: single source of truth for headline numbers
  (`snapshot_correct_10run_10m` only; `summary_table.md` exploratory).
- **Experiment**: canonical `model_type=` / `sample=`; legacy
  `model_class=` / `parameters=` emit `DeprecationWarning` (remove in 1.0).
- **ParallelRunner** / **GPUEnsembleRunner** docs: single-run `.run()` is never
  parallel; ensemble “× vs loop” wording de-hyped as exploratory.

## v0.4.5 - 2026-07-29

Research-grade package hygiene: honest 10M headline evidence, pair-keyed GPU
SIR counter-tape tests, software citation metadata, and optional `ambr[gpu]`.

### Added

- **Tests:** `tests/test_sir_counter_tape.py` locks the SplitMix64 counter-tape
  reference used by production GPU SIR infection draws (pair-keyed
  `(global_seed, step, EVT_INFECTION, min(i,j), max(i,j))`), documents
  `sir_kernel_step(..., global_seed=...)`, and exercises FLAME NVRTC preload
  configuration without requiring pyflamegpu.
- **Tracked headline evidence:** `benchmarks/results/benchmark_results_snapshot_correct_10run_10m.json`
  and `summary_table_snapshot_correct_10run_10m.md` (README / Sphinx source of truth).
- **`CITATION.cff`** for repository citation metadata.
- Optional **`ambr[gpu]`** extra (CuPy) for the NVIDIA GPU lane.

### Changed

- **GPU SIR scale kernels** (`benchmarks/models/amber_gpu_scale_models.py`) and
  **vectorized SIR wiring** (`amber_models.py`): infection draws use pair-keyed
  SplitMix64 with explicit `global_seed` / step (order-invariant RVs for
  cross-backend attestation).
- **`benchmarks/run_all_frameworks.py`:** FLAME GPU 2 CUDA 13 NVRTC/nvJitLink
  preload via `ctypes` (glibc does not re-read `LD_LIBRARY_PATH` after start);
  Agents.jl subprocess timeout raised for long scale runs.
- **README performance section:** single committed source of truth
  (`benchmark_results_snapshot_correct_10run_10m.json`); Schelling ratio
  labeled setup-inclusive/exploratory; multi-framework cells not imputed.
- **Benchmarks docs:** optional dependency matrix; AMBER-only vs multi-framework
  paths; missing OOM/budget cells are not zeros;
  `summary_table.md` bannered exploratory/historical.
- **Installation / going_faster:** document `ambr[gpu]`; README How to cite
  references `CITATION.cff`; Sphinx index points at 0.4.5; calibration
  throughput wording de-hyped as exploratory.

### Notes

- Paper / AAMAS materials stay **outside** the library tree (e.g.
  `~/Documents/AMBER_AAMAS`); `.gitignore` blocks `paper-fix-work/` and
  `AMBER_AAMAS/` under the package repo.

## v0.4.4 - 2026-07-18

Honest execution lanes, operational contract wording, and opt-in private GPU
fast paths. Builds on the 0.4.3 placement API without overselling “one
unchanged `step` body” or schedule proofs.

### Added

- **Lane hooks:** `step_vectorized()` for vectorized CPU/GPU runs and
  `step_oop()` for CPU Agent-object runs. Legacy `step()` remains the
  fallback when a lane hook is not defined. GPU placement is
  **vectorized-only**; Python Agent objects use `cpu(mode="oop")`.
- **`approve_fast_path(evidence)` / `revoke_fast_path_approval()`:** private
  model-specific GPU loops run only with `contract="off"` **and** an explicit
  per-instance evidence label (caller-supplied provenance; not verified by
  AMBER). Without approval, `gpu().run()` uses the instrumented general path.
- Contract hazard **`uncertified_mutable_borrow`** for `agents.array(...)`
  (in-place mutations after a raw borrow are not fully reconstructible).
- Benchmarks: native lane models, evidence-labeled GPU rows, and a cautionary
  `benchmarks/try_polars_gpu.py` probe (not a product path).

### Changed

- **Contract semantics (docs + runtime posture):** the snapshot-view contract
  is an **operational monitor** at instrumented seams — not a proof that
  arbitrary NumPy/CuPy or private kernels preserve an intended activation
  schedule. `cert.clean` means no monitored error/warning, not completeness.
- GPU teardown tracks **dirty columns** more carefully for host sync.
- Docs / README: remove “same `step` body only” oversell; document lanes,
  `approve_fast_path`, and honest benchmark claims.

### Notes

- Private GPU loops and `ambr.gpu_kernels` remain **non-public** internals.
- Polars Lazy `engine="gpu"` is **not** AMBER’s agent GPU runtime.

## v0.4.3 - 2026-07-17

Native GPU path and Keras-style device placement on one `Model` /
view-API step (superseded wording refined in 0.4.4).

### Added

- **Keras-style placement:** `model.cpu(mode=...)` / `model.gpu(mode=...)`
  set device and optional run mode (default `vectorized`); `run(mode=...)`
  still overrides. Implementation in `ambr.execution`.
- **Device-resident columns** for the native view API under `Model.gpu()`:
  `device_columns`, GPU write path in sequences, `DeviceRNG`, `scatter_add`,
  and `DeviceColumn.mean()` so the same `where` / column write /
  `scatter_add` step runs on CPU or GPU.
- Tests: `tests/test_execution_api.py`; expanded `tests/test_gpu_backend.py`.

### Changed

- **AMBER (GPU) benchmarks** use the same vectorized model classes as
  AMBER (vectorized) via `model.gpu().run()` (no separate kernel models for
  the main harness). Vectorized models use `self.xp` + `agents.array(...)`
  where needed for dual CPU/GPU.
- Fused wealth throughput variant moved to `amber_fused_models.py` (not the
  documented view idiom).
- FLAME GPU 2 Schelling workload + docs note on the GPU chart page.

## v0.4.2 - 2026-07-11

Hygiene, CI quality, and docs release on top of 0.4.1.

### Added

- **GPU ensemble tests** (`tests/test_gpu_ensemble.py`) covering
  `GPUEnsembleRunner` / `BatchedWellMixedSIR` on the NumPy fallback path, plus
  a tiny optional `smac_batch_calibrate` smoke when SMAC is installed.
- Public exports for `GPUEnsembleRunner`, `BatchedWellMixedSIR`,
  `smac_batch_calibrate`.
- CI installs `ambr[perf]` (Numba) on Ubuntu + Python 3.12 so scatter JIT
  paths are exercised in the matrix.
- **Canonical Schelling example** `examples/schelling_vectorized.py` and
  guide `docs/environments_schelling.rst`.
- **Deprecations guide** `docs/deprecations.rst` (canonical vs legacy → 1.0).
- **Ruff + mypy CI job**; gradual mypy module set in `pyproject.toml`.
- **pre-commit** hooks: nbstripout + ruff (``pre-commit install``).

### Changed

- **Git history rewrite** (purge paper drafts, notebook bloat, build artifacts
  from history; re-clone or ``git reset --hard origin/dev``).
- **Branch protection** on ``main`` / ``dev``: require ``Ruff + mypy`` and
  Ubuntu 3.12 tests; no force-push.
- Docs / README: public arXiv **citation** for the paper; do **not** co-locate
  manuscript drafts (``.gitignore`` excludes ``paper/`` and build artifacts).
- Ruff-clean `src/ambr` (unused imports / small fixes); SIR batch status
  constant renamed `INFECTED`.
- **Repo hygiene:** clear outputs from example notebooks (~26 MB → ~50 KB);
  drop unused root ``architecture_diagram.png``; tighten ``.gitignore`` /
  ``MANIFEST.in``; point ``requirements*.txt`` at ``pyproject.toml`` extras;
  drop tracked ``benchmarks/results/*.json`` (regenerate locally; charts/md stay).

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
