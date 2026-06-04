# Changelog

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
