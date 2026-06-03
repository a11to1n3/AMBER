# Changelog

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
