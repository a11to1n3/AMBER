Release gates
=============

Tag-driven releases (``vX.Y.Z``) use ``.github/workflows/release.yml``.
Publish waits on the CPU wheel matrix, not on GitHub CUDA.

Pipeline
--------

1. **Validate tag/version** — ``vX.Y.Z`` must equal ``project.version`` in
   ``pyproject.toml``. The version must **not** already exist on PyPI
   (there is no ``skip-existing`` publish path).
2. **Build wheel once** — single ``python -m build``; the same ``ambr-dist``
   artifact is passed to every downstream job.
3. **Install the exact wheel** — test jobs install the wheel into a clean
   environment and hide ``src/ambr`` so imports cannot resolve from the tree.
4. **CPU matrix** — OS × Python matrix runs package-surface / quickstart
   checks and the full pytest suite against the wheel.
5. **CUDA tests (optional)** — run only when repository variable
   ``GPU_RUNNER`` is set to a single self-hosted runner label. The job
   does **not** block publish. An unset variable skips the job; that is
   **not** a CUDA pass. **0.5.0:** GPU claims were verified locally on an
   NVIDIA RTX 3090 (Ubuntu x86_64, driver 560.35.05, CuPy 14.1.1) at
   ``ce79082`` via ``scripts/run_gpu_claims.py --quick`` (7/7 PASS) and
   the GPU pytest modules (25 passed, 2 expected skips). Not verified by
   GitHub Actions.
6. **SBOM** — SPDX SBOM generated from the built wheel during the build job.
7. **Protected publish** — GitHub Environment ``pypi`` with **required
   reviewers** (maintainer approval). Only the publish job has
   ``id-token: write``: OIDC Trusted Publishing to PyPI **and** build
   provenance attestation (``actions/attest-build-provenance``) over the
   exact ``dist/*`` artifact.

Maintainer setup
----------------

* PyPI trusted publisher: owner ``a11to1n3``, repo ``AMBER``, workflow
  ``release.yml``, environment ``pypi``.
* GitHub Environment ``pypi``: enable required reviewers.
* Optional repository variable ``GPU_RUNNER``: a **single** self-hosted
  runner label with CUDA (for example ``cuda``). The ``runs-on`` expression
  is scalar, so do not set ``self-hosted,gpu``. If unset, GitHub CUDA jobs
  are skipped and publish still proceeds.

Security defaults
-----------------

* All Actions are pinned to full commit SHAs.
* Build and test jobs: ``permissions.contents: read`` only.
* Publish job: ``id-token: write`` (OIDC) and ``contents: write`` (GitHub
  Release assets). No long-lived PyPI API token is required when Trusted
  Publishing is configured.

Definition of done (0.5.0 production-candidate)
-----------------------------------------------

Before tagging **0.5.0** (and later production tags), require **all** of::

   pytest -q
   ruff check src/ambr
   mypy
   sphinx-build -W --keep-going -b html docs docs/_build/html
   python -m build
   twine check dist/*

Plus:

* Fresh wheel installation passes CPU quick starts.
* GPU lane: local claim script + GPU pytest on a real NVIDIA host, **or**
  a GitHub ``GPU_RUNNER`` job. 0.5.0 used the local RTX 3090 run above.
  Do not claim GitHub CUDA verification unless that job actually ran.
* Persistence traversal/staleness tests pass.
* Importing AMBER does not alter Matplotlib configuration.
* Intentional model/optimizer failures remain visible and diagnosable.

See also :doc:`contributing` and ``SECURITY.md``.
