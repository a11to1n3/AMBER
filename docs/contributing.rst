Contributing
============

We welcome contributions to AMBER! This guide will help you get started.

Getting Started
---------------

1. **Fork the Repository**

   Fork the AMBER repository on GitHub and clone your fork:

   .. code-block:: bash

      git clone https://github.com/your-username/AMBER.git
      cd AMBER

2. **Set Up Development Environment**

   Create a virtual environment and install development dependencies:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e ".[dev]"

3. **Run Tests**

   Make sure all tests pass before making changes:

   .. code-block:: bash

      pytest tests/

Types of Contributions
----------------------

**Bug Reports**
   Report bugs using GitHub Issues. Include:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, AMBER version)

**Feature Requests**
   Suggest new features using GitHub Issues. Include:
   - Clear description of the feature
   - Use case and motivation
   - Proposed API (if applicable)

**Code Contributions**
   - Bug fixes
   - New features
   - Performance improvements
   - Documentation improvements

**Documentation**
   - Fix typos and improve clarity
   - Add examples
   - Translate documentation

Development Workflow
--------------------

1. **Create a Branch**

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make Changes**

   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**

   .. code-block:: bash

      # Run all tests
      pytest tests/

      # Run specific test file
      pytest tests/test_model.py

      # Run with coverage
      pytest tests/ --cov=ambr

4. **Commit Changes**

   .. code-block:: bash

      git add .
      git commit -m "Add feature: description of changes"

5. **Push and Create Pull Request**

   .. code-block:: bash

      git push origin feature/your-feature-name

   Then create a pull request on GitHub.

Code Style Guidelines
---------------------

**Python Style**
   - Follow PEP 8
   - Use type hints where appropriate
   - Write docstrings for all public functions and classes
   - Use meaningful variable and function names

**Testing**
   - Write tests for all new functionality
   - Aim for high test coverage
   - Use descriptive test names
   - Include both unit and integration tests

**Documentation**
   - Update docstrings for API changes
   - Add examples for new features
   - Update tutorials if relevant

Pull Request Guidelines
-----------------------

**Before Submitting**
   - Ensure all tests pass
   - Update documentation
   - Add entry to changelog (if applicable)
   - Rebase on latest main branch

**Pull Request Description**
   - Clear title summarizing the change
   - Detailed description of what was changed and why
   - Link to related issues
   - Screenshots for UI changes (if applicable)

**Review Process**
   - All PRs require review from maintainers
   - Address feedback promptly
   - Keep PRs focused and reasonably sized
   - Be patient - reviews take time

Release Process
---------------

AMBER follows semantic versioning with an explicit **pre-1.0** policy.
Full details: :doc:`versioning` and :doc:`roadmap_1_0`.

**Before 1.0 (0.x):**

- Breaking changes may appear in minor bumps when required for honesty/API cleanup.
- Deprecated names keep working until **1.0** (see :doc:`deprecations`).

**After 1.0:**

- **Major** (x.0.0): Breaking public API (``ambr.__all__`` / documented behaviour)
- **Minor** (x.y.0): Backward-compatible features
- **Patch** (x.y.z): Backward-compatible fixes

Before tagging a release:

1. Make sure ``dev`` is up to date with ``origin/dev`` and merged or fast
   forwarded into ``main`` for release.
2. Run ``pytest`` (include doc fences + deprecation inventory).
3. If GPU claims changed, run ``scripts/run_gpu_claims.py`` on a CUDA host.
4. Update ``CHANGELOG.md`` and version metadata.
2. Bump the package version in ``pyproject.toml`` and update
   ``CHANGELOG.md`` plus ``docs/changelog.rst`` (also keep
   ``src/ambr/__init__.py`` and ``docs/conf.py`` fallbacks in sync).
3. Run ``make release-check`` from a clean checkout. This builds the wheel and
   source distribution, runs ``twine check``, and executes the test suite.
4. Inspect the source distribution: only package metadata and ``src/ambr``
   should ship (benchmarks, docs sources, examples stay out of the sdist via
   ``MANIFEST.in``; local paper drafts stay untracked via ``.gitignore`` —
   cite the public paper, do not commit drafts).
5. Create and push an annotated ``vX.Y.Z`` tag from the release commit on
   ``main``. The ``Release`` workflow (``.github/workflows/release.yml``):

   * builds and validates the sdist/wheel
   * attaches them to a GitHub Release
   * publishes to **PyPI** via Trusted Publishing (OIDC)

PyPI Trusted Publishing (one-time, preferred)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do **not** paste API tokens into chat or commit them. Configure OIDC once:

1. On PyPI: `Publishing settings for ambr
   <https://pypi.org/manage/project/ambr/settings/publishing/>`_
2. Add a **GitHub** trusted publisher with:

   * Owner: ``a11to1n3``
   * Repository: ``AMBER``
   * Workflow name: ``release.yml``
   * Environment name: ``pypi``

3. On GitHub: Environment ``pypi`` already exists on this repo
   (Settings → Environments). Optional: require reviewers before deploys.

**Status (repo side):** GitHub Environment ``pypi`` is configured; the
``Release`` workflow requests OIDC (``id-token: write``). **You must still
add the matching Trusted Publisher row on PyPI** (step 1–2) if the next tag
should upload automatically — that step requires project-owner login on
pypi.org and cannot be done from CI alone.

If Trusted Publishing is not configured yet, you can still upload manually
with a *project-scoped* token (``twine upload``) and then switch to OIDC.
Revoke any token that has been exposed.

Lint & type checks
~~~~~~~~~~~~~~~~~~

CI runs a dedicated **Ruff + mypy** job on every PR:

.. code-block:: bash

   ruff check src/ambr
   mypy   # module list configured in pyproject.toml [tool.mypy]

``make lint`` / ``make type-check`` wrap the same tools for local use.


History rewrite (2026-07)
~~~~~~~~~~~~~~~~~~~~~~~~~

The repository history was rewritten once to drop large notebook outputs,
paper drafts, and other local-only blobs (``.git`` ~160 MB → ~3 MB on a
fresh clone). If your local clone predates that rewrite::

   git fetch origin
   git checkout dev
   git reset --hard origin/dev
   # or re-clone: git clone https://github.com/a11to1n3/AMBER.git

Do **not** merge old local branches that still contain the pre-rewrite
history without rebasing onto the new ``origin/dev``.

Repo hygiene
~~~~~~~~~~~~

* Install git hooks once: ``pip install -e ".[dev]" && pre-commit install``
  (or ``make pre-commit-install``). Commits then run **nbstripout** (strip
  notebook outputs) and **ruff** on ``src/ambr``.
* Do **not** commit notebook outputs. Example ``.ipynb`` files in
  ``examples/`` should stay small.
* Do **not** commit manuscript drafts (``paper/`` is gitignored); cite the
  public arXiv paper only.
* Prefer ``pyproject.toml`` extras over hand-editing ``requirements*.txt``.
* Package surface is enforced by ``MANIFEST.in`` and the release workflow
  (sdist must not contain ``benchmarks/``, ``docs/``, ``paper/``, ``tests/``).

Community Guidelines
--------------------

**Be Respectful**
   - Use welcoming and inclusive language
   - Respect differing viewpoints
   - Focus on constructive feedback

**Be Collaborative**
   - Help others learn and contribute
   - Share knowledge and expertise
   - Acknowledge contributions

**Be Patient**
   - Maintainers are volunteers
   - Reviews and responses take time
   - Complex changes require thorough review

Getting Help
------------

If you need help:

1. Check existing documentation
2. Search GitHub Issues
3. Ask questions in discussions
4. Contact maintainers directly

Thank you for contributing to AMBER!
