Versioning policy (SemVer)
==========================

AMBER uses `Semantic Versioning <https://semver.org/>`_ with an explicit
**pre-1.0** and **post-1.0** contract.

Before 1.0 (current: 0.x)
-------------------------

* ``0.MAJOR.PATCH`` in practice: **breaking changes may land in minor bumps**
  (e.g. ``0.4 → 0.5``) when required for honesty or API cleanup.
* Deprecated names keep working with :class:`DeprecationWarning` until **1.0**.
* Public claim surfaces (README fences, Host B GPU script, doc-fence tests)
  must stay green on each release.

After 1.0
---------

* **MAJOR** (``x.0.0``): breaking public API changes (anything in
  :doc:`public_api` without a documented compatibility shim).
* **MINOR** (``0.x.0`` / ``x.y.0``): backward-compatible features.
* **PATCH** (``x.y.z``): backward-compatible bug fixes and docs.

What is "public API"?
---------------------

* Names exported in ``ambr.__all__`` (see :doc:`public_api`).
* Documented behaviour in the Sphinx **User Guide** and **API Reference**.
* Stable CLI/scripts that are part of the package story
  (``scripts/run_host_b_gpu_claims.py`` behaviour for GPU verification).

Internal modules (``ambr._*``, private ``Model._*`` helpers, benchmark-only
kernels under ``benchmarks/models/``) are **not** SemVer-guaranteed.

Deprecations → 1.0
------------------

* Inventory: :mod:`ambr.deprecation_inventory` and :doc:`deprecations`.
* Silence in harnesses: ``AMBER_SUPPRESS_DEPRECATIONS=1``.
* **1.0.0** removes every entry in ``DEPRECATIONS_TO_REMOVE_IN_1_0``.

Release checklist (maintainers)
-------------------------------

1. Changelog entry under the target version.
2. ``pytest`` green (incl. doc fences + deprecation inventory).
3. Version in ``pyproject.toml`` / package metadata.
4. Tag + Trusted Publishing release workflow.
5. If GPU claims changed: run ``scripts/run_host_b_gpu_claims.py`` on Host B.

See also :doc:`contributing`, :doc:`roadmap_1_0`.
