Roadmap to 1.0
==============

Status: **0.4.x / Unreleased P0–P2 complete**; this page is the **P3 freeze
plan** (dry-run). Nothing listed under “Remove in 1.0” is deleted until the
``1.0.0`` tag.

Goals for 1.0
-------------

1. **Honest claims** — README/docs fences and GPU claim verification script stay green.
2. **Stable public surface** — :doc:`public_api` + :doc:`versioning`.
3. **Legacy purge** — remove every entry in
   :mod:`ambr.deprecation_inventory`.
4. **One story** — package docs are the software source of truth
   (:doc:`paper_and_package`).

Remove in 1.0 (inventory)
-------------------------

See :doc:`deprecations` and :mod:`ambr.deprecation_inventory`
(``DEPRECATIONS_TO_REMOVE_IN_1_0``). Tests enforce that every
``warn_deprecated(...)`` call site is listed.

Soft legacy **kept** unless a later 0.9 decision says otherwise:

* ``self.rng`` / ``self.random`` (canonical)
* ``self.nprandom`` (compat shim — prefer ``rng``)

Out of scope for 1.0
--------------------

Explicit **non-goals** (will not block 1.0):

* Full Mesa Solara / browser dashboard product
* Apple Metal / MPS GPU backend
* Distributed multi-node runtime
* Bit-identical CPU ↔ GPU trajectories for all models
* Contract proving activation-schedule confluence
* Mesa-complete space API (hex multi-grid property layers) unless demanded

Suggested version train
-----------------------

==========  ============================================================
Version     Intent
==========  ============================================================
0.4.x       P0–P2 honesty, activation, viz, RunResults I/O (current work)
0.5.x       Optional: deeper schedulers / viz polish if needed
0.9.0       API freeze candidate: no new deprecations without inventory;
            migration guide; release candidate tags
1.0.0       Delete deprecated aliases; SemVer post-1.0 rules apply
==========  ============================================================

Checklist before tagging 1.0.0
------------------------------

* [ ] ``tests/test_deprecation_inventory.py`` green
* [ ] No remaining ``warn_deprecated`` call sites (inventory empty after purge)
* [ ] Doc fences + README smokes green
* [ ] ``scripts/run_gpu_claims.py`` green on CUDA hardware
* [ ] :doc:`paper_and_package` reviewed
* [ ] CHANGELOG ``v1.0.0`` with migration notes

Contributors: prefer canonical APIs only in new code and examples.
