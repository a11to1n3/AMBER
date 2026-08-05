Reproducibility policy
======================

What AMBER guarantees
---------------------

* **Seeded CPU runs** with the same package version, parameters, and code
  path: ``seed`` initializes both the stdlib ``random`` module and the
  canonical NumPy ``Generator`` on the model (``self.rng``). Environment
  helpers should use ``model.rng``, not global ``np.random``.
* **Explicit lanes**: ``cpu(mode="vectorized"|"oop")`` vs ``gpu()`` are
  different execution paths. Prefer recording ``results.info`` (device, mode)
  with any published number.

What AMBER does **not** guarantee
---------------------------------

* **CPU vs GPU bit-identical** trajectories for arbitrary models. CuPy RNG,
  floating-point reduction order, and device-resident columns can differ from
  NumPy/Polars CPU even under the same seed.
* **Cross-framework trajectory identity** (Mesa / AgentPy / FLAME). Benchmark
  gates are often **structural** (conservation, bounds, compartment totals),
  not step-wise state equality.
* **Schedule confluence** from the snapshot-view contract. ``contract=`` is an
  **operational monitor** at instrumented seams — ``cert.clean`` is not a proof
  that every activation order is equivalent.
* **Private GPU fast paths** (``approve_fast_path``): the library only checks
  that a non-empty evidence **label** is present; it does not verify
  equivalence to the instrumented path.

Recommended practice
--------------------

1. Pin ``ambr`` (and CuPy when used) versions in your environment lockfile.
2. Log ``am.__version__``, ``am.print_status()`` output, and ``results.info``.
3. Publish the **lane** used (OOP / vectorized / GPU / ensemble).
4. For multi-run studies, use ``Experiment``, ``ParallelRunner``, or
   ``GPUEnsembleRunner`` explicitly — never assume ``.run()`` is parallel.
5. For GPU claims, re-verify on NVIDIA + CuPy with
   ``python scripts/run_host_b_gpu_claims.py``.

Seed semantics (summary)
------------------------

==========  =========================================================
API         Role
==========  =========================================================
``seed``    Model parameter → seeds ``self.random`` and ``self.rng``
``self.rng`` Canonical NumPy ``Generator`` (prefer this)
``self.random`` stdlib Random (AgentPy-shaped)
``nprandom`` Legacy alias of ``rng`` (deprecated → 1.0)
==========  =========================================================

See also :doc:`going_faster`, :doc:`api/contract`, :doc:`benchmarks`.
