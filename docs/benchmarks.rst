Benchmarks & Performance
========================

AMBER stores the whole population as a columnar Polars DataFrame and compiles
each step into a handful of vectorized expressions. From **0.4.4**, the
vectorized lane (``step_vectorized`` / legacy ``step``) runs on GPU via
``model.gpu().run()`` (:doc:`api/gpu`) with device-resident columns. Published
AMBER (GPU) rows may use an explicit ``approve_fast_path(evidence)`` label
before placement; the label is caller provenance, not a runtime certificate.
Batched calibration is separate (:doc:`api/gpu_ensemble`).

The full, reproducible suite — correctness checks, raw timings, and per-model
tables — lives under ``benchmarks/`` (see ``benchmarks/README.md``). Reproduce
with ``python benchmarks/run_all_frameworks.py``.

Headline: AMBER (GPU) vs FLAME GPU 2 at 10M
--------------------------------------------

**Source of truth** (committed JSON, all 10 samples retained — no outlier
trim): ``benchmarks/results/benchmark_results_snapshot_correct_10run_10m.json``
and ``benchmarks/results/summary_table_snapshot_correct_10run_10m.md``.

**Protocol:** NVIDIA RTX 5090; 10M agents; 50 steps; 10 runs; one untimed
warm-up; timed scope = construct + setup + steps + assemble. Implementation
comparison under that protocol — not byte-identical dynamics across frameworks.

================  =============  =============  ==============================
Model             AMBER (GPU)    FLAME GPU 2    Speedup (FLAME / AMBER)
================  =============  =============  ==============================
Wealth            94 ms          194 ms         ~2.05×
Random walk       80 ms          161 ms         ~2.00×
SIR (cell-list)   2.08 s         3.68 s         ~1.77×
Schelling         295 ms         18.7 s         ~63× (setup-inclusive; exploratory)
================  =============  =============  ==============================

- **Wealth / walk / SIR** are the comparable headline class (~1.8–2.1×).
- **Schelling** includes heavy Python-side setup in the FLAME harness; do not
  treat ~63× as a pure step-kernel speedup.
- Multi-framework scale-out charts (Mesa, mesa-frames, Agents.jl, …) under
  ``benchmarks/results/summary_table.md`` are **exploratory** (trimmed means,
  older stack in places). Missing cells (OOM / budget) are not zeros.
- **Reproduce:**
  ``python benchmarks/run_all_frameworks.py --agents 10000000 --steps 50 --runs 10 --frameworks "AMBER (GPU)" "FLAME GPU 2"``

.. image:: ../benchmarks/results/scaling_chart.png
   :alt: Multi-framework scaling chart (exploratory full sweep)
   :width: 100%

Calibration throughput
----------------------

For derivative-free calibration, the GPU *ensemble* axis batches ``B``
simulations of ``N`` agents into one device pass (:func:`ambr.gpu_ensemble.smac_batch_calibrate`).
On a well-mixed SIR recovery task, AMBER's GPU batched ensemble reaches the best
held-out validation loss at roughly **270× the slowest framework** and ~3× the
fastest CPU competitor, evaluating ~970 candidate parameter sets per second.
