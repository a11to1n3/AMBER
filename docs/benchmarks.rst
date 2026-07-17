Benchmarks & Performance
========================

AMBER stores the whole population as a columnar Polars DataFrame and compiles
each step into a handful of vectorized expressions, so per-step cost stays
near-constant in the number of agents. This page summarises how that plays out
against other agent-based-modelling frameworks. The full, reproducible suite —
correctness checks, raw timings, and per-model speedups — lives under
``benchmarks/`` (see ``benchmarks/README.md``); reproduce the headline numbers
with ``python benchmarks/run_all_frameworks.py``.

Headline (5,000 agents)
-----------------------

At 5,000 agents over 50 steps, **AMBER (vectorized) is the fastest Python-hosted
framework** on the wealth-transfer and random-walk workloads, and competitive on
SIR. Against ``Agents.jl`` (Julia, compiled dispatch) AMBER trails on the two
cheapest per-step kernels, where fixed overhead dominates. Every framework is
checked against output invariants (wealth conservation, boundary clamping,
S+I+R conservation) **before** timing.

Scaling to 10M agents with the GPU backend
------------------------------------------

From **0.4.3**, the product API for a single large run is the same vectorized
``Model`` + view-API ``step`` under ``model.gpu().run()`` (:doc:`api/gpu`) —
not a separate kernel rewrite. The main harness times that path in
``benchmarks/run_all_frameworks.py``. Batched calibration is separate
(:doc:`api/gpu_ensemble`).

**Published figure (historical — not 0.4.3 native timings).** The chart below is
a multi-framework 1k→10M sweep (four models, including Schelling) measured with
the **pre–0.4.3** AMBER (GPU) harness on an NVIDIA RTX 3090 (hand-rolled
on-device loops / scale helpers). Keep it only as a qualitative large-N
comparison. **Do not cite per-point milliseconds or speedups as current
AMBER (GPU) performance under** ``model.gpu().run()``.

.. image:: ../benchmarks/results/scaling_chart_gpu_schelling.png
   :alt: Historical pre-0.4.3 GPU harness scaling chart (not native view-API timings)
   :width: 100%

- **API today:** write the view-API ``step`` once; place with ``.gpu().run()``
  (NVIDIA + CuPy). FLAME GPU 2 Schelling uses a ``MessageArray2D`` grid model in
  ``benchmarks/models/flamegpu_models.py``.
- **Refresh numbers:** re-run on CUDA with the current harness, then replot with
  ``python benchmarks/plot_scaling_with_gpu_schelling.py`` once a matching JSON
  is regenerated. Machine-local ``*5090*`` / interim JSON under
  ``benchmarks/results/`` is gitignored and is not the published baseline until
  checked in with an updated chart and prose.

Calibration throughput
----------------------

For derivative-free calibration, the GPU *ensemble* axis batches ``B``
simulations of ``N`` agents into one device pass (:func:`ambr.gpu_ensemble.smac_batch_calibrate`).
On a well-mixed SIR recovery task, AMBER's GPU batched ensemble reaches the best
held-out validation loss at roughly **270× the slowest framework** and ~3× the
fastest CPU competitor, evaluating ~970 candidate parameter sets per second.
