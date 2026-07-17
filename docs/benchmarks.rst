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

From **0.4.3**, AMBER (GPU) is the same vectorized ``Model`` + view-API
``step`` under ``model.gpu().run()`` (:doc:`api/gpu`, :doc:`api/gpu_ensemble`) —
not a separate kernel rewrite. The chart below sweeps **1k → 10M agents across
10 frameworks and four models** (Wealth Transfer, Random Walk, SIR Epidemic,
and Schelling segregation), adding **AMBER (GPU)** as a series (NVIDIA RTX 3090):

.. image:: ../benchmarks/results/scaling_chart_gpu_schelling.png
   :alt: AMBER GPU + Schelling scaling to 10M agents across 10 frameworks
   :width: 100%

- **AMBER (GPU) reaches 10M agents on all four models** while CPU frameworks blow
  up. Wealth transfer: ~14 ms at 1M (≈330× the fastest CPU framework), 199 ms at
  10M (**3.1× faster than FLAME GPU 2**). Schelling: the **only** framework to
  reach 10M (847 ms; 19× AMBER-vectorized and 225× ``Agents.jl`` at 1M). SIR:
  5.98 s at 10M (**~2× faster than FLAME GPU 2**).
- **It is a large-N win, not a small-N one.** A ~90 ms fixed device cost means
  AMBER (GPU) only leads at scale: below ~100k–1M agents, AMBER (vectorized) or
  ``Agents.jl`` are faster, and on SIR FLAME GPU 2 wins at 1k–10k before AMBER
  overtakes it from 100k up. FLAME GPU 2 also runs Schelling via a
  ``MessageArray2D`` grid model in ``benchmarks/models/flamegpu_models.py``.

Regenerate this chart from recorded data with
``python benchmarks/plot_scaling_with_gpu_schelling.py``. Machine-local JSON /
``*5090*`` reruns under ``benchmarks/results/`` are gitignored — do not treat
interim split logs as the published baseline.

Calibration throughput
----------------------

For derivative-free calibration, the GPU *ensemble* axis batches ``B``
simulations of ``N`` agents into one device pass (:func:`ambr.gpu_ensemble.smac_batch_calibrate`).
On a well-mixed SIR recovery task, AMBER's GPU batched ensemble reaches the best
held-out validation loss at roughly **270× the slowest framework** and ~3× the
fastest CPU competitor, evaluating ~970 candidate parameter sets per second.
