Benchmarks & Performance
========================

AMBER stores the whole population as a columnar Polars DataFrame and compiles
each step into a handful of vectorized expressions. From **0.4.3**, the same
vectorized ``Model`` + view-API ``step`` runs on GPU via ``model.gpu().run()``
(:doc:`api/gpu`) — not a separate kernel rewrite. Batched calibration is
separate (:doc:`api/gpu_ensemble`).

The full, reproducible suite — correctness checks, raw timings, and per-model
tables — lives under ``benchmarks/`` (see ``benchmarks/README.md``). Reproduce
with ``python benchmarks/run_all_frameworks.py``.

Large-N multi-framework scaling (1k→10M)
---------------------------------------

**Protocol:** NVIDIA RTX 5090, 50 steps, 10 runs (trimmed mean). Ten
frameworks: AMBER (GPU / vectorized / loop), mesa-frames, FLAME GPU 2,
Agents.jl, SimPy, Melodie, AgentPy, Mesa. AMBER (GPU) is the same view-API
classes under ``model.gpu().run()``. Full table:
``benchmarks/results/summary_table.md``.

.. image:: ../benchmarks/results/scaling_chart.png
   :alt: Multi-framework scaling from 1k to 10M agents on RTX 5090
   :width: 100%

**At 1M / 10M agents (where each framework still finishes):**

================  =================  =====================  =================  ===========================
Model             AMBER (GPU)        AMBER (vectorized)     FLAME GPU 2        Next best CPU-scale peer
================  =================  =====================  =================  ===========================
Wealth            3.91 s / 193 s     6.44 s / 214 s         28 ms / 226 ms     Agents.jl 8.53 s @ 1M
Random walk       198 ms / 2.04 s    531 ms / 6.23 s        20 ms / 201 ms     mesa-frames 3.55 s / 20.8 s
Schelling         428 ms / 5.17 s    2.64 s / 59.8 s        2.06 s / 20.8 s    mesa-frames 4.33 s / 86.9 s
SIR (cell-list)   882 ms / 9.39 s    31.3 s / —             108 ms / 3.80 s    —
================  =================  =====================  =================  ===========================

- **Schelling:** AMBER (GPU) is the fastest measured row at 1M and 10M.
- **Wealth / random walk:** FLAME GPU 2 leads; AMBER (GPU) still leads other
  Python-hosted stacks that reach those scales.
- **SIR:** AMBER uses **cell-list** infection (GPU reaches 10M at 9.39 s).
  FLAME still leads; vectorized 10M exceeded the 120 s/run budget in this pass.
- **Reproduce:**
  ``python benchmarks/run_all_frameworks.py --agents 1000 10000 100000 1000000 10000000 --steps 50 --runs 10``
  then ``python benchmarks/plot_scaling_with_gpu_schelling.py``.

Calibration throughput
----------------------

For derivative-free calibration, the GPU *ensemble* axis batches ``B``
simulations of ``N`` agents into one device pass (:func:`ambr.gpu_ensemble.smac_batch_calibrate`).
On a well-mixed SIR recovery task, AMBER's GPU batched ensemble reaches the best
held-out validation loss at roughly **270× the slowest framework** and ~3× the
fastest CPU competitor, evaluating ~970 candidate parameter sets per second.
