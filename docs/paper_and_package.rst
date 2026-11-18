Paper vs package
================

The research paper and the installable **PyPI package** are related but
**not interchangeable** as evidence sources.

Package (source of truth for software claims)
---------------------------------------------

* Name: ``ambr`` on PyPI (Beta classifier).
* The 0.6.0 automatic batching benchmark uses two ordinary agent models
  at 1,000 and 10,000 agents, with samples in
  ``benchmarks/results/automatic_batching_0_6_0.json``.
* The historical 10-million-agent comparison uses explicit GPU kernels
  with the committed snapshot::

     benchmarks/results/benchmark_results_snapshot_correct_10run_10m.json

  (AMBER GPU vs FLAME GPU 2, 10M agents, RTX 5090 protocol).
* GPU claims require **NVIDIA + CuPy**; verified with
  ``scripts/run_gpu_claims.py``.
* API and lanes: dual OOP + vectorized, ``model.gpu()``, ensemble calibration,
  contract monitor — as documented in this tree.

Original arXiv evaluation
-------------------------

* arXiv: https://arxiv.org/abs/2601.16292
* The paper’s **main empirical table** evaluates an **earlier** package line
  (reported as AMBER 0.1.x era) at **much smaller N** (≤10k) on **CPU**
  (e.g. Apple M2), with GPU called out as future work in that setting.
* Do **not** cite the paper table as the same claim as the 10M FLAME GPU
  README numbers without stating the different protocol.

Round 73 and Round 75 reviewer evaluations
-------------------------------------------

Round 73 records GPU timings from research commit ``a29811d80283`` using
approved workload-specific hooks on an RTX 5090. The historical 10-million-agent
comparison in the README comes from a July experiment based on ``cdb4f135``. The Round 73
600-second guard was enforced by the CPU and peer-framework timing drivers.
All 140 archived GPU timing attempts completed in less than three seconds.

Round 75 reran the paper's timing and correctness checks with the
``0.5.2+fix.11`` research wheel. The independent SIR transition comparisons
use that same wheel and reach one million agents. Cite each result with its
recorded version and protocol.

How to cite
-----------

* **Software / reproducibility of the library:** use ``CITATION.cff`` and the
  installed ``ambr`` version (``import ambr; print(ambr.__version__)``).
* **Academic narrative / architecture:** cite the paper **and** name the
  package version you actually ran.

Alignment policy
----------------

1. User-facing package docs must not silently present paper-era numbers as
   current package defaults.
2. When package claims change, update README + committed snapshot (or clearly
   mark exploratory tables).
3. Paper errata / updates live outside the library tree unless maintainers
   deliberately re-export figures into ``benchmarks/results/``.

See :doc:`benchmarks`, :doc:`reproducibility`, :doc:`roadmap_1_0`.
