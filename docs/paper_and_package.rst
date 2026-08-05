Paper vs package
================

The research paper and the installable **PyPI package** are related but
**not interchangeable** as evidence sources.

Package (source of truth for software claims)
---------------------------------------------

* Name: ``ambr`` on PyPI (current line: **0.4.x**, Beta classifier).
* Performance headlines in the **README / Sphinx** cite the committed
  snapshot::

     benchmarks/results/benchmark_results_snapshot_correct_10run_10m.json

  (AMBER GPU vs FLAME GPU 2, 10M agents, RTX 5090 protocol).
* GPU claims require **NVIDIA + CuPy**; verified with
  ``scripts/run_gpu_claims.py``.
* API and lanes: dual OOP + vectorized, ``model.gpu()``, ensemble calibration,
  contract monitor — as documented in this tree.

Paper (historical / academic evaluation setting)
------------------------------------------------

* arXiv: https://arxiv.org/abs/2601.16292
* The paper’s **main empirical table** evaluates an **earlier** package line
  (reported as AMBER 0.1.x era) at **much smaller N** (≤10k) on **CPU**
  (e.g. Apple M2), with GPU called out as future work in that setting.
* Do **not** cite the paper table as the same claim as the 10M FLAME GPU
  README numbers without stating the different protocol.

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
