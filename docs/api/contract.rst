Snapshot-view contract
=======================

The snapshot-view contract certifies that AMBER's columnar fast path preserves
the intended update schedule (e.g. that every agent reads the start-of-step
snapshot rather than a half-mutated frame). Run with a contract mode and inspect
the per-step certificates:

.. code-block:: python

   results = model.run(steps=100, contract="check")   # "off" | "check" | "warn" | "raise"
   for cert in results["contract"]:
       if not cert.ok:
           print(cert.step, cert.violations)

``check`` records a :class:`~ambr.contract.ContractCertificate` per step; ``warn``
also emits a warning per violation; ``raise`` stops on the first one. Mode
``off`` (the default) adds zero overhead. Contract modes apply under both
``model.cpu(...)`` and ``model.gpu()`` (certificates use the CPU snapshot at
step boundaries).

Write paths the monitor sees
----------------------------

* **Buffered (OOP)** -- ``agent.col = value`` / ``Model._queue_write`` (per-cell
  duplicate detection).
* **Lane / view** -- ``agents.col = ...``, ``agents.set(...)``,
  ``agents.commit(...)``, and :class:`~ambr.tensor_lane.TensorLane` commits
  (per-column commit counting + borrow-after-commit).
* **Cross-path** -- the same column written via *both* OOP and lane/view in one
  step (``cross_path_write``).

``scatter_add`` is the sanctioned multi-write reducer and is **not** counted as
an ordinary lane/view commit.

Prefer those APIs over assigning ``population.data`` directly: raw population
assignment is invisible to the contract (documented escape hatch).

Runtime bookkeeping lives in :class:`~ambr.contract.ContractMonitor`;
:class:`~ambr.model.Model` only owns the thin write seams
(``_queue_write``, ``_set_frame(..., written_columns=...)``, ``run_step``).

.. automodule:: ambr.contract
   :members:
   :undoc-members:
   :show-inheritance:
