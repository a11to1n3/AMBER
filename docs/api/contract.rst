Snapshot-view contract
=======================

**Operational monitor, not schedule proof.** The snapshot-view contract reports
selected hazards at AMBER's instrumented read/write seams (for example a borrow
after a same-step commit). ``cert.clean`` means no monitored hazard was observed;
it is **not** a proof that a vectorized/GPU rewrite preserves an intended
activation schedule, confluence, or bit-identical trajectories vs an OOP loop.
Private GPU fast paths under ``approve_fast_path`` are unmonitored. Run with a
contract mode and inspect the per-step records:

.. code-block:: python

   import ambr as am

   class WealthModel(am.Model):
       def setup(self):
           self.add_agents(50, wealth=1)

       def step_vectorized(self):
           donors = self.agents.where(self.agents.wealth > 0)
           donors.wealth -= 1
           rec = self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
           self.agents.at[rec].scatter_add(wealth=1)

   model = WealthModel({"steps": 20, "seed": 0, "show_progress": False})
   results = model.run(steps=20, contract="check")   # "off" | "check" | "warn" | "raise"
   for cert in results["contract"]:
       if not cert.ok:
           print(cert.step, cert.violations)

``check`` records a :class:`~ambr.contract.ContractCertificate` per step; ``warn``
also emits a warning per violation; ``raise`` stops on the first one. Mode
``off`` (the default) adds no monitor bookkeeping. Contract modes apply under
both ``model.cpu(...)`` and ``model.gpu()`` on the instrumented general runner.
An optional private GPU fast loop is selected only with ``contract="off"``
and an explicit per-instance deployment declaration such as
``model.approve_fast_path("smoke-report-2026-07")``. The string is a
caller-supplied provenance label, not evidence AMBER verifies. Without that
declaration the general native runner is used. The private loop is not
monitored. ``cert.clean`` means no monitored error or warning was
observed; it is not a completeness or confluence claim.

Write paths the monitor sees
----------------------------

* **Buffered (OOP)** -- ``agent.col = value`` / ``Model._queue_write`` (per-cell
  duplicate detection).
* **Lane / view** -- ``agents.col = ...``, ``agents.set(...)``,
  ``agents.commit(...)``, and :class:`~ambr.tensor_lane.TensorLane` commits
  (per-column commit counting + borrow-after-commit).
* **Mutable raw arrays** -- ``agents.array(...)`` records
  ``uncertified_mutable_borrow`` because later indexing and in-place mutations
  cannot be reconstructed reliably.
* **Cross-path** -- the same column written via *both* OOP and lane/view in one
  step (``cross_path_write``).

``scatter_add`` is the sanctioned multi-write reducer and is **not** counted as
an ordinary lane/view commit.

Prefer those APIs over assigning ``population.data`` directly: direct frame
replacement is deprecated and remains outside the write-conflict seam, although
endpoint schema/id changes can still be reported.

Runtime bookkeeping lives in :class:`~ambr.contract.ContractMonitor`;
:class:`~ambr.model.Model` only owns the thin write seams
(``_queue_write``, ``_set_frame(..., written_columns=...)``, ``run_step``).

.. automodule:: ambr.contract
   :members:
   :undoc-members:
   :show-inheritance:
