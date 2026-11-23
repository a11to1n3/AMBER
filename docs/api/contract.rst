Snapshot-view contract
=======================

The snapshot-view contract reports hazards at AMBER's instrumented read/write
seams, such as a borrow after a same-step commit. ``cert.clean`` means no
monitored hazard was observed. Inspect ``cert.conditional_guarantee`` for the
order-independence verdict. This example uses a supported CPU activation sweep.

.. code-block:: python

   import ambr as am

   class Person(am.Agent):
       def step(self):
           if self.draw("earn") < 0.25:
               self.wealth += 1

   class WealthModel(am.Model):
       def setup(self):
           self.add_agents(50, agent_class=Person, wealth=1)

       def step(self):
           self.agents.step()

   results = WealthModel({"steps": 20, "seed": 7}).run(contract="check")
   cert = results["contract"][0]
   assert cert.conditional_guarantee == "established"
   print(cert.certification)

``check`` records a :class:`~ambr.contract.ContractCertificate` per step.
``warn`` also emits a warning per violation. ``raise`` stops on the first error.
Mode ``off`` (the default) disables monitor bookkeeping. Contract modes apply under
both ``model.cpu(...)`` and ``model.gpu()`` on the instrumented general runner.
An optional private GPU fast loop is selected only with ``contract="off"``
and an explicit per-instance deployment declaration such as
``model.approve_fast_path("smoke-report-2026-07")``. The string is a
caller-supplied provenance label. Private loops bypass the monitor.

When the guarantee is established
---------------------------------

The verdict applies to permutations within one sequential or shuffled CPU
sweep over the tracked root population at its observed entry state. A
supported method adds or subtracts integer constants in its own row,
optionally under one local or keyed draw condition, or calls ``scatter_add``
or ``scatter_max`` with a constant integer and a single fixed target shared
by the agents. AMBER checks unique IDs, committed inputs, non-null ``Int64``
arithmetic, safe bounds, and agreement between the predicted and observed
cell effects and draw values.
``cert.certification`` records the scope and premises in the saved results.

For other methods and execution paths, the verdict is ``not_established``
and ``cert.certification_reason`` explains the first failing check. Shared
random streams, multi-stage execution, and GPU runs are outside the current
proof. Separately written vectorized or GPU kernels require reference comparisons.

Keyed draws
-----------

``self.draw("earn")`` on an Agent returns a uniform value in ``[0, 1)``
from the model seed, current step, agent ID, event name, and draw index.
Reusing a key gives the same value. Use different event names for separate
decisions or pass ``index=1``, ``index=2``, and so on for multiple draws
within an event. Pass ``seed`` in the model parameters to repeat draws across
model instances. Existing ``random``, ``rng``, and ``nprandom`` retain their
shared-stream behavior.

The verifier supports one comparison of ``self.draw("event", index=0)``
to a constant in ``[0, 1]`` guarding row-local integer updates. Saved
certificates include the algorithm, seed, step, event, index, and draw count
under ``cert.certification["keyed_randomness"]``. The same draw API works in
Python agent methods under ``model.gpu()``; those methods currently compute
the keyed draws on the host.

For swapped-order replay, AMBER compares the model-owned random streams
immediately before and after the sweep. A stream change can reveal draws
through a cached method. Replay leaves the witness inconclusive when a
stream changes or its state cannot be verified. The scheduler's shuffle
occurs before this check. External random sources remain outside this trace.
The comparison includes the final agent frame, supported public model state,
and values recorded with ``record_model()`` during the step. The swapped
replay starts from each agent's private attributes as they were at step
entry. After the replay, each agent's instance attributes, including
underscore-prefixed ones, are restored to their forward bindings. Private
lists, dicts, sets, and numeric arrays also get their top-level contents back
in place; nested containers inside them are not copied.

Write paths the monitor sees
----------------------------

* **Buffered (OOP)** -- ``agent.col = value`` / ``Model._queue_write`` (per-cell
  duplicate detection).
* **Attributed activation reads and writes** -- ordinary agent attribute
  access and view writes with known targets during method broadcasts,
  ``activate_agents``, and staged activation, used to report ``activation_race``.
  Reading ``model.agents_df``, ``Agent.get_data()``, or
  ``Agent.get_neighbors(...)`` inside an activation counts as a read of every
  column.
* **Lane / view** -- ``agents.col = ...``, ``agents.set(...)``,
  ``agents.commit(...)``, and :class:`~ambr.tensor_lane.TensorLane` commits
  (per-column commit counting + borrow-after-commit). Inside an activation,
  a borrow is checked only against commits made during the same sweep or
  stage. A value committed by an earlier phase of the step is the sweep's
  input; a value written by another activation of the sweep is reported as
  ``activation_race``.
* **Mutable raw arrays** -- ``agents.array(...)`` records
  ``uncertified_mutable_borrow`` because later indexing and in-place mutations
  cannot be reconstructed reliably.
* **Cross-path** -- the same column written via *both* OOP and lane/view in one
  step (``cross_path_write``).

Ordinary view assignments require unique target IDs. ``scatter_add`` and
``scatter_max`` accept repeated IDs. The monitor checks ID multiplicity at
both step endpoints.

Prefer those APIs over assigning ``population.data`` directly: direct frame
replacement is deprecated and observed through its schema and ID changes at
step endpoints. ``model.agents_df = ...`` inside an activation reports an
incomplete trace because its effects cannot be attributed to individual cells.

Runtime bookkeeping lives in :class:`~ambr.contract.ContractMonitor`;
:class:`~ambr.model.Model` only owns the thin write seams
(``_queue_write``, ``_set_frame(..., written_columns=...)``, ``run_step``).

.. automodule:: ambr.contract
   :members:
   :undoc-members:
   :show-inheritance:
