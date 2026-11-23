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
cell effects and draw values. The step must be that sweep alone. The verdict
is withheld when, outside the sweep, the step commits or reduces a column,
replaces the agent frame, queues a write with ``update_agent_data()``,
changes the agent frame by any other path (for example an in-place
``agents_df.replace_column(...)`` or a ``Population`` method), draws from a
model random stream, records a value with ``record_model()``, changes the
agent roster or an agent's ID binding, or changes model attributes or
private agent attributes between step entry and step end. The frame and the
streams are compared at step entry, at the sweep's entry and exit, and at
step end, so these checks do not depend on the write path. Model and private
agent state is compared with the replay snapshot described below, and
``cert.certification_reason`` names the first attribute that changed. When
that state reaches a container the snapshot cannot represent, a change
cannot be ruled out, the verdict is withheld, and the reason names the
attribute. An object the snapshot keeps by identity does not withhold the
verdict while it is still bound and its own recorded attributes are
unchanged; ``cert.assumptions`` lists each such attribute, because a change
inside the object cannot be seen.
The verdict covers ``step()`` only: model reporters and ``update()`` run
after the certificate is issued.
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

A step runs at most five swapped-order replays, each swapping two
activations of its single recorded sweep, so many reported pairs may not be
replayed. The budget goes first to the reported pairs in report order, one
replay per distinct pair. When three or more activations write one cell,
AMBER reports a swap of the last writer with one earlier writer per
distinct recorded value, nearest first; the nearest of these is replayed in
report order and the others are coverage pairs. Values are recorded for
agent attribute writes and for ``agents.at[ids].set(...)`` with scalar or
list values on up to 64 agents. Only built-in and NumPy scalars are
recorded, so the monitor never runs a user ``__eq__`` or ``__hash__``; any
other value, or one whose key cannot be built (for example
``np.longdouble``), counts as unrecorded. A writer whose value was not
recorded could hold any value, so such writers are sampled across the
sweep, earliest first. When activations reduce into one cell with the same mix of
operators, the first two are paired, and each later reducer whose recorded
operators and scalar operands differ is paired with the first; these pairs
are coverage pairs. Coverage pairs use the replays left after the reported
pairs, so a two-writer report is replayed unless the reports before it used
all five replays. A reported pair that was not replayed says so in
``divergence_reason``, including whether earlier reports or coverage pairs
used the budget. When a cell has more candidate pairs than were replayed and
the replayed pairs show no divergence, ``divergence_reason`` gives the
number of candidates and the number replayed.

For swapped-order replay, AMBER compares the model-owned random streams
immediately before and after the sweep. A stream change can reveal draws
through a cached method. Replay leaves the witness inconclusive when a
stream changes or its state cannot be verified. The scheduler's shuffle
occurs before this check. Other random sources remain outside this trace.

What a replay restores
~~~~~~~~~~~~~~~~~~~~~~

A swapped-order replay runs the step's agent code a second time. Before it
runs, AMBER puts back the step-entry state listed here; afterwards it puts
back the forward state, so the replay leaves no trace in this state:

* the agent frame, pending writes, the per-agent attribute cache, and the
  agent roster;
* the model's random streams (``random``, ``rng``, ``nprandom``) and the
  global ``random`` and legacy ``numpy.random`` streams;
* the model's public and underscore-prefixed attributes (framework-owned
  attributes excluded) and each agent's underscore-prefixed attributes,
  with the state they reach: lists, deques, dicts (including
  ``defaultdict`` with its ``default_factory``, and ``Counter`` and
  ``OrderedDict`` without instance attributes), sets, tuples and named
  tuples, NumPy arrays of numeric, boolean, datetime, timedelta, or
  fixed-width string dtype (a view through the array that owns its memory,
  with its shape), dataclass instances and ``SimpleNamespace`` objects
  field by field, Polars ``DataFrame`` and ``Series`` objects,
  ``random.Random``, NumPy ``Generator``, and ``RandomState`` objects,
  functions (code, defaults, attributes, and closure cells), plain classes
  (namespace and bases), bound methods, and slot attributes (``__slots__``).
  Containers are updated in place, so aliases are kept. Immutable values
  (numbers, strings, bytes, ``None``, enum members, dates, times, time
  deltas, ``Decimal``, ``Fraction``, ``range``, paths, and tuples or
  frozensets of these) are kept as they are.

Any other object reached this way, for example a ``GridEnvironment``, a
``Logger``, a lock, a file, or an instance of your own class, is kept by
identity. When its class is written in Python (logging objects excepted),
its own attributes are also recorded one level deep and put back, with the
containers listed above that they hold; the other objects they refer to are
kept by identity only. Containers are matched by exact type. The swap is not
run, ``divergence_reason`` names the attribute, and the certificate is
withheld, when this state reaches a container that cannot be restored
without calling user code or at all: a subclass of ``list``, ``dict``,
``set``, ``deque``, or ``ndarray``, a ``bytearray`` or ``memoryview``, an
array that does not own its memory or holds Python objects, a set item or
dict key that is not an immutable value, or a module.

A replay does not restore module-level variables (including those a stored
function reads or writes), class attributes of classes reached only through
``type(obj)`` or not held by the model, the internal state of objects kept by
identity beyond the attributes described above, files, network connections,
printed or logged output, or any other effect outside the state listed
above. When a replayed activation changes such state, the change happens
again in the forward run, so ``contract="check"`` can then give different
results from ``contract="off"``, and a witness can be wrong because the
replay started from changed state. Do not rely on replay for a model whose
activations have such effects. For every other model, ``contract="check"``
and ``"warn"`` give the same agent frame, model and agent attributes,
recorded values, and random streams as ``contract="off"``.

The witness compares the final agent frame, the restored model and private
agent state, and values recorded with ``record_model()`` during the step.
Objects kept by identity are compared by identity and by their recorded
attributes.

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
  ``activation_race``. Consecutive sweeps over the whole agent list
  (``model.agents.step()``, ``model.agents.call(...)``,
  ``activate_agents()``, or ``am.activate(model.agents, ...)``) each have
  their own activation trace, so a later such sweep that reads an earlier
  one's writes is not a race. A sweep over a subset or view
  (``agents.at[ids].call(...)``, a filtered view, or a list of agents) joins
  the open trace instead: the model, not AMBER, chooses the order of such
  sweeps, for example by shuffling one-agent sweeps, so their reads and
  writes are checked against every activation of the trace. A step with
  more than one sweep is not replayed, and its races give
  ``divergence_reason`` ``"replay requires a single recorded activation
  sweep"``.
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
