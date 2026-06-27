Tensor lane
===========

The tensor lane provides zero-copy ``borrow`` / ``commit`` over the Polars frame
for kernels that are easier (or faster) to express as array math. Borrow a
numeric column as a NumPy view, compute, and commit derived columns back
atomically — routed through the snapshot-view contract so the writes stay
observable.

.. code-block:: python

   x, is_view = model.agents.borrow("x")   # zero-copy where the layout allows
   model.agents.commit(x=x * 2.0)          # atomic write-back, contract-observed

.. automodule:: ambr.tensor_lane
   :members:
   :undoc-members:
   :show-inheritance:
