Speed lanes
===========

Progressive “how do I go faster?” helpers and the single-run array model.
See also the user guide :doc:`../going_faster`.

For a **view-API model on GPU**, prefer
``MyModel(...).gpu().run()`` (0.4.3) rather than rewriting the step as an
:class:`~ambr.lanes.ArrayKernelModel`. Use ``ArrayKernelModel`` when the
state is naturally a pure array dict and you do not need the Polars view API.

Status & recommendations
------------------------

.. autofunction:: ambr.lanes.status

.. autofunction:: ambr.lanes.print_status

.. autofunction:: ambr.lanes.recommend

Array kernel model
------------------

.. autoclass:: ambr.lanes.ArrayKernelModel
   :members:
   :undoc-members:
   :show-inheritance:
