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
``off`` (the default) adds zero overhead.

.. automodule:: ambr.contract
   :members:
   :undoc-members:
   :show-inheritance:
