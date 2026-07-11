GPU batched ensemble
====================

Batches a ``(B simulations × N agents)`` ensemble into a single device pass —
the natural fit for derivative-free calibration, where you evaluate thousands of
small replicate runs.

.. code-block:: python

   import ambr as am
   import numpy as np

   # Evaluate B parameter sets in one (B, N) pass (NumPy or CuPy).
   B = 8
   runner = am.GPUEnsembleRunner(am.BatchedWellMixedSIR())
   traj = runner.run(
       n_agents=1_000,
       steps=20,
       params={
           "beta": np.linspace(0.1, 0.4, B),
           "gamma": np.full(B, 0.05),
           "i0_frac": np.full(B, 0.02),
       },
       seed=0,
   )
   # traj["I_frac"] has shape (B, steps)

.. automodule:: ambr.gpu_ensemble
   :members:
   :undoc-members:
   :show-inheritance:
