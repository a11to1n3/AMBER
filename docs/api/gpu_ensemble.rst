GPU batched ensemble
====================

Batches a ``(B simulations × N agents)`` ensemble into a single device pass —
the natural fit for derivative-free calibration, where you evaluate thousands of
small replicate runs.

.. code-block:: python

   from ambr.gpu_ensemble import GPUEnsembleRunner, BatchedWellMixedSIR, smac_batch_calibrate

   # Evaluate B parameter sets in one (B, N) pass.
   runner = GPUEnsembleRunner(BatchedWellMixedSIR())
   traj = runner.run(n_agents=100_000, steps=60,
                     params={"beta": betas, "gamma": gammas, "i0_frac": i0})

   # SMAC ask -> one batched evaluation -> tell.
   best, history = smac_batch_calibrate(BatchedWellMixedSIR(), bounds, loss_fn,
                                        n_agents=100_000, steps=60)

.. automodule:: ambr.gpu_ensemble
   :members:
   :undoc-members:
   :show-inheritance:
