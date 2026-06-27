"""Batched SMAC calibration on the GPU ensemble (ask -> batch-evaluate -> tell).

Calibrates a well-mixed SIR to a target curve two ways at the SAME GPU budget:
  * SMAC: each round proposes a batch of configs (ask x B), the GPU evaluates
    all of them in one (B, N) pass, losses are told back (tell x B).
  * Random search: one batch of the full budget, evaluated in one GPU pass.
SMAC's surrogate-guided batches should reach a lower loss for equal evaluations.
"""

import os
import sys
import time

os.environ.setdefault("CUDA_PATH", os.path.expanduser("~/cuda-12.0"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ambr.gpu import GPU_AVAILABLE, get_array_module, synchronize
from ambr.gpu_ensemble import GPUEnsembleRunner, BatchedWellMixedSIR, smac_batch_calibrate

N = 2000
STEPS = 60
TRUE = dict(beta=0.35, gamma=0.08)
I0 = 0.01
BOUNDS = dict(beta=(0.10, 0.60), gamma=(0.02, 0.20))
ROUNDS, BATCH = 8, 64          # total budget = 512 evaluations


def main():
    if not GPU_AVAILABLE:
        print("No GPU available."); return
    xp = get_array_module()
    model = BatchedWellMixedSIR()
    runner = GPUEnsembleRunner(model)

    # target curve from the true parameters
    target = runner.run(N, STEPS, {**{k: [v] for k, v in TRUE.items()}, "i0_frac": [I0]})["I_frac"][0]

    def loss_fn(traj):
        return ((traj["I_frac"] - target[None, :]) ** 2).sum(axis=1)   # (B,) on device

    budget = ROUNDS * BATCH

    # --- SMAC: batched ask -> evaluate -> tell ---
    synchronize(); t0 = time.perf_counter()
    best, history = smac_batch_calibrate(
        model, BOUNDS, loss_fn, N, STEPS,
        rounds=ROUNDS, batch_size=BATCH, fixed_params={"i0_frac": I0}, seed=0,
    )
    synchronize(); t_smac = time.perf_counter() - t0

    # --- Random search at the same budget (one big batch) ---
    rs = np.random.default_rng(0)
    cand = dict(
        beta=rs.uniform(*BOUNDS["beta"], budget),
        gamma=rs.uniform(*BOUNDS["gamma"], budget),
        i0_frac=np.full(budget, I0),
    )
    synchronize(); t0 = time.perf_counter()
    traj = runner.run(N, STEPS, cand)
    rloss = loss_fn(traj)
    rbest = int(rloss.argmin())
    synchronize(); t_rand = time.perf_counter() - t0

    print(f"Calibrating well-mixed SIR  (N={N}, steps={STEPS}, budget={budget} evals)")
    print(f"  TRUE             beta={TRUE['beta']:.3f}  gamma={TRUE['gamma']:.3f}\n")
    print(f"  SMAC (batched)   beta={best['beta']:.3f}  gamma={best['gamma']:.3f}  "
          f"| best loss {min(history):.5f} | {t_smac*1000:6.0f} ms ({ROUNDS} rounds x {BATCH})")
    print(f"  Random search    beta={cand['beta'][rbest]:.3f}  gamma={cand['gamma'][rbest]:.3f}  "
          f"| best loss {float(rloss[rbest]):.5f} | {t_rand*1000:6.0f} ms (1 batch of {budget})")
    print(f"\n  SMAC per-round best loss: {[round(h, 4) for h in history]}")


if __name__ == "__main__":
    main()
