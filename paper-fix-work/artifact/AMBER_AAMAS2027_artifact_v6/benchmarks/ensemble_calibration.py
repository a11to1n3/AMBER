"""GPU-batched calibration demo for AMBER's GPUEnsembleRunner.

Calibrates a well-mixed SIR to a target infection curve by evaluating thousands
of (beta, gamma) candidates *simultaneously* as one (B, N) tensor batch -- the
calibration / SMAC use case. Reports the recovered parameters and the speedup
of the batched ensemble over a sequential loop.
"""

import os
import sys
import time

os.environ.setdefault("CUDA_PATH", os.path.expanduser("~/cuda-12.0"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ambr.gpu import GPU_AVAILABLE, get_array_module, synchronize, to_host
from ambr.gpu_ensemble import GPUEnsembleRunner, BatchedWellMixedSIR

N = 2000
STEPS = 60
TRUE = dict(beta=0.35, gamma=0.08, i0_frac=0.01)


def main():
    if not GPU_AVAILABLE:
        print("No GPU available."); return
    xp = get_array_module()
    runner = GPUEnsembleRunner(BatchedWellMixedSIR())

    # warm up the device (CUDA context + kernel compilation) so timings are clean
    runner.run(N, STEPS, {k: np.full(64, 0.1, np.float32) for k in ("beta", "gamma", "i0_frac")})
    synchronize()

    # 1. target curve from the true parameters (one run)
    target = runner.run(N, STEPS, {k: [v] for k, v in TRUE.items()})["I_frac"][0]  # (STEPS,)

    # 2. calibrate: evaluate B candidate (beta, gamma) as ONE batch
    B = 4096
    rs = np.random.default_rng(0)
    cand = dict(
        beta=rs.uniform(0.10, 0.60, B),
        gamma=rs.uniform(0.02, 0.20, B),
        i0_frac=np.full(B, TRUE["i0_frac"]),
    )
    synchronize(); t0 = time.perf_counter()
    traj = runner.run(N, STEPS, cand)["I_frac"]            # (B, STEPS) on device
    loss = ((traj - target[None, :]) ** 2).sum(axis=1)     # (B,) batched objective
    best = int(loss.argmin())
    synchronize(); t_batched = time.perf_counter() - t0

    print("GPU-batched SIR calibration")
    print(f"  agents/run N={N}, steps={STEPS}, candidates B={B}")
    print(f"  TRUE      beta={TRUE['beta']:.3f}  gamma={TRUE['gamma']:.3f}")
    print(f"  RECOVERED beta={cand['beta'][best]:.3f}  gamma={cand['gamma'][best]:.3f}  "
          f"(loss {float(loss[best]):.4f})")
    print(f"  evaluated {B} candidates in {t_batched * 1000:.1f} ms "
          f"({B / t_batched:,.0f} candidate-runs/s)\n")

    # 3. batched vs sequential at a fixed B (measured, not extrapolated)
    print("Batched ensemble vs sequential loop:")
    for Bc in (256, 1024):
        sub = {k: v[:Bc] for k, v in cand.items()}
        synchronize(); t0 = time.perf_counter()
        runner.run(N, STEPS, sub); synchronize()
        t_bat = time.perf_counter() - t0

        synchronize(); t0 = time.perf_counter()
        for b in range(Bc):
            one = {k: v[b:b + 1] for k, v in sub.items()}
            runner.run(N, STEPS, one)
        synchronize()
        t_seq = time.perf_counter() - t0
        print(f"  B={Bc:>4}: batched {t_bat*1000:7.1f} ms | sequential {t_seq*1000:8.1f} ms "
              f"| {t_seq/t_bat:5.0f}x")


if __name__ == "__main__":
    main()
