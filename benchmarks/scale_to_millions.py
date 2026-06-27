"""Scale AMBER's GPU spatial backend to millions of agents (constant density).

A proper scalability test holds *density* constant -- the world grows with N
(world = sqrt(N / density)) so each agent's neighbour count stays bounded and we
measure true O(N) scaling, not the artefact of cramming N agents into a fixed
box. Compares three SIR implementations at each N:

  * AMBER (GPU, cell-list)  -- O(N) uniform-grid neighbour query, chunked.
  * FLAME GPU 2             -- spatial messaging (the reference at scale).
  * AMBER (GPU, all-pairs)  -- O(N^2); shown until it OOMs.

Reports ms/step and throughput (million agent-steps / second).
"""

import math
import os
import sys
import time

os.environ.setdefault("CUDA_PATH", os.path.expanduser("~/cuda-12.0"))
os.environ.setdefault("FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from models.amber_gpu_scale_models import GPUSIRBinnedModel, GPUSIRKernelModel
from models.amber_gpu_models import GPUSIRModel as GPUSIRAllPairs
from models.flamegpu_models import SIRModel as FlameSIR

DENSITY = 0.1
STEPS = 50
SIZES = [10_000, 100_000, 1_000_000, 4_000_000, 8_000_000]


def cfg_for(n):
    return dict(
        world_size=math.sqrt(n / DENSITY),
        movement_speed=2.0,
        infection_radius=5.0,
        transmission_rate=0.1,
        recovery_time=14,
        initial_infected=max(5, n // 200),   # 0.5% seed so the epidemic spreads
        max_per_cell=24,
        chunk=500_000,
    )


def measure(make, n):
    """Warm once at a small size, then time a full run at n. Returns seconds or error str."""
    try:
        make(10_000).run()
    except Exception:
        pass
    try:
        t0 = time.perf_counter()
        make(n).run()
        return time.perf_counter() - t0
    except Exception as e:
        return f"{type(e).__name__}"


def fmt(sec):
    if isinstance(sec, str):
        return f"{sec:>14}"
    ms_step = sec / STEPS * 1000
    return f"{ms_step:8.1f} ms/st"


def thr(sec, n):
    if isinstance(sec, str):
        return "—"
    return f"{n * STEPS / sec / 1e6:.0f}M"


if __name__ == "__main__":
    # correctness sanity: both AMBER GPU SIR variants spread the epidemic
    c = cfg_for(100_000)
    for label, M in (("cell-list", GPUSIRBinnedModel), ("kernel", GPUSIRKernelModel)):
        st = M(100_000, STEPS, c).run(return_state=True)
        print(f"sanity @100k {label:9s}: S/I/R = {st['S']}/{st['I']}/{st['R']} "
              f"(recovered {st['R']/100_000:.1%})")
    print()

    print(f"Constant-density SIR scaling  (density={DENSITY}/unit^2, {STEPS} steps)")
    hdr = ("N", "AMBER kernel", "AMBER cell-list", "FLAME GPU 2", "AMBER all-pairs", "kernel thrpt")
    print(f"{hdr[0]:>9} | {hdr[1]:>15} | {hdr[2]:>15} | {hdr[3]:>15} | {hdr[4]:>15} | {hdr[5]:>14}")
    print("-" * 105)
    for n in SIZES:
        c = cfg_for(n)
        t_ker = measure(lambda nn, c=c: GPUSIRKernelModel(nn, STEPS, {**c}), n)
        t_bin = measure(lambda nn, c=c: GPUSIRBinnedModel(nn, STEPS, {**c}), n)
        t_flame = measure(lambda nn, c=c: FlameSIR(nn, STEPS, {**c}), n)
        t_all = measure(lambda nn, c=c: GPUSIRAllPairs(nn, STEPS, {**c}), n) if n <= 1_000_000 else "skipped"
        print(f"{n:>9} | {fmt(t_ker):>15} | {fmt(t_bin):>15} | {fmt(t_flame):>15} | {fmt(t_all):>15} | {thr(t_ker, n):>9} a-s/s")
