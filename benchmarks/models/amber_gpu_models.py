"""AMBER GPU-backend benchmark models (CuPy).

The *same vectorized algorithms* as AMBER's CPU vectorized models
(``amber_models.AMBERVectorized*``), but with agent state held resident on the
GPU (CuPy) and every per-step kernel running on-device. This is AMBER's
columnar / tensor execution path running on the GPU backend (``ambr.gpu``):
state is initialised on the device, the step loop never touches the host, and
we synchronise once at the end for honest timing.

Notes on scaling:
* wealth_transfer / random_walk are O(N) per step -> scale to N = 1e6 easily.
* sir_epidemic is benchmarked with the O(N) spatial-binning kernel from
  amber_gpu_scale_models (counting-sort group-by + fixed-radius self-join CUDA
  kernel, no per-cell cap), which scales to N = 1e7 on a 24 GB GPU. The naive
  all-pairs (S x I) matrix version is kept as GPUSIRModel /
  AMBER_GPU_MODELS['sir_epidemic_naive']; it OOMs the device at N >= 1e5.
"""

import os
import sys

from ambr.gpu import GPU_AVAILABLE

if GPU_AVAILABLE:
    import cupy as cp
    from cupyx import scatter_add

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _schelling_core import schelling_setup, schelling_step

# Scalable O(N) spatial-binning SIR (counting-sort group-by + fixed-radius
# self-join CUDA kernel). The naive GPUSIRModel below materialises a dense
# (S x I) contact matrix and OOMs the device at N >= 1e5; the kernel model is
# the same fixed-radius SIR dynamics with NO per-cell cap, so it is
# density-faithful and scales to N = 1e7 on a 24 GB GPU. See
# amber_gpu_scale_models.GPUSIRKernelModel.
from amber_gpu_scale_models import GPUSIRKernelModel

SEED = 42


class _Base:
    def __init__(self, n, steps, cfg):
        self.n, self.steps, self.cfg = n, steps, cfg


class GPUWealthModel(_Base):
    def run(self):
        rng = cp.random.default_rng(SEED)
        n = self.n
        wealth = cp.full(n, int(self.cfg.get("initial_wealth", 1)), dtype=cp.int64)
        for _ in range(self.steps):
            donor = wealth > 0
            recipients = rng.integers(0, n, size=n)
            wealth = wealth + cp.where(donor, -1, 0).astype(cp.int64)   # donors give 1
            scatter_add(wealth, recipients, donor.astype(cp.int64))     # recipients receive
        cp.cuda.Stream.null.synchronize()


class GPUWalkModel(_Base):
    def run(self):
        rng = cp.random.default_rng(SEED)
        n = self.n
        ws = float(self.cfg.get("world_size", 100))
        speed = float(self.cfg.get("speed", 1.0))
        x = rng.random(n) * ws
        y = rng.random(n) * ws
        for _ in range(self.steps):
            x = cp.clip(x + rng.uniform(-speed, speed, n), 0.0, ws)
            y = cp.clip(y + rng.uniform(-speed, speed, n), 0.0, ws)
        cp.cuda.Stream.null.synchronize()


class GPUSIRModel(_Base):
    def run(self):
        rng = cp.random.default_rng(SEED)
        n = self.n
        ws = float(self.cfg.get("world_size", 100))
        speed = float(self.cfg.get("movement_speed", 2.0))
        r2 = float(self.cfg.get("infection_radius", 5.0)) ** 2
        trans = float(self.cfg.get("transmission_rate", 0.1))
        recovery = int(self.cfg.get("recovery_time", 14))
        ii = int(self.cfg.get("initial_infected", 5))

        status = cp.zeros(n, dtype=cp.int64)
        status[:ii] = 1
        inf_time = cp.zeros(n, dtype=cp.int64)
        x = rng.random(n) * ws
        y = rng.random(n) * ws

        for _ in range(self.steps):
            # movement
            x = cp.clip(x + rng.uniform(-speed, speed, n), 0.0, ws)
            y = cp.clip(y + rng.uniform(-speed, speed, n), 0.0, ws)

            # infection: all-pairs susceptible x infected within radius
            inf = status == 1
            sus = status == 0
            xi, yi = x[inf], y[inf]
            xs, ys = x[sus], y[sus]
            if xi.size and xs.size:
                dx = xs[:, None] - xi[None, :]
                dy = ys[:, None] - yi[None, :]
                within = (dx * dx + dy * dy) <= r2
                draws = rng.random((xs.size, xi.size)) < trans
                hit = (within & draws).any(axis=1)
                newly = cp.nonzero(sus)[0][hit]
                status[newly] = 1
                inf_time[newly] = 0

            # recovery
            inf_now = status == 1
            inf_time = cp.where(inf_now, inf_time + 1, inf_time)
            status = cp.where(inf_now & (inf_time >= recovery), 2, status)
        cp.cuda.Stream.null.synchronize()


class GPUSchellingModel(_Base):
    """Schelling segregation on the GPU via the shared grid-vectorized core."""

    def run(self):
        rng = cp.random.default_rng(SEED)
        x, y, t, G = schelling_setup(
            self.n, self.cfg.get("density", 0.8), self.cfg.get("fraction_a", 0.5), rng, cp)
        tol = float(self.cfg.get("tolerance", 0.3))
        for _ in range(self.steps):
            x, y = schelling_step(x, y, t, G, tol, rng, cp)
        cp.cuda.Stream.null.synchronize()


AMBER_GPU_MODELS = {
    "wealth_transfer": GPUWealthModel,
    "random_walk": GPUWalkModel,
    # sir_epidemic: use the O(N) spatial-binning kernel (GPUSIRModel below is
    # the naive O(N^2) all-pairs version and OOMs at N >= 1e5).
    "sir_epidemic": GPUSIRKernelModel,
    "sir_epidemic_naive": GPUSIRModel,
    "schelling": GPUSchellingModel,
}
