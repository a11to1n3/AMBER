"""Boids flocking on the Polars-backed tensor lane, verified by the contract.

This is an end-to-end demonstration of two AMBER ideas working together:

* the **tensor lane** (``ambr.tensor_lane``): the agent state lives in Polars,
  but the O(N^2) interaction kernel reads it as a zero-copy NumPy view and runs
  as dense tensor contractions -- the regime where columnar relational joins are
  slowest and tensors are fastest.
* the **snapshot-view contract** (``Model.run(contract=...)``): the step borrows
  every input read-only and commits each output column exactly once at the end,
  so the per-step certificate comes back clean. The certificate verifies, for
  this lane-based step, (a) column-name and dtype stability, (b) population
  (id-set) stability, (c) no column committed more than once per step, and
  (d) no read-after-write (no column borrowed after it was committed this step).
  It does not prove the *physics* correct -- only that the columnar execution
  did not violate simultaneous-update semantics.

The Boids rules (Reynolds 1987) are cohesion + alignment + separation on a
toroidal world. The interaction is computed two equivalent ways -- a vectorized
tensor form (used by the model) and an explicit double loop (the reference) --
so the tensor formulation can be checked for correctness.

Run directly for a demo + micro-benchmark::

    python examples/flocking_tensor.py
"""

import numpy as np

import ambr as am
from ambr.tensor_lane import borrow_numeric, commit_columns


# --- the interaction kernel, two equivalent ways ---------------------------

def boids_forces_vectorized(x, y, vx, vy, L, r, w_c, w_a, w_s):
    """Steering accelerations for every agent as dense tensor contractions.

    Returns ``(ax, ay)``. All inputs are 1-D float arrays of length N; ``L`` is
    the (square) world size, ``r`` the separation radius, and ``w_*`` the rule
    weights. Distances use the minimum-image convention on a torus.
    """
    N = len(x)
    if N < 2:
        return np.zeros(N), np.zeros(N)

    # Minimum-image pairwise displacement: d_ij points from i to j.
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    dx = dx - L * np.round(dx / L)
    dy = dy - L * np.round(dy / L)

    dist2 = dx * dx + dy * dy
    eye = np.eye(N, dtype=bool)
    dist2_ns = np.where(eye, np.inf, dist2)  # drop self-pairs

    # Cohesion: steer toward the centroid of the others (mean displacement).
    # The diagonal (self) term is exactly 0 under the min-image convention, so
    # summing over all j equals summing over j != i.
    cohes_x = dx.sum(axis=1) / (N - 1)
    cohes_y = dy.sum(axis=1) / (N - 1)

    # Alignment: steer toward the mean velocity of the others.
    sum_vx, sum_vy = vx.sum(), vy.sum()
    align_x = (sum_vx - vx) / (N - 1) - vx
    align_y = (sum_vy - vy) / (N - 1) - vy

    # Separation: repel from neighbours within radius r (inverse-square).
    # Exclude both the self-pair (diagonal) *and* coincident distinct agents
    # (dist2 == 0): the reciprocal must never be evaluated where dist2 == 0, or
    # 0 * inf = NaN would poison every agent via the row-sum. This matches the
    # loop reference's ``0.0 < d2 < r*r`` guard.
    near = (dist2_ns < (r * r)) & (dist2_ns > 0.0)
    inv = np.where(near, 1.0 / np.where(near, dist2_ns, 1.0), 0.0)
    sep_x = -(dx * inv).sum(axis=1)
    sep_y = -(dy * inv).sum(axis=1)

    ax = w_c * cohes_x + w_a * align_x + w_s * sep_x
    ay = w_c * cohes_y + w_a * align_y + w_s * sep_y
    return ax, ay


def boids_forces_loop(x, y, vx, vy, L, r, w_c, w_a, w_s):
    """Reference implementation of :func:`boids_forces_vectorized` (slow, clear)."""
    N = len(x)
    ax = np.zeros(N)
    ay = np.zeros(N)
    if N < 2:
        return ax, ay
    sum_vx, sum_vy = vx.sum(), vy.sum()
    for i in range(N):
        cx = cy = spx = spy = 0.0
        for j in range(N):
            if i == j:
                continue
            ddx = x[j] - x[i]
            ddy = y[j] - y[i]
            ddx -= L * round(ddx / L)
            ddy -= L * round(ddy / L)
            cx += ddx
            cy += ddy
            d2 = ddx * ddx + ddy * ddy
            if 0.0 < d2 < r * r:
                spx -= ddx / d2
                spy -= ddy / d2
        cohes_x = cx / (N - 1)
        cohes_y = cy / (N - 1)
        align_x = (sum_vx - vx[i]) / (N - 1) - vx[i]
        align_y = (sum_vy - vy[i]) / (N - 1) - vy[i]
        ax[i] = w_c * cohes_x + w_a * align_x + w_s * spx
        ay[i] = w_c * cohes_y + w_a * align_y + w_s * spy
    return ax, ay


# --- the model -------------------------------------------------------------

class FlockingModel(am.Model):
    """Boids flocking whose interaction step runs on the tensor lane."""

    # Typed parameter schema: self.p.<name> is pre-coerced at init (no
    # int()/float() boilerplate), and missing values fall back to the default.
    params = {
        "n_agents": (int, 200),
        "width": (float, 100.0),
        "radius": (float, 8.0),
        "w_cohesion": (float, 0.02),
        "w_align": (float, 0.05),
        "w_separation": (float, 1.0),
        "vmax": (float, 2.0),
    }

    def setup(self):
        N = self.p.n_agents
        self.L = self.p.width
        self.r = self.p.radius
        self.w_c = self.p.w_cohesion
        self.w_a = self.p.w_align
        self.w_s = self.p.w_separation
        self.vmax = self.p.vmax

        rng = self.rng
        x = rng.random(N) * self.L
        y = rng.random(N) * self.L
        ang = rng.random(N) * 2.0 * np.pi
        # Pre-declare every column the step touches so the schema is stable.
        self.add_agents(
            N,
            x=x.astype(np.float64),
            y=y.astype(np.float64),
            vx=np.cos(ang),
            vy=np.sin(ang),
        )

    def step(self):
        # Zero-copy borrows of the Polars buffers (read-only views).
        x, _ = borrow_numeric(self, "x")
        y, _ = borrow_numeric(self, "y")
        vx, _ = borrow_numeric(self, "vx")
        vy, _ = borrow_numeric(self, "vy")

        ax, ay = boids_forces_vectorized(
            x, y, vx, vy, self.L, self.r, self.w_c, self.w_a, self.w_s
        )

        # Build new arrays (never mutate the read-only borrowed views).
        nvx = vx + ax
        nvy = vy + ay
        speed = np.sqrt(nvx * nvx + nvy * nvy)
        scale = np.where(speed > self.vmax, self.vmax / np.maximum(speed, 1e-12), 1.0)
        nvx = nvx * scale
        nvy = nvy * scale
        nx = (x + nvx) % self.L
        ny = (y + nvy) % self.L

        # One atomic write-back of the four existing columns -> schema stable.
        commit_columns(self, x=nx, y=ny, vx=nvx, vy=nvy)


if __name__ == "__main__":
    import time

    # --- demo: run under the contract checker --------------------------------
    m = FlockingModel({"n_agents": 300, "steps": 50, "seed": 1, "show_progress": False})
    res = m.run(contract="check")
    certs = res["contract"]
    clean = sum(c.clean for c in certs)
    print(f"flocking: {len(certs)} steps, {clean} clean certificates, "
          f"all ok={all(c.ok for c in certs)}")
    print(f"final positions sample:\n{res['agents'].select('id', 'x', 'y').head(3)}")

    # --- micro-benchmark: tensor-lane step vs the loop reference -------------
    for N in (200, 1000):
        mm = FlockingModel({"n_agents": N, "seed": 2, "show_progress": False})
        mm.setup()
        x, _ = borrow_numeric(mm, "x"); y, _ = borrow_numeric(mm, "y")
        vx, _ = borrow_numeric(mm, "vx"); vy, _ = borrow_numeric(mm, "vy")
        args = (x, y, vx, vy, mm.L, mm.r, mm.w_c, mm.w_a, mm.w_s)

        t0 = time.perf_counter()
        for _ in range(10):
            boids_forces_vectorized(*args)
        t_vec = (time.perf_counter() - t0) / 10

        reps = 3 if N <= 1000 else 1
        t0 = time.perf_counter()
        for _ in range(reps):
            boids_forces_loop(*args)
        t_loop = (time.perf_counter() - t0) / reps

        print(f"N={N:>5}: tensor {t_vec*1e3:7.2f} ms | loop {t_loop*1e3:8.2f} ms "
              f"| {t_loop/t_vec:6.0f}x")
