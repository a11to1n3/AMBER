"""AMBER GPU spatial-binning models — scaling spatial interaction to millions.

The all-pairs neighbour test in ``amber_gpu_models`` is O(N^2) and OOMs around
N=1e5. This module replaces it with a **uniform-grid cell list**:

* Each agent is binned into a cell of side >= interaction radius, so every
  neighbour within the radius lives in the 3x3 block of surrounding cells.
* The capped-table prototype comparison-sorts agents by cell, then uses
  bincount/prefix-sum and scatters into a (num_cells, K) table. Its build is
  O(N log N), its query is O(N * 9K), and it may omit neighbours when
  occupancy exceeds K.
* The exact private kernel uses counting-sort bins and scans all candidate
  pairs. Its cost is O(N + C + P), where C is the number of cells and P the
  candidate pairs examined; P can be quadratic when density grows in a fixed
  domain. Its ten-million-agent result is therefore empirical, not an O(N)
  worst-case guarantee.

Positions are float32, status int8 -> state is ~13 bytes/agent, so tens of
millions of agents fit in 24 GB; the query chunk bounds the transient memory.
This is the GPU spatial backend AMBER needs to match FLAME GPU's spatial
messaging on interaction-heavy models. (Prototype here; would be promoted to
``ambr.gpu_spatial``.)
"""

import numpy as np

from ambr.gpu import GPU_AVAILABLE

if GPU_AVAILABLE:
    import cupy as cp

SEED = 42


def _build_cell_table(cx, cy, ncell, K):
    """Scatter agent indices into a (ncell*ncell, K) cell-occupancy table."""
    n = cx.size
    cell = cx.astype(cp.int64) * ncell + cy
    order = cp.argsort(cell)
    sorted_cell = cell[order]
    counts = cp.bincount(cell, minlength=ncell * ncell)
    cell_start = cp.cumsum(counts) - counts
    slot = cp.arange(n) - cell_start[sorted_cell]
    keep = slot < K
    table = cp.full((ncell * ncell, K), -1, dtype=cp.int64)
    table[sorted_cell[keep], slot[keep].astype(cp.int64)] = order[keep]
    return table


class GPUSIRBinnedModel:
    """Capped SIR cell list: O(N log N + NK) per step for fixed K."""

    def __init__(self, n, steps, cfg):
        self.n, self.steps, self.cfg = n, steps, cfg

    def run(self, return_state=False):
        rng = cp.random.default_rng(SEED)
        n = self.n
        ws = float(self.cfg["world_size"])
        speed = float(self.cfg.get("movement_speed", 2.0))
        r = float(self.cfg.get("infection_radius", 5.0))
        r2 = r * r
        trans = float(self.cfg.get("transmission_rate", 0.1))
        recovery = int(self.cfg.get("recovery_time", 14))
        ii = int(self.cfg.get("initial_infected", 5))
        K = int(self.cfg.get("max_per_cell", 24))
        chunk = int(self.cfg.get("chunk", 500_000))

        ncell = max(1, int(ws // r))     # cell side = ws/ncell >= r
        cs = ws / ncell

        status = cp.zeros(n, dtype=cp.int8)
        status[:ii] = 1
        inf_time = cp.zeros(n, dtype=cp.int32)
        x = (rng.random(n, dtype=cp.float32) * ws)
        y = (rng.random(n, dtype=cp.float32) * ws)

        for _ in range(self.steps):
            # movement
            x = cp.clip(x + rng.uniform(-speed, speed, n, dtype=cp.float32), 0.0, ws)
            y = cp.clip(y + rng.uniform(-speed, speed, n, dtype=cp.float32), 0.0, ws)

            cx = cp.minimum((x / cs).astype(cp.int32), ncell - 1)
            cy = cp.minimum((y / cs).astype(cp.int32), ncell - 1)
            table = _build_cell_table(cx, cy, ncell, K)

            # infection — chunked cell-list neighbour query
            newly = []
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                m = e - s
                cxi, cyi = cx[s:e], cy[s:e]
                xi, yi = x[s:e], y[s:e]
                cand = cp.empty((m, 9 * K), dtype=cp.int64)
                c = 0
                for dxc in (-1, 0, 1):
                    for dyc in (-1, 0, 1):
                        ncx = cp.clip(cxi + dxc, 0, ncell - 1)
                        ncy = cp.clip(cyi + dyc, 0, ncell - 1)
                        cand[:, c * K:(c + 1) * K] = table[ncx.astype(cp.int64) * ncell + ncy]
                        c += 1
                valid = cand >= 0
                cc = cp.where(valid, cand, 0)
                within = ((x[cc] - xi[:, None]) ** 2 + (y[cc] - yi[:, None]) ** 2) <= r2
                is_inf = (status[cc] == 1) & valid & within
                draws = rng.random((m, 9 * K), dtype=cp.float32) < trans
                hit = (is_inf & draws).any(axis=1) & (status[s:e] == 0)
                newly.append(cp.nonzero(hit)[0] + s)
            newly = cp.concatenate(newly)
            status[newly] = 1
            inf_time[newly] = 0

            # recovery
            inf_now = status == 1
            inf_time = cp.where(inf_now, inf_time + 1, inf_time)
            status = cp.where(inf_now & (inf_time >= recovery), cp.int8(2), status)

        cp.cuda.Stream.null.synchronize()
        if return_state:
            return {
                "S": int((status == 0).sum()),
                "I": int((status == 1).sum()),
                "R": int((status == 2).sum()),
            }


# --------------------------------------------------------------------------- #
# Columnar fixed-radius self-join on the GPU.
#
# Architecturally this is *not* FLAME GPU's message-passing model. The spatial
# interaction is expressed the way AMBER expresses everything else -- as a
# columnar / relational operation:
#
#   1. GROUP-BY the agent table by cell, built in O(N) with a counting sort:
#      bincount(cell) -> per-cell counts; exclusive prefix sum -> `cell_start`
#      (a group-boundary *column*, CSR offsets); a single atomic-scatter pass
#      (`bucketize`) places agent indices into their group -> `order`
#      (a permutation *column*). No comparison sort, no multi-pass radix.
#   2. SEGMENTED SELF-JOIN: each susceptible agent joins against the agents in
#      its 3x3 neighbour cell-groups (segments of `order` delimited by
#      `cell_start`), filtered by radius -> a fixed-radius spatial join.
#
# The whole index is two columns (`cell_start`, `order`); the kernels are the
# *execution* of columnar primitives (group-by, segmented join) over them, the
# same way Polars compiles group-by/join to kernels over Arrow columns. It is
# snapshot-structured: it reads step-entry columns and writes `new_inf`
# separately (race-free), but this private path is not runtime-monitored.
# Per-pair infection draws use the shared SplitMix64 counter tape
# (global_seed, step, EVT_INFECTION=4, min(i,j), max(i,j), draw_index=0),
# matching experiments/rng/counter_rng.py for cross-backend attestation.
# --------------------------------------------------------------------------- #

_MODULE_VERSION = 2  # bump when _MODULE_SRC changes (forces RawModule reload)

_MODULE_SRC = r'''
extern "C" {
// SplitMix64 — must match experiments/rng/counter_rng.py bit-for-bit.
__device__ __forceinline__ unsigned long long mix64(unsigned long long z){
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
__device__ __forceinline__ float counter_u01(
    unsigned long long global_seed,
    unsigned int step,
    unsigned int event_type,
    unsigned int agent_id,
    unsigned int partner_id,
    unsigned int draw_index)
{
    unsigned long long x = global_seed;
    x = mix64(x ^ (unsigned long long)step);
    x = mix64(x ^ (unsigned long long)event_type);
    x = mix64(x ^ (unsigned long long)agent_id);
    x = mix64(x ^ (unsigned long long)partner_id);
    x = mix64(x ^ (unsigned long long)draw_index);
    unsigned long long u = mix64(x);
    // top 53 mantissa bits -> U(0,1), same as Python (u >> 11) / 2^53
    return (float)((u >> 11) * (1.0 / 9007199254740992.0));
}
// Legacy 32-bit hashrand retained only for non-attested diagnostics if needed.
__device__ __forceinline__ float hashrand(unsigned int a, unsigned int b, unsigned int c){
    unsigned int h = a * 747796405u + 2891336453u;
    h ^= (b + 0x9e3779b9u + (h << 6) + (h >> 2));
    h ^= (c + 0x9e3779b9u + (h << 6) + (h >> 2));
    h = (h ^ (h >> 15)) * 0x2c1b3c6du;
    h = (h ^ (h >> 12)) * 0x297a2d39u;
    h = h ^ (h >> 15);
    return (h & 0x00ffffffu) * (1.0f / 16777216.0f);
}

// Pass 1 of the counting-sort group-by, fused: compute each agent's cell id
// AND histogram it in one pass (a custom atomic histogram -- ~5x faster than
// cupy.bincount, which dominated the step otherwise).
__global__ void cell_and_count(
    const float* __restrict__ x, const float* __restrict__ y,
    float cs, int ncell, long long* __restrict__ cell,
    unsigned long long* counts, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cx = (int)(x[i] / cs); cx = max(0, min(cx, ncell - 1));
    int cy = (int)(y[i] / cs); cy = max(0, min(cy, ncell - 1));
    long long c = (long long)cx * ncell + cy;
    cell[i] = c;
    atomicAdd(&counts[c], 1ULL);
}

// Pass 2, fused: scatter agent indices into per-cell groups (counting sort)
// AND physically reorder the state columns into cell order in the same pass,
// so the join reads coalesce. `cursor` is pre-seeded to the cell start offsets.
__global__ void bucketize_reorder(
    const long long* __restrict__ cell, unsigned long long* cursor,
    const float* __restrict__ x, const float* __restrict__ y,
    const signed char* __restrict__ status,
    long long* __restrict__ order, float* __restrict__ xs,
    float* __restrict__ ys, signed char* __restrict__ ss, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long pos = atomicAdd(&cursor[cell[i]], 1ULL);
    order[pos] = (long long)i;
    xs[pos] = x[i];
    ys[pos] = y[i];
    ss[pos] = status[i];
}

// Segmented fixed-radius self-join over CELL-SORTED columns. The position
// columns are physically permuted into cell order (xs/ys/ss), so an agent's
// neighbour cell-groups are *contiguous* ranges -> the inner-loop reads are
// memory-coalesced (the dominant cost otherwise). `order` carries the original
// agent id. Infection Bernoulli draws are keyed by the unordered pair
// (min(i,j), max(i,j)) so visit order cannot change the assigned RV.
// Reads old `ss`, writes new_inf_sorted (race-free snapshot).
// EVT_INFECTION = 4 (must match experiments/rng/counter_rng.py).
__global__ void sir_join_sorted(
    const float* __restrict__ xs, const float* __restrict__ ys,
    const signed char* __restrict__ ss, const long long* __restrict__ order,
    const long long* __restrict__ cell_start,
    int ncell, float cs, float r2, float trans,
    unsigned int step, unsigned long long global_seed,
    int n, signed char* __restrict__ new_inf_sorted)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    if (ss[t] != 0) return;                      // susceptible only
    float xi = xs[t], yi = ys[t];
    int cx = (int)(xi / cs); cx = max(0, min(cx, ncell - 1));
    int cy = (int)(yi / cs); cy = max(0, min(cy, ncell - 1));
    unsigned int id = (unsigned int)order[t];    // original agent id
    for (int dx = -1; dx <= 1; ++dx){
        int ncx = cx + dx; if (ncx < 0 || ncx >= ncell) continue;
        for (int dy = -1; dy <= 1; ++dy){
            int ncy = cy + dy; if (ncy < 0 || ncy >= ncell) continue;
            long long c = (long long)ncx * ncell + ncy;
            long long s = cell_start[c], e = cell_start[c + 1];
            for (long long k = s; k < e; ++k){    // contiguous -> coalesced
                if (ss[k] != 1) continue;         // infected sources only
                float ddx = xs[k] - xi, ddy = ys[k] - yi;
                if (ddx * ddx + ddy * ddy <= r2){
                    unsigned int jid = (unsigned int)order[k];
                    unsigned int lo = id < jid ? id : jid;
                    unsigned int hi = id < jid ? jid : id;
                    // EVT_INFECTION = 4
                    if (counter_u01(global_seed, step, 4u, lo, hi, 0u) < trans){
                        new_inf_sorted[t] = 1;
                        return;
                    }
                }
            }
        }
    }
}
}
'''


def _load_sir_module():
    """Compile/load the SIR CUDA module; reload when source version changes."""
    import cupy as cp

    if (
        GPUSIRKernelModel._module is None
        or GPUSIRKernelModel._module_version != _MODULE_VERSION
    ):
        GPUSIRKernelModel._module = cp.RawModule(code=_MODULE_SRC)
        GPUSIRKernelModel._module_version = _MODULE_VERSION
    return GPUSIRKernelModel._module


class GPUSIRKernelModel:
    """SIR as a columnar fixed-radius self-join (counting-sort group-by + kernel)."""

    _module = None
    _module_version = None

    def __init__(self, n, steps, cfg):
        self.n, self.steps, self.cfg = n, steps, cfg

    def run(self, return_state=False):
        mod = _load_sir_module()
        cell_and_count = mod.get_function("cell_and_count")
        bucketize_reorder = mod.get_function("bucketize_reorder")
        sir_join_sorted = mod.get_function("sir_join_sorted")

        rng = cp.random.default_rng(SEED)
        n = self.n
        ws = float(self.cfg["world_size"])
        speed = float(self.cfg.get("movement_speed", 2.0))
        r = float(self.cfg.get("infection_radius", 5.0))
        r2 = r * r
        trans = float(self.cfg.get("transmission_rate", 0.1))
        recovery = int(self.cfg.get("recovery_time", 14))
        ii = int(self.cfg.get("initial_infected", 5))

        ncell = max(1, int(ws // r))
        cs = ws / ncell
        ncell2 = ncell * ncell

        status = cp.zeros(n, dtype=cp.int8)
        status[:ii] = 1
        inf_time = cp.zeros(n, dtype=cp.int32)
        x = rng.random(n, dtype=cp.float32) * ws
        y = rng.random(n, dtype=cp.float32) * ws

        tpb = 256
        blocks = (n + tpb - 1) // tpb
        for step in range(self.steps):
            x = cp.clip(x + rng.uniform(-speed, speed, n, dtype=cp.float32), 0.0, ws)
            y = cp.clip(y + rng.uniform(-speed, speed, n, dtype=cp.float32), 0.0, ws)

            # --- columnar group-by, pass 1: cell id + histogram (fused) ---
            cell = cp.empty(n, dtype=cp.int64)
            counts = cp.zeros(ncell2, dtype=cp.int64)
            cell_and_count((blocks,), (tpb,), (
                x, y, np.float32(cs), np.int32(ncell), cell, counts, np.int32(n)))
            cell_start = cp.zeros(ncell2 + 1, dtype=cp.int64)
            cell_start[1:] = cp.cumsum(counts)
            cursor = cell_start[:-1].copy()

            # --- group-by pass 2: bucketize + reorder state columns (fused) ---
            order = cp.empty(n, dtype=cp.int64)
            xs = cp.empty(n, dtype=cp.float32)
            ys = cp.empty(n, dtype=cp.float32)
            ss = cp.empty(n, dtype=cp.int8)
            bucketize_reorder((blocks,), (tpb,), (
                cell, cursor, x, y, status, order, xs, ys, ss, np.int32(n)))

            # --- segmented fixed-radius self-join (infection), in sorted space ---
            new_inf_sorted = cp.zeros(n, dtype=cp.int8)
            sir_join_sorted((blocks,), (tpb,), (
                xs, ys, ss, order, cell_start,
                np.int32(ncell), np.float32(cs), np.float32(r2), np.float32(trans),
                np.uint32(step), np.uint64(SEED), np.int32(n), new_inf_sorted,
            ))
            # scatter results back to original agent order
            new_inf = cp.zeros(n, dtype=cp.int8)
            new_inf[order] = new_inf_sorted
            newly = new_inf.astype(cp.bool_)
            status[newly] = 1
            inf_time[newly] = 0

            inf_now = status == 1
            inf_time = cp.where(inf_now, inf_time + 1, inf_time)
            status = cp.where(inf_now & (inf_time >= recovery), cp.int8(2), status)

        cp.cuda.Stream.null.synchronize()
        if return_state:
            return {
                "S": int((status == 0).sum()),
                "I": int((status == 1).sum()),
                "R": int((status == 2).sum()),
            }


def sir_kernel_step(
    x,
    y,
    status,
    infection_time,
    *,
    step: int,
    world_size: float,
    radius: float,
    transmission: float,
    recovery_time: int,
    global_seed: int = 0,
):
    """Run one exact cell-binned SIR spatial join on caller-owned columns.

    Infection Bernoulli draws use the shared SplitMix64 counter tape keyed by
    ``(global_seed, step, EVT_INFECTION=4, min(i,j), max(i,j), 0)`` so that
    candidate visit order cannot change the random values assigned to pairs.

    Counting/bucketization costs O(N + C); exact neighbour traversal adds P,
    the number of candidate pairs examined. Thus total work is O(N + C + P),
    with a quadratic worst case under unbounded density in a fixed domain.

    This is an internal backend hook.  The public model still enters through
    ``model.gpu().run()``; this function only replaces the generic CuPy cell
    list when the GPU lane is active.  The caller's arrays are returned so
    AMBER's normal device-column lifecycle and final Polars result remain
    unchanged.
    """
    import cupy as cp

    mod = _load_sir_module()
    cell_and_count = mod.get_function("cell_and_count")
    bucketize_reorder = mod.get_function("bucketize_reorder")
    sir_join_sorted = mod.get_function("sir_join_sorted")

    # The CUDA join is intentionally typed for compact GPU state.  Conversion
    # happens once when a model first enters this private fast path; subsequent
    # steps stay in these device dtypes.
    x = cp.asarray(x, dtype=cp.float32)
    y = cp.asarray(y, dtype=cp.float32)
    status = cp.asarray(status, dtype=cp.int8)
    infection_time = cp.asarray(infection_time, dtype=cp.int32)
    n = int(x.size)

    ncell = max(1, int(float(world_size) // float(radius)))
    cs = float(world_size) / ncell
    ncell2 = ncell * ncell
    tpb = 256
    blocks = (n + tpb - 1) // tpb

    cell = cp.empty(n, dtype=cp.int64)
    counts = cp.zeros(ncell2, dtype=cp.int64)
    cell_and_count(
        (blocks,), (tpb,),
        (x, y, np.float32(cs), np.int32(ncell), cell, counts, np.int32(n)),
    )
    cell_start = cp.zeros(ncell2 + 1, dtype=cp.int64)
    cell_start[1:] = cp.cumsum(counts)
    cursor = cell_start[:-1].copy()

    order = cp.empty(n, dtype=cp.int64)
    xs = cp.empty(n, dtype=cp.float32)
    ys = cp.empty(n, dtype=cp.float32)
    ss = cp.empty(n, dtype=cp.int8)
    bucketize_reorder(
        (blocks,), (tpb,),
        (cell, cursor, x, y, status, order, xs, ys, ss, np.int32(n)),
    )

    new_inf_sorted = cp.zeros(n, dtype=cp.int8)
    sir_join_sorted(
        (blocks,), (tpb,),
        (
            xs, ys, ss, order, cell_start,
            np.int32(ncell), np.float32(cs), np.float32(float(radius) ** 2),
            np.float32(transmission), np.uint32(int(step)),
            np.uint64(int(global_seed) & ((1 << 64) - 1)),
            np.int32(n),
            new_inf_sorted,
        ),
    )
    new_inf = cp.zeros(n, dtype=cp.int8)
    new_inf[order] = new_inf_sorted
    newly = new_inf.astype(cp.bool_)
    status[newly] = 1
    infection_time[newly] = 0

    infected = status == 1
    infection_time = cp.where(infected, infection_time + 1, infection_time)
    status = cp.where(
        infected & (infection_time >= int(recovery_time)),
        cp.int8(2),
        status,
    )
    return x, y, status, infection_time


AMBER_GPU_SCALE_MODELS = {
    "sir_epidemic": GPUSIRBinnedModel,
    "sir_epidemic_kernel": GPUSIRKernelModel,
}
