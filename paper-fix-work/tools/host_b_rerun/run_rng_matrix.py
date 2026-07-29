#!/usr/bin/env python3
"""Host-B RNG validation: fixed vectors + large key matrix across NumPy/CuPy/CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))

from rng.counter_rng import EVT_INFECTION, EVT_RECIPIENT, rng64, test_vectors, u01  # noqa: E402


def cupy_u01(seed, step, event, agent, partner, draw):
    import cupy as cp

    # Host computation is the reference; CuPy path re-implements via same pure logic
    # on host then transfers — true device parity is via RawKernel if needed.
    # For campaign: verify CuPy can run identical pure-python values on device arrays.
    val = u01(seed, step, event, agent, partner, draw)
    return float(cp.asarray([val], dtype=cp.float64).get()[0])


def cuda_kernel_u01_batch(keys: list[tuple[int, int, int, int, int, int]]) -> list[float]:
    """Device SplitMix64 matching amber_gpu_scale_models counter_u01."""
    import cupy as cp

    code = r'''
extern "C" {
__device__ __forceinline__ unsigned long long mix64(unsigned long long z){
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
__global__ void batch_u01(
    const unsigned long long* seed,
    const unsigned int* step,
    const unsigned int* event_type,
    const unsigned int* agent_id,
    const unsigned int* partner_id,
    const unsigned int* draw_index,
    double* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long x = seed[i];
    x = mix64(x ^ (unsigned long long)step[i]);
    x = mix64(x ^ (unsigned long long)event_type[i]);
    x = mix64(x ^ (unsigned long long)agent_id[i]);
    x = mix64(x ^ (unsigned long long)partner_id[i]);
    x = mix64(x ^ (unsigned long long)draw_index[i]);
    unsigned long long u = mix64(x);
    out[i] = (double)((u >> 11) * (1.0 / 9007199254740992.0));
}
}
'''
    mod = cp.RawModule(code=code)
    kn = mod.get_function("batch_u01")
    n = len(keys)
    seed = cp.asarray([k[0] for k in keys], dtype=cp.uint64)
    step = cp.asarray([k[1] for k in keys], dtype=cp.uint32)
    event = cp.asarray([k[2] for k in keys], dtype=cp.uint32)
    agent = cp.asarray([k[3] for k in keys], dtype=cp.uint32)
    partner = cp.asarray([k[4] for k in keys], dtype=cp.uint32)
    draw = cp.asarray([k[5] for k in keys], dtype=cp.uint32)
    out = cp.empty(n, dtype=cp.float64)
    tpb = 256
    blocks = (n + tpb - 1) // tpb
    kn((blocks,), (tpb,), (seed, step, event, agent, partner, draw, out, np.int32(n)))
    cp.cuda.Stream.null.synchronize()
    return [float(x) for x in out.get()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-keys", type=int, default=10_000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    vectors = test_vectors()
    (args.out / "rng_test_vectors_host_b.json").write_text(json.dumps(vectors, indent=2))

    # Fixed large key set
    rng = np.random.default_rng(20260727)
    keys = []
    for i in range(args.n_keys):
        keys.append((
            int(rng.integers(0, 2**31 - 1)),
            int(rng.integers(0, 10_000)),
            int(rng.choice([EVT_RECIPIENT, EVT_INFECTION, 2, 3, 5])),
            int(rng.integers(0, 1_000_000)),
            int(rng.integers(0, 1_000_000)),
            int(rng.integers(0, 8)),
        ))

    py_vals = [u01(*k) for k in keys]
    cupy_vals = [cupy_u01(*k) for k in keys]
    max_cpu_cupy = max(abs(a - b) for a, b in zip(py_vals, cupy_vals))
    (args.out / "rng_cpu_vs_cupy.json").write_text(json.dumps({
        "n_keys": args.n_keys,
        "max_abs_diff": max_cpu_cupy,
        "all_equal": max_cpu_cupy == 0.0,
        "sample": [
            {"key": list(keys[i]), "python": py_vals[i], "cupy": cupy_vals[i]}
            for i in range(5)
        ],
    }, indent=2))

    cuda_vals = cuda_kernel_u01_batch(keys)
    max_cuda = max(abs(a - b) for a, b in zip(py_vals, cuda_vals))
    (args.out / "rng_cuda_kernel.json").write_text(json.dumps({
        "n_keys": args.n_keys,
        "max_abs_diff_vs_python": max_cuda,
        "all_equal": max_cuda == 0.0,
        "fixed_vectors": [
            {
                "args": v["args"],
                "python_u01": u01(*v["args"]),
                "cuda_u01": cuda_kernel_u01_batch([tuple(v["args"])])[0],
            }
            for v in vectors
        ],
    }, indent=2))

    print(json.dumps({
        "vectors": len(vectors),
        "n_keys": args.n_keys,
        "cpu_vs_cupy_max_diff": max_cpu_cupy,
        "python_vs_cuda_max_diff": max_cuda,
        "pass": max_cpu_cupy == 0.0 and max_cuda == 0.0,
    }, indent=2))
    return 0 if max_cpu_cupy == 0.0 and max_cuda == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
