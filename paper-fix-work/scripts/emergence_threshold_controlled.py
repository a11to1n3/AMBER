#!/usr/bin/env python3
"""Matched-randomness SIR threshold experiment used by the paper.

The row-wise snapshot and in-place ordered states are advanced side by side.
For each (tau, seed), both receive the same initial state, per-step activation
orders, infection draws, and recovery draws.  The headline schedule reshuffles
the shared order at every step; a fixed-order condition is retained as a
robustness check.  This isolates activation semantics from random-stream
consumption.  A separate batched CuPy implementation checks the snapshot result
with an independent device RNG.
"""
from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numba import njit


N = 4_000
STEPS = 120
GAMMA = 0.10
RADIUS = 3
I0 = 10
TAUS = np.round(np.linspace(0.02, 0.40, 16), 3)
SEEDS_CPU = 48
SEEDS_GPU = 96
N_BIG = 100_000
SEEDS_BIG = 24
SHIFTS = np.array([d for d in range(-RADIUS, RADIUS + 1) if d], dtype=np.int64)


@njit(cache=True)
def _simulate_matched_pair(
    tau: float,
    seed: int,
    reshuffle_each_step: bool,
) -> tuple[float, float]:
    np.random.seed(seed)
    initial_order = np.random.permutation(N)
    activation_order = np.random.permutation(N)

    snapshot = np.zeros(N, dtype=np.int8)
    snapshot[initial_order[:I0]] = 1
    ordered = snapshot.copy()

    for _ in range(STEPS):
        if reshuffle_each_step:
            activation_order = np.random.permutation(N)
        infection_draw = np.random.random(N)
        recovery_draw = np.random.random(N)

        entry = snapshot
        next_snapshot = entry.copy()
        for p in range(N):
            i = activation_order[p]
            if entry[i] == 0:
                infected_neighbors = 0
                for d in SHIFTS:
                    if entry[(i + d) % N] == 1:
                        infected_neighbors += 1
                if infected_neighbors:
                    prob = 1.0 - (1.0 - tau) ** infected_neighbors
                    if infection_draw[i] < prob:
                        next_snapshot[i] = 1
            elif entry[i] == 1 and recovery_draw[i] < GAMMA:
                next_snapshot[i] = 2
        snapshot = next_snapshot

        for p in range(N):
            i = activation_order[p]
            if ordered[i] == 0:
                infected_neighbors = 0
                for d in SHIFTS:
                    if ordered[(i + d) % N] == 1:
                        infected_neighbors += 1
                if infected_neighbors:
                    prob = 1.0 - (1.0 - tau) ** infected_neighbors
                    if infection_draw[i] < prob:
                        ordered[i] = 1
            elif ordered[i] == 1 and recovery_draw[i] < GAMMA:
                ordered[i] = 2

    snapshot_attack = np.mean(snapshot != 0)
    ordered_attack = np.mean(ordered != 0)
    return snapshot_attack, ordered_attack


def _pair_task(args: tuple[float, int, bool]) -> tuple[float, float]:
    tau, seed, reshuffle_each_step = args
    return _simulate_matched_pair(
        float(tau), int(seed), bool(reshuffle_each_step)
    )


def _cpu_sweep(reshuffle_each_step: bool) -> tuple[np.ndarray, np.ndarray]:
    # Compile before forking so workers inherit the machine code.
    _simulate_matched_pair(float(TAUS[0]), 0, reshuffle_each_step)
    jobs = [
        (float(tau), seed, reshuffle_each_step)
        for tau in TAUS
        for seed in range(SEEDS_CPU)
    ]
    with Pool(32) as pool:
        rows = pool.map(_pair_task, jobs, chunksize=1)
    paired = np.asarray(rows, dtype=float).reshape(len(TAUS), SEEDS_CPU, 2)
    return paired[:, :, 0], paired[:, :, 1]


def _gpu_snapshot_batched(n: int, seeds: int) -> np.ndarray:
    import cupy as cp

    out = np.zeros((len(TAUS), seeds), dtype=float)
    n_i0 = max(1, I0 * n // N)
    for tau_index, tau in enumerate(TAUS):
        rng = cp.random.default_rng(1000 + tau_index)
        state = cp.zeros((seeds, n), dtype=cp.int8)
        for seed in range(seeds):
            idx = np.random.default_rng(seed).choice(n, size=n_i0, replace=False)
            state[seed, cp.asarray(idx)] = 1
        for _ in range(STEPS):
            infected = state == 1
            count = cp.zeros((seeds, n), dtype=cp.int32)
            for d in SHIFTS:
                count += cp.roll(infected, int(d), axis=1)
            probability = 1.0 - (1.0 - float(tau)) ** count
            newly = (state == 0) & (rng.random((seeds, n)) < probability)
            recovered = (state == 1) & (rng.random((seeds, n)) < GAMMA)
            next_state = state.copy()
            next_state[newly] = 1
            next_state[recovered] = 2
            state = next_state
        out[tau_index] = cp.asnumpy((state != 0).mean(axis=1))
    cp.cuda.Stream.null.synchronize()
    return out


def _gpu_results() -> tuple[np.ndarray, np.ndarray, str]:
    """Run the device cross-check, or retain the recorded device-only result.

    CPU-only reproduction environments can still regenerate every headline
    paired statistic.  They must not silently synthesize the GPU evidence, so
    the fallback is labeled explicitly in the output metadata.
    """
    try:
        return (
            _gpu_snapshot_batched(N, SEEDS_GPU),
            _gpu_snapshot_batched(N_BIG, SEEDS_BIG),
            "regenerated in this run",
        )
    except (ImportError, RuntimeError) as exc:
        recorded_path = Path("artifacts/emergence_threshold_controlled.json")
        if not recorded_path.exists():
            raise RuntimeError(
                "CuPy/GPU unavailable and no recorded GPU cross-check exists"
            ) from exc
        recorded = json.loads(recorded_path.read_text())
        raw = recorded["raw"]
        return (
            np.asarray(raw["batched_gpu_snapshot"], dtype=float),
            np.asarray(raw["batched_gpu_snapshot_100k"], dtype=float),
            f"retained recorded device result ({type(exc).__name__})",
        )


def _critical_tau(curve: np.ndarray, level: float = 0.5) -> float:
    for j in range(1, len(curve)):
        if curve[j - 1] < level <= curve[j]:
            frac = (level - curve[j - 1]) / (curve[j] - curve[j - 1] + 1e-12)
            return float(TAUS[j - 1] + frac * (TAUS[j] - TAUS[j - 1]))
    return float("nan")


def _bootstrap_tc(
    values: np.ndarray,
    *,
    level: float = 0.5,
    nboot: int = 2_000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    estimates = []
    n_seeds = values.shape[1]
    for _ in range(nboot):
        picked = rng.integers(0, n_seeds, size=n_seeds)
        estimate = _critical_tau(values[:, picked].mean(axis=1), level=level)
        if np.isfinite(estimate):
            estimates.append(estimate)
    arr = np.asarray(estimates)
    return {
        "median": float(np.median(arr)),
        "lo": float(np.percentile(arr, 2.5)),
        "hi": float(np.percentile(arr, 97.5)),
        "sd": float(np.std(arr)),
        "nboot_valid": int(arr.size),
    }


def _bootstrap_paired_shift(
    snapshot: np.ndarray,
    ordered: np.ndarray,
    *,
    nboot: int = 2_000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    shifts = []
    n_seeds = snapshot.shape[1]
    for _ in range(nboot):
        picked = rng.integers(0, n_seeds, size=n_seeds)
        tc_snapshot = _critical_tau(snapshot[:, picked].mean(axis=1))
        tc_ordered = _critical_tau(ordered[:, picked].mean(axis=1))
        if np.isfinite(tc_snapshot) and np.isfinite(tc_ordered):
            shifts.append(tc_snapshot - tc_ordered)
    arr = np.asarray(shifts)
    return {
        "median_absolute": float(np.median(arr)),
        "lo_absolute": float(np.percentile(arr, 2.5)),
        "hi_absolute": float(np.percentile(arr, 97.5)),
        "median_relative_to_snapshot": None,
        "nboot_valid": int(arr.size),
    }


def main() -> None:
    started = time.time()
    # The headline schedule is random sequential activation: the order is
    # reshuffled at every step and coupled between the two regimes.  Retain a
    # fixed-order condition as a schedule-robustness check rather than relying
    # on one quenched ordering for the full trajectory.
    snapshot, ordered = _cpu_sweep(reshuffle_each_step=True)
    snapshot_fixed, ordered_fixed = _cpu_sweep(reshuffle_each_step=False)
    gpu_snapshot, gpu_snapshot_big, gpu_result_status = _gpu_results()

    tc = {
        "rowwise_snapshot": _bootstrap_tc(snapshot),
        "batched_gpu_snapshot": _bootstrap_tc(gpu_snapshot),
        "inplace_ordered": _bootstrap_tc(ordered),
        "batched_gpu_snapshot_100k": _bootstrap_tc(gpu_snapshot_big),
    }
    paired_shift = _bootstrap_paired_shift(snapshot, ordered)
    paired_shift["median_relative_to_snapshot"] = float(
        paired_shift["median_absolute"] / tc["rowwise_snapshot"]["median"]
    )
    fixed_tc_snapshot = _bootstrap_tc(snapshot_fixed)
    fixed_paired_shift = _bootstrap_paired_shift(snapshot_fixed, ordered_fixed)
    fixed_paired_shift["median_relative_to_snapshot"] = float(
        fixed_paired_shift["median_absolute"] / fixed_tc_snapshot["median"]
    )

    crossing_sensitivity = {}
    for level in (0.3, 0.5, 0.7):
        crossing_sensitivity[str(level)] = {
            "rowwise_snapshot": _bootstrap_tc(snapshot, level=level),
            "batched_gpu_snapshot": _bootstrap_tc(gpu_snapshot, level=level),
            "inplace_ordered": _bootstrap_tc(ordered, level=level),
        }

    result = {
        "protocol": {
            "N": N,
            "N_big": N_BIG,
            "steps": STEPS,
            "gamma": GAMMA,
            "radius": RADIUS,
            "initial_infected": I0,
            "taus": TAUS.tolist(),
            "seeds_cpu_paired": SEEDS_CPU,
            "seeds_gpu": SEEDS_GPU,
            "seeds_big": SEEDS_BIG,
            "bootstrap_resamples": 2_000,
            "cpu_pairing": "same initial state, per-step activation order, infection draws, and recovery draws",
            "headline_schedule": "random sequential; activation order reshuffled at every step",
            "robustness_schedule": "one fixed activation order reused across steps",
            "gpu_rng": "independent CuPy stream; distributional snapshot cross-check",
            "gpu_result_status": gpu_result_status,
        },
        "curves": {
            "rowwise_snapshot_mean": snapshot.mean(axis=1).tolist(),
            "rowwise_snapshot_sd": snapshot.std(axis=1).tolist(),
            "inplace_ordered_mean": ordered.mean(axis=1).tolist(),
            "inplace_ordered_sd": ordered.std(axis=1).tolist(),
            "batched_gpu_snapshot_mean": gpu_snapshot.mean(axis=1).tolist(),
            "batched_gpu_snapshot_sd": gpu_snapshot.std(axis=1).tolist(),
            "batched_gpu_snapshot_100k_mean": gpu_snapshot_big.mean(axis=1).tolist(),
        },
        "tau_c": tc,
        "crossing_sensitivity": crossing_sensitivity,
        "paired_shift": paired_shift,
        "schedule_robustness": {
            "fixed_order_tau_c_snapshot": fixed_tc_snapshot,
            "fixed_order_paired_shift": fixed_paired_shift,
        },
        "raw": {
            "rowwise_snapshot": snapshot.tolist(),
            "inplace_ordered": ordered.tolist(),
            "batched_gpu_snapshot": gpu_snapshot.tolist(),
            "batched_gpu_snapshot_100k": gpu_snapshot_big.tolist(),
        },
        "elapsed_seconds": time.time() - started,
    }
    Path("emergence_threshold_controlled.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps({"tau_c": tc, "paired_shift": paired_shift}, indent=2))


if __name__ == "__main__":
    main()
