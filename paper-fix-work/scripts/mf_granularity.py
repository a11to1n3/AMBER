#!/usr/bin/env python3
"""mesa-frames SIR sensitivity to local update-block granularity.

Every variant performs exactly one public ``AgentSetPolars.set`` per model
step.  ``nblocks`` changes only how a local NumPy copy is processed before that
final framework commit: one block is snapshot-like, while later blocks in a
multi-block pass observe earlier local writes.
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
from mesa_frames import AgentSetPolars, ModelDF

os.environ.setdefault("AMBER_SUPPRESS_DEPRECATIONS", "1")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "mf_granularity.json"

N = 4000
STEPS = 120
GAMMA = 0.10
RADIUS = 3
I0 = 10
SEEDS = 24
TAUS = np.round(np.linspace(0.02, 0.40, 16), 3)
SHIFTS = [d for d in range(-RADIUS, RADIUS + 1) if d != 0]
# One local whole-array pass, then increasingly fine local in-place blocks.
NBLOCKS = [1, 10, 40, 160, 640]


def init_status(rng):
    status = np.zeros(N, dtype=np.int64)
    status[rng.choice(N, size=I0, replace=False)] = 1
    return status


def neighbour_count(infected):
    count = np.zeros(N, dtype=np.int64)
    for shift in SHIFTS:
        count += np.roll(infected, shift)
    return count


class SIRBlock(AgentSetPolars):
    """Real AgentSetPolars model with one final framework commit per step."""

    def __init__(self, model, status0, tau, rng, nblocks):
        super().__init__(model)
        self.tau = tau
        self.rng = rng
        self.nblocks = nblocks
        self.add(pl.DataFrame({"unique_id": np.arange(N), "status": status0}))

    def step(self):
        status = self.agents["status"].to_numpy().copy()
        count = neighbour_count((status == 1).astype(np.int64))
        random_infect = self.rng.random(N)
        random_recover = self.rng.random(N)
        edges = np.linspace(0, N, self.nblocks + 1).astype(int)

        for block in range(self.nblocks):
            lo, hi = edges[block], edges[block + 1]
            indices = np.arange(lo, hi)
            probability = 1 - (1 - self.tau) ** count[lo:hi]
            newly = indices[
                (status[lo:hi] == 0) & (random_infect[lo:hi] < probability)
            ]
            recovered = indices[
                (status[lo:hi] == 1) & (random_recover[lo:hi] < GAMMA)
            ]

            # Local writes update neighbour counts seen by later local blocks.
            for index in newly:
                status[index] = 1
                for shift in SHIFTS:
                    count[(index + shift) % N] += 1
            for index in recovered:
                status[index] = 2
                for shift in SHIFTS:
                    count[(index + shift) % N] -= 1

        # The sole framework-level status commit in every variant.
        self.set("status", pl.Series(status))

    def attack(self):
        status = self.agents["status"].to_numpy()
        return float((status == 2).mean() + (status == 1).mean())


def run(args):
    nblocks, tau, seed = args
    rng = np.random.default_rng(seed)
    model = ModelDF()
    agents = SIRBlock(model, init_status(rng), tau, rng, nblocks)
    for _ in range(STEPS):
        agents.step()
    return agents.attack()


def locate(means, level=0.5):
    means = np.asarray(means)
    for index in range(len(TAUS) - 1):
        if means[index] < level <= means[index + 1]:
            fraction = (level - means[index]) / (means[index + 1] - means[index])
            return float(TAUS[index] + fraction * (TAUS[index + 1] - TAUS[index]))
    return float("nan")


def bootstrap(all_values, level=0.5, repetitions=2000, seed=0):
    rng = np.random.default_rng(seed)
    all_values = np.asarray(all_values)
    crossings = []
    n_seeds = all_values.shape[1]
    for _ in range(repetitions):
        # Preserve the trajectory across tau: one seed-index resample is shared
        # by every row, rather than independently scrambling each tau value.
        picked = rng.integers(0, n_seeds, size=n_seeds)
        resampled = all_values[:, picked].mean(axis=1)
        crossing = locate(resampled, level)
        if not np.isnan(crossing):
            crossings.append(crossing)
    crossings = np.asarray(crossings)
    return float(np.median(crossings)), [
        float(np.percentile(crossings, 2.5)),
        float(np.percentile(crossings, 97.5)),
    ]


def main():
    started = time.time()
    output = {
        "taus": TAUS.tolist(),
        "nblocks": NBLOCKS,
        "by_nblocks": {},
        "meta": {
            "N": N,
            "STEPS": STEPS,
            "RADIUS": RADIUS,
            "GAMMA": GAMMA,
            "SEEDS": SEEDS,
            "framework_commits_per_step": 1,
            "varied_quantity": "local_numpy_update_blocks",
        },
    }
    with Pool(10) as pool:
        for nblocks in NBLOCKS:
            jobs = [
                (nblocks, float(tau), int(seed))
                for tau in TAUS
                for seed in range(SEEDS)
            ]
            values = np.asarray(pool.map(run, jobs)).reshape(len(TAUS), SEEDS)
            crossing, interval = bootstrap(values)
            output["by_nblocks"][str(nblocks)] = {
                "curve": values.mean(axis=1).tolist(),
                "tc": crossing,
                "tc_ci": interval,
                "allvals": values.tolist(),
            }
            print(
                f"local_blocks={nblocks}: tc={crossing:.4f} "
                f"CI=[{interval[0]:.4f},{interval[1]:.4f}] "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
