#!/usr/bin/env python3
"""Measure current AMBER monitor cost with all timing samples retained."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

import ambr as am


STEPS = 20
RUNS = 5
POPULATIONS = (10_000, 100_000, 1_000_000)
COLUMN_COUNTS = (1, 8)


def make_model(column_count: int):
    names = tuple(f"x{i}" for i in range(column_count))

    class Synthetic(am.Model):
        def setup(self):
            n = int(self.p["n"])
            self.add_agents(n, **{
                name: self.rng.random(n) for name in names
            })

        def step_vectorized(self):
            outputs = {}
            for name in names:
                values, _token = self.agents.borrow(name)
                outputs[name] = values + 1.0
            self.agents.commit(**outputs)

    return Synthetic


def timed_run(model_class, n: int, mode: str, seed: int) -> tuple[float, bool]:
    model = model_class({
        "n": n,
        "steps": STEPS,
        "seed": seed,
        "show_progress": False,
    })
    start = time.perf_counter()
    result = model.cpu(mode="vectorized").run(contract=mode)
    elapsed = time.perf_counter() - start
    clean = True if mode == "off" else all(c.clean for c in result["contract"])
    return elapsed, clean


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "raw_seconds": samples,
        "mean_seconds": statistics.fmean(samples),
        "median_seconds": statistics.median(samples),
        "iqr_seconds": q3 - q1,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def main() -> None:
    rows = []
    for column_count in COLUMN_COUNTS:
        model_class = make_model(column_count)
        for n in POPULATIONS:
            samples = {"off": [], "check": []}
            all_clean = True
            # Warm each path once before retaining samples.
            timed_run(model_class, n, "off", seed=999)
            _, clean = timed_run(model_class, n, "check", seed=999)
            all_clean &= clean

            for repeat in range(RUNS):
                order = ("off", "check") if repeat % 2 == 0 else ("check", "off")
                for mode in order:
                    elapsed, clean = timed_run(
                        model_class, n, mode, seed=100 + repeat
                    )
                    samples[mode].append(elapsed)
                    all_clean &= clean

            off = summarize(samples["off"])
            check = summarize(samples["check"])
            added_ms = (
                check["mean_seconds"] - off["mean_seconds"]
            ) * 1_000.0 / STEPS
            rows.append({
                "n": n,
                "columns": column_count,
                "steps": STEPS,
                "runs": RUNS,
                "all_check_records_clean": all_clean,
                "off": off,
                "check": check,
                "mean_ratio": check["mean_seconds"] / off["mean_seconds"],
                "added_ms_per_step": added_ms,
            })
            print(
                f"N={n:>8} c={column_count} "
                f"ratio={rows[-1]['mean_ratio']:.3f} "
                f"added={added_ms:.3f} ms/step"
            )

    output = {
        "protocol": {
            "steps": STEPS,
            "runs": RUNS,
            "retention": "all measured samples retained",
            "timing_scope": "model construction, setup, step loop, and result assembly",
            "workload": "q=0 whole-column borrow/commit; one commit per column",
        },
        "rows": rows,
    }
    destination = Path(__file__).resolve().parents[1] / "monitor_cost_current.json"
    destination.write_text(json.dumps(output, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
