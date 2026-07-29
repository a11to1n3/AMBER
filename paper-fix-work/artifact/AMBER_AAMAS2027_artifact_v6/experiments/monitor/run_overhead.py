#!/usr/bin/env python3
"""E2b — Monitor overhead surface over N, q, c.

q = number of concurrent column commits per step (write intensity).
c = schema columns present (checked at step boundary).
Retains every timing sample; separates setup from step loop.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "src"))

import ambr as am  # noqa: E402


def make_model(column_count: int, write_count: int):
    names = tuple(f"x{i}" for i in range(column_count))
    writes = names[: max(1, min(write_count, column_count))]

    class Synthetic(am.Model):
        def setup(self):
            n = int(self.p["n"])
            self.add_agents(n, **{name: self.rng.random(n) for name in names})

        def step_vectorized(self):
            outputs = {}
            for name in writes:
                values, _ = self.agents.borrow(name)
                outputs[name] = values + 1.0
            self.agents.commit(**outputs)

    return Synthetic


def time_run(cls, n, steps, contract, seed):
    model = cls({"n": n, "steps": steps, "seed": seed, "show_progress": False})
    t0 = time.perf_counter()
    model.cpu(mode="vectorized").run(contract=contract)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "raw" / "monitor")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.quick:
        populations = (1_000, 10_000, 100_000)
        columns = (1, 4)
        writes = (1, 4)
        runs = 3
        steps = 10
    else:
        populations = (1_000, 10_000, 100_000, 1_000_000)
        columns = (1, 4, 8)
        writes = (1, 4, 8)
        runs = 5
        steps = 20

    records = []
    for n in populations:
        for c in columns:
            for q in writes:
                if q > c:
                    continue
                cls = make_model(c, q)
                for contract in ("off", "check"):
                    samples = []
                    # one untimed warm-up
                    try:
                        time_run(cls, n, steps, contract, seed=0)
                    except Exception as exc:
                        records.append({
                            "n": n, "c": c, "q": q, "contract": contract,
                            "status": "error", "error": str(exc),
                        })
                        continue
                    for r in range(runs):
                        samples.append(time_run(cls, n, steps, contract, seed=r + 1))
                    records.append({
                        "n": n,
                        "c": c,
                        "q": q,
                        "contract": contract,
                        "steps": steps,
                        "runs": runs,
                        "samples_s": samples,
                        "mean_s": statistics.mean(samples),
                        "median_s": statistics.median(samples),
                        "stdev_s": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
                        "status": "success",
                    })
                    print(
                        f"N={n:8d} c={c} q={q} contract={contract:5} "
                        f"median={statistics.median(samples):.4f}s",
                        flush=True,
                    )

    # Pair off/check for overhead ratio
    pairs = {}
    for rec in records:
        if rec.get("status") != "success":
            continue
        key = (rec["n"], rec["c"], rec["q"])
        pairs.setdefault(key, {})[rec["contract"]] = rec
    overhead = []
    for key, d in sorted(pairs.items()):
        if "off" in d and "check" in d:
            off_m = d["off"]["median_s"]
            chk_m = d["check"]["median_s"]
            overhead.append({
                "n": key[0],
                "c": key[1],
                "q": key[2],
                "median_off_s": off_m,
                "median_check_s": chk_m,
                "abs_overhead_s": chk_m - off_m,
                "ratio": chk_m / off_m if off_m > 0 else None,
                "per_step_ms": 1000.0 * (chk_m - off_m) / steps,
            })

    report = {
        "tag": args.tag,
        "host": platform.node(),
        "records": records,
        "overhead": overhead,
        "notes": "q=0 whole-column regime is NOT claimed; this varies q and c.",
    }
    path = args.out / f"overhead_{args.tag}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"wrote {path} ({len(overhead)} overhead points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
