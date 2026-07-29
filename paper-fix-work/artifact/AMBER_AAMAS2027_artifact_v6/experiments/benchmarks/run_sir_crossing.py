#!/usr/bin/env python3
"""Priority 2 — SIR cumulative attack-rate crossing under shared counter tape.

Protocol (restored):
  N=4000, steps=120, 16 τ values, 48 paired seeds,
  snapshot vs sequential in-place activation,
  identical initial state / orders / infection+recovery draws.

Primary outcomes: attack rate A_T = (I_T + R_T)/N and isotonic τ crossings
at 0.3, 0.5, 0.7.  Final I prevalence remains secondary.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "semantic" / "references"))

from rng.counter_rng import EVT_INFECTION, EVT_RECOVERY, u01  # noqa: E402

S, I, R = 0, 1, 2


def snapshot_step(status, seed, t, radius, tau, recovery):
    n = status.shape[0]
    entry = status.copy()
    out = status.copy()
    for i in range(n):
        if entry[i] != S:
            continue
        for d in range(-radius, radius + 1):
            if d == 0:
                continue
            j = (i + d) % n
            if entry[j] != I:
                continue
            a, b = (i, j) if i < j else (j, i)
            if u01(seed, t, EVT_INFECTION, a, b, 0) < tau:
                out[i] = I
                break
    for j in range(n):
        if entry[j] == I and u01(seed, t, EVT_RECOVERY, j, 0, 0) < recovery:
            out[j] = R
    return out


def sequential_step(status, seed, t, radius, tau, recovery, order):
    # recovery from entry
    entry = status.copy()
    for j in range(status.shape[0]):
        if entry[j] == I and u01(seed, t, EVT_RECOVERY, j, 0, 0) < recovery:
            status[j] = R
    for i in order:
        i = int(i)
        if status[i] != S:
            continue
        for d in range(-radius, radius + 1):
            if d == 0:
                continue
            j = (i + d) % status.shape[0]
            if status[j] != I:
                continue
            a, b = (i, j) if i < j else (j, i)
            if u01(seed, t, EVT_INFECTION, a, b, 0) < tau:
                status[i] = I
                break
    return status


def simulate_pair(n, steps, seed, tau, radius, recovery, i0, reshuffle=True):
    rng = np.random.default_rng(seed + 99_991)
    init_order = rng.permutation(n)
    snap = np.zeros(n, dtype=np.int8)
    snap[init_order[:i0]] = I
    seq = snap.copy()
    order = np.arange(n)
    peak_s = peak_q = 0.0
    t_peak_s = t_peak_q = 0
    for t in range(steps):
        if reshuffle:
            order = rng.permutation(n)
        snap = snapshot_step(snap, seed, t, radius, tau, recovery)
        seq = sequential_step(seq, seed, t, radius, tau, recovery, order)
        fi_s = float((snap == I).mean())
        fi_q = float((seq == I).mean())
        if fi_s > peak_s:
            peak_s, t_peak_s = fi_s, t
        if fi_q > peak_q:
            peak_q, t_peak_q = fi_q, t
    A_s = float(((snap == I) | (snap == R)).mean())
    A_q = float(((seq == I) | (seq == R)).mean())
    I_s = float((snap == I).mean())
    I_q = float((seq == I).mean())
    return {
        "A_snap": A_s, "A_seq": A_q,
        "I_snap": I_s, "I_seq": I_q,
        "peak_I_snap": peak_s, "peak_I_seq": peak_q,
        "t_peak_snap": t_peak_s, "t_peak_seq": t_peak_q,
    }


def isotonic_fit(taus, values):
    """Pool-adjacent-violators isotonic regression (non-decreasing)."""
    taus = np.asarray(taus, dtype=float)
    y = np.asarray(values, dtype=float)
    order = np.argsort(taus)
    x = taus[order]
    v = y[order].copy()
    n = len(v)
    # simple PAV
    level = v.copy()
    weight = np.ones(n)
    i = 0
    while i < n - 1:
        if level[i] <= level[i + 1] + 1e-15:
            i += 1
            continue
        # merge pool
        j = i
        while j >= 0 and level[j] > level[j + 1] + 1e-15:
            w = weight[j] + weight[j + 1]
            avg = (level[j] * weight[j] + level[j + 1] * weight[j + 1]) / w
            level[j] = level[j + 1] = avg
            weight[j] = weight[j + 1] = w
            j -= 1
        i = max(j + 1, 0)
    # re-expand equals
    out = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(level[j + 1] - level[i]) < 1e-12:
            j += 1
        out[i : j + 1] = level[i]
        i = j + 1
    return x, out


def invert_crossing(taus_sorted, iso_vals, level: float):
    """First τ where isotonic curve reaches level; nan if never."""
    if iso_vals[-1] < level - 1e-12:
        return float("nan")
    if iso_vals[0] >= level:
        return float(taus_sorted[0])
    for i in range(1, len(iso_vals)):
        if iso_vals[i] >= level:
            # linear interp in τ
            y0, y1 = iso_vals[i - 1], iso_vals[i]
            x0, x1 = taus_sorted[i - 1], taus_sorted[i]
            if y1 <= y0:
                return float(x1)
            frac = (level - y0) / (y1 - y0)
            return float(x0 + frac * (x1 - x0))
    return float("nan")


def bootstrap_crossings(curve_by_seed, taus, level, n_boot=2000, seed=0):
    """curve_by_seed: list of arrays shape (len(taus),) for each seed."""
    rng = np.random.default_rng(seed)
    n = len(curve_by_seed)
    taus = np.asarray(taus, float)
    crossings = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mean_curve = np.mean([curve_by_seed[i] for i in idx], axis=0)
        xs, iso = isotonic_fit(taus, mean_curve)
        crossings.append(invert_crossing(xs, iso, level))
    arr = np.asarray(crossings, float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "median": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "n_in_range": 0,
            "n_boot": n_boot,
        }
    finite.sort()
    return {
        "median": float(np.median(finite)),
        "lo": float(np.percentile(finite, 2.5)),
        "hi": float(np.percentile(finite, 97.5)),
        "n_in_range": int(finite.size),
        "n_boot": n_boot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "raw" / "semantic")
    ap.add_argument("--tag", default="host_a")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.quick:
        n, steps, seeds = 400, 40, 12
        taus = list(np.round(np.linspace(0.02, 0.40, 8), 3))
    else:
        n, steps, seeds = 4000, 120, 48
        taus = list(np.round(np.linspace(0.02, 0.40, 16), 3))
    radius, recovery, i0 = 3, 0.10, 10

    t0 = time.time()
    rows = []
    A_snap_by_seed = [[] for _ in range(seeds)]
    A_seq_by_seed = [[] for _ in range(seeds)]

    for tau in taus:
        dA, dI = [], []
        for s in range(seeds):
            r = simulate_pair(n, steps, s, float(tau), radius, recovery, i0, reshuffle=True)
            A_snap_by_seed[s].append(r["A_snap"])
            A_seq_by_seed[s].append(r["A_seq"])
            dA.append(r["A_seq"] - r["A_snap"])
            dI.append(r["I_seq"] - r["I_snap"])
        dA = np.asarray(dA)
        dI = np.asarray(dI)
        rows.append({
            "tau": float(tau),
            "mean_A_snap": float(np.mean([A_snap_by_seed[s][-1] for s in range(seeds)])),
            "mean_A_seq": float(np.mean([A_seq_by_seed[s][-1] for s in range(seeds)])),
            "delta_A": {
                "mean": float(dA.mean()),
                "lo": float(np.percentile(dA, 2.5)),
                "hi": float(np.percentile(dA, 97.5)),
            },
            "delta_I_secondary": {
                "mean": float(dI.mean()),
                "lo": float(np.percentile(dI, 2.5)),
                "hi": float(np.percentile(dI, 97.5)),
            },
        })
        print(
            f"tau={tau:.3f} A_snap={rows[-1]['mean_A_snap']:.3f} "
            f"A_seq={rows[-1]['mean_A_seq']:.3f} "
            f"ΔA={rows[-1]['delta_A']['mean']:.4f} "
            f"[{rows[-1]['delta_A']['lo']:.4f},{rows[-1]['delta_A']['hi']:.4f}]",
            flush=True,
        )

    # convert to arrays per seed
    A_snap_curves = [np.asarray(A_snap_by_seed[s], float) for s in range(seeds)]
    A_seq_curves = [np.asarray(A_seq_by_seed[s], float) for s in range(seeds)]

    crossings = {}
    for level in (0.3, 0.5, 0.7):
        c_snap = bootstrap_crossings(A_snap_curves, taus, level)
        c_seq = bootstrap_crossings(A_seq_curves, taus, level)
        # paired shift: bootstrap mean curve of A_seq - A_snap is wrong for crossing;
        # bootstrap paired difference of inverted crossings when both finite
        rng = np.random.default_rng(int(level * 1000))
        shifts = []
        for _ in range(2000):
            idx = rng.integers(0, seeds, size=seeds)
            ms = np.mean([A_snap_curves[i] for i in idx], axis=0)
            mq = np.mean([A_seq_curves[i] for i in idx], axis=0)
            xs, iso_s = isotonic_fit(taus, ms)
            _, iso_q = isotonic_fit(taus, mq)
            ts = invert_crossing(xs, iso_s, level)
            tq = invert_crossing(xs, iso_q, level)
            if np.isfinite(ts) and np.isfinite(tq):
                shifts.append(tq - ts)
        shifts = np.asarray(shifts, float)
        if shifts.size:
            shifts.sort()
            shift_sum = {
                "median": float(np.median(shifts)),
                "lo": float(np.percentile(shifts, 2.5)),
                "hi": float(np.percentile(shifts, 97.5)),
                "n_paired": int(shifts.size),
                "excludes_zero": bool(np.percentile(shifts, 2.5) > 0 or np.percentile(shifts, 97.5) < 0),
            }
            rel = {
                "median": float(np.median(shifts / np.median([invert_crossing(*isotonic_fit(taus, np.mean(A_snap_curves, axis=0)), level) or np.nan])) if False else np.nan),
            }
            # relative to snapshot crossing median
            if c_snap["median"] and np.isfinite(c_snap["median"]) and abs(c_snap["median"]) > 1e-12:
                rel_shift = {
                    "median": float(np.median(shifts) / c_snap["median"]),
                    "lo": float(np.percentile(shifts, 2.5) / c_snap["median"]),
                    "hi": float(np.percentile(shifts, 97.5) / c_snap["median"]),
                }
            else:
                rel_shift = {"median": float("nan"), "lo": float("nan"), "hi": float("nan")}
        else:
            shift_sum = {
                "median": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_paired": 0, "excludes_zero": False,
            }
            rel_shift = {"median": float("nan"), "lo": float("nan"), "hi": float("nan")}
        crossings[str(level)] = {
            "snapshot": c_snap,
            "sequential": c_seq,
            "paired_shift_seq_minus_snap": shift_sum,
            "paired_relative_shift": rel_shift,
        }
        print(f"crossing@{level}: snap={c_snap} seq={c_seq} shift={shift_sum}", flush=True)

    primary = crossings["0.5"]
    # Seed-level attack curves (taus-aligned) for reproducibility export.
    seed_level = {
        "seeds": list(range(seeds)),
        "taus": taus,
        "A_snap_by_seed": [list(map(float, A_snap_by_seed[s])) for s in range(seeds)],
        "A_seq_by_seed": [list(map(float, A_seq_by_seed[s])) for s in range(seeds)],
    }
    report = {
        "tag": args.tag,
        "host_label": "host_a",
        "platform": platform.platform(),
        "elapsed_s": time.time() - t0,
        "protocol": {
            "N": n, "steps": steps, "seeds": seeds, "taus": taus,
            "radius": radius, "recovery": recovery, "i0": i0,
            "primary_outcome": "cumulative_attack_A_T=(I+R)/N",
            "secondary_outcome": "final_I_prevalence",
            "schedule": "per-step reshuffled activation order for sequential",
            "rng": "shared counter tape EVT_INFECTION/EVT_RECOVERY",
        },
        "rows": rows,
        "crossings": crossings,
        "seed_level": seed_level,
        "acceptance": {
            "crossing_0.5_shift_excludes_zero": primary["paired_shift_seq_minus_snap"]["excludes_zero"],
            "note": "Do not require exactly 6.9%; report new honest estimate.",
        },
        "scope": "Supplementary final-I 80-step 3090 sweep remains secondary; this is the main crossing experiment.",
    }
    path = args.out / f"sir_crossing_{args.tag}.json"
    path.write_text(json.dumps(report, indent=2))
    # Compact companion for seed-level curves only (easier to ship).
    seed_path = args.out / f"sir_crossing_seed_level_{args.tag}.json"
    seed_path.write_text(json.dumps({"tag": args.tag, "protocol": report["protocol"], "seed_level": seed_level}, indent=2))
    print(json.dumps(report["acceptance"], indent=2))
    print("wrote", path)
    print("wrote", seed_path)
    return 0 if primary["paired_shift_seq_minus_snap"]["excludes_zero"] else 0  # still write even if null


if __name__ == "__main__":
    raise SystemExit(main())
