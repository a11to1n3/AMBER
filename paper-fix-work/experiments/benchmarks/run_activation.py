#!/usr/bin/env python3
"""E3 — Activation semantics can change ABM conclusions.

Paired SIR (snapshot vs sequential in-place) with shared counter RNG, plus a
matched Schelling contrast (three-stage sync vs last-writer).
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

from sir_reference import (  # noqa: E402
    I,
    run_sir,
    run_sir_inplace_sequential,
    step_sir,
)
from schelling_reference import (  # noqa: E402
    cell_disagreement_frac,
    mean_same_neighbor_frac,
    run_schelling,
    run_schelling_sequential,
    segregation_index,
)
from rng.counter_rng import EVT_INFECTION, EVT_RECOVERY, u01  # noqa: E402


def crossing_time(status_traj, thresh=0.5):
    """First step where infected fraction >= thresh; nan if never."""
    for t, st in enumerate(status_traj):
        if (st == I).mean() >= thresh:
            return float(t)
    return float("nan")


def sir_traj_snapshot(n, steps, seed, tau, radius=3, recovery=0.1, i0=10):
    status = np.zeros(n, dtype=np.int8)
    status[:i0] = I
    traj = [status.copy()]
    for t in range(steps):
        status = step_sir(status, seed, t, radius=radius, transmission=tau, recovery_prob=recovery)
        traj.append(status.copy())
    return traj


def sir_traj_sequential(n, steps, seed, tau, radius=3, recovery=0.1, i0=10, reshuffle=True):
    status = np.zeros(n, dtype=np.int8)
    status[:i0] = I
    traj = [status.copy()]
    rng = np.random.default_rng(seed + 999)  # only for order permutation
    order = np.arange(n)
    for t in range(steps):
        if reshuffle:
            order = rng.permutation(n)
        # recovery from entry
        entry = status.copy()
        for j in np.flatnonzero(entry == I):
            if u01(seed, t, EVT_RECOVERY, int(j), 0, 0) < recovery:
                status[j] = 2  # R
        for i in order:
            if status[i] != 0:
                continue
            for d in range(-radius, radius + 1):
                if d == 0:
                    continue
                j = (int(i) + d) % n
                if status[j] != I:
                    continue
                a, b = (int(i), j) if int(i) < j else (j, int(i))
                if u01(seed, t, EVT_INFECTION, a, b, 0) < tau:
                    status[i] = I
                    break
        traj.append(status.copy())
    return traj


def bootstrap_mean_diff(diffs, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=diffs.size, replace=True)
        boots.append(sample.mean())
    boots = np.sort(boots)
    return {
        "mean": float(diffs.mean()),
        "lo": float(boots[int(0.025 * n_boot)]),
        "hi": float(boots[int(0.975 * n_boot)]),
        "n": int(diffs.size),
    }


def run_sir_activation(quick=False):
    n = 400 if quick else 2000
    steps = 40 if quick else 80
    seeds = 24 if quick else 48
    taus = [0.05, 0.1, 0.15, 0.2, 0.3] if quick else list(np.round(np.linspace(0.02, 0.40, 12), 3))
    radius = 3
    recovery = 0.1
    i0 = max(1, n // 400)

    rows = []
    for tau in taus:
        snap_cross = []
        seq_cross = []
        final_I_snap = []
        final_I_seq = []
        for s in range(seeds):
            tr_s = sir_traj_snapshot(n, steps, s, tau, radius=radius, recovery=recovery, i0=i0)
            tr_q = sir_traj_sequential(n, steps, s, tau, radius=radius, recovery=recovery, i0=i0, reshuffle=True)
            snap_cross.append(crossing_time(tr_s, 0.2))
            seq_cross.append(crossing_time(tr_q, 0.2))
            final_I_snap.append(float((tr_s[-1] == I).mean()))
            final_I_seq.append(float((tr_q[-1] == I).mean()))
        d_cross = np.array(seq_cross) - np.array(snap_cross)
        d_final = np.array(final_I_seq) - np.array(final_I_snap)
        rows.append({
            "tau": float(tau),
            "crossing_diff": bootstrap_mean_diff(d_cross),
            "final_I_diff": bootstrap_mean_diff(d_final),
            "mean_final_I_snap": float(np.mean(final_I_snap)),
            "mean_final_I_seq": float(np.mean(final_I_seq)),
        })
        print(
            f"tau={tau:.3f} ΔfinalI={rows[-1]['final_I_diff']['mean']:.4f} "
            f"[{rows[-1]['final_I_diff']['lo']:.4f},{rows[-1]['final_I_diff']['hi']:.4f}]",
            flush=True,
        )
    # Primary at tau closest to 0.15
    primary = min(rows, key=lambda r: abs(r["tau"] - 0.15))
    return {
        "n": n,
        "steps": steps,
        "seeds": seeds,
        "rows": rows,
        "primary_tau": primary["tau"],
        "primary_final_I_diff": primary["final_I_diff"],
        "effect_excludes_zero": (
            primary["final_I_diff"]["lo"] > 0 or primary["final_I_diff"]["hi"] < 0
        ),
    }


def run_schelling_activation(quick=False):
    """Contrast three-stage snapshot sync vs sequential activation.

    Primary outcome: cell-disagreement fraction between paired finals (same
    seed/init). Secondary: Δ mean same-neighbor fraction and Δ happiness
    segregation index. Parameters chosen to create contention (higher empty
    ratio and threshold) so activation order can change destinies.
    """
    sides = [10, 14] if quick else [12, 16, 20]
    steps = 15 if quick else 30
    seeds = 24 if quick else 48
    threshold = 0.6
    empty_ratio = 0.30
    rows = []
    for side in sides:
        disagree = []
        d_same = []
        d_seg = []
        for s in range(seeds):
            g_sync = run_schelling(
                side, steps, seed=s, threshold=threshold, empty_ratio=empty_ratio,
            )
            g_seq = run_schelling_sequential(
                side, steps, seed=s, threshold=threshold, empty_ratio=empty_ratio,
                reshuffle=True,
            )
            disagree.append(cell_disagreement_frac(g_sync, g_seq))
            d_same.append(mean_same_neighbor_frac(g_seq) - mean_same_neighbor_frac(g_sync))
            d_seg.append(
                segregation_index(g_seq, threshold) - segregation_index(g_sync, threshold)
            )
        row = {
            "side": side,
            "threshold": threshold,
            "empty_ratio": empty_ratio,
            "cell_disagreement": bootstrap_mean_diff(disagree),
            "same_neighbor_diff_seq_minus_sync": bootstrap_mean_diff(d_same),
            "segregation_diff_seq_minus_sync": bootstrap_mean_diff(d_seg),
        }
        rows.append(row)
        print(
            f"side={side} disagree={row['cell_disagreement']} "
            f"Δsame={row['same_neighbor_diff_seq_minus_sync']}",
            flush=True,
        )
    # Primary = largest grid; effect if mean disagreement > 0 with CI above 0
    primary = rows[-1]
    d = primary["cell_disagreement"]
    return {
        "steps": steps,
        "seeds": seeds,
        "threshold": threshold,
        "empty_ratio": empty_ratio,
        "contrast": "three_stage_snapshot_vs_sequential_activation",
        "primary_outcome": "cell_disagreement_frac",
        "rows": rows,
        "primary": d,
        "effect_excludes_zero": d["lo"] > 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "raw" / "semantic")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=== SIR activation ===", flush=True)
    sir = run_sir_activation(quick=args.quick)
    print("=== Schelling activation ===", flush=True)
    sch = run_schelling_activation(quick=args.quick)

    report = {
        "tag": args.tag,
        "host": platform.node(),
        "elapsed_s": time.time() - t0,
        "sir": sir,
        "schelling": sch,
        "acceptance": {
            "C3_activation_effect_sir": sir["effect_excludes_zero"],
            "C3_activation_effect_schelling": sch["effect_excludes_zero"],
        },
    }
    path = args.out / f"activation_{args.tag}.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["acceptance"], indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
