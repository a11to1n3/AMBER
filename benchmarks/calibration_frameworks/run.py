"""Cross-framework calibration & validation benchmark driver.

Every framework recovers SIR (beta, gamma) from the same observed curve using
the same shared candidate set (a fair common optimiser). Reports per-framework
recovery accuracy, out-of-sample validation loss, and -- the headline --
calibration wall-clock / throughput. AMBER additionally runs its GPU batched
ensemble (all candidates per GPU pass) and its native SMAC+GPU calibrator.

Usage:
    AMBER_SUPPRESS_DEPRECATIONS=1 python run.py [--budget 128] [--quick]
"""

import argparse
import json
import os
import time
import traceback

import numpy as np

import frameworks
import task

TRAIN_SEEDS = [0, 1, 2, 3]
EVAL_SEED = 100
VAL_SEEDS = [200, 201, 202, 203]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _finish(r, name, mode, val_fn, observed):
    r["framework"] = name
    r["mode"] = mode
    r["recovery_error"] = task.recovery_error(r["theta_hat"])
    r["val_loss"] = val_fn(r["theta_hat"]) if r["theta_hat"] else float("nan")
    print(f"  {name:22s} [{mode}] rec_err={r['recovery_error']:.3f} "
          f"val={r['val_loss']:.4f}  {r['wall_s']:.2f}s  {r['evals_per_s']:.1f} ev/s")
    return r


def run(budget):
    observed = task.make_observed(TRAIN_SEEDS)
    candidates = task.make_candidates(budget, seed=0)
    rows, curves = [], {}

    print(f"\n=== common optimiser ({budget} shared candidates) ===")
    for name, fn in frameworks.CURVE_FNS.items():
        try:
            r = task.run_common_optimizer(fn, observed, candidates, task.N, task.STEPS, EVAL_SEED)
            curves[name] = r.pop("curve")
            rows.append(_finish(r, name, "common-opt",
                                lambda th, fn=fn: task.validation_loss(fn, observed, th, VAL_SEEDS),
                                observed))
        except Exception as e:
            print(f"  {name:22s} FAILED: {e}")
            traceback.print_exc()

    # Agents.jl (Julia subprocess -- self-contained, reports its own validation)
    try:
        r = frameworks.agentsjl_calibrate(observed, candidates, task.N, task.STEPS,
                                          EVAL_SEED, VAL_SEEDS, task.GROUND_TRUTH, task.BOUNDS)
        r["framework"] = "Agents.jl"
        r["mode"] = "common-opt"
        print(f"  {'Agents.jl':22s} [common-opt] rec_err={r['recovery_error']:.3f} "
              f"val={r['val_loss']:.4f}  {r['wall_s']:.2f}s  {r['evals_per_s']:.1f} ev/s")
        rows.append(r)
    except Exception as e:
        print(f"  {'Agents.jl':22s} FAILED: {e}")

    # AMBER GPU batched ensemble -- same candidates, evaluated in GPU passes
    try:
        r = frameworks.amber_gpu_calibrate(observed, candidates, task.N, task.STEPS, EVAL_SEED)
        r.pop("curve", None)
        rows.append(_finish(r, "AMBER (GPU ensemble)", "batched",
                            lambda th: task.validation_loss(frameworks.amber_sir_curve, observed, th, VAL_SEEDS),
                            observed))
    except Exception as e:
        print(f"  AMBER (GPU ensemble)   FAILED: {e}")
        traceback.print_exc()

    # AMBER native calibrator: SMAC ask/tell + GPU batched evaluation
    try:
        r = _amber_native(observed, budget)
        rows.append(_finish(r, "AMBER (native SMAC+GPU)", "native",
                            lambda th: task.validation_loss(frameworks.amber_sir_curve, observed, th, VAL_SEEDS),
                            observed))
    except Exception as e:
        print(f"  AMBER (native SMAC+GPU) FAILED: {e}")

    # FLAME GPU 2 last: its CUDA context would otherwise perturb cupy's timings.
    # One GPU simulation per candidate (no cross-candidate batching); warm the
    # RTC compile before timing.
    try:
        frameworks.flamegpu_sir_curve(0.3, 0.1, task.N, task.STEPS, EVAL_SEED)  # warmup
        r = task.run_common_optimizer(frameworks.flamegpu_sir_curve, observed,
                                      candidates, task.N, task.STEPS, EVAL_SEED)
        curves["FLAME GPU 2"] = r.pop("curve")
        rows.append(_finish(r, "FLAME GPU 2", "common-opt",
                            lambda th: task.validation_loss(frameworks.flamegpu_sir_curve, observed, th, VAL_SEEDS),
                            observed))
    except Exception as e:
        print(f"  {'FLAME GPU 2':22s} FAILED: {e}")

    return rows, curves


def _amber_native(observed, budget):
    """AMBER's native calibration: smac_batch_calibrate (SMAC + GPU ensemble)."""
    from ambr.gpu import get_array_module, to_host
    from ambr.gpu_ensemble import BatchedWellMixedSIR, smac_batch_calibrate

    obs = np.asarray(observed, dtype=np.float64)

    def loss_fn(traj):
        xp = get_array_module()
        return ((traj["I_frac"] - xp.asarray(obs, dtype=traj["I_frac"].dtype).reshape(1, -1)) ** 2).sum(axis=1)

    batch = 32
    rounds = max(1, budget // batch)
    t0 = time.perf_counter()
    best, _ = smac_batch_calibrate(
        BatchedWellMixedSIR(), task.BOUNDS, loss_fn, task.N, task.STEPS,
        rounds=rounds, batch_size=batch, fixed_params={"i0_frac": task.I0_FRAC}, seed=0)
    wall = time.perf_counter() - t0
    return {"theta_hat": {k: float(best[k]) for k in task.BOUNDS},
            "best_loss": float("nan"), "n_evals": rounds * batch,
            "wall_s": wall, "evals_per_s": rounds * batch / max(wall, 1e-9)}


def write_table(rows, path):
    rows = sorted(rows, key=lambda r: r["wall_s"])
    lines = [
        "# Cross-framework calibration & validation benchmark", "",
        f"Task: recover well-mixed SIR (beta, gamma) from an observed infected-"
        f"fraction curve (N={task.N}, {task.STEPS} steps). Every framework "
        "evaluates the same shared candidate set (fair common optimiser); "
        "AMBER also runs its GPU batched ensemble and native SMAC+GPU calibrator. "
        "Recovery error is normalised RMS distance to truth; validation loss is "
        "out-of-sample (held-out seeds).", "",
        "| Framework | Mode | Recovery err | Val loss | Evals | Wall (s) | Evals/s | Speedup |",
        "|-----------|------|-------------:|---------:|------:|---------:|--------:|--------:|",
    ]
    slowest = max(r["wall_s"] for r in rows)
    for r in rows:
        lines.append(
            f"| {r['framework']} | {r['mode']} | {r['recovery_error']:.3f} | "
            f"{r['val_loss']:.4f} | {r['n_evals']} | {r['wall_s']:.2f} | "
            f"{r['evals_per_s']:.1f} | {slowest / max(r['wall_s'], 1e-9):.1f}x |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_curves(curves, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping curves: {e})")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, curve in curves.items():
        if curve:
            ax.plot(range(1, len(curve) + 1), curve, marker=".", ms=3, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("calibration evaluations")
    ax.set_ylabel("best loss so far")
    ax.set_title("Calibration progress per framework (common optimiser)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    budget = 24 if args.quick else args.budget

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.perf_counter()
    rows, curves = run(budget)
    print(f"\nTotal wall: {time.perf_counter() - t0:.1f}s")

    with open(os.path.join(RESULTS_DIR, "calibration_frameworks_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    write_table(rows, os.path.join(RESULTS_DIR, "summary_table_calibration_frameworks.md"))
    write_curves(curves, os.path.join(RESULTS_DIR, "calibration_frameworks_curves.png"))


if __name__ == "__main__":
    main()
