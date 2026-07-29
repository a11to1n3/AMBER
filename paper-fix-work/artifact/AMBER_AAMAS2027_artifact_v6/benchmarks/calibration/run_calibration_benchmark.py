"""Calibration & validation benchmark across model families and methods.

For each (model family, method) it:
  1. generates synthetic *observed* data from the model at known ground-truth
     parameters (averaged over training seeds);
  2. calibrates -- recovers the parameters by minimising the loss vs observed
     under a fixed evaluation budget;
  3. validates -- re-simulates at the recovered parameters on *held-out* seeds
     and measures out-of-sample loss (and the overfitting gap).

Reports recovery accuracy, sample efficiency, wall-clock throughput, and the
GPU batched-ensemble advantage (SIR). Outputs a markdown table, a JSON dump,
and sample-efficiency curves.

Usage:
    python run_calibration_benchmark.py [--budget 48] [--quick]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import calib_models as cm          # noqa: E402
from methods import calibrate      # noqa: E402

TRAIN_SEEDS = [0, 1, 2, 3]
EVAL_SEED = 100
VAL_SEEDS = [200, 201, 202, 203]
CPU_METHODS = ["grid", "random", "bayesian", "smac"]
GPU_MODELS = {"sir"}               # only SIR has a batched model for the ensemble
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def recovery_error(theta_hat, ground_truth, bounds):
    """Normalised RMS parameter-recovery error (0 = exact, fraction of range)."""
    errs = [((theta_hat[k] - ground_truth[k]) / (bounds[k][1] - bounds[k][0])) ** 2
            for k in ground_truth if k in theta_hat]
    return float(np.sqrt(np.mean(errs))) if errs else float("nan")


def validation_loss(problem, observed, theta_hat, seeds):
    """Mean out-of-sample loss of the recovered parameters on held-out seeds."""
    problem.OBSERVED = np.asarray(observed, dtype=np.float64)
    theta = {k: theta_hat[k] for k in problem.GROUND_TRUTH if k in theta_hat}
    losses = []
    for s in seeds:
        m = problem({**problem.FIXED, **theta, "seed": int(s), "show_progress": False})
        losses.append(float(m.run()["model"]["loss"].to_numpy()[-1]))
    return float(np.mean(losses))


def run(budget):
    rows, curves = [], {}
    for model_name, problem in cm.PROBLEMS.items():
        print(f"\n=== {model_name} (recover {list(problem.GROUND_TRUTH)}) ===")
        observed = cm.make_observed(problem, TRAIN_SEEDS)
        methods = CPU_METHODS + (["gpu_ensemble"] if model_name in GPU_MODELS else [])
        for method in methods:
            r = calibrate(method, problem, observed, budget, seed=0, eval_seed=EVAL_SEED)
            rec_err = recovery_error(r["theta_hat"], problem.GROUND_TRUTH, problem.BOUNDS)
            val = validation_loss(problem, observed, r["theta_hat"], VAL_SEEDS)
            row = {
                "model": model_name, "method": method,
                "recovery_error": rec_err, "train_loss": r["best_loss"],
                "val_loss": val, "overfit_gap": val - r["best_loss"],
                "n_evals": r["n_evals"], "wall_s": r["wall_time"],
                "evals_per_s": r["n_evals"] / max(r["wall_time"], 1e-9),
                "theta_hat": r["theta_hat"],
            }
            rows.append(row)
            curves[(model_name, method)] = r["curve"]
            print(f"  {method:13s} rec_err={rec_err:.3f}  val_loss={val:.4f}  "
                  f"{r['wall_time']:.2f}s  {row['evals_per_s']:.1f} ev/s")
    return rows, curves


def write_table(rows, path):
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    lines = ["# Calibration & validation benchmark", "",
             "Recovery error = normalised RMS distance of recovered to true "
             "parameters (fraction of search range). Validation loss is "
             "out-of-sample (held-out seeds); overfit gap = val - train loss.", ""]
    for model, rs in by_model.items():
        gt = cm.PROBLEMS[model].GROUND_TRUTH
        lines += [f"## {model}  (truth: {gt})", "",
                  "| Method | Recovery err | Train loss | Val loss | Overfit gap | Evals | Wall (s) | Evals/s |",
                  "|--------|-------------:|-----------:|---------:|------------:|------:|---------:|--------:|"]
        for r in sorted(rs, key=lambda x: x["recovery_error"]):
            lines.append(
                f"| {r['method']} | {r['recovery_error']:.3f} | {r['train_loss']:.4f} | "
                f"{r['val_loss']:.4f} | {r['overfit_gap']:+.4f} | {r['n_evals']} | "
                f"{r['wall_s']:.2f} | {r['evals_per_s']:.1f} |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_curves(curves, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping curves: {e})")
        return
    models = list(cm.PROBLEMS)
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        for (m, method), curve in curves.items():
            if m != model:
                continue
            ax.plot(range(1, len(curve) + 1), curve, marker=".", ms=3, label=method)
        ax.set_yscale("log")
        ax.set_title(f"{model}: best loss vs evaluations")
        ax.set_xlabel("evaluations")
        ax.set_ylabel("best loss so far")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=48, help="evaluations per method")
    ap.add_argument("--quick", action="store_true", help="small budget smoke run")
    args = ap.parse_args()
    budget = 16 if args.quick else args.budget

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.perf_counter()
    rows, curves = run(budget)
    print(f"\nTotal wall: {time.perf_counter() - t0:.1f}s")

    with open(os.path.join(RESULTS_DIR, "calibration_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    write_table(rows, os.path.join(RESULTS_DIR, "summary_table_calibration.md"))
    write_curves(curves, os.path.join(RESULTS_DIR, "calibration_curves.png"))


if __name__ == "__main__":
    main()
