#!/usr/bin/env python3
"""Re-plot the multi-framework scaling chart from an existing benchmark JSON,
adding AMBER (GPU) as a framework series and Schelling Segregation as a model.

This is a *pure re-plot* of data already measured by ``run_all_frameworks.py``
(no benchmarks are run here). It reuses the exact visual style of
``_write_chart`` in ``run_all_frameworks.py`` — matplotlib, log-log, one
subplot per model, one line per framework, "o" markers, per-subplot legends —
but extends the colour map so the GPU / mesa-frames / FLAME GPU 2 series get
distinct colours instead of all falling back to grey.

Source data: benchmarks/results/benchmark_results_10M.json, which contains the
full 10-framework x 4-model sweep (agents 1k -> 10M) including AMBER (GPU) and
the Schelling Segregation workload.

Usage::

    python benchmarks/plot_scaling_with_gpu_schelling.py \
        --input benchmarks/results/benchmark_results.json \
        --output benchmarks/results/scaling_chart.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Style constants (mirrors run_all_frameworks.py) ----------------------- #

# Plot order: AMBER family first (GPU is the new headline), then GPU peers,
# then the CPU frameworks. Matches the server's extended FRAMEWORK_ORDER.
FRAMEWORK_ORDER = [
    "AMBER (GPU)",
    "AMBER (vectorized)",
    "AMBER (loop)",
    "mesa-frames",
    "FLAME GPU 2",
    "Agents.jl",
    "SimPy",
    "Melodie",
    "AgentPy",
    "Mesa",
]

# Colours: the 7 from run_all_frameworks.py verbatim, plus 3 added so the
# previously grey-fallback series (GPU, mesa-frames, FLAME GPU 2) are legible.
FRAMEWORK_COLORS = {
    "AMBER (GPU)":         "#0b3d91",   # deep navy — the new headline series
    "AMBER (vectorized)": "#2563eb",   # blue — CPU star
    "AMBER (loop)":       "#60a5fa",   # light blue
    "mesa-frames":        "#06b6d4",   # cyan (GPU-capable dataframe engine)
    "FLAME GPU 2":        "#d946ef",   # magenta (the other GPU framework)
    "Agents.jl":          "#16a34a",   # green
    "SimPy":              "#a855f7",   # purple
    "Melodie":            "#f97316",   # orange
    "AgentPy":            "#ef4444",   # red
    "Mesa":               "#78716c",   # stone
}

MODEL_ORDER = ["wealth_transfer", "random_walk", "sir_epidemic", "schelling"]
MODEL_LABELS = {
    "wealth_transfer": "Wealth Transfer",
    "random_walk": "Random Walk",
    "sir_epidemic": "SIR Epidemic",
    "schelling": "Schelling Segregation",
}


def _load_results(
    path: Path,
) -> Tuple[Dict[Tuple[str, str, int], float], List[int], str]:
    """Return {(framework, model, n_agents): execution_time}, sorted agent
    counts actually present, and the JSON's generated_at timestamp."""
    data = json.loads(path.read_text())
    results: Dict[Tuple[str, str, int], float] = {}
    agent_counts: set[int] = set()
    for row in data["results"]:
        t = row.get("execution_time")
        if t is None or t <= 0:
            continue
        key = (row["framework"], row["model"], row["n_agents"])
        results[key] = t
        agent_counts.add(row["n_agents"])
    return results, sorted(agent_counts), data.get("generated_at", "unknown")


def _write_chart(
    results: Dict[Tuple[str, str, int], float],
    agent_counts: List[int],
    models: List[str],
    path: Path,
    generated_at: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
    if len(models) == 1:
        axes = [axes]

    present_frameworks: set[str] = set()
    for ax, model in zip(axes, models):
        for framework in FRAMEWORK_ORDER:
            xs: List[int] = []
            ys: List[float] = []
            for n in agent_counts:
                t = results.get((framework, model, n))
                if t is not None and t > 0:
                    xs.append(n)
                    ys.append(t)
            if not xs:
                continue
            present_frameworks.add(framework)
            # Emphasise the two AMBER headline series (GPU + vectorized).
            if framework == "AMBER (GPU)":
                lw, alpha = 3.0, 1.0
            elif framework == "AMBER (vectorized)":
                lw, alpha = 2.5, 1.0
            else:
                lw, alpha = 1.5, 0.8
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=lw,
                alpha=alpha,
                color=FRAMEWORK_COLORS.get(framework, "gray"),
                label=framework,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of agents")
        ax.set_ylabel("Execution time (s)")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"ABM framework scaling — {len(present_frameworks)} frameworks "
        f"(incl. AMBER GPU) across {len(models)} models",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    print(f"  source generated_at: {generated_at}")
    print(f"  frameworks plotted: {sorted(present_frameworks)}")
    print(f"  models plotted: {models}")
    print(f"  agent counts: {agent_counts}")


def _parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=repo / "benchmarks/results/benchmark_results_10M.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=repo / "benchmarks/results/scaling_chart.png",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=MODEL_ORDER,
        help="subset/order of models to plot (default: all 4).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results, agent_counts, generated_at = _load_results(args.input)
    models = [m for m in args.models if any(
        (fw, m, n) in results for fw in FRAMEWORK_ORDER for n in agent_counts
    )]
    _write_chart(results, agent_counts, models, args.output, generated_at)


if __name__ == "__main__":
    main()
