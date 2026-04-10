#!/usr/bin/env python3
"""Master benchmark runner: all six frameworks side by side.

Produces a single reproducer for the headline performance table that
compares AMBER against every ABM framework the README / paper reference:

  * AMBER (loop)        — per-agent Python loops (``benchmarks/models/amber_models.py``)
  * AMBER (vectorized)  — view API (same file, the new classes)
  * AgentPy             — ``benchmarks/models/agentpy_models.py``
  * Mesa                — ``benchmarks/models/mesa_models.py``
  * SimPy               — ``benchmarks/models/simpy_models.py``
  * Melodie             — ``benchmarks/models/melodie_models.py``
  * Agents.jl           — ``benchmarks/models/agentsjl_models.jl`` (via ``julia`` subprocess)

The Python frameworks are timed in-process. Agents.jl is invoked as a
subprocess and its stdout is parsed. All numbers are averages of multiple
runs per configuration.

Outputs (in ``benchmarks/results/``):

  * ``benchmark_results_all.json``   — raw per-run timings
  * ``summary_table_all.md``         — side-by-side markdown table
  * ``scaling_chart_all.png``        — log-log scaling plot per model

Usage::

    python benchmarks/run_all_frameworks.py
    python benchmarks/run_all_frameworks.py --quick   # shorter run
    python benchmarks/run_all_frameworks.py --agents 500 1000 5000 --steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
MODELS_DIR = BENCH_DIR / "models"
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(BENCH_DIR))


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "wealth_transfer": {"initial_wealth": 1},
    "random_walk": {"world_size": 100, "speed": 1.0},
    "sir_epidemic": {
        "initial_infected": 5,
        "world_size": 100,
        "movement_speed": 2.0,
        "infection_radius": 5.0,
        "transmission_rate": 0.1,
        "recovery_time": 14,
    },
}

# --------------------------------------------------------------------------- #
# Timing primitives
# --------------------------------------------------------------------------- #

def _time(callable_: Callable[[], None], runs: int) -> float:
    """Run ``callable_`` ``runs`` times and return the mean wall-clock time."""
    ts: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        callable_()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    # Trim the slowest run when we have >= 3 samples; noise is asymmetric.
    if len(ts) >= 3:
        ts = ts[:-1]
    return sum(ts) / len(ts)


# --------------------------------------------------------------------------- #
# Framework: AMBER (loop) and AMBER (vectorized)
# --------------------------------------------------------------------------- #

def _bench_amber(
    model_name: str,
    n: int,
    steps: int,
    runs: int,
    variant: str,  # "loop" or "vectorized"
) -> Optional[float]:
    from models.amber_models import AMBER_MODELS, AMBER_VECTORIZED_MODELS

    registry = AMBER_MODELS if variant == "loop" else AMBER_VECTORIZED_MODELS
    cls = registry[model_name]
    cfg = {"n": n, "steps": steps, "show_progress": False}
    cfg.update(MODEL_CONFIGS[model_name])

    def _run():
        cls(cfg).run()

    _run()  # warm up (polars LazyFrame caches, AMBER add_agents schema)
    return _time(_run, runs)


# --------------------------------------------------------------------------- #
# Framework: AgentPy
# --------------------------------------------------------------------------- #

def _bench_agentpy(
    model_name: str, n: int, steps: int, runs: int
) -> Optional[float]:
    try:
        from models.agentpy_models import AGENTPY_MODELS
    except ImportError:
        return None
    cls = AGENTPY_MODELS[model_name]
    cfg = {"n": n, "steps": steps}
    cfg.update(MODEL_CONFIGS[model_name])

    def _run():
        cls(cfg).run()

    _run()
    return _time(_run, runs)


# --------------------------------------------------------------------------- #
# Framework: Mesa
# --------------------------------------------------------------------------- #

def _bench_mesa(
    model_name: str, n: int, steps: int, runs: int
) -> Optional[float]:
    try:
        from models.mesa_models import MESA_MODELS
    except ImportError:
        return None
    cls = MESA_MODELS[model_name]
    cfg = {"n": n, "steps": steps}
    cfg.update(MODEL_CONFIGS[model_name])

    def _run():
        cls(**cfg).run()

    _run()
    return _time(_run, runs)


# --------------------------------------------------------------------------- #
# Framework: SimPy
# --------------------------------------------------------------------------- #

def _bench_simpy(
    model_name: str, n: int, steps: int, runs: int
) -> Optional[float]:
    try:
        from models import simpy_models
    except ImportError:
        return None
    fn_map = {
        "wealth_transfer": simpy_models.run_wealth_benchmark,
        "random_walk": simpy_models.run_walk_benchmark,
        "sir_epidemic": simpy_models.run_sir_benchmark,
    }
    fn = fn_map[model_name]

    def _run():
        # simpy's sir helper prints "Final Infected: .../..." — silence it.
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            fn(n=n, steps=steps)

    _run()
    return _time(_run, runs)


# --------------------------------------------------------------------------- #
# Framework: Melodie
# --------------------------------------------------------------------------- #

def _bench_melodie(
    model_name: str, n: int, steps: int, runs: int
) -> Optional[float]:
    try:
        import numpy as np
        import Melodie  # noqa: F401  (surface probe)
        from models import melodie_models as mm
    except ImportError:
        return None

    config = mm.Melodie.Config(
        project_name="MelodieBenchmark",
        project_root=".",
        sqlite_folder=".",
        output_folder=".",
        input_folder=".",
    )

    scenario_map = {
        "wealth_transfer": (mm.WealthModel, mm.WealthScenario),
        "random_walk": (mm.WalkModel, mm.WalkScenario),
        "sir_epidemic": (mm.SIRModel, mm.SIRScenario),
    }
    model_cls, scenario_cls = scenario_map[model_name]

    def _run_once():
        sqlite_path = Path("MelodieBenchmark.sqlite")
        if sqlite_path.exists():
            sqlite_path.unlink()
        scenario = scenario_cls()
        scenario.manager = None
        scenario.periods = steps
        scenario.agent_num = n
        scenario.id = 0

        model = model_cls(config, scenario)
        model.setup()
        for i in range(n):
            agent = model.agent_list.add()
            agent.id = i
            agent.setup()
        for i in range(n):
            agent = model.agent_list[i]
            agent.id = i
            if hasattr(agent, "wealth"):
                agent.wealth = 1
            if hasattr(agent, "x"):
                agent.x = np.random.uniform(0, 100)
                agent.y = np.random.uniform(0, 100)
            if hasattr(agent, "status"):
                agent.status = 1 if i < 5 else 0
        model.run()

    _run_once()  # warm up
    return _time(_run_once, runs)


# --------------------------------------------------------------------------- #
# Framework: Agents.jl (subprocess)
# --------------------------------------------------------------------------- #

_AGENTSJL_LINE = re.compile(r"^\s*(\d+)\s+agents:\s+([\d.e+-]+)s\s*$")


def _run_agentsjl(agent_counts: List[int], steps: int) -> Dict[Tuple[str, int], float]:
    """Run the Agents.jl standalone script once and parse all timings.

    Returns a dict keyed by (model, n) → seconds. Because Julia bears
    JIT compilation on the first call per benchmark function, we run the
    script twice and keep the second run's numbers.
    """
    results: Dict[Tuple[str, int], float] = {}
    jl_path = MODELS_DIR / "agentsjl_models.jl"
    if not jl_path.exists():
        return results

    # Check julia is on PATH
    try:
        subprocess.run(
            ["julia", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return results

    # The Julia script accepts ``--agents`` and ``--steps`` so we can hold
    # it to the exact same configuration as the Python frameworks. Run it
    # twice: the first run warms Julia's JIT, the second reflects
    # steady-state performance (which is what we compare against the
    # Python frameworks' warm-up-then-measure protocol).
    agents_arg = ",".join(str(n) for n in agent_counts)
    def _run_once() -> str:
        proc = subprocess.run(
            [
                "julia",
                str(jl_path),
                "--agents",
                agents_arg,
                "--steps",
                str(steps),
            ],
            cwd=str(MODELS_DIR),
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Agents.jl subprocess failed: {proc.stderr[:500]}"
            )
        return proc.stdout

    try:
        _run_once()  # warm (JIT)
        stdout = _run_once()
    except Exception as e:
        print(f"  Agents.jl: SKIPPED ({e})")
        return results

    # The script prints sections like:
    #   wealth_transfer:
    #     100 agents: 0.0s
    #     500 agents: 0.001s
    # Map the section header onto our model names.
    current_model: Optional[str] = None
    for line in stdout.splitlines():
        stripped = line.strip().rstrip(":").lower()
        if stripped in MODEL_CONFIGS:
            current_model = stripped
            continue
        m = _AGENTSJL_LINE.match(line)
        if m and current_model is not None:
            n = int(m.group(1))
            sec = float(m.group(2))
            if n in agent_counts:
                results[(current_model, n)] = sec
    return results


# --------------------------------------------------------------------------- #
# Chart and table generation
# --------------------------------------------------------------------------- #

FRAMEWORK_ORDER = [
    "AMBER (vectorized)",
    "AMBER (loop)",
    "Agents.jl",
    "SimPy",
    "Melodie",
    "AgentPy",
    "Mesa",
]

FRAMEWORK_COLORS = {
    "AMBER (vectorized)": "#2563eb",   # blue — the star
    "AMBER (loop)":        "#60a5fa",  # light blue
    "Agents.jl":           "#16a34a",  # green
    "SimPy":               "#a855f7",  # purple
    "Melodie":             "#f97316",  # orange
    "AgentPy":             "#ef4444",  # red
    "Mesa":                "#78716c",  # stone
}


def _write_json(
    results: Dict[Tuple[str, str, int], Optional[float]],
    steps: int,
    agent_counts: List[int],
    path: Path,
) -> None:
    flat = []
    for (framework, model, n), t in results.items():
        if t is None:
            continue
        flat.append(
            {
                "framework": framework,
                "model": model,
                "n_agents": n,
                "n_steps": steps,
                "execution_time": round(t, 4),
                "time_per_step": round(t / steps, 6),
            }
        )
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "agent_counts": agent_counts,
                "n_steps": steps,
                "results": flat,
            },
            indent=2,
        )
    )


def _write_markdown(
    results: Dict[Tuple[str, str, int], Optional[float]],
    agent_counts: List[int],
    path: Path,
) -> None:
    def _cell(val: Optional[float]) -> str:
        if val is None:
            return "—"
        if val >= 1:
            return f"{val:.2f}s"
        return f"{val * 1000:.0f}ms"

    lines: List[str] = []
    lines.append("# Benchmark results — all frameworks")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} on "
        f"`python {sys.version_info.major}.{sys.version_info.minor}`. "
        f"Lower is better. Times are wall-clock, averaged over multiple runs._"
    )
    lines.append("")

    for model in MODEL_CONFIGS:
        pretty = model.replace("_", " ").title()
        lines.append(f"## {pretty}")
        lines.append("")
        header = ["Framework"] + [str(n) for n in agent_counts]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for framework in FRAMEWORK_ORDER:
            row_vals = [
                _cell(results.get((framework, model, n))) for n in agent_counts
            ]
            if all(v == "—" for v in row_vals):
                continue
            label = framework
            if framework == "AMBER (vectorized)":
                label = "**AMBER (vectorized)**"
            lines.append("| " + label + " | " + " | ".join(row_vals) + " |")
        lines.append("")

    # Speedup summary (AMBER vectorized baseline)
    lines.append("## Speedup of AMBER (vectorized) vs other frameworks")
    lines.append("")
    lines.append("| Framework | wealth_transfer | random_walk | sir_epidemic |")
    lines.append("|---|---|---|---|")
    for framework in FRAMEWORK_ORDER:
        if framework == "AMBER (vectorized)":
            continue
        cells: List[str] = [framework]
        for model in MODEL_CONFIGS:
            speedups = []
            for n in agent_counts:
                vec_t = results.get(("AMBER (vectorized)", model, n))
                other_t = results.get((framework, model, n))
                if vec_t and other_t:
                    speedups.append(other_t / vec_t)
            if speedups:
                geo = (sum(speedups) / len(speedups))  # arithmetic mean on ratios
                cells.append(f"{geo:.1f}×")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


def _write_chart(
    results: Dict[Tuple[str, str, int], Optional[float]],
    agent_counts: List[int],
    path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except ImportError:
        print("  matplotlib not available — skipping chart")
        return

    fig, axes = plt.subplots(1, len(MODEL_CONFIGS), figsize=(5 * len(MODEL_CONFIGS), 4.5))
    if len(MODEL_CONFIGS) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODEL_CONFIGS):
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
            lw = 2.5 if framework == "AMBER (vectorized)" else 1.5
            alpha = 1.0 if framework == "AMBER (vectorized)" else 0.8
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
        ax.set_title(model.replace("_", " ").title())
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"ABM framework scaling — {sum(1 for f in FRAMEWORK_ORDER if any((f, m, n) in results for m in MODEL_CONFIGS for n in agent_counts))} frameworks",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", type=int, nargs="+", default=None)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--frameworks", type=str, nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.quick:
        agent_counts = [100, 500, 1000]
        steps = 20
        runs = 2
    else:
        agent_counts = args.agents or [500, 1000, 5000]
        steps = args.steps
        runs = args.runs

    selected_frameworks = set(args.frameworks) if args.frameworks else set(FRAMEWORK_ORDER)

    print(f"Benchmark configuration")
    print(f"  agent counts : {agent_counts}")
    print(f"  steps        : {steps}")
    print(f"  runs/config  : {runs}")
    print(f"  frameworks   : {sorted(selected_frameworks)}")
    print()

    results: Dict[Tuple[str, str, int], Optional[float]] = {}

    # Python-hosted frameworks ----------------------------------------------
    py_frameworks: List[Tuple[str, Callable[..., Optional[float]]]] = []
    if "AMBER (loop)" in selected_frameworks:
        py_frameworks.append(("AMBER (loop)", lambda m, n, s, r: _bench_amber(m, n, s, r, "loop")))
    if "AMBER (vectorized)" in selected_frameworks:
        py_frameworks.append(("AMBER (vectorized)", lambda m, n, s, r: _bench_amber(m, n, s, r, "vectorized")))
    if "AgentPy" in selected_frameworks:
        py_frameworks.append(("AgentPy", _bench_agentpy))
    if "Mesa" in selected_frameworks:
        py_frameworks.append(("Mesa", _bench_mesa))
    if "SimPy" in selected_frameworks:
        py_frameworks.append(("SimPy", _bench_simpy))
    if "Melodie" in selected_frameworks:
        py_frameworks.append(("Melodie", _bench_melodie))

    for model in MODEL_CONFIGS:
        print(f"[{model}]")
        for n in agent_counts:
            print(f"  n={n}")
            for framework, fn in py_frameworks:
                try:
                    t = fn(model, n, steps, runs)
                except Exception as e:
                    print(f"    {framework:22s} ERROR  ({type(e).__name__}: {e})")
                    t = None
                results[(framework, model, n)] = t
                if t is not None:
                    print(f"    {framework:22s} {t * 1000:>9.1f} ms")
                else:
                    print(f"    {framework:22s} —")

    # Agents.jl -------------------------------------------------------------
    if "Agents.jl" in selected_frameworks:
        print("[Agents.jl] running julia subprocess…")
        jl_results = _run_agentsjl(agent_counts, steps)
        for (model, n), sec in jl_results.items():
            results[("Agents.jl", model, n)] = sec
        if jl_results:
            for (model, n), sec in sorted(jl_results.items()):
                print(f"    Agents.jl  {model:<16s} n={n:<6d}  {sec * 1000:>9.1f} ms")
        else:
            print("    Agents.jl  — (not available)")

    # Outputs ---------------------------------------------------------------
    print()
    json_path = RESULTS_DIR / "benchmark_results_all.json"
    md_path = RESULTS_DIR / "summary_table_all.md"
    chart_path = RESULTS_DIR / "scaling_chart_all.png"

    _write_json(results, steps, agent_counts, json_path)
    print(f"  wrote {json_path.relative_to(REPO_ROOT)}")
    _write_markdown(results, agent_counts, md_path)
    print(f"  wrote {md_path.relative_to(REPO_ROOT)}")
    _write_chart(results, agent_counts, chart_path)
    print(f"  wrote {chart_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
