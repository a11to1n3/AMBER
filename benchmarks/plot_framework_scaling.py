"""Plot the Round 75 matched-rule RTX 5090 framework benchmark."""

import argparse
from io import StringIO
import json
from pathlib import Path
from statistics import median

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, StrMethodFormatter


DATA = Path(__file__).resolve().parent / "results" / "framework_scaling_fix11.json"
POPULATIONS = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
WORKLOADS = (
    ("wealth", "Wealth transfer"),
    ("walk", "Random walk"),
    ("sir", "SIR epidemic"),
    ("schelling", "Schelling segregation"),
)
SERIES = (
    ("AMBER GPU", "AMBER GPU", "#00659C", "o", "-"),
    ("AMBER vectorized CPU", "AMBER vectorized CPU", "#00659C", "s", "--"),
    ("AMBER oop CPU", "AMBER loop CPU", "#00659C", "^", ":"),
    ("FLAME GPU 2", "FLAME GPU 2", "#BC4700", "D", "-"),
    ("mesa-frames", "mesa-frames", "#987200", "h", "--"),
    ("Agents.jl", "Agents.jl", "#007E67", "v", ":"),
    ("AgentPy", "AgentPy", "#8D4D83", "p", "-."),
    ("Mesa", "Mesa", "#287697", "P", "--"),
    ("Melodie", "Melodie", "#565656", ">", ":"),
    ("SimPy", "SimPy", "#443187", "<", "-."),
)


def plot(report):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "svg.hashsalt": "ambr-framework-scaling-fix11",
    })
    cells = {
        (row["workload"], row["implementation"], row["agents"]): row
        for row in report["timing_cells"]
    }
    figure, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.5, 5.4),
                                layout="constrained")
    for panel, (axis, (workload, label)) in enumerate(zip(axes.flat, WORKLOADS)):
        axis.text(0, 1.03, f"({chr(97 + panel)}) {label}", transform=axis.transAxes,
                  fontweight="bold", ha="left", va="bottom")
        for key, name, color, marker, line in SERIES:
            populations, centers, lower, upper = [], [], [], []
            for population in POPULATIONS:
                row = cells[workload, key, population]
                if row["status"] != "complete":
                    break
                samples = row["samples"]
                center = median(samples)
                populations.append(population)
                centers.append(center)
                lower.append(center - min(samples))
                upper.append(max(samples) - center)
            amber = key.startswith("AMBER ")
            axis.errorbar(populations, centers, yerr=[lower, upper], label=name, color=color,
                          marker=marker, markersize=4.4 if amber else 3.7,
                          markerfacecolor=color if amber else "white", markeredgewidth=0.85,
                          linestyle=line, linewidth=1.65 if amber else 0.95,
                          elinewidth=0.65, capsize=1.15, zorder=4 if amber else 3)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(760, 13_000_000)
        axis.set_ylim(0.001, 1_000)
        axis.xaxis.set_major_locator(FixedLocator(POPULATIONS))
        axis.set_xticklabels(["1k", "10k", "100k", "1M", "10M"])
        axis.yaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 1, 10, 100, 1_000]))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_locator(NullLocator())
        axis.grid(which="major", color="#DDDDDD", linewidth=0.45)
        axis.tick_params(length=3, width=0.65)
        if panel >= 2:
            axis.set_xlabel("Agents")
        if panel % 2 == 0:
            axis.set_ylabel("End-to-end time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=5, frameon=False,
                  fontsize=8, handlelength=2.3, columnspacing=1.25)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA.with_suffix(".svg"))
    args = parser.parse_args()
    figure = plot(json.loads(DATA.read_text()))
    with StringIO() as svg:
        figure.savefig(svg, format="svg", metadata={"Date": None})
        args.output.write_text("\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
