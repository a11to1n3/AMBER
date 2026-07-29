#!/usr/bin/env python3
"""Plots + detailed MD for the multi-framework scaling campaign (10 frameworks)."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Prefer paper-fix-work reconciled artifact; fall back to benchmarks/
REPO = Path(__file__).resolve().parents[2]
CANDIDATES = [
    REPO / "paper-fix-work" / "artifacts" / "benchmark_results_all5090_reconciled.json",
    REPO / "paper-fix-work" / "artifacts" / "benchmark_results_all5090.json",
    REPO / "benchmarks" / "results" / "benchmark_results_all.json",
]
OUT_DIR = REPO / "paper-fix-work" / "campaign_results"
FIGS = OUT_DIR / "figs_all_frameworks"

BLUE = "#1f4e79"
COPPER = "#c4713b"
GREEN = "#2f6b4f"
RED = "#a33b3b"
GRAY = "#5c6670"
GRAY_LIGHT = "#c5ccd3"
BG = "#fbfaf8"

# Stable colors / markers for 10 frameworks
FRAMEWORK_STYLE = {
    "AMBER (GPU)": (BLUE, "o", 2.4),
    "AMBER (vectorized)": ("#5b8fbf", "s", 1.8),
    "AMBER (loop)": ("#8aa9c4", "^", 1.4),
    "FLAME GPU 2": (COPPER, "D", 2.4),
    "mesa-frames": (GREEN, "v", 1.6),
    "Agents.jl": ("#6b4c9a", "P", 1.8),
    "SimPy": ("#b07d4f", "x", 1.2),
    "Melodie": ("#4a7c7a", ">", 1.2),
    "AgentPy": ("#8b5e6b", "<", 1.2),
    "Mesa": (GRAY, "h", 1.2),
}

MODEL_TITLE = {
    "wealth_transfer": "Wealth transfer",
    "random_walk": "Random walk",
    "sir_epidemic": "SIR epidemic (cell-list)",
    "schelling": "Schelling segregation",
}
MODEL_ORDER = ["wealth_transfer", "random_walk", "sir_epidemic", "schelling"]
FW_ORDER = [
    "AMBER (GPU)", "AMBER (vectorized)", "AMBER (loop)",
    "FLAME GPU 2", "mesa-frames", "Agents.jl",
    "SimPy", "Melodie", "AgentPy", "Mesa",
]


def style():
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": "#1a1a1a",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 7.5,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRAY_LIGHT,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.75,
        "lines.linewidth": 1.6,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": BG,
    })


def load_data():
    for p in CANDIDATES:
        if p.exists():
            data = json.loads(p.read_text())
            return p, data
    raise FileNotFoundError("No all-framework benchmark JSON found")


def index_rows(results):
    by = defaultdict(dict)  # model -> framework -> sorted list of (n, time, row)
    for r in results:
        n = r.get("n_agents") or r.get("n") or r.get("population")
        t = r.get("execution_time")
        if n is None or t is None:
            continue
        by[r["model"]][r["framework"]] = by[r["model"]].get(r["framework"], [])
        by[r["model"]][r["framework"]].append((int(n), float(t), r))
    for model in by:
        for fw in by[model]:
            by[model][fw].sort(key=lambda x: x[0])
    return by


def fmt_time(t: float) -> str:
    if t is None:
        return "—"
    if t < 0.001:
        return f"{t * 1e6:.0f} µs"
    if t < 1:
        return f"{t * 1e3:.1f} ms"
    if t < 100:
        return f"{t:.2f} s"
    return f"{t:.1f} s"


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", FIGS / f"{name}.png")


def plot_scaling_panels(by, meta):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    axes = axes.ravel()
    for ax, model in zip(axes, MODEL_ORDER):
        for fw in FW_ORDER:
            pts = by.get(model, {}).get(fw)
            if not pts:
                continue
            color, marker, lw = FRAMEWORK_STYLE.get(fw, (GRAY, "o", 1.2))
            ns = [p[0] for p in pts]
            ts = [p[1] for p in pts]
            ax.plot(ns, ts, marker=marker, color=color, lw=lw, markersize=5, label=fw)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(MODEL_TITLE.get(model, model))
        ax.set_xlabel("N agents")
        ax.set_ylabel("wall time (s)")
    # shared legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    # collect from all
    hmap = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            hmap[lab] = h
    ordered = [(hmap[f], f) for f in FW_ORDER if f in hmap]
    fig.legend(
        [h for h, _ in ordered], [lab for _, lab in ordered],
        loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02),
    )
    gpu = meta.get("note", "")
    fig.suptitle(
        "Multi-framework scaling (50 steps) — reconciled all-framework campaign",
        fontsize=12, y=1.06,
    )
    fig.tight_layout()
    save(fig, "allfw_01_scaling_4panel")


def plot_1m_bar(by):
    """Bar chart at 1M where available."""
    n_target = 1_000_000
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    axes = axes.ravel()
    for ax, model in zip(axes, MODEL_ORDER):
        vals = []
        labels = []
        colors = []
        for fw in FW_ORDER:
            pts = by.get(model, {}).get(fw) or []
            hit = next((t for n, t, _ in pts if n == n_target), None)
            if hit is None:
                continue
            vals.append(hit)
            labels.append(fw)
            colors.append(FRAMEWORK_STYLE.get(fw, (GRAY, "o", 1))[0])
        if not vals:
            ax.set_title(MODEL_TITLE[model] + " (no 1M rows)")
            ax.axis("off")
            continue
        y = np.arange(len(vals))
        ax.barh(y, vals, color=colors, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("wall time (s)")
        ax.set_title(MODEL_TITLE[model] + " @ 1M")
        for i, v in enumerate(vals):
            ax.text(v * 1.05, i, fmt_time(v), va="center", fontsize=7, color=GRAY)
    fig.suptitle("Framework comparison at 1,000,000 agents", fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "allfw_02_bar_1m")


def plot_10m_endpoints(by):
    n_target = 10_000_000
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    models = []
    amber = []
    flame = []
    for model in MODEL_ORDER:
        a = next((t for n, t, _ in (by.get(model, {}).get("AMBER (GPU)") or []) if n == n_target), None)
        f = next((t for n, t, _ in (by.get(model, {}).get("FLAME GPU 2") or []) if n == n_target), None)
        if a is None and f is None:
            continue
        models.append(MODEL_TITLE[model])
        amber.append(a if a is not None else np.nan)
        flame.append(f if f is not None else np.nan)
    x = np.arange(len(models))
    w = 0.36
    ax.bar(x - w / 2, amber, width=w, color=BLUE, label="AMBER (GPU)")
    ax.bar(x + w / 2, flame, width=w, color=COPPER, label="FLAME GPU 2")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("wall time (s)")
    ax.set_title("10M-agent endpoints (reconciled campaign)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    for i, (a, f) in enumerate(zip(amber, flame)):
        if not np.isnan(a):
            ax.text(i - w / 2, a * 1.08, fmt_time(a), ha="center", fontsize=7, color=BLUE)
        if not np.isnan(f):
            ax.text(i + w / 2, f * 1.08, fmt_time(f), ha="center", fontsize=7, color=COPPER)
    fig.tight_layout()
    save(fig, "allfw_03_10m_amber_flame")


def plot_speedup_heatmap(by):
    """Speedup of AMBER (GPU) vs each peer at largest shared N."""
    peers = [f for f in FW_ORDER if f != "AMBER (GPU)"]
    data = np.full((len(MODEL_ORDER), len(peers)), np.nan)
    annot = [[""] * len(peers) for _ in MODEL_ORDER]
    for i, model in enumerate(MODEL_ORDER):
        amber_pts = {n: t for n, t, _ in (by.get(model, {}).get("AMBER (GPU)") or [])}
        for j, peer in enumerate(peers):
            peer_pts = {n: t for n, t, _ in (by.get(model, {}).get(peer) or [])}
            shared = sorted(set(amber_pts) & set(peer_pts))
            if not shared:
                continue
            n = shared[-1]
            sp = peer_pts[n] / amber_pts[n]
            data[i, j] = sp
            annot[i][j] = f"{sp:.1f}×\n@{n/1e6:.0f}M" if n >= 1e6 else f"{sp:.1f}×\n@{n/1e3:.0f}k"

    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    # log-ish color via log10, clip
    plot = np.log10(np.clip(data, 0.05, 200))
    im = ax.imshow(plot, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=2)
    ax.set_xticks(range(len(peers)))
    ax.set_xticklabels(peers, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_TITLE[m] for m in MODEL_ORDER])
    for i in range(len(MODEL_ORDER)):
        for j in range(len(peers)):
            if annot[i][j]:
                ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=7, color="#1a1a1a")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log10(peer / AMBER GPU)  >0 AMBER faster")
    ax.set_title("AMBER (GPU) speedup vs peers (largest shared N)")
    fig.tight_layout()
    save(fig, "allfw_04_speedup_heatmap")


def write_md(path_src, data, by):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = data["results"]
    recon = data.get("reconciliation") or {}
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# Multi-framework scaling results (all frameworks)")
    w()
    w(f"**Source artifact:** `{path_src}`  ")
    w(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    w(f"**Hardware (campaign):** NVIDIA **RTX 5090** (paper all-framework / reconciled endpoints)  ")
    w(f"**Steps:** {data.get('n_steps', 50)}  ")
    w(f"**Runs (nominal):** {data.get('runs', 10)}  ")
    w(f"**Timing:** {data.get('timing', '')}  ")
    w(f"**Agent counts:** {data.get('agent_counts', [])}  ")
    w(f"**Rows:** {len(results)}")
    w()
    w("This is the **full multi-framework** benchmark (10 implementations), distinct from the")
    w("semantic/monitor package on host_a (RTX 3090) (RTX 3090). The 10M AMBER/FLAME endpoints")
    w("in the reconciled file supersede historical trimmed means for those eight keys.")
    w()
    if recon:
        w("### Reconciliation")
        w()
        w(f"- Historical source: `{recon.get('historical_source')}`  ")
        w(f"- Authoritative endpoints: `{recon.get('authoritative_endpoint_source')}`  ")
        w(f"- Rule: {recon.get('rule')}  ")
        w(f"- Historical timing: {recon.get('historical_timing_scope')}  ")
        w(f"- Final endpoints: {recon.get('final_endpoint_timing')}")
        w()

    w("## Frameworks")
    w()
    w("| Framework | Role |")
    w("|---|---|")
    w("| AMBER (GPU) | Native GPU path (`.gpu().run()` / private fast) |")
    w("| AMBER (vectorized) | Columnar CPU view API |")
    w("| AMBER (loop) | OOP / per-agent Python loop |")
    w("| FLAME GPU 2 | RTC CUDA reference accelerator |")
    w("| mesa-frames | Polars AgentSet |")
    w("| Agents.jl | Julia ABM |")
    w("| SimPy | Discrete-event |")
    w("| Melodie | Python ABM |")
    w("| AgentPy | Python ABM |")
    w("| Mesa | Python ABM |")
    w()

    w("## Scaling plots")
    w()
    w("![4-panel scaling](figs_all_frameworks/allfw_01_scaling_4panel.png)")
    w()
    w("![1M bars](figs_all_frameworks/allfw_02_bar_1m.png)")
    w()
    w("![10M AMBER vs FLAME](figs_all_frameworks/allfw_03_10m_amber_flame.png)")
    w()
    w("![Speedup heatmap](figs_all_frameworks/allfw_04_speedup_heatmap.png)")
    w()

    # Full tables per model
    all_ns = sorted({
        n for model in by.values() for pts in model.values() for n, _, _ in pts
    })

    for model in MODEL_ORDER:
        w(f"## {MODEL_TITLE[model]}")
        w()
        header = "| Framework | " + " | ".join(str(n) for n in all_ns) + " |"
        sep = "|---|" + "|".join(["---:"] * len(all_ns)) + "|"
        w(header)
        w(sep)
        for fw in FW_ORDER:
            pts = {n: t for n, t, _ in (by.get(model, {}).get(fw) or [])}
            if not pts:
                continue
            cells = [fmt_time(pts[n]) if n in pts else "—" for n in all_ns]
            w(f"| {fw} | " + " | ".join(cells) + " |")
        w()
        # speedup vs AMBER GPU at max shared
        w(f"### Relative to AMBER (GPU) (largest shared N)")
        w()
        w("| Peer | Shared N | Peer time | AMBER GPU | Peer / AMBER |")
        w("|---|---:|---:|---:|---:|")
        amber_pts = {n: t for n, t, _ in (by.get(model, {}).get("AMBER (GPU)") or [])}
        for fw in FW_ORDER:
            if fw == "AMBER (GPU)":
                continue
            peer_pts = {n: t for n, t, _ in (by.get(model, {}).get(fw) or [])}
            shared = sorted(set(amber_pts) & set(peer_pts))
            if not shared:
                continue
            n = shared[-1]
            w(
                f"| {fw} | {n} | {fmt_time(peer_pts[n])} | {fmt_time(amber_pts[n])} | "
                f"**{peer_pts[n]/amber_pts[n]:.2f}×** |"
            )
        w()

    # 10M detail from snapshot if present
    w("## 10M AMBER / FLAME endpoints (authoritative)")
    w()
    w("| Model | AMBER (GPU) | FLAME GPU 2 | FLAME / AMBER | AMBER / FLAME |")
    w("|---|---:|---:|---:|---:|")
    for model in MODEL_ORDER:
        a = next((t for n, t, _ in (by.get(model, {}).get("AMBER (GPU)") or []) if n == 10_000_000), None)
        f = next((t for n, t, _ in (by.get(model, {}).get("FLAME GPU 2") or []) if n == 10_000_000), None)
        if a is None and f is None:
            continue
        if a and f:
            w(f"| {MODEL_TITLE[model]} | {fmt_time(a)} | {fmt_time(f)} | {f/a:.2f}× | {a/f:.2f}× |")
        else:
            w(f"| {MODEL_TITLE[model]} | {fmt_time(a) if a else '—'} | {fmt_time(f) if f else '—'} | — | — |")
    w()
    w("Paper headline range for wealth / random walk / SIR is typically **1.77×–2.05×** "
      "FLAME-over-AMBER or the inverse depending on framing; Schelling is exploratory "
      "because FLAME setup includes Python per-agent init inside the timed region.")
    w()

    w("## Per-row provenance (excerpt)")
    w()
    w("| Framework | Model | N | mean (s) | median | runs | source_campaign |")
    w("|---|---|---:|---:|---:|---:|---|")
    # show only 1M and 10M for brevity in excerpt, full JSON has all
    for r in sorted(results, key=lambda x: (x["model"], x["framework"], x.get("n_agents") or 0)):
        n = r.get("n_agents")
        if n not in (1_000_000, 10_000_000):
            continue
        w(
            f"| {r['framework']} | {r['model']} | {n} | {r.get('execution_time')} | "
            f"{r.get('median')} | {r.get('runs')} | {r.get('source_campaign', '')} |"
        )
    w()
    w("Full 142 rows: see JSON. Intervals (`ci95_median`, `iqr`, `raw_samples`) are in the artifact.")
    w()
    w("## How to reproduce")
    w()
    w("```bash")
    w("export CUDA_PATH=/usr/local/cuda   # FLAME RTC")
    w("python benchmarks/run_all_frameworks.py \\")
    w("  --agents 1000 10000 100000 1000000 10000000 \\")
    w("  --steps 50 --runs 10")
    w("# 10M AMBER+FLAME only (snapshot campaign):")
    w("python benchmarks/run_all_frameworks.py \\")
    w("  --agents 10000000 --steps 50 --runs 10 \\")
    w("  --frameworks \"AMBER (GPU)\" \"FLAME GPU 2\" \\")
    w("  --tag snapshot_correct_10run_10m --budget 120")
    w("python paper-fix-work/scripts/reconcile_all_framework_benchmark.py")
    w("```")
    w()
    w("## Relation to host_a (RTX 3090) campaign")
    w()
    w("| Campaign | Host | Focus | Report |")
    w("|---|---|---|---|")
    w("| **All frameworks (this file)** | RTX **5090** | 10 frameworks × 4 models × scale | this MD |")
    w("| Semantic / monitor package | RTX **3090** (`host_a (RTX 3090)`) | C1–C4 attestation, activation, overhead | `DETAILED_RESULTS_host_a.md` |")
    w()
    w("Do not mix 3090 and 5090 wall-clock numbers in the same speedup sentence without labeling the host.")
    w()

    out = OUT_DIR / "ALL_FRAMEWORKS_RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)
    return out


def main():
    style()
    path_src, data = load_data()
    print("loaded", path_src)
    by = index_rows(data["results"])
    FIGS.mkdir(parents=True, exist_ok=True)
    plot_scaling_panels(by, data)
    plot_1m_bar(by)
    plot_10m_endpoints(by)
    plot_speedup_heatmap(by)
    write_md(path_src, data, by)


if __name__ == "__main__":
    main()
