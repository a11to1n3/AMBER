#!/usr/bin/env python3
"""Deterministic dynamic graph coordination benchmark runner.

This runner promotes the paper-local dynamic graph audits into a root
benchmark artifact.  The workload is a synchronous bounded-confidence opinion
dynamic over a deterministic sparse directed graph that is regenerated every
step.  Every row is checked against the same NumPy reference trajectory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
MODELS_DIR = BENCH_DIR / "models"
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(BENCH_DIR))

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB
C3 = 0xD6E8FEB86659FD93
C4 = 0xA5A3564E27F886D3

TAG_INIT = 101
TAG_RANDOM_EDGE = 109
TAG_DYNAMIC_EDGE = 211

FINAL_AGENT_COUNTS = [500, 1000, 5000]
FINAL_SEEDS = [42, 77, 123]
FINAL_STEPS = 20
FINAL_RAW_SAMPLE_MIN = 5


@dataclass(frozen=True)
class GraphConfig:
    n: int
    steps: int
    seed: int = 42
    degree: int = 8
    confidence: float = 0.18
    alpha: float = 0.45


@dataclass(frozen=True)
class GraphOutcome:
    opinions: np.ndarray
    mean: float
    std: float
    min_opinion: float
    max_opinion: float
    active_edges_last: int
    checksum: float


Runner = Callable[[GraphConfig], GraphOutcome]
TimedRunner = Callable[[GraphConfig, int, int], tuple[dict[str, Any], GraphOutcome]]


@dataclass(frozen=True)
class RunnerSpec:
    mode: str
    framework: str
    source: str
    fn: Runner | None = None
    timed_fn: TimedRunner | None = None


def _splitmix64_int(x: int) -> int:
    x = (x + GOLDEN) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * C1) & MASK64
    z = ((z ^ (z >> 27)) * C2) & MASK64
    return (z ^ (z >> 31)) & MASK64


def _splitmix64_np(x: np.ndarray) -> np.ndarray:
    z = x + np.uint64(GOLDEN)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(C1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(C2)
    return z ^ (z >> np.uint64(31))


def _uniform01_int(seed: int, tag: int, step: int, a: int, b: int = 0) -> float:
    x = (
        seed
        ^ ((tag * C1) & MASK64)
        ^ (((step + 1_000_003) * C2) & MASK64)
        ^ ((a * C3) & MASK64)
        ^ ((b * C4) & MASK64)
    )
    return (_splitmix64_int(x & MASK64) >> 11) * (1.0 / (1 << 53))


def _uniform01_np(seed: int, tag: int, step: int, a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    a64 = np.asarray(a, dtype=np.uint64)
    b64 = np.zeros_like(a64, dtype=np.uint64) if b is None else np.asarray(b, dtype=np.uint64)
    x = (
        np.uint64(seed & MASK64)
        ^ np.uint64((tag * C1) & MASK64)
        ^ np.uint64(((step + 1_000_003) * C2) & MASK64)
        ^ (a64 * np.uint64(C3))
        ^ (b64 * np.uint64(C4))
    )
    return (_splitmix64_np(x) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def _initial_opinions(cfg: GraphConfig) -> np.ndarray:
    ids = np.arange(cfg.n, dtype=np.int64)
    return _uniform01_np(cfg.seed, TAG_INIT, -1, ids)


def _validate_degree(cfg: GraphConfig) -> int:
    if cfg.n < 2:
        raise ValueError("n must be at least 2")
    if cfg.degree < 2 or cfg.degree % 2:
        raise ValueError("degree must be an even integer >= 2")
    degree = min(cfg.degree, cfg.n - 1)
    if degree % 2:
        degree -= 1
    if degree < 2:
        raise ValueError("effective degree must be at least 2")
    return degree


def dynamic_edges(cfg: GraphConfig, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic sparse directed edges for one step."""
    degree = _validate_degree(cfg)
    edges: set[tuple[int, int]] = set()
    step_part = ((step + 1_000_003) * C4) & MASK64
    for src in range(cfg.n):
        chosen: set[int] = set()
        salt = 0
        while len(chosen) < degree:
            raw = _splitmix64_int(
                cfg.seed
                ^ ((TAG_DYNAMIC_EDGE * C1) & MASK64)
                ^ ((TAG_RANDOM_EDGE * C2) & MASK64)
                ^ ((src * C3) & MASK64)
                ^ step_part
                ^ ((salt * C4) & MASK64)
            )
            dst = raw % cfg.n
            salt += 1
            if dst != src:
                chosen.add(dst)
        for dst in sorted(chosen):
            edges.add((src, dst))
    srcs, dsts = zip(*sorted(edges))
    return np.asarray(srcs, dtype=np.int64), np.asarray(dsts, dtype=np.int64)


def _dynamic_neighbors(cfg: GraphConfig, step: int) -> list[list[int]]:
    src, dst = dynamic_edges(cfg, step)
    neighbors: list[list[int]] = [[] for _ in range(cfg.n)]
    for s, d in zip(src.tolist(), dst.tolist()):
        neighbors[s].append(d)
    return neighbors


def _outcome(opinions: np.ndarray, active_edges_last: int) -> GraphOutcome:
    weights = np.arange(1, opinions.size + 1, dtype=np.float64)
    return GraphOutcome(
        opinions=opinions,
        mean=float(opinions.mean()),
        std=float(opinions.std()),
        min_opinion=float(opinions.min()),
        max_opinion=float(opinions.max()),
        active_edges_last=int(active_edges_last),
        checksum=float(np.dot(opinions, weights)),
    )


def run_numpy_reference(cfg: GraphConfig) -> GraphOutcome:
    opinions = _initial_opinions(cfg)
    active_edges_last = 0
    for step in range(cfg.steps):
        src, dst = dynamic_edges(cfg, step)
        src_op = opinions[src]
        dst_op = opinions[dst]
        active = np.abs(dst_op - src_op) <= cfg.confidence
        active_edges_last = int(active.sum())
        sums = np.bincount(src[active], weights=dst_op[active], minlength=cfg.n)
        counts = np.bincount(src[active], minlength=cfg.n)
        mask = counts > 0
        new_opinions = opinions.copy()
        neighbor_mean = np.zeros(cfg.n, dtype=np.float64)
        neighbor_mean[mask] = sums[mask] / counts[mask]
        new_opinions[mask] = opinions[mask] + cfg.alpha * (neighbor_mean[mask] - opinions[mask])
        opinions = new_opinions
    return _outcome(opinions, active_edges_last)


def run_columnar_polars(cfg: GraphConfig) -> GraphOutcome:
    agents = pl.DataFrame({"id": np.arange(cfg.n, dtype=np.int64), "opinion": _initial_opinions(cfg)})
    active_edges_last = 0
    for step in range(cfg.steps):
        src, dst = dynamic_edges(cfg, step)
        edges = pl.DataFrame({"src": src, "dst": dst})
        src_view = agents.select(pl.col("id").alias("src"), pl.col("opinion").alias("src_opinion"))
        dst_view = agents.select(pl.col("id").alias("dst"), pl.col("opinion").alias("dst_opinion"))
        accepted = (
            edges.join(src_view, on="src")
            .join(dst_view, on="dst")
            .filter((pl.col("dst_opinion") - pl.col("src_opinion")).abs() <= cfg.confidence)
        )
        active_edges_last = accepted.height
        if active_edges_last:
            updates = accepted.group_by("src").agg(pl.col("dst_opinion").mean().alias("neighbor_mean"))
            agents = (
                agents.join(updates, left_on="id", right_on="src", how="left")
                .with_columns(
                    pl.when(pl.col("neighbor_mean").is_not_null())
                    .then(pl.col("opinion") + cfg.alpha * (pl.col("neighbor_mean") - pl.col("opinion")))
                    .otherwise(pl.col("opinion"))
                    .alias("opinion")
                )
                .drop("neighbor_mean")
            )
    return _outcome(agents["opinion"].to_numpy(), active_edges_last)


def run_object_loop(cfg: GraphConfig) -> GraphOutcome:
    opinions = _initial_opinions(cfg)
    active_edges_last = 0
    for step in range(cfg.steps):
        neighbors = _dynamic_neighbors(cfg, step)
        new_opinions = opinions.copy()
        active_edges_last = 0
        for i, nbrs in enumerate(neighbors):
            accepted_sum = 0.0
            accepted_count = 0
            oi = opinions[i]
            for j in nbrs:
                oj = opinions[j]
                if abs(oj - oi) <= cfg.confidence:
                    accepted_sum += oj
                    accepted_count += 1
            active_edges_last += accepted_count
            if accepted_count:
                neighbor_mean = accepted_sum / accepted_count
                new_opinions[i] = oi + cfg.alpha * (neighbor_mean - oi)
        opinions = new_opinions
    return _outcome(opinions, active_edges_last)


def _snapshot_agent_opinions(agents: list[object]) -> np.ndarray:
    ordered = sorted(agents, key=lambda agent: agent.semantic_id)
    return np.array([agent.opinion for agent in ordered], dtype=np.float64)


def _assign_agent_opinions(agents: list[object], opinions: np.ndarray) -> None:
    for agent in sorted(agents, key=lambda item: item.semantic_id):
        agent.opinion = float(opinions[agent.semantic_id])


def _sync_step_objects(cfg: GraphConfig, agents: list[object], neighbors: list[list[int]]) -> int:
    opinions = _snapshot_agent_opinions(agents)
    new_opinions = opinions.copy()
    active_edges = 0
    for i, nbrs in enumerate(neighbors):
        accepted_sum = 0.0
        accepted_count = 0
        oi = opinions[i]
        for j in nbrs:
            oj = opinions[j]
            if abs(oj - oi) <= cfg.confidence:
                accepted_sum += oj
                accepted_count += 1
        active_edges += accepted_count
        if accepted_count:
            neighbor_mean = accepted_sum / accepted_count
            new_opinions[i] = oi + cfg.alpha * (neighbor_mean - oi)
    _assign_agent_opinions(agents, new_opinions)
    return active_edges


def _outcome_from_agents(agents: list[object], active_edges_last: int) -> GraphOutcome:
    return _outcome(_snapshot_agent_opinions(agents), active_edges_last)


def run_amber_object_container(cfg: GraphConfig) -> GraphOutcome:
    try:
        import ambr as am
    except Exception as exc:  # pragma: no cover - dependency-gated path
        raise RuntimeError(f"AMBER unavailable: {exc}") from exc

    class DynamicGraphAMBERAgent(am.Agent):
        pass

    class DynamicGraphAMBERModel(am.Model):
        def __init__(self, config: GraphConfig):
            super().__init__({"seed": config.seed, "show_progress": False})
            self.config = config

        def setup(self) -> None:
            opinions = _initial_opinions(self.config)
            for idx, opinion in enumerate(opinions):
                agent = DynamicGraphAMBERAgent(self, idx)
                agent.semantic_id = idx
                agent.opinion = float(opinion)
                self.add_agent(agent)
            _ = self.agents_df

        def run_aligned(self) -> GraphOutcome:
            self.setup()
            active_edges_last = 0
            for step in range(self.config.steps):
                active_edges_last = _sync_step_objects(self.config, self.agents, _dynamic_neighbors(self.config, step))
                _ = self.agents_df
            return _outcome_from_agents(self.agents, active_edges_last)

    return DynamicGraphAMBERModel(cfg).run_aligned()


def run_mesa_object_container(cfg: GraphConfig) -> GraphOutcome:
    try:
        import mesa
    except Exception as exc:  # pragma: no cover - dependency-gated path
        raise RuntimeError(f"Mesa unavailable: {exc}") from exc

    class DynamicGraphMesaAgent(mesa.Agent):
        def __init__(self, model: mesa.Model, semantic_id: int, opinion: float):
            super().__init__(model)
            self.semantic_id = semantic_id
            self.opinion = float(opinion)

    class DynamicGraphMesaModel(mesa.Model):
        def __init__(self, config: GraphConfig):
            super().__init__(seed=config.seed)
            self.config = config
            opinions = _initial_opinions(config)
            self.agent_by_id = [
                DynamicGraphMesaAgent(self, idx, float(opinion))
                for idx, opinion in enumerate(opinions)
            ]

        def run_aligned(self) -> GraphOutcome:
            active_edges_last = 0
            for step in range(self.config.steps):
                active_edges_last = _sync_step_objects(self.config, self.agent_by_id, _dynamic_neighbors(self.config, step))
            return _outcome_from_agents(self.agent_by_id, active_edges_last)

    return DynamicGraphMesaModel(cfg).run_aligned()


def run_agentpy_object_container(cfg: GraphConfig) -> GraphOutcome:
    try:
        import agentpy as ap
    except Exception as exc:  # pragma: no cover - dependency-gated path
        raise RuntimeError(f"AgentPy unavailable: {exc}") from exc

    class DynamicGraphAgentPyAgent(ap.Agent):
        pass

    class DynamicGraphAgentPyModel(ap.Model):
        def setup(self) -> None:
            config: GraphConfig = self.p["config"]
            self.config = config
            self.agents = ap.AgentList(self, config.n, DynamicGraphAgentPyAgent)
            opinions = _initial_opinions(config)
            for idx, agent in enumerate(self.agents):
                agent.semantic_id = idx
                agent.opinion = float(opinions[idx])

        def run_aligned(self) -> GraphOutcome:
            self.setup()
            active_edges_last = 0
            for step in range(self.config.steps):
                active_edges_last = _sync_step_objects(self.config, self.agents, _dynamic_neighbors(self.config, step))
            return _outcome_from_agents(self.agents, active_edges_last)

    return DynamicGraphAgentPyModel({"config": cfg}).run_aligned()


def _outcome_from_agentsjl_payload(payload: dict[str, Any]) -> GraphOutcome:
    opinions = np.asarray(payload["opinions"], dtype=np.float64)
    return GraphOutcome(
        opinions=opinions,
        mean=float(payload["mean"]),
        std=float(payload["std"]),
        min_opinion=float(payload["min_opinion"]),
        max_opinion=float(payload["max_opinion"]),
        active_edges_last=int(payload["active_edges_last"]),
        checksum=float(payload["checksum"]),
    )


def _time_agentsjl_object_container(cfg: GraphConfig, runs: int, timeout: int) -> tuple[dict[str, Any], GraphOutcome]:
    script = MODELS_DIR / "agentsjl_dynamic_graph_variants.jl"
    if not script.exists():
        raise RuntimeError(f"Agents.jl dynamic graph script missing: {script}")

    cmd = [
        "julia",
        str(script),
        "--n",
        str(cfg.n),
        "--steps",
        str(cfg.steps),
        "--seed",
        str(cfg.seed),
        "--degree",
        str(cfg.degree),
        "--confidence",
        str(cfg.confidence),
        "--alpha",
        str(cfg.alpha),
        "--runs",
        str(runs),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"Agents.jl dynamic graph failed: {proc.stderr.strip()[-1200:]}")
    stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    json_lines = [line for line in stdout_lines if line.startswith("{") and line.endswith("}")]
    if not json_lines:
        raise RuntimeError(f"Agents.jl dynamic graph produced no JSON output: {proc.stdout[-1200:]}")
    payload = json.loads(json_lines[-1])
    samples = [float(x) for x in payload["raw_samples_s"]]
    if not samples:
        raise RuntimeError("Agents.jl dynamic graph produced no timing samples")
    return _timing_from_samples(samples), _outcome_from_agentsjl_payload(payload)


def _runner_specs() -> list[RunnerSpec]:
    return [
        RunnerSpec("dynamic_numpy_reference", "NumPy", "benchmarks/run_dynamic_graph_variants.py::run_numpy_reference", fn=run_numpy_reference),
        RunnerSpec("dynamic_columnar_polars", "Polars", "benchmarks/run_dynamic_graph_variants.py::run_columnar_polars", fn=run_columnar_polars),
        RunnerSpec("dynamic_object_loop", "Python object loop", "benchmarks/run_dynamic_graph_variants.py::run_object_loop", fn=run_object_loop),
        RunnerSpec("dynamic_amber_object_container", "AMBER", "benchmarks/run_dynamic_graph_variants.py::run_amber_object_container", fn=run_amber_object_container),
        RunnerSpec("dynamic_mesa_object_container", "Mesa", "benchmarks/run_dynamic_graph_variants.py::run_mesa_object_container", fn=run_mesa_object_container),
        RunnerSpec("dynamic_agentpy_object_container", "AgentPy", "benchmarks/run_dynamic_graph_variants.py::run_agentpy_object_container", fn=run_agentpy_object_container),
        RunnerSpec("dynamic_agentsjl_object_container", "Agents.jl", "benchmarks/models/agentsjl_dynamic_graph_variants.jl::DynamicGraphAgent", timed_fn=_time_agentsjl_object_container),
    ]


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def _timing_from_samples(samples: list[float]) -> dict[str, Any]:
    return {
        "sample_count": len(samples),
        "median_s": statistics.median(samples),
        "mean_s": statistics.mean(samples),
        "iqr_s": _percentile(samples, 75) - _percentile(samples, 25),
        "min_s": min(samples),
        "max_s": max(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "raw_samples_s": samples,
    }


def _time_python_runner(fn: Runner, cfg: GraphConfig, runs: int) -> tuple[dict[str, Any], GraphOutcome]:
    fn(cfg)
    samples: list[float] = []
    last: GraphOutcome | None = None
    for _ in range(runs):
        start = time.perf_counter()
        last = fn(cfg)
        samples.append(time.perf_counter() - start)
    assert last is not None
    return _timing_from_samples(samples), last


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(row["mode"]),
        int(row["seed"]),
        int(row["n_agents"]),
        int(row["n_steps"]),
        int(row["runs"]),
    )


def _load_existing_rows(path: Path, runs: int) -> dict[tuple[str, int, int, int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    existing: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    for row in payload.get("results", []):
        if row.get("skipped"):
            continue
        if int(row.get("timing", {}).get("sample_count", 0)) < runs:
            continue
        existing[_row_key(row)] = row
    return existing


def _make_row(
    spec: RunnerSpec,
    cfg: GraphConfig,
    runs: int,
    timing: dict[str, Any],
    outcome: GraphOutcome,
    reference: GraphOutcome,
) -> dict[str, Any]:
    max_abs_diff = 0.0 if spec.mode == "dynamic_numpy_reference" else float(np.max(np.abs(outcome.opinions - reference.opinions)))
    checksum_abs_diff = 0.0 if spec.mode == "dynamic_numpy_reference" else abs(outcome.checksum - reference.checksum)
    return {
        "mode": spec.mode,
        "framework": spec.framework,
        "source": spec.source,
        "topology": "dynamic_random",
        "semantics": "synchronous bounded-confidence update with step-varying sparse graph",
        "seed": cfg.seed,
        "n_agents": cfg.n,
        "n_steps": cfg.steps,
        "runs": runs,
        "skipped": False,
        "timing": timing,
        "mean_opinion": outcome.mean,
        "std_opinion": outcome.std,
        "min_opinion": outcome.min_opinion,
        "max_opinion": outcome.max_opinion,
        "active_edges_last": outcome.active_edges_last,
        "checksum": outcome.checksum,
        "reference_checksum": reference.checksum,
        "checksum_abs_diff_to_numpy": checksum_abs_diff,
        "max_abs_diff_to_numpy": max_abs_diff,
        "matches_numpy_reference": max_abs_diff < 1e-10,
    }


def build_rows(
    agent_counts: list[int],
    seeds: list[int],
    steps: int,
    runs: int,
    modes: set[str],
    timeout: int,
    existing_rows: dict[tuple[str, int, int, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    specs = [spec for spec in _runner_specs() if spec.mode in modes]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for n in agent_counts:
            cfg = GraphConfig(n=n, steps=steps, seed=seed)
            reference = run_numpy_reference(cfg)
            for spec in specs:
                key = (spec.mode, seed, n, steps, runs)
                if key in existing_rows:
                    rows.append(existing_rows[key])
                    continue
                try:
                    if spec.timed_fn is not None:
                        timing, outcome = spec.timed_fn(cfg, runs, timeout)
                    else:
                        assert spec.fn is not None
                        timing, outcome = _time_python_runner(spec.fn, cfg, runs)
                    rows.append(_make_row(spec, cfg, runs, timing, outcome, reference))
                except Exception as exc:
                    rows.append(
                        {
                            "mode": spec.mode,
                            "framework": spec.framework,
                            "source": spec.source,
                            "topology": "dynamic_random",
                            "semantics": "synchronous bounded-confidence update with step-varying sparse graph",
                            "seed": seed,
                            "n_agents": n,
                            "n_steps": steps,
                            "runs": runs,
                            "skipped": True,
                            "skip_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
    return rows


def summarize(rows: list[dict[str, Any]], steps: int, runs: int) -> dict[str, Any]:
    checked = [row for row in rows if not row.get("skipped")]
    skipped = [row for row in rows if row.get("skipped")]
    mismatches = [row for row in checked if not row.get("matches_numpy_reference")]
    raw_sample_counts = [
        int(row.get("timing", {}).get("sample_count", 0))
        for row in checked
    ]
    rows_with_raw_min = [
        row
        for row in checked
        if int(row.get("timing", {}).get("sample_count", 0)) >= FINAL_RAW_SAMPLE_MIN
    ]
    final_like = [
        row
        for row in rows_with_raw_min
        if int(row["n_agents"]) in FINAL_AGENT_COUNTS
        and int(row["seed"]) in FINAL_SEEDS
        and int(row["n_steps"]) == FINAL_STEPS
    ]
    final_like_by_framework = Counter(str(row["framework"]) for row in final_like)
    expected_final_rows = len(FINAL_AGENT_COUNTS) * len(FINAL_SEEDS) * len(_runner_specs())
    ok = (
        len(skipped) == 0
        and len(mismatches) == 0
        and len(final_like) == expected_final_rows
        and steps == FINAL_STEPS
        and runs >= FINAL_RAW_SAMPLE_MIN
    )
    return {
        "status": "dynamic_graph_benchmark_runner_available" if ok else "dynamic_graph_benchmark_runner_partial",
        "rows": len(rows),
        "checked_rows": len(checked),
        "skipped_rows": len(skipped),
        "reference_mismatches": len(mismatches),
        "rows_with_raw_n_ge_5": len(rows_with_raw_min),
        "raw_sample_count_distribution": dict(sorted(Counter(raw_sample_counts).items())),
        "agent_counts": sorted({int(row["n_agents"]) for row in checked}),
        "seeds": sorted({int(row["seed"]) for row in checked}),
        "step_counts": sorted({int(row["n_steps"]) for row in checked}),
        "run_counts": sorted({int(row["runs"]) for row in checked}),
        "frameworks": sorted({str(row["framework"]) for row in checked}),
        "final_like_rows": len(final_like),
        "expected_final_rows": expected_final_rows,
        "final_like_rows_by_framework": dict(sorted(final_like_by_framework.items())),
        "max_abs_diff_to_numpy_max": max((float(row.get("max_abs_diff_to_numpy", 0.0)) for row in checked), default=0.0),
        "skipped_modes": [
            f"{row['mode']}@seed={row['seed']},n={row['n_agents']}"
            for row in skipped
        ],
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    lines = [
        "# Dynamic Graph Benchmark Runner Results",
        "",
        "Synchronous bounded-confidence opinion dynamics with a deterministic sparse edge relation regenerated every step. "
        "Times are median wall-clock milliseconds; every non-skipped row is checked against the NumPy reference trajectory.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Checked rows: `{summary['checked_rows']}`",
        f"- Skipped rows: `{summary['skipped_rows']}`",
        f"- Reference mismatches: `{summary['reference_mismatches']}`",
        f"- Rows with raw n >= 5: `{summary['rows_with_raw_n_ge_5']}`",
        f"- Final-like rows: `{summary['final_like_rows']}` of `{summary['expected_final_rows']}`",
        f"- Frameworks: `{', '.join(summary['frameworks'])}`",
        "",
        "| seed | framework | mode | agents | raw n | median ms | IQR ms | final std | active edges | max abs diff vs NumPy | match |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        if row.get("skipped"):
            reason = str(row.get("skip_reason", "skipped")).replace("|", "/")
            lines.append(
                f"| {row['seed']} | {row['framework']} | {row['mode']} | {row['n_agents']} | 0 | skipped | - | - | - | {reason} | false |"
            )
            continue
        timing = row["timing"]
        lines.append(
            "| {seed} | {framework} | {mode} | {n_agents} | {raw_n} | {median_ms:.1f} | {iqr_ms:.1f} | {std:.4f} | {active_edges} | {diff:.2e} | {match} |".format(
                seed=row["seed"],
                framework=row["framework"],
                mode=row["mode"],
                n_agents=row["n_agents"],
                raw_n=timing["sample_count"],
                median_ms=float(timing["median_s"]) * 1000.0,
                iqr_ms=float(timing["iqr_s"]) * 1000.0,
                std=float(row["std_opinion"]),
                active_edges=row["active_edges_last"],
                diff=float(row["max_abs_diff_to_numpy"]),
                match=str(bool(row["matches_numpy_reference"])).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The runner separates the dynamic graph workload from paper-only audits and records reproducible root benchmark rows.",
            "- A zero-mismatch result supports semantic equivalence for synchronous step-varying graph coordination under this protocol.",
            "- This remains one-machine timing evidence until platform replication is added.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, nargs="+", default=FINAL_AGENT_COUNTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=FINAL_SEEDS)
    parser.add_argument("--steps", type=int, default=FINAL_STEPS)
    parser.add_argument("--runs", type=int, default=FINAL_RAW_SAMPLE_MIN)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--modes", nargs="+", default=[spec.mode for spec in _runner_specs()])
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--output-json", type=Path, default=RESULTS_DIR / "dynamic_graph_results.json")
    parser.add_argument("--output-md", type=Path, default=RESULTS_DIR / "dynamic_graph_results.md")
    args = parser.parse_args()

    valid_modes = {spec.mode for spec in _runner_specs()}
    requested_modes = set(args.modes)
    unknown_modes = sorted(requested_modes - valid_modes)
    if unknown_modes:
        raise SystemExit(f"Unknown modes: {', '.join(unknown_modes)}")

    existing_rows = _load_existing_rows(args.output_json, args.runs) if args.resume_existing else {}
    rows = build_rows(
        agent_counts=args.agents,
        seeds=args.seeds,
        steps=args.steps,
        runs=args.runs,
        modes=requested_modes,
        timeout=args.timeout,
        existing_rows=existing_rows,
    )
    payload = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "agent_counts": args.agents,
            "seeds": args.seeds,
            "steps": args.steps,
            "runs": args.runs,
            "degree": 8,
            "confidence": 0.18,
            "alpha": 0.45,
            "topology": "dynamic_random",
            "timing": "median wall-clock seconds; no trimming",
            "scope": "Root benchmark-runner dynamic graph coordination rows.",
        },
        "summary": summarize(rows, args.steps, args.runs),
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.output_md)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
