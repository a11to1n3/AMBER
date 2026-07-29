#!/usr/bin/env python3
"""Deterministic sync/async SIR benchmark runner.

The headline SIR benchmark compares framework implementations that do not all
use the same activation semantics.  This runner makes the schedule explicit:
every row starts from the same positions and uses deterministic movement and
pair-level transmission draws.  The output is a benchmark-runner artifact,
not a prose-only paper-local audit.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import polars as pl


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
MODELS_DIR = BENCH_DIR / "models"
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(BENCH_DIR))

STATUS_S = 0
STATUS_I = 1
STATUS_R = 2

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB
C3 = 0xD6E8FEB86659FD93
C4 = 0xA5A3564E27F886D3

TAG_INIT = 11
TAG_MOVE_X = 23
TAG_MOVE_Y = 29
TAG_TRANSMIT = 37

FINAL_AGENT_COUNTS = [500, 1000, 5000]
FINAL_STEPS = 50
FINAL_RAW_SAMPLE_MIN = 10
FINAL_SCHEDULES = ["sync", "async"]


@dataclass(frozen=True)
class SirConfig:
    n: int
    steps: int
    seed: int = 42
    initial_infected: int = 5
    world_size: float = 100.0
    movement_speed: float = 2.0
    infection_radius: float = 5.0
    transmission_rate: float = 0.1
    recovery_time: int = 14


@dataclass(frozen=True)
class SirOutcome:
    counts: tuple[int, int, int]
    status_checksum: int
    infection_time_checksum: int
    trajectory: list[tuple[int, int, int]]


Runner = Callable[[SirConfig], SirOutcome]


def _splitmix64_int(x: int) -> int:
    x = (x + GOLDEN) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * C1) & MASK64
    z = ((z ^ (z >> 27)) * C2) & MASK64
    return (z ^ (z >> 31)) & MASK64


def _hash_int(seed: int, tag: int, step: int, a: int, b: int = 0) -> int:
    x = (
        seed
        ^ ((tag * C1) & MASK64)
        ^ (((step + 1_000_003) * C2) & MASK64)
        ^ ((a * C3) & MASK64)
        ^ ((b * C4) & MASK64)
    )
    return _splitmix64_int(x & MASK64)


def _uniform01_int(seed: int, tag: int, step: int, a: int, b: int = 0) -> float:
    return (_hash_int(seed, tag, step, a, b) >> 11) * (1.0 / (1 << 53))


def _splitmix64_np(x: np.ndarray) -> np.ndarray:
    z = x + np.uint64(GOLDEN)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(C1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(C2)
    return z ^ (z >> np.uint64(31))


def _uniform01_np(
    seed: int,
    tag: int,
    step: int,
    a: np.ndarray,
    b: np.ndarray | None = None,
) -> np.ndarray:
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


def _initial_arrays(cfg: SirConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.arange(cfg.n, dtype=np.int64)
    x = _uniform01_np(cfg.seed, TAG_INIT, -1, ids, np.zeros_like(ids)) * cfg.world_size
    y = _uniform01_np(cfg.seed, TAG_INIT, -1, ids, np.ones_like(ids)) * cfg.world_size
    status = np.full(cfg.n, STATUS_S, dtype=np.int8)
    status[: cfg.initial_infected] = STATUS_I
    infection_time = np.zeros(cfg.n, dtype=np.int16)
    return x, y, status, infection_time


def _move_arrays(cfg: SirConfig, step: int, x: np.ndarray, y: np.ndarray) -> None:
    ids = np.arange(cfg.n, dtype=np.int64)
    dx = (_uniform01_np(cfg.seed, TAG_MOVE_X, step, ids) * 2.0 - 1.0) * cfg.movement_speed
    dy = (_uniform01_np(cfg.seed, TAG_MOVE_Y, step, ids) * 2.0 - 1.0) * cfg.movement_speed
    np.clip(x + dx, 0.0, cfg.world_size, out=x)
    np.clip(y + dy, 0.0, cfg.world_size, out=y)


def _counts(status: np.ndarray) -> tuple[int, int, int]:
    return (
        int((status == STATUS_S).sum()),
        int((status == STATUS_I).sum()),
        int((status == STATUS_R).sum()),
    )


def _outcome(
    status: np.ndarray,
    infection_time: np.ndarray,
    trajectory: list[tuple[int, int, int]],
) -> SirOutcome:
    weights = np.arange(1, len(status) + 1, dtype=np.int64)
    return SirOutcome(
        counts=_counts(status),
        status_checksum=int(np.dot(status.astype(np.int64), weights)),
        infection_time_checksum=int(np.dot(infection_time.astype(np.int64), weights)),
        trajectory=trajectory,
    )


def run_sync_numpy_reference(cfg: SirConfig) -> SirOutcome:
    x, y, status, infection_time = _initial_arrays(cfg)
    trajectory: list[tuple[int, int, int]] = []

    for step in range(cfg.steps):
        _move_arrays(cfg, step, x, y)
        infected_ids = np.flatnonzero(status == STATUS_I)
        susceptible_ids = np.flatnonzero(status == STATUS_S)
        newly_infected = np.zeros(cfg.n, dtype=bool)

        if infected_ids.size and susceptible_ids.size:
            sx = x[susceptible_ids]
            sy = y[susceptible_ids]
            radius_sq = cfg.infection_radius**2
            for source_id in infected_ids:
                dx = x[source_id] - sx
                dy = y[source_id] - sy
                candidates = susceptible_ids[(dx * dx + dy * dy) <= radius_sq]
                if candidates.size:
                    sources = np.full(candidates.size, source_id, dtype=np.int64)
                    draws = _uniform01_np(cfg.seed, TAG_TRANSMIT, step, sources, candidates)
                    newly_infected[candidates[draws < cfg.transmission_rate]] = True

        if newly_infected.any():
            status[newly_infected] = STATUS_I
            infection_time[newly_infected] = 0
        infection_time[status == STATUS_I] += 1
        status[(status == STATUS_I) & (infection_time >= cfg.recovery_time)] = STATUS_R
        trajectory.append(_counts(status))

    return _outcome(status, infection_time, trajectory)


def run_async_spatial_reference(cfg: SirConfig) -> SirOutcome:
    x, y, status, infection_time = _initial_arrays(cfg)
    trajectory: list[tuple[int, int, int]] = []
    radius_sq = cfg.infection_radius**2
    cell_size = cfg.infection_radius

    for step in range(cfg.steps):
        _move_arrays(cfg, step, x, y)
        cell_x = np.floor(x / cell_size).astype(np.int64)
        cell_y = np.floor(y / cell_size).astype(np.int64)
        buckets: dict[tuple[int, int], list[int]] = {}
        for idx in range(cfg.n):
            buckets.setdefault((int(cell_x[idx]), int(cell_y[idx])), []).append(idx)

        for source_id in range(cfg.n):
            if status[source_id] != STATUS_I:
                continue
            candidates: list[int] = []
            cx = int(cell_x[source_id])
            cy = int(cell_y[source_id])
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    candidates.extend(buckets.get((cx + ox, cy + oy), ()))
            if not candidates:
                continue
            target_ids = np.array(sorted(set(candidates)), dtype=np.int64)
            target_ids = target_ids[(target_ids != source_id) & (status[target_ids] == STATUS_S)]
            if target_ids.size == 0:
                continue
            dx = x[source_id] - x[target_ids]
            dy = y[source_id] - y[target_ids]
            near = target_ids[(dx * dx + dy * dy) <= radius_sq]
            if near.size:
                sources = np.full(near.size, source_id, dtype=np.int64)
                draws = _uniform01_np(cfg.seed, TAG_TRANSMIT, step, sources, near)
                hits = near[draws < cfg.transmission_rate]
                status[hits] = STATUS_I
                infection_time[hits] = 0

        infection_time[status == STATUS_I] += 1
        status[(status == STATUS_I) & (infection_time >= cfg.recovery_time)] = STATUS_R
        trajectory.append(_counts(status))

    return _outcome(status, infection_time, trajectory)


def run_sync_columnar_view(cfg: SirConfig) -> SirOutcome:
    x, y, status, infection_time = _initial_arrays(cfg)
    ids = np.arange(cfg.n, dtype=np.int64)
    df = pl.DataFrame({"id": ids, "x": x, "y": y, "status": status, "infection_time": infection_time})
    trajectory: list[tuple[int, int, int]] = []

    for step in range(cfg.steps):
        x = df["x"].to_numpy().copy()
        y = df["y"].to_numpy().copy()
        _move_arrays(cfg, step, x, y)
        df = df.with_columns(pl.Series("x", x), pl.Series("y", y))

        infected = df.filter(pl.col("status") == STATUS_I).select(
            pl.col("id").alias("source_id"),
            pl.col("x").alias("source_x"),
            pl.col("y").alias("source_y"),
        )
        susceptible = df.filter(pl.col("status") == STATUS_S).select("id", "x", "y")
        newly_infected: list[int] = []
        if infected.height and susceptible.height:
            pairs = (
                susceptible.join(infected, how="cross")
                .with_columns(
                    (
                        (pl.col("x") - pl.col("source_x")) ** 2
                        + (pl.col("y") - pl.col("source_y")) ** 2
                    ).alias("dist_sq")
                )
                .filter(pl.col("dist_sq") <= cfg.infection_radius**2)
            )
            if pairs.height:
                target_ids = pairs["id"].to_numpy()
                source_ids = pairs["source_id"].to_numpy()
                draws = _uniform01_np(cfg.seed, TAG_TRANSMIT, step, source_ids, target_ids)
                hits = pairs.with_columns(pl.Series("draw", draws)).filter(
                    pl.col("draw") < cfg.transmission_rate
                )
                newly_infected = hits["id"].unique().to_list() if hits.height else []

        if newly_infected:
            df = df.with_columns(
                pl.when(pl.col("id").is_in(newly_infected))
                .then(STATUS_I)
                .otherwise(pl.col("status"))
                .alias("status"),
                pl.when(pl.col("id").is_in(newly_infected))
                .then(0)
                .otherwise(pl.col("infection_time"))
                .alias("infection_time"),
            )

        df = df.with_columns(
            pl.when(pl.col("status") == STATUS_I)
            .then(pl.col("infection_time") + 1)
            .otherwise(pl.col("infection_time"))
            .alias("infection_time")
        )
        df = df.with_columns(
            pl.when((pl.col("status") == STATUS_I) & (pl.col("infection_time") >= cfg.recovery_time))
            .then(STATUS_R)
            .otherwise(pl.col("status"))
            .alias("status")
        )
        trajectory.append(_counts(df["status"].to_numpy()))

    return _outcome(df["status"].to_numpy(), df["infection_time"].to_numpy(), trajectory)


def _object_snapshot(agents: Iterable[object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(agents, key=lambda agent: agent.semantic_id)
    x = np.array([agent.x for agent in ordered], dtype=np.float64)
    y = np.array([agent.y for agent in ordered], dtype=np.float64)
    status = np.array([agent.status for agent in ordered], dtype=np.int8)
    infection_time = np.array([agent.infection_time for agent in ordered], dtype=np.int16)
    return x, y, status, infection_time


def _assign_object_state(agents: Iterable[object], cfg: SirConfig) -> None:
    x, y, status, infection_time = _initial_arrays(cfg)
    for idx, agent in enumerate(agents):
        agent.semantic_id = idx
        agent.x = float(x[idx])
        agent.y = float(y[idx])
        agent.status = int(status[idx])
        agent.infection_time = int(infection_time[idx])


def _assign_positions(agents: Iterable[object], x: np.ndarray, y: np.ndarray) -> None:
    for agent in sorted(agents, key=lambda item: item.semantic_id):
        idx = agent.semantic_id
        agent.x = float(x[idx])
        agent.y = float(y[idx])


def _assign_status(agents: Iterable[object], status: np.ndarray, infection_time: np.ndarray) -> None:
    for agent in sorted(agents, key=lambda item: item.semantic_id):
        idx = agent.semantic_id
        agent.status = int(status[idx])
        agent.infection_time = int(infection_time[idx])


def _sync_step_objects(cfg: SirConfig, step: int, agents: Iterable[object]) -> None:
    ordered = sorted(agents, key=lambda item: item.semantic_id)
    x, y, status, infection_time = _object_snapshot(ordered)
    _move_arrays(cfg, step, x, y)
    infected_ids = np.flatnonzero(status == STATUS_I)
    susceptible_ids = np.flatnonzero(status == STATUS_S)
    newly_infected: set[int] = set()
    radius_sq = cfg.infection_radius**2

    for source_id in infected_ids:
        for target_id in susceptible_ids:
            dx = x[source_id] - x[target_id]
            dy = y[source_id] - y[target_id]
            if dx * dx + dy * dy <= radius_sq:
                draw = _uniform01_int(cfg.seed, TAG_TRANSMIT, step, int(source_id), int(target_id))
                if draw < cfg.transmission_rate:
                    newly_infected.add(int(target_id))

    if newly_infected:
        new_ids = np.fromiter(newly_infected, dtype=np.int64)
        status[new_ids] = STATUS_I
        infection_time[new_ids] = 0
    infection_time[status == STATUS_I] += 1
    status[(status == STATUS_I) & (infection_time >= cfg.recovery_time)] = STATUS_R
    _assign_positions(ordered, x, y)
    _assign_status(ordered, status, infection_time)


def _async_step_objects(cfg: SirConfig, step: int, agents: Iterable[object]) -> None:
    ordered = sorted(agents, key=lambda item: item.semantic_id)
    x, y, _, _ = _object_snapshot(ordered)
    _move_arrays(cfg, step, x, y)
    _assign_positions(ordered, x, y)
    radius_sq = cfg.infection_radius**2

    for source in ordered:
        if source.status != STATUS_I:
            continue
        for target in ordered:
            if source.semantic_id == target.semantic_id or target.status != STATUS_S:
                continue
            dx = source.x - target.x
            dy = source.y - target.y
            if dx * dx + dy * dy <= radius_sq:
                draw = _uniform01_int(
                    cfg.seed,
                    TAG_TRANSMIT,
                    step,
                    int(source.semantic_id),
                    int(target.semantic_id),
                )
                if draw < cfg.transmission_rate:
                    target.status = STATUS_I
                    target.infection_time = 0

    for agent in ordered:
        if agent.status == STATUS_I:
            agent.infection_time += 1
            if agent.infection_time >= cfg.recovery_time:
                agent.status = STATUS_R


def _outcome_from_objects(agents: Iterable[object], trajectory: list[tuple[int, int, int]]) -> SirOutcome:
    _, _, status, infection_time = _object_snapshot(agents)
    return _outcome(status, infection_time, trajectory)


def _object_trajectory_outcome(
    agents: Iterable[object],
    cfg: SirConfig,
    schedule: str,
) -> SirOutcome:
    ordered = list(agents)
    trajectory: list[tuple[int, int, int]] = []
    for step in range(cfg.steps):
        if schedule == "sync":
            _sync_step_objects(cfg, step, ordered)
        elif schedule == "async":
            _async_step_objects(cfg, step, ordered)
        else:
            raise ValueError(f"unknown schedule: {schedule}")
        _, _, status, _ = _object_snapshot(ordered)
        trajectory.append(_counts(status))
    return _outcome_from_objects(ordered, trajectory)


def _base_params(cfg: SirConfig) -> dict[str, object]:
    return {
        "n": cfg.n,
        "steps": cfg.steps,
        "initial_infected": cfg.initial_infected,
        "world_size": cfg.world_size,
        "movement_speed": cfg.movement_speed,
        "infection_radius": cfg.infection_radius,
        "transmission_rate": cfg.transmission_rate,
        "recovery_time": cfg.recovery_time,
        "seed": cfg.seed,
        "show_progress": False,
    }


def run_amber_loop_actual(cfg: SirConfig, schedule: str) -> SirOutcome:
    from models.amber_models import AMBERSIRModel

    model = AMBERSIRModel(_base_params(cfg))
    model.setup()
    _assign_object_state(model.agent_objects_list, cfg)
    return _object_trajectory_outcome(model.agent_objects_list, cfg, schedule)


def run_mesa_actual(cfg: SirConfig, schedule: str) -> SirOutcome:
    from models.mesa_models import MesaSIRModel

    params = _base_params(cfg)
    params.pop("show_progress", None)
    model = MesaSIRModel(**params)
    agents = list(model.agents)
    _assign_object_state(agents, cfg)
    return _object_trajectory_outcome(agents, cfg, schedule)


def run_agentpy_actual(cfg: SirConfig, schedule: str) -> SirOutcome:
    from models.agentpy_models import AgentPySIRModel

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = AgentPySIRModel(_base_params(cfg))
        model.setup()
    agents = list(model.agents)
    _assign_object_state(agents, cfg)
    return _object_trajectory_outcome(agents, cfg, schedule)


def run_melodie_actual(cfg: SirConfig, schedule: str) -> SirOutcome:
    from models import melodie_models as mm

    with tempfile.TemporaryDirectory(prefix="amber_melodie_sir_runner_") as tmp:
        tmp_path = Path(tmp)
        config = mm.Melodie.Config(
            project_name="MelodieSIRScheduleRunner",
            project_root=str(tmp_path),
            sqlite_folder=str(tmp_path),
            output_folder=str(tmp_path),
            input_folder=str(tmp_path),
        )
        scenario = mm.SIRScenario()
        scenario.manager = None
        scenario.periods = cfg.steps
        scenario.agent_num = cfg.n
        scenario.id = 0
        model = mm.SIRModel(config, scenario)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.setup()
            for idx in range(cfg.n):
                agent = model.agent_list.add()
                agent.id = idx
                agent.setup()
        agents = list(model.agent_list)
        _assign_object_state(agents, cfg)
        return _object_trajectory_outcome(agents, cfg, schedule)


def run_simpy_refactored(cfg: SirConfig, schedule: str) -> SirOutcome:
    import simpy

    x, y, status, infection_time = _initial_arrays(cfg)
    trajectory: list[tuple[int, int, int]] = []
    env = simpy.Environment()
    radius_sq = cfg.infection_radius**2

    def driver():
        nonlocal x, y, status, infection_time
        for step in range(cfg.steps):
            _move_arrays(cfg, step, x, y)
            if schedule == "sync":
                infected_ids = np.flatnonzero(status == STATUS_I)
                susceptible_ids = np.flatnonzero(status == STATUS_S)
                newly_infected: set[int] = set()
                for source_id in infected_ids:
                    for target_id in susceptible_ids:
                        dx = x[source_id] - x[target_id]
                        dy = y[source_id] - y[target_id]
                        if dx * dx + dy * dy <= radius_sq:
                            draw = _uniform01_int(
                                cfg.seed,
                                TAG_TRANSMIT,
                                step,
                                int(source_id),
                                int(target_id),
                            )
                            if draw < cfg.transmission_rate:
                                newly_infected.add(int(target_id))
                if newly_infected:
                    new_ids = np.fromiter(newly_infected, dtype=np.int64)
                    status[new_ids] = STATUS_I
                    infection_time[new_ids] = 0
            elif schedule == "async":
                for source_id in range(cfg.n):
                    if status[source_id] != STATUS_I:
                        continue
                    for target_id in range(cfg.n):
                        if source_id == target_id or status[target_id] != STATUS_S:
                            continue
                        dx = x[source_id] - x[target_id]
                        dy = y[source_id] - y[target_id]
                        if dx * dx + dy * dy <= radius_sq:
                            draw = _uniform01_int(cfg.seed, TAG_TRANSMIT, step, source_id, target_id)
                            if draw < cfg.transmission_rate:
                                status[target_id] = STATUS_I
                                infection_time[target_id] = 0
            else:
                raise ValueError(f"unknown schedule: {schedule}")

            infection_time[status == STATUS_I] += 1
            status[(status == STATUS_I) & (infection_time >= cfg.recovery_time)] = STATUS_R
            trajectory.append(_counts(status))
            yield env.timeout(1)

    env.process(driver())
    env.run(until=cfg.steps)
    return _outcome(status, infection_time, trajectory)


def _agentsjl_script_path() -> Path:
    return MODELS_DIR / "agentsjl_sir_schedule_variants.jl"


def _outcome_from_agentsjl_payload(payload: dict[str, Any]) -> SirOutcome:
    return SirOutcome(
        counts=tuple(int(value) for value in payload["counts"]),  # type: ignore[arg-type]
        status_checksum=int(payload["status_checksum"]),
        infection_time_checksum=int(payload["infection_time_checksum"]),
        trajectory=[
            tuple(int(value) for value in point)
            for point in payload["trajectory"]
        ],
    )


def _time_agentsjl_actual_source(
    cfg: SirConfig,
    schedule: str,
    runs: int,
    timeout: int = 900,
) -> tuple[dict[str, object], SirOutcome]:
    script = _agentsjl_script_path()
    if not script.exists():
        raise RuntimeError(f"Agents.jl split-SIR script missing: {script}")

    cmd = [
        "julia",
        str(script),
        "--n",
        str(cfg.n),
        "--steps",
        str(cfg.steps),
        "--seed",
        str(cfg.seed),
        "--initial-infected",
        str(cfg.initial_infected),
        "--world-size",
        str(cfg.world_size),
        "--movement-speed",
        str(cfg.movement_speed),
        "--infection-radius",
        str(cfg.infection_radius),
        "--transmission-rate",
        str(cfg.transmission_rate),
        "--recovery-time",
        str(cfg.recovery_time),
        "--schedule",
        schedule,
        "--runs",
        str(runs),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(MODELS_DIR),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Agents.jl split-SIR failed: {proc.stderr.strip()[-1200:]}")

    stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    json_lines = [line for line in stdout_lines if line.startswith("{") and line.endswith("}")]
    if not json_lines:
        raise RuntimeError(f"Agents.jl split-SIR produced no JSON output: {proc.stdout[-1200:]}")

    payload = json.loads(json_lines[-1])
    samples = [float(value) for value in payload["raw_samples_s"]]
    if not samples:
        raise RuntimeError("Agents.jl split-SIR produced no timing samples")
    return (
        {
            "sample_count": len(samples),
            "median_s": statistics.median(samples),
            "mean_s": statistics.mean(samples),
            "min_s": min(samples),
            "max_s": max(samples),
            "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "raw_samples_s": samples,
        },
        _outcome_from_agentsjl_payload(payload),
    )


def _runner_specs() -> list[dict[str, object]]:
    object_specs = [
        ("amber_loop_actual", "AMBER (loop)", "benchmarks/models/amber_models.py::AMBERSIRModel", run_amber_loop_actual),
        ("mesa_actual", "Mesa", "benchmarks/models/mesa_models.py::MesaSIRModel", run_mesa_actual),
        ("agentpy_actual", "AgentPy", "benchmarks/models/agentpy_models.py::AgentPySIRModel", run_agentpy_actual),
        ("melodie_actual", "Melodie", "benchmarks/models/melodie_models.py::SIRModel", run_melodie_actual),
        ("simpy_refactored", "SimPy", "benchmarks/models/simpy_models.py::refactored_driver", run_simpy_refactored),
    ]
    return [
        {
            "mode": "sync_numpy_reference",
            "framework": "NumPy",
            "schedule": "sync",
            "source": "benchmarks/run_sir_schedule_variants.py::run_sync_numpy_reference",
            "fn": run_sync_numpy_reference,
        },
        {
            "mode": "async_spatial_reference",
            "framework": "Spatial reference",
            "schedule": "async",
            "source": "benchmarks/run_sir_schedule_variants.py::run_async_spatial_reference",
            "fn": run_async_spatial_reference,
        },
        {
            "mode": "sync_amber_vectorized_view",
            "framework": "AMBER (vectorized)",
            "schedule": "sync",
            "source": "benchmarks/run_sir_schedule_variants.py::run_sync_columnar_view",
            "fn": run_sync_columnar_view,
        },
        {
            "mode": "agentsjl_actual_source_sync",
            "framework": "Agents.jl",
            "schedule": "sync",
            "source": "benchmarks/models/agentsjl_sir_schedule_variants.jl::SIRAgent2/sir_step2_sync!",
            "timed_fn": lambda cfg, runs: _time_agentsjl_actual_source(cfg, "sync", runs),
        },
        {
            "mode": "agentsjl_actual_source_async",
            "framework": "Agents.jl",
            "schedule": "async",
            "source": "benchmarks/models/agentsjl_sir_schedule_variants.jl::SIRAgent2/sir_step2_async!",
            "timed_fn": lambda cfg, runs: _time_agentsjl_actual_source(cfg, "async", runs),
        },
    ] + [
        {
            "mode": f"{schedule}_{label}",
            "framework": framework,
            "schedule": schedule,
            "source": source,
            "fn": lambda cfg, schedule=schedule, runner=runner: runner(cfg, schedule),
        }
        for label, framework, source, runner in object_specs
        for schedule in FINAL_SCHEDULES
    ]


def _time_runner(fn: Runner, cfg: SirConfig, runs: int) -> tuple[dict[str, object], SirOutcome]:
    fn(cfg)
    samples: list[float] = []
    last: SirOutcome | None = None
    for _ in range(runs):
        start = time.perf_counter()
        last = fn(cfg)
        samples.append(time.perf_counter() - start)
    if last is None:
        raise RuntimeError("runner produced no outcome")
    return (
        {
            "sample_count": len(samples),
            "median_s": statistics.median(samples),
            "mean_s": statistics.mean(samples),
            "min_s": min(samples),
            "max_s": max(samples),
            "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "raw_samples_s": samples,
        },
        last,
    )


def _result_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return str(row["mode"]), int(row["n_agents"]), int(row["n_steps"]), int(row["runs"])


def _raw_sample_count(row: dict[str, Any]) -> int:
    timing = row.get("timing") or {}
    return int(timing.get("sample_count", 0))


def _is_final_like(row: dict[str, Any]) -> bool:
    return (
        not row.get("skipped")
        and row.get("schedule") in FINAL_SCHEDULES
        and int(row.get("n_agents", 0)) in FINAL_AGENT_COUNTS
        and int(row.get("n_steps", 0)) == FINAL_STEPS
        and _raw_sample_count(row) >= FINAL_RAW_SAMPLE_MIN
        and row.get("reference_match") is not False
    )


def _budget_skip_row(spec: dict[str, object], n_agents: int, steps: int, runs: int, reason: str) -> dict[str, Any]:
    return {
        "mode": str(spec["mode"]),
        "framework": spec["framework"],
        "schedule": spec["schedule"],
        "source": spec["source"],
        "n_agents": n_agents,
        "n_steps": steps,
        "runs": runs,
        "skipped": True,
        "skip_type": "budget",
        "skip_reason": reason,
    }


def _selected_specs(selected_modes: set[str] | None, skip_modes: set[str]) -> list[dict[str, object]]:
    specs = _runner_specs()
    known_modes = {str(spec["mode"]) for spec in specs}
    unknown = (selected_modes or set()) - known_modes
    if unknown:
        raise ValueError(f"unknown mode(s): {', '.join(sorted(unknown))}")
    requested = set(selected_modes) if selected_modes else known_modes
    requested_schedules = {
        str(spec["schedule"])
        for spec in specs
        if str(spec["mode"]) in requested and str(spec["mode"]) not in skip_modes
    }
    reference_modes: set[str] = set()
    if "sync" in requested_schedules:
        reference_modes.add("sync_numpy_reference")
    if "async" in requested_schedules:
        reference_modes.add("async_spatial_reference")
    included = requested | reference_modes
    return [
        spec
        for spec in specs
        if str(spec["mode"]) in included and str(spec["mode"]) not in skip_modes
    ]


def build_rows(
    agent_counts: list[int],
    steps: int,
    runs: int,
    *,
    selected_modes: set[str] | None = None,
    skip_modes: set[str] | None = None,
    budget_skip_modes: set[str] | None = None,
    budget_skip_reason: str = "",
    existing_rows: list[dict[str, Any]] | None = None,
    progress_writer: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {
        _result_key(row): row for row in (existing_rows or [])
    }
    refs: dict[tuple[int, str], SirOutcome] = {}
    active_skip_modes = skip_modes or set()
    active_budget_skip_modes = budget_skip_modes or set()
    active_budget_skip_reason = budget_skip_reason or "Excluded by the declared run-budget policy."

    for n in agent_counts:
        cfg = SirConfig(n=n, steps=steps)
        for spec in _selected_specs(selected_modes, active_skip_modes):
            mode = str(spec["mode"])
            schedule = str(spec["schedule"])
            fn = spec.get("fn")
            timed_fn = spec.get("timed_fn")
            key = (mode, n, steps, runs)
            existing = rows_by_key.get(key)
            if mode in active_budget_skip_modes:
                rows_by_key[key] = _budget_skip_row(spec, n, steps, runs, active_budget_skip_reason)
                if progress_writer is not None:
                    progress_writer(list(rows_by_key.values()))
                continue
            if existing and not existing.get("skipped"):
                if mode == "sync_numpy_reference":
                    refs[(n, "sync")] = fn(cfg)  # type: ignore[operator]
                elif mode == "async_spatial_reference":
                    refs[(n, "async")] = fn(cfg)  # type: ignore[operator]
                continue
            try:
                if timed_fn is not None:
                    timing, outcome = timed_fn(cfg, runs)  # type: ignore[operator]
                else:
                    timing, outcome = _time_runner(fn, cfg, runs)  # type: ignore[arg-type]
            except Exception as exc:
                rows_by_key[key] = {
                    "mode": mode,
                    "framework": spec["framework"],
                    "schedule": schedule,
                    "source": spec["source"],
                    "n_agents": n,
                    "n_steps": steps,
                    "runs": runs,
                    "skipped": True,
                    "skip_reason": f"{type(exc).__name__}: {exc}",
                }
                if progress_writer is not None:
                    progress_writer(list(rows_by_key.values()))
                continue

            if mode == "sync_numpy_reference":
                refs[(n, "sync")] = outcome
            elif mode == "async_spatial_reference":
                refs[(n, "async")] = outcome
            ref = refs.get((n, schedule))
            reference_match = None
            if ref is not None:
                reference_match = (
                    outcome.status_checksum == ref.status_checksum
                    and outcome.infection_time_checksum == ref.infection_time_checksum
                    and outcome.trajectory == ref.trajectory
                )
            s, i, r = outcome.counts
            rows_by_key[key] = {
                "mode": mode,
                "framework": spec["framework"],
                "schedule": schedule,
                "source": spec["source"],
                "n_agents": n,
                "n_steps": steps,
                "runs": runs,
                "skipped": False,
                "timing": timing,
                "susceptible": s,
                "infected": i,
                "recovered": r,
                "attack_rate": (i + r) / n,
                "status_checksum": outcome.status_checksum,
                "infection_time_checksum": outcome.infection_time_checksum,
                "trajectory": outcome.trajectory,
                "reference_match": reference_match,
            }
            if progress_writer is not None:
                progress_writer(list(rows_by_key.values()))
    return list(rows_by_key.values())


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {str(spec["mode"]): idx for idx, spec in enumerate(_runner_specs())}
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("n_steps", 0)),
            int(row.get("n_agents", 0)),
            str(row.get("schedule", "")),
            order.get(str(row.get("mode", "")), 999),
            str(row.get("mode", "")),
            int(row.get("runs", 0)),
        ),
    )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = [row for row in rows if not row.get("skipped")]
    skipped = [row for row in rows if row.get("skipped")]
    budget_skipped = [row for row in skipped if row.get("skip_type") == "budget"]
    mismatches = [row for row in checked if row.get("reference_match") is False]
    final_like = [row for row in checked if _is_final_like(row)]
    return {
        "status": "sir_schedule_benchmark_runner_available",
        "rows": len(rows),
        "checked_rows": len(checked),
        "skipped_rows": len(skipped),
        "budget_skipped_rows": len(budget_skipped),
        "reference_mismatches": len(mismatches),
        "rows_with_raw_n_ge_10": sum(1 for row in checked if _raw_sample_count(row) >= FINAL_RAW_SAMPLE_MIN),
        "raw_sample_count_distribution": dict(sorted(Counter(_raw_sample_count(row) for row in checked).items())),
        "agent_counts": sorted({int(row["n_agents"]) for row in rows}),
        "step_counts": sorted({int(row["n_steps"]) for row in rows}),
        "run_counts": sorted({int(row["runs"]) for row in rows}),
        "frameworks": sorted({str(row["framework"]) for row in checked}),
        "schedules": sorted({str(row["schedule"]) for row in rows}),
        "final_like_rows": len(final_like),
        "final_like_rows_by_framework": dict(
            sorted(Counter(str(row["framework"]) for row in final_like).items())
        ),
        "budget_skipped_modes": sorted(
            {
                f"{row.get('mode')}@n={row.get('n_agents')},steps={row.get('n_steps')},runs={row.get('runs')}"
                for row in budget_skipped
            }
        ),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    lines = [
        "# SIR Schedule Benchmark Runner",
        "",
        "Deterministic split sync/async SIR rows emitted from the root benchmark runner.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Checked rows: `{summary['checked_rows']}`",
        f"- Skipped rows: `{summary['skipped_rows']}`",
        f"- Budget-skipped rows: `{summary['budget_skipped_rows']}`",
        f"- Reference mismatches: `{summary['reference_mismatches']}`",
        f"- Rows with raw n >= 10: `{summary['rows_with_raw_n_ge_10']}`",
        f"- Headline-size 10-run rows: `{summary['final_like_rows']}`",
        f"- Headline-size 10-run rows by framework: `{summary['final_like_rows_by_framework']}`",
        "",
        "| mode | framework | schedule | steps | agents | raw n | median ms | final S/I/R | reference match |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["results"]:
        if row.get("skipped"):
            reason = str(row.get("skip_reason", "skipped")).replace("|", "/")
            lines.append(
                f"| {row['mode']} | {row['framework']} | {row['schedule']} | {row['n_steps']} | {row['n_agents']} | 0 | skipped | - | {reason} |"
            )
            continue
        timing = row["timing"]
        lines.append(
            "| {mode} | {framework} | {schedule} | {n_steps} | {n_agents} | {raw_n} | {median_ms:.1f} | {s}/{i}/{r} | {match} |".format(
                mode=row["mode"],
                framework=row["framework"],
                schedule=row["schedule"],
                n_steps=row["n_steps"],
                n_agents=row["n_agents"],
                raw_n=timing["sample_count"],
                median_ms=float(timing["median_s"]) * 1000.0,
                s=row["susceptible"],
                i=row["infected"],
                r=row["recovered"],
                match=row["reference_match"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_payload(
    rows: list[dict[str, Any]],
    *,
    agent_counts: list[int],
    steps: int,
    runs: int,
    selected_modes: set[str] | None,
    skip_modes: set[str],
    budget_skip_modes: set[str],
    budget_skip_reason: str,
    output_json: Path,
    output_md: Path,
    resume_existing: bool,
) -> dict[str, Any]:
    ordered = _ordered_rows(rows)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "requested_agent_counts": agent_counts,
            "requested_steps": steps,
            "requested_runs": runs,
            "selected_modes": sorted(selected_modes) if selected_modes else None,
            "skip_modes": sorted(skip_modes),
            "budget_skip_modes": sorted(budget_skip_modes),
            "budget_skip_reason": budget_skip_reason,
            "resume_existing": resume_existing,
            "sir_defaults": asdict(SirConfig(n=0, steps=steps)),
            "scope": (
                "Root benchmark-runner deterministic sync/async SIR rows for "
                "Python-hosted frameworks plus Agents.jl source-shaped rows."
            ),
        },
        "summary": _summarize(ordered),
        "results": ordered,
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(payload, output_md)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, nargs="+", default=[200, 500])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--skip-modes", nargs="+", default=[])
    parser.add_argument("--budget-skip-modes", nargs="+", default=[])
    parser.add_argument("--budget-skip-reason", default="Excluded by the declared run-budget policy.")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESULTS_DIR / "sir_schedule_results.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=RESULTS_DIR / "sir_schedule_results.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_modes = set(args.modes) if args.modes else None
    skip_modes = set(args.skip_modes)
    budget_skip_modes = set(args.budget_skip_modes)
    existing_rows: list[dict[str, Any]] = []
    if args.resume_existing and args.output_json.exists():
        existing_rows = json.loads(args.output_json.read_text(encoding="utf-8")).get("results", [])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    def progress_writer(rows: list[dict[str, Any]]) -> None:
        _write_payload(
            rows,
            agent_counts=args.agents,
            steps=args.steps,
            runs=args.runs,
            selected_modes=selected_modes,
            skip_modes=skip_modes,
            budget_skip_modes=budget_skip_modes,
            budget_skip_reason=args.budget_skip_reason,
            output_json=args.output_json,
            output_md=args.output_md,
            resume_existing=args.resume_existing,
        )

    rows = build_rows(
        args.agents,
        args.steps,
        args.runs,
        selected_modes=selected_modes,
        skip_modes=skip_modes,
        budget_skip_modes=budget_skip_modes,
        budget_skip_reason=args.budget_skip_reason,
        existing_rows=existing_rows,
        progress_writer=progress_writer,
    )
    payload = _write_payload(
        rows,
        agent_counts=args.agents,
        steps=args.steps,
        runs=args.runs,
        selected_modes=selected_modes,
        skip_modes=skip_modes,
        budget_skip_modes=budget_skip_modes,
        budget_skip_reason=args.budget_skip_reason,
        output_json=args.output_json,
        output_md=args.output_md,
        resume_existing=args.resume_existing,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
