#!/usr/bin/env python3
"""Correctness check: do all the frameworks compute the same thing?

For each benchmark model, this script runs every framework at a fixed
``(n, steps)`` configuration and extracts observable output statistics:

* **wealth_transfer**: total wealth (should equal ``n * initial_wealth``
  — strict conservation invariant), mean wealth, max wealth, number of
  agents with zero wealth.
* **random_walk**: mean/std of ``x`` and ``y``, min/max (should all be
  inside the world bounds).
* **sir_epidemic**: final counts of S, I, R (sum should equal ``n``).

Each row is either verified in-process for Python frameworks, or parsed
out of a Julia subprocess for Agents.jl.

This answers the question "are the benchmark runtimes measuring the same
thing across frameworks?"  When a framework produces nonsensical stats
(e.g. SimPy SIR starts 100% infected) or violates an invariant, it shows
up here.

Usage::

    python benchmarks/correctness_check.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
MODELS_DIR = BENCH_DIR / "models"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(BENCH_DIR))

# Fixed config for the correctness audit — small enough that every
# framework finishes quickly, large enough that statistics are meaningful.
N = 500
STEPS = 50
INITIAL_WEALTH = 1
INITIAL_INFECTED = 5
WORLD_SIZE = 100


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.2f}"
    return str(v)


def _row(label: str, vals: Dict[str, Any], cols: List[str]) -> str:
    cells = [label] + [_fmt(vals.get(c)) for c in cols]
    return "| " + " | ".join(cells) + " |"


def _hdr(cols: List[str]) -> str:
    return "| framework | " + " | ".join(cols) + " |"


def _sep(n: int) -> str:
    return "|" + "|".join(["---"] * n) + "|"


# --------------------------------------------------------------------------- #
# Wealth Transfer
# --------------------------------------------------------------------------- #

def wealth_amber_loop() -> Dict[str, Any]:
    from models.amber_models import AMBERWealthTransfer

    m = AMBERWealthTransfer(
        {"n": N, "steps": STEPS, "initial_wealth": INITIAL_WEALTH, "show_progress": False}
    )
    m.run()
    wealths = [a.wealth for a in m.agent_objects_list]
    return {
        "total": sum(wealths),
        "mean": sum(wealths) / N,
        "max": max(wealths),
        "zeros": sum(1 for w in wealths if w == 0),
    }


def wealth_amber_vec() -> Dict[str, Any]:
    from models.amber_models import AMBERVectorizedWealthTransfer

    m = AMBERVectorizedWealthTransfer(
        {"n": N, "steps": STEPS, "initial_wealth": INITIAL_WEALTH, "show_progress": False}
    )
    m.run()
    wealth = m.agents.wealth.to_numpy()
    return {
        "total": int(wealth.sum()),
        "mean": float(wealth.mean()),
        "max": int(wealth.max()),
        "zeros": int((wealth == 0).sum()),
    }


def wealth_agentpy() -> Dict[str, Any]:
    from models.agentpy_models import AgentPyWealthTransfer

    m = AgentPyWealthTransfer(
        {"n": N, "steps": STEPS, "initial_wealth": INITIAL_WEALTH}
    )
    m.run()
    wealths = [a.wealth for a in m.agents]
    return {
        "total": sum(wealths),
        "mean": sum(wealths) / N,
        "max": max(wealths),
        "zeros": sum(1 for w in wealths if w == 0),
    }


def wealth_mesa() -> Dict[str, Any]:
    from models.mesa_models import MesaWealthTransfer

    m = MesaWealthTransfer(n=N, steps=STEPS, initial_wealth=INITIAL_WEALTH)
    m.run()
    wealths = [a.wealth for a in m.agents]
    return {
        "total": sum(wealths),
        "mean": sum(wealths) / N,
        "max": max(wealths),
        "zeros": sum(1 for w in wealths if w == 0),
    }


def wealth_simpy() -> Dict[str, Any]:
    import contextlib
    import io

    import simpy
    from models import simpy_models

    # Inline the standalone runner so we can recover the agents_data at the end.
    env = simpy.Environment()
    agents_data = [{"id": i, "wealth": INITIAL_WEALTH} for i in range(N)]
    for i in range(N):
        env.process(simpy_models.wealth_agent(env, i, agents_data))
    with contextlib.redirect_stdout(io.StringIO()):
        env.run(until=STEPS)
    wealths = [a["wealth"] for a in agents_data]
    return {
        "total": sum(wealths),
        "mean": sum(wealths) / N,
        "max": max(wealths),
        "zeros": sum(1 for w in wealths if w == 0),
    }


def wealth_melodie() -> Dict[str, Any]:
    import numpy as np
    from models import melodie_models as mm

    sqlite_path = Path("MelodieBenchmark.sqlite")
    if sqlite_path.exists():
        sqlite_path.unlink()

    config = mm.Melodie.Config(
        project_name="MelodieBenchmark",
        project_root=".",
        sqlite_folder=".",
        output_folder=".",
        input_folder=".",
    )
    scenario = mm.WealthScenario()
    scenario.manager = None
    scenario.periods = STEPS
    scenario.agent_num = N
    scenario.id = 0
    model = mm.WealthModel(config, scenario)
    model.setup()
    for i in range(N):
        a = model.agent_list.add()
        a.id = i
        a.setup()
    for i in range(N):
        model.agent_list[i].wealth = 1
    model.run()
    wealths = [a.wealth for a in model.agent_list]
    return {
        "total": sum(wealths),
        "mean": sum(wealths) / N,
        "max": max(wealths),
        "zeros": sum(1 for w in wealths if w == 0),
    }


# --------------------------------------------------------------------------- #
# Random Walk
# --------------------------------------------------------------------------- #

def walk_amber_loop() -> Dict[str, Any]:
    from models.amber_models import AMBERRandomWalk

    m = AMBERRandomWalk(
        {"n": N, "steps": STEPS, "world_size": WORLD_SIZE, "speed": 1.0, "show_progress": False}
    )
    m.run()
    xs = [a.x for a in m.agent_objects_list]
    ys = [a.y for a in m.agent_objects_list]
    return _walk_stats(xs, ys)


def walk_amber_vec() -> Dict[str, Any]:
    from models.amber_models import AMBERVectorizedRandomWalk

    m = AMBERVectorizedRandomWalk(
        {"n": N, "steps": STEPS, "world_size": WORLD_SIZE, "speed": 1.0, "show_progress": False}
    )
    m.run()
    xs = m.agents.x.to_numpy().tolist()
    ys = m.agents.y.to_numpy().tolist()
    return _walk_stats(xs, ys)


def walk_agentpy() -> Dict[str, Any]:
    from models.agentpy_models import AgentPyRandomWalk

    m = AgentPyRandomWalk({"n": N, "steps": STEPS, "world_size": WORLD_SIZE, "speed": 1.0})
    m.run()
    xs = [a.x for a in m.agents]
    ys = [a.y for a in m.agents]
    return _walk_stats(xs, ys)


def walk_mesa() -> Dict[str, Any]:
    from models.mesa_models import MesaRandomWalk

    m = MesaRandomWalk(n=N, steps=STEPS, world_size=WORLD_SIZE, speed=1.0)
    m.run()
    xs = [a.x for a in m.agents]
    ys = [a.y for a in m.agents]
    return _walk_stats(xs, ys)


def walk_simpy() -> Dict[str, Any]:
    import contextlib
    import io

    import simpy
    from models import simpy_models

    env = simpy.Environment()
    agents_data = [{"id": i, "x": 0, "y": 0} for i in range(N)]
    for i in range(N):
        env.process(simpy_models.walk_agent(env, i, agents_data))
    with contextlib.redirect_stdout(io.StringIO()):
        env.run(until=STEPS)
    xs = [a["x"] for a in agents_data]
    ys = [a["y"] for a in agents_data]
    return _walk_stats(xs, ys)


def walk_melodie() -> Dict[str, Any]:
    import numpy as np
    from models import melodie_models as mm

    sqlite_path = Path("MelodieBenchmark.sqlite")
    if sqlite_path.exists():
        sqlite_path.unlink()
    config = mm.Melodie.Config(
        project_name="MelodieBenchmark",
        project_root=".",
        sqlite_folder=".",
        output_folder=".",
        input_folder=".",
    )
    scenario = mm.WalkScenario()
    scenario.manager = None
    scenario.periods = STEPS
    scenario.agent_num = N
    scenario.id = 0
    model = mm.WalkModel(config, scenario)
    model.setup()
    for i in range(N):
        a = model.agent_list.add()
        a.id = i
        a.setup()
    for i in range(N):
        a = model.agent_list[i]
        a.id = i
        a.x = np.random.uniform(0, WORLD_SIZE)
        a.y = np.random.uniform(0, WORLD_SIZE)
    model.run()
    xs = [a.x for a in model.agent_list]
    ys = [a.y for a in model.agent_list]
    return _walk_stats(xs, ys)


def _walk_stats(xs, ys) -> Dict[str, Any]:
    import statistics as st

    return {
        "mean_x": st.fmean(xs),
        "mean_y": st.fmean(ys),
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "out_of_bounds": sum(
            1 for x, y in zip(xs, ys) if not (0 <= x <= WORLD_SIZE and 0 <= y <= WORLD_SIZE)
        ),
    }


# --------------------------------------------------------------------------- #
# SIR Epidemic
# --------------------------------------------------------------------------- #

def sir_amber_loop() -> Dict[str, Any]:
    from models.amber_models import AMBERSIRModel, SIRAgent

    m = AMBERSIRModel(
        {
            "n": N,
            "steps": STEPS,
            "initial_infected": INITIAL_INFECTED,
            "world_size": WORLD_SIZE,
            "movement_speed": 2.0,
            "infection_radius": 5.0,
            "transmission_rate": 0.1,
            "recovery_time": 14,
            "show_progress": False,
        }
    )
    m.run()
    s = sum(1 for a in m.agent_objects_list if a.status == SIRAgent.STATUS_S)
    i = sum(1 for a in m.agent_objects_list if a.status == SIRAgent.STATUS_I)
    r = sum(1 for a in m.agent_objects_list if a.status == SIRAgent.STATUS_R)
    return {"S": s, "I": i, "R": r, "total": s + i + r}


def sir_amber_vec() -> Dict[str, Any]:
    from models.amber_models import AMBERVectorizedSIRModel

    m = AMBERVectorizedSIRModel(
        {
            "n": N,
            "steps": STEPS,
            "initial_infected": INITIAL_INFECTED,
            "world_size": WORLD_SIZE,
            "movement_speed": 2.0,
            "infection_radius": 5.0,
            "transmission_rate": 0.1,
            "recovery_time": 14,
            "show_progress": False,
        }
    )
    m.run()
    status = m.agents.status.to_numpy()
    return {
        "S": int((status == AMBERVectorizedSIRModel.STATUS_S).sum()),
        "I": int((status == AMBERVectorizedSIRModel.STATUS_I).sum()),
        "R": int((status == AMBERVectorizedSIRModel.STATUS_R).sum()),
        "total": int(status.size),
    }


def sir_agentpy() -> Dict[str, Any]:
    from models.agentpy_models import AgentPySIRModel, APSIRAgent

    m = AgentPySIRModel(
        {
            "n": N,
            "steps": STEPS,
            "initial_infected": INITIAL_INFECTED,
            "world_size": WORLD_SIZE,
            "movement_speed": 2.0,
            "infection_radius": 5.0,
            "transmission_rate": 0.1,
            "recovery_time": 14,
        }
    )
    m.run()
    s = sum(1 for a in m.agents if a.status == APSIRAgent.STATUS_S)
    i = sum(1 for a in m.agents if a.status == APSIRAgent.STATUS_I)
    r = sum(1 for a in m.agents if a.status == APSIRAgent.STATUS_R)
    return {"S": s, "I": i, "R": r, "total": s + i + r}


def sir_mesa() -> Dict[str, Any]:
    from models.mesa_models import MesaSIRAgent, MesaSIRModel

    m = MesaSIRModel(
        n=N,
        steps=STEPS,
        initial_infected=INITIAL_INFECTED,
        world_size=WORLD_SIZE,
        movement_speed=2.0,
        infection_radius=5.0,
        transmission_rate=0.1,
        recovery_time=14,
    )
    m.run()
    s = sum(1 for a in m.agents if a.status == MesaSIRAgent.STATUS_S)
    i = sum(1 for a in m.agents if a.status == MesaSIRAgent.STATUS_I)
    r = sum(1 for a in m.agents if a.status == MesaSIRAgent.STATUS_R)
    return {"S": s, "I": i, "R": r, "total": s + i + r}


def sir_simpy() -> Dict[str, Any]:
    import contextlib
    import io

    import simpy
    from models import simpy_models

    env = simpy.Environment()
    agents_data = [
        {"id": i, "status": 0, "x": 0, "y": 0, "infection_time": 0} for i in range(N)
    ]
    for i in range(N):
        env.process(simpy_models.sir_agent(env, i, agents_data, {}))
    with contextlib.redirect_stdout(io.StringIO()):
        env.run(until=STEPS)
    s = sum(1 for a in agents_data if a["status"] == 0)
    i = sum(1 for a in agents_data if a["status"] == 1)
    r = sum(1 for a in agents_data if a["status"] == 2)
    return {"S": s, "I": i, "R": r, "total": s + i + r}


def sir_melodie() -> Dict[str, Any]:
    import numpy as np
    from models import melodie_models as mm

    sqlite_path = Path("MelodieBenchmark.sqlite")
    if sqlite_path.exists():
        sqlite_path.unlink()
    config = mm.Melodie.Config(
        project_name="MelodieBenchmark",
        project_root=".",
        sqlite_folder=".",
        output_folder=".",
        input_folder=".",
    )
    scenario = mm.SIRScenario()
    scenario.manager = None
    scenario.periods = STEPS
    scenario.agent_num = N
    scenario.id = 0
    model = mm.SIRModel(config, scenario)
    model.setup()
    for i in range(N):
        a = model.agent_list.add()
        a.id = i
        a.setup()
    for i in range(N):
        a = model.agent_list[i]
        a.id = i
        a.x = np.random.uniform(0, WORLD_SIZE)
        a.y = np.random.uniform(0, WORLD_SIZE)
        a.status = 1 if i < INITIAL_INFECTED else 0
    model.run()
    s = sum(1 for a in model.agent_list if a.status == 0)
    i = sum(1 for a in model.agent_list if a.status == 1)
    r = sum(1 for a in model.agent_list if a.status == 2)
    return {"S": s, "I": i, "R": r, "total": s + i + r}


# --------------------------------------------------------------------------- #
# Agents.jl runner — uses a small inline Julia script we write to /tmp
# --------------------------------------------------------------------------- #

def agentsjl_check() -> Dict[str, Dict[str, Any]]:
    jl = f"""
using Agents, Random
Random.seed!(42)

# Wealth
@agent struct WA(NoSpaceAgent); wealth::Float64; end
function wealth_step!(a, m)
    if a.wealth > 0
        p = random_agent(m)
        if p !== nothing && p.id != a.id
            a.wealth -= 1
            p.wealth += 1
        end
    end
end
mw = StandardABM(WA; agent_step! = wealth_step!)
for _ in 1:{N}; add_agent!(mw, 1.0); end
step!(mw, {STEPS})
ws = [a.wealth for a in allagents(mw)]
total_w = Int(sum(ws))
mean_w = sum(ws) / length(ws)
max_w = Int(maximum(ws))
zeros_w = count(==(0), ws)
println("WEALTH $total_w $mean_w $max_w $zeros_w")

# Random walk
@agent struct WkA(NoSpaceAgent); x::Float64; y::Float64; end
function walk_step!(a, m)
    θ = rand() * 2π
    a.x = clamp(a.x + cos(θ), 0.0, {WORLD_SIZE}.0)
    a.y = clamp(a.y + sin(θ), 0.0, {WORLD_SIZE}.0)
end
mwk = StandardABM(WkA; agent_step! = walk_step!)
for _ in 1:{N}; add_agent!(mwk, rand()*{WORLD_SIZE}, rand()*{WORLD_SIZE}); end
step!(mwk, {STEPS})
xs = [a.x for a in allagents(mwk)]
ys = [a.y for a in allagents(mwk)]
mean_x = sum(xs)/length(xs)
mean_y = sum(ys)/length(ys)
min_x = minimum(xs); max_x = maximum(xs)
min_y = minimum(ys); max_y = maximum(ys)
oob = count(xy -> !(0 <= xy[1] <= {WORLD_SIZE} && 0 <= xy[2] <= {WORLD_SIZE}), zip(xs, ys))
println("WALK $mean_x $mean_y $min_x $max_x $min_y $max_y $oob")

# SIR
@agent struct SA(NoSpaceAgent); x::Float64; y::Float64; status::Symbol; infection_time::Int; end
function sir_step!(a, m)
    θ = rand() * 2π
    a.x = clamp(a.x + 2.0*cos(θ), 0.0, {WORLD_SIZE}.0)
    a.y = clamp(a.y + 2.0*sin(θ), 0.0, {WORLD_SIZE}.0)
    if a.status == :I
        for o in allagents(m)
            if o.status == :S
                dx = a.x - o.x; dy = a.y - o.y
                if sqrt(dx*dx + dy*dy) <= 5.0 && rand() < 0.1
                    o.status = :I
                    o.infection_time = 0
                end
            end
        end
        a.infection_time += 1
        if a.infection_time >= 14
            a.status = :R
        end
    end
end
ms = StandardABM(SA; agent_step! = sir_step!)
for i in 1:{N}
    s = i <= {INITIAL_INFECTED} ? :I : :S
    add_agent!(ms, rand()*{WORLD_SIZE}, rand()*{WORLD_SIZE}, s, 0)
end
step!(ms, {STEPS})
ss = [a.status for a in allagents(ms)]
sc = count(==(:S), ss); ic = count(==(:I), ss); rc = count(==(:R), ss)
println("SIR $sc $ic $rc $(sc+ic+rc)")
"""
    tmp = Path("/tmp/_ambr_correctness.jl")
    tmp.write_text(jl)
    try:
        proc = subprocess.run(
            ["julia", str(tmp)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError:
        return {}
    if proc.returncode != 0:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "WEALTH":
            _, total, mean, mx, zeros = parts
            out["wealth"] = {
                "total": int(total),
                "mean": float(mean),
                "max": int(mx),
                "zeros": int(zeros),
            }
        elif parts[0] == "WALK":
            _, mx, my, minx, maxx, miny, maxy, oob = parts
            out["walk"] = {
                "mean_x": float(mx),
                "mean_y": float(my),
                "min_x": float(minx),
                "max_x": float(maxx),
                "min_y": float(miny),
                "max_y": float(maxy),
                "out_of_bounds": int(oob),
            }
        elif parts[0] == "SIR":
            _, s, i, r, total = parts
            out["sir"] = {"S": int(s), "I": int(i), "R": int(r), "total": int(total)}
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    print(f"Correctness check: n={N}, steps={STEPS}, initial_wealth={INITIAL_WEALTH}, initial_infected={INITIAL_INFECTED}")
    print()

    wealth_runs = [
        ("AMBER (loop)",       wealth_amber_loop),
        ("AMBER (vectorized)", wealth_amber_vec),
        ("AgentPy",            wealth_agentpy),
        ("Mesa",               wealth_mesa),
        ("SimPy",              wealth_simpy),
        ("Melodie",            wealth_melodie),
    ]
    walk_runs = [
        ("AMBER (loop)",       walk_amber_loop),
        ("AMBER (vectorized)", walk_amber_vec),
        ("AgentPy",            walk_agentpy),
        ("Mesa",               walk_mesa),
        ("SimPy",              walk_simpy),
        ("Melodie",            walk_melodie),
    ]
    sir_runs = [
        ("AMBER (loop)",       sir_amber_loop),
        ("AMBER (vectorized)", sir_amber_vec),
        ("AgentPy",            sir_agentpy),
        ("Mesa",               sir_mesa),
        ("SimPy",              sir_simpy),
        ("Melodie",            sir_melodie),
    ]

    print("== wealth_transfer ==")
    print("Invariant: total should equal", N * INITIAL_WEALTH)
    cols = ["total", "mean", "max", "zeros"]
    print(_hdr(cols))
    print(_sep(len(cols) + 1))
    for label, fn in wealth_runs:
        try:
            stats = fn()
        except Exception as e:
            stats = {"total": f"ERROR: {type(e).__name__}"}
        print(_row(label, stats, cols))
    print()

    print("== random_walk ==")
    print(f"Invariant: all positions in [0, {WORLD_SIZE}] (out_of_bounds should be 0)")
    cols = ["mean_x", "mean_y", "min_x", "max_x", "min_y", "max_y", "out_of_bounds"]
    print(_hdr(cols))
    print(_sep(len(cols) + 1))
    for label, fn in walk_runs:
        try:
            stats = fn()
        except Exception as e:
            stats = {}
            print(f"| {label} | ERROR: {type(e).__name__}: {e} |")
            continue
        print(_row(label, stats, cols))
    print()

    print("== sir_epidemic ==")
    print(f"Invariant: S+I+R == {N}. Initial I == {INITIAL_INFECTED}.")
    cols = ["S", "I", "R", "total"]
    print(_hdr(cols))
    print(_sep(len(cols) + 1))
    for label, fn in sir_runs:
        try:
            stats = fn()
        except Exception as e:
            stats = {"S": f"ERROR: {type(e).__name__}"}
        print(_row(label, stats, cols))
    print()

    # Agents.jl
    print("== Agents.jl (Julia subprocess) ==")
    try:
        jl = agentsjl_check()
    except Exception as e:
        print(f"  Agents.jl unavailable: {e}")
        jl = {}
    if jl:
        if "wealth" in jl:
            print("wealth:", jl["wealth"])
        if "walk" in jl:
            print("walk:  ", jl["walk"])
        if "sir" in jl:
            print("sir:   ", jl["sir"])
    else:
        print("  (Julia not available or run failed)")


if __name__ == "__main__":
    main()
