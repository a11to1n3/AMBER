"""Thin activation helpers for the OOP agent lane (Mesa-inspired, optional).

Vectorized models encode activation order inside ``step_vectorized`` (columnar
ops act on the whole population). These helpers target **tracked Python agents**
when you write OOP-style ``agent.step()`` loops.

Modes
-----
* ``sequential`` — agents in current view order (stable id order for
  :class:`~ambr.sequences.AgentList`).
* ``random`` — shuffle agents (uses ``model.rng`` when available), then step.
* ``simultaneous`` — Mesa-style two-phase activation: call ``step()`` on every
  agent, then ``advance()`` when defined. Without ``advance``, this is
  sequential ``step()`` over a fixed list (no mid-step order effects from
  re-shuffling), **not** a proof of schedule equivalence.

The snapshot-view contract can certify order independence for supported
sequential and shuffled single-phase sweeps.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Iterable, List, Optional

import numpy as np

ACTIVATION_MODES = ("sequential", "random", "simultaneous")


def normalize_activation(mode: Optional[str]) -> str:
    """Return a valid activation mode or raise ``ValueError``."""
    if mode is None or mode == "":
        return "sequential"
    m = str(mode).lower().strip()
    if m not in ACTIVATION_MODES:
        raise ValueError(
            f"Unknown activation mode {mode!r}; expected one of {ACTIVATION_MODES}"
        )
    return m


def _agent_list(agents: Iterable[Any]) -> List[Any]:
    return list(agents)


def _rng_for(agents: Any, rng: Any = None):
    if rng is not None:
        return rng
    model = getattr(agents, "model", None)
    if model is not None:
        r = getattr(model, "rng", None)
        if r is not None:
            return r
    return np.random.default_rng()


def _replay_order(ordered, model, override):
    id_map = {agent.id: agent for agent in ordered}
    if len(ordered) != len(override) or id_map.keys() != set(override):
        raise ValueError("the activation sweep differs from the recorded sweep")
    model._activation_override_order = None
    return [id_map[agent_id] for agent_id in override]


def activate(
    agents: Iterable[Any],
    mode: str = "sequential",
    method: str = "step",
    rng: Any = None,
) -> None:
    """Call ``method`` on each agent under the given activation order.

    Args:
        agents: Iterable of agent objects (typically ``model.agents``).
        mode: ``'sequential'``, ``'random'``, or ``'simultaneous'``.
        method: Name of the per-agent method to call (default ``'step'``).
        rng: NumPy Generator for ``random`` mode; defaults to ``model.rng``.
    """
    mode = normalize_activation(mode)
    if mode == "sequential":
        from ._auto_batch import try_batch

        if try_batch(agents, method):
            return
    ordered = _agent_list(agents)
    if not ordered:
        return

    model = getattr(agents, "model", None)
    if model is None:
        model = getattr(ordered[0], "model", None)
    mon = getattr(model, "_contract", None) if model is not None else None
    override = getattr(model, "_activation_override_order", None)
    if mon is not None and mon._in_replay and (override is None or mode == "simultaneous"):
        raise ValueError("replay requires one sequential or random activation sweep")
    if mode == "random":
        execution = getattr(model, "_execution", None)
        r = _rng_for(agents, rng)
        if rng is None and execution is not None and execution.tracked_agents and r is execution.device_rng:
            r = model._host_rng
        if override is not None and r is not model.rng:
            raise ValueError("random activation uses a stream outside the replay snapshot")
        order = list(range(len(ordered)))
        r.shuffle(order)
        ordered = [ordered[i] for i in order]

    if override is not None:
        ordered = _replay_order(ordered, model, override)
    if mon is not None and mon.active:
        mon.begin_sweep(agents, method, single_phase=mode != "simultaneous")

    def call(agent: Any, name: str) -> None:
        aid = getattr(agent, "id", None)
        replaying = mon is not None and mon._in_replay
        traced = mon is not None and mon.active and aid is not None
        if traced:
            mon.begin_activation(aid)
        elif replaying:
            mon._current_activation = aid
        try:
            fn = getattr(agent, name, None)
            if callable(fn):
                fn()
        finally:
            if traced:
                mon.end_activation()
            elif replaying:
                mon._current_activation = None

    if mode == "simultaneous":
        staged = mon is not None and mon.active and mon._current_activation is None and not mon._stage_open
        if staged:
            mon.begin_stage(method)
        try:
            for agent in ordered:
                call(agent, method)
            if staged:
                mon.begin_stage("advance")
            for agent in ordered:
                call(agent, "advance")
        finally:
            if staged:
                mon.end_stage()
        return

    watch_rng = model._watch_sweep_rng() if mon is not None and (mon.active or mon._in_replay) else nullcontext()
    with watch_rng:
        for agent in ordered:
            call(agent, method)


def shuffled_ids(agents: Any, rng: Any = None) -> np.ndarray:
    """Return a shuffled copy of agent ids (vectorized-lane helper).

    Use inside ``step_vectorized`` when you want a random activation order over
    ids without OOP loops::

        for aid in am.shuffled_ids(self.agents, self.rng):
            ...
    """
    if hasattr(agents, "ids"):
        ids = np.asarray(agents.ids.to_numpy() if hasattr(agents.ids, "to_numpy") else agents.ids)
    elif hasattr(agents, "_ids_series"):
        ids = np.asarray(agents._ids_series().to_numpy())
    else:
        ids = np.asarray([getattr(a, "id", a) for a in agents])
    out = ids.copy()
    r = _rng_for(agents, rng)
    r.shuffle(out)
    return out


class Activation:
    """Callable activation policy (optional OOP helper).

    Example::

        act = am.Activation("random")

        def step_oop(self):
            act(self.agents)
    """

    __slots__ = ("mode", "method")

    def __init__(self, mode: str = "sequential", method: str = "step"):
        self.mode = normalize_activation(mode)
        self.method = method

    def __call__(self, agents: Iterable[Any], rng: Any = None) -> None:
        activate(agents, mode=self.mode, method=self.method, rng=rng)

    def __repr__(self) -> str:
        return f"Activation(mode={self.mode!r}, method={self.method!r})"


# Mesa-shaped aliases (thin constructors)
def SequentialActivation(method: str = "step") -> Activation:
    return Activation("sequential", method=method)


def RandomActivation(method: str = "step") -> Activation:
    return Activation("random", method=method)


def SimultaneousActivation(method: str = "step") -> Activation:
    return Activation("simultaneous", method=method)
