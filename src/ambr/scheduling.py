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

This module does **not** prove activation-schedule confluence; the snapshot-view
contract remains an operational monitor only.
"""

from __future__ import annotations

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
    ordered = _agent_list(agents)
    if not ordered:
        return

    model = getattr(agents, "model", None)
    if model is None:
        model = getattr(ordered[0], "model", None)
    override = getattr(model, "_activation_override_order", None)
    if override is not None:
        id_map = {getattr(agent, "id", None): agent for agent in ordered}
        ordered = [id_map[aid] for aid in override if aid in id_map]
    elif mode == "random":
        r = _rng_for(agents, rng)
        # Local list shuffle — do not mutate AgentList structure.
        order = list(range(len(ordered)))
        r.shuffle(order)
        ordered = [ordered[i] for i in order]

    mon = getattr(model, "_contract", None) if model is not None else None

    def call(agent: Any, name: str) -> None:
        aid = getattr(agent, "id", None)
        traced = (
            mon is not None
            and getattr(mon, "active", False)
            and not getattr(mon, "_in_replay", False)
            and aid is not None
        )
        if traced:
            mon.begin_activation(aid)
        try:
            fn = getattr(agent, name, None)
            if callable(fn):
                fn()
        finally:
            if traced:
                mon.end_activation()

    if mode == "simultaneous":
        # Phase 1 is the traced sweep. advance() stays untraced, matching the
        # historical Mesa-style second pass.
        for agent in ordered:
            call(agent, method)
        for agent in ordered:
            adv = getattr(agent, "advance", None)
            if callable(adv):
                adv()
        return

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
