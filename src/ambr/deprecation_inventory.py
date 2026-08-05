"""Machine-readable inventory of APIs scheduled for removal in 1.0.

This is the **dry-run checklist** for the 1.0 purge: every
:func:`ambr._deprecation.warn_deprecated` call site should map to a row
here (and to ``docs/deprecations.rst``). Adding a new deprecation without
updating this list fails :mod:`tests.test_deprecation_inventory`.

Rows are ``(what, replacement, since)`` matching the first two arguments
to ``warn_deprecated`` (``what`` is matched as a prefix when the call site
appends ``()`` for callables).
"""

from __future__ import annotations

from typing import List, Tuple

# Exact ``what`` strings passed to warn_deprecated (keep in sync with call sites).
DEPRECATIONS_TO_REMOVE_IN_1_0: List[Tuple[str, str, str]] = [
    # agent.py
    ("Agent.record(name, value)", "agent.<name> = value", "0.4"),
    ("Agent.update_data(data)", "direct attribute assignment", "0.4"),
    # model.py
    ("Model.record(key, value)", "record_model(key, value) or model_reporters", "0.4"),
    ("Model.update_agent_data(...)", "agent.<col> = value or agents.at[id].set(**cols)", "0.4"),
    ("Model.batch_update_agents(...)", "agents.at[ids].set(**cols) or agents.where(...).set(**cols)", "0.4"),
    # sequences.py
    ("AgentList.select(...)", "agents.where(expr) / agents.at[ids] / agents[mask]", "0.4"),
    ("AgentList.record(name, value)", "view.<name> = value (or view.set(...))", "0.4"),
    ("AgentList.update_data(data)", "view.set(**cols)", "0.4"),
    ("AgentList.agents", "iterating model.agents (or agents.by_id / agents.ids)", "0.4"),
    ("AgentList.agent_ids", "agents.ids", "0.4"),
    # population.py
    ("assigning Population.data", "agents.set(**cols) / agents.col = values (or Model agents_df)", "0.4"),
    ("Population.set_agent_value(...)", "agent.<col> = value or agents.at[id].set(**cols)", "0.4"),
    ("Population.batch_update(...)", "agents.set(**cols) or agents.where(expr).set(**cols)", "0.4"),
    ("Population.batch_update_by_ids(...)", "agents.at[ids].set(**cols)", "0.4"),
    ("Population.create_batch_context()", "agents.set(**cols) / column assign", "0.4"),
    # environments.py
    ("GridEnvironment(wrap=...)", "torus=", "0.4"),
    ("GridEnvironment.wrap", "GridEnvironment.torus", "0.4"),
    # execution.py
    ("run(backend=...)", "model.cpu()/model.gpu() or run(device=...)", "0.4"),
    # experiment.py
    ("Experiment(model_class=...)", "Experiment(model_type=...)", "0.4"),
    ("Experiment(parameters=...)", "Experiment(sample=...)", "0.4"),
]

# Soft legacy kept for AgentPy feel — *not* hard-removed in 1.0 unless noted.
# Documented so reviewers do not "forget" them during the freeze.
SOFT_LEGACY_KEEP_FOR_NOW: List[Tuple[str, str]] = [
    (
        "Model.nprandom",
        "Compat alias over model.rng; prefer self.rng. May stay as thin alias after 1.0.",
    ),
    (
        "Model.random",
        "stdlib Random seeded from seed — canonical AgentPy-shaped surface; keep.",
    ),
]


def inventory_whats() -> List[str]:
    """Return the ``what`` strings for tests / docs generation."""
    return [row[0] for row in DEPRECATIONS_TO_REMOVE_IN_1_0]
