"""Verify activation methods against their entry state and keyed draws."""

import dis
from functools import lru_cache
from inspect import CO_ASYNC_GENERATOR, CO_COROUTINE, CO_GENERATOR, CO_VARARGS, CO_VARKEYWORDS
from inspect import getattr_static
from itertools import islice
from typing import NamedTuple, Optional

import polars as pl

from ._auto_batch import (
    _DrawCondition,
    _MAX_I64,
    _MIN_I64,
    _MISSING,
    _agent_plan,
    _call_shape,
    _condition_mask,
    _draw_context,
    _draw_mask,
    _interpreter_instrumented,
    _plan_fields,
    _read_column,
    _tracked_sweep_method,
    _tracked_sweep_rows,
)
from .agent import _AGENT_PROTECTED_ATTRS
from ._keyed_rng import DRAW_ALGORITHM, keyed_uniform
from .model import Model
from .sequences import _SCATTER_SEAMS


_MODEL_SEAMS = {
    name: getattr_static(Model, name)
    for name in ("_apply_pending_writes", "_flush_pending_writes", "_frame_for_columns",
                 "_set_frame", "_contract_record_reduction", "agents_df")
}
_MODEL_SET = Model.__setattr__


class _KeyedDraws(NamedTuple):
    seed: int
    step: int
    event: str
    draw_index: int
    trace: dict


class SweepProof(NamedTuple):
    ids: tuple
    method: str
    reads: dict
    writes: dict
    reduction: frozenset
    draws: Optional[_KeyedDraws] = None

    def matches(self, monitor):
        return (
            set(monitor._activation_order) == set(self.ids)
            and monitor._activation_reads == self.reads
            and monitor._activation_writes == self.writes
            and not monitor._activation_column_reads
            and monitor._activation_draws == (self.draws.trace if self.draws else {})
            and monitor._activation_reductions == (
                dict.fromkeys(self.ids, self.reduction) if self.reduction else {}
            )
        )

    def evidence(self):
        evidence = {
            "scope": "activation_sweep",
            "method": self.method,
            "activation_count": len(self.ids),
            "premises": ["fixed unique IDs", "complete activation trace",
                         "exact Int64 arithmetic"],
        }
        if self.draws:
            seed, step, event, index, trace = self.draws
            evidence["keyed_randomness"] = {
                "algorithm": DRAW_ALGORITHM,
                "seed": seed,
                "step": step,
                "event": event,
                "index": index,
                "draw_count": len(trace),
            }
            evidence["premises"].append("deterministic rules given the verified keyed draws")
        else:
            evidence["premises"].append("deterministic rules with no random draws")
        if self.reduction:
            column, target, operator = next(iter(self.reduction))
            evidence["reduction"] = {"column": column, "target_id": target, "operator": operator}
            evidence["premises"].append("overlaps use one associative and commutative reducer")
        else:
            evidence["premises"].append("activations only read and write their own rows")
        return evidence


# Match the bytecode produced by the current interpreter, including its calling
# convention. The only substitutions allowed are two integer literals and a keyword.
def _add_example(agent):
    agent.model.agents.at[[0]].scatter_add(x=1)


def _max_example(agent):
    agent.model.agents.at[[0]].scatter_max(x=1)


_REDUCTION_SHAPES = (
    ("+", _call_shape(_add_example.__code__, dis.get_instructions(_add_example))),
    ("max", _call_shape(_max_example.__code__, dis.get_instructions(_max_example))),
)


@lru_cache(maxsize=256)
def _reduction_plan(code):
    flags = CO_VARARGS | CO_VARKEYWORDS | CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR
    if (code.co_argcount != 1 or code.co_kwonlyargcount or code.co_flags & flags
            or getattr(code, "co_exceptiontable", b"")):
        return None
    actual = _call_shape(code, islice(dis.get_instructions(code), 41))
    for operator, template in _REDUCTION_SHAPES:
        if len(template) != len(actual):
            continue
        literals = []
        keyword = None
        for (name, value), (want_name, want_value) in zip(actual, template):
            if name != want_name:
                break
            if want_name == "constant" and type(want_value) is int:
                if type(value) is not int or not _MIN_I64 <= value <= _MAX_I64:
                    break
                literals.append(value)
            elif want_name in {"constant", "keyword"} and want_value == ("x",):
                if not (type(value) is tuple and len(value) == 1 and type(value[0]) is str):
                    break
                keyword = value[0]
            elif value != want_value:
                break
        else:
            if keyword and not keyword.startswith("_") and keyword not in _AGENT_PROTECTED_ATTRS:
                target, operand = literals
                return target, keyword, operator, operand
    return None


def verify_sweep(agents, method_name):
    """Return a proof candidate and the reason when an entry check fails."""
    if _interpreter_instrumented():
        return None, "An interpreter trace or profiler can affect activation execution."
    tracked = _tracked_sweep_method(agents, method_name)
    if tracked is None or method_name.startswith("_"):
        return None, "Certification requires one built-in root view with a plain agent method."
    state, cls, method = tracked
    model_type = type(agents.model)
    execution = state.get("_execution")
    if execution is None or execution.config.device != "cpu":
        return None, "The activation proof currently supports the CPU backend."
    if (model_type.__setattr__ is not _MODEL_SET
            or any(getattr_static(model_type, name, _MISSING) is not _MISSING
                   for name in ("agents", "population", "_pending_writes"))):
        return None, "A model override changes the tracked state bindings."
    for name, original in _MODEL_SEAMS.items():
        if name in state or getattr_static(model_type, name, None) is not original:
            return None, "A model override changes a write or read path used by the proof."
    if any(getattr_static(cls, name, _MISSING) is not _MISSING for name in ("id", "model")):
        return None, "The agent class overrides its identity or model binding."
    pending = state["_pending_writes"]
    if type(pending) is not dict or pending:
        return None, "Certification requires committed input at sweep entry."

    plan = _agent_plan(method.__code__)
    reduction = None if plan else _reduction_plan(method.__code__)
    if plan is None and reduction is None:
        return None, "The method is outside the verified integer rules and keyed draw conditions."
    rows = _tracked_sweep_rows(agents, method_name, state, cls)
    if rows is None or rows[0].is_empty():
        return None, "Tracked agents and committed rows do not have the same fixed, unique IDs."
    frame, ids, bindings = rows
    reads = {aid: set() for aid in ids}
    writes = {aid: set() for aid in ids}
    if reduction is not None:
        if any(getattr_static(view_type, name, None) is not original
               for view_type, name, original in _SCATTER_SEAMS):
            return None, "Certification requires the built-in scatter reducer."
        target, column, operator, operand = reduction
        source = _read_column(frame, pending, ids, column)
        if target not in ids or source is None:
            return None, "The reduction requires a fixed target and a non-null Int64 column."
        if operator == "+":
            end = source[ids.index(target)] + len(ids) * operand
            if not _MIN_I64 <= end <= _MAX_I64:
                return None, "The integer sum can leave the Int64 domain."
        return SweepProof(tuple(ids), method_name, reads, writes,
                          frozenset({(column, target, operator)})), None

    fields = _plan_fields(plan)
    if (any(getattr_static(cls, field, _MISSING) is not _MISSING for field in fields)
            or any(callable(attrs.get(field)) for attrs in bindings for field in fields)):
        return None, "An agent field overrides a tracked column."
    selected = ids
    mask = None
    draws = None
    condition = plan.condition
    if isinstance(condition, _DrawCondition):
        context, reason = _draw_context(model_type, cls, state, frame, bindings)
        if context is None:
            return None, reason
        seed, step = context
        event, index = condition.event, condition.draw_index
        try:
            values = [keyed_uniform(seed, step, aid, event, index) for aid in ids]
        except UnicodeEncodeError:
            return None, "The event name must be representable in UTF-8."
        draws = _KeyedDraws(seed, step, event, index, {
            aid: [(seed, step, aid, event, index, value)] for aid, value in zip(ids, values)
        })
        mask = pl.Series(_draw_mask(condition, values))
        selected = [aid for aid, enabled in zip(ids, mask) if enabled]
    elif condition is not None:
        mask = _condition_mask(condition, frame, pending, ids)
        if mask is None:
            return None, "The branch is outside the supported integer and boolean domain."
        selected = [ids[i] for i in mask.arg_true()]
        if condition.field != "id":
            for aid in ids:
                reads[aid].add((condition.field, aid))
    for field, delta in plan.adjustments:
        source = _read_column(frame, pending, ids, field)
        if source is None:
            return None, "Ordinary arithmetic requires non-null Int64 columns."
        if selected:
            if mask is not None:
                source = source.filter(mask)
            if source.min() + delta < _MIN_I64 or source.max() + delta > _MAX_I64:
                return None, "An ordinary update can leave the Int64 domain."
            for aid in selected:
                reads[aid].add((field, aid))
                writes[aid].add((field, aid))
    return SweepProof(tuple(ids), method_name, reads, writes, frozenset(), draws), None
