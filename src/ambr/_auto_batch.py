"""Batch a small, verified subset of ordinary Python agent methods."""

import dis
from functools import lru_cache
from inspect import CO_ASYNC_GENERATOR, CO_COROUTINE, CO_GENERATOR, CO_VARARGS, CO_VARKEYWORDS
from inspect import getattr_static
from itertools import islice, takewhile
from math import isfinite
import operator
import sys
from sys import getprofile, gettrace
from types import CodeType, FunctionType
from typing import NamedTuple, Optional

import numpy as np
import polars as pl

from ._keyed_rng import keyed_uniform
from .agent import Agent, _AGENT_PROTECTED_ATTRS
from .execution import reconcile_host_frame
from .model import Model
from .population import Population


_MIN_AGENTS = 64
_MIN_I64 = -(1 << 63)
_MAX_I64 = (1 << 63) - 1
_AGENT_GET = Agent.__getattribute__
_AGENT_SET = Agent.__setattr__
_AGENT_DRAW = Agent.draw
_MODEL_GET = Model.__getattribute__
_QUEUE_WRITE = Model._queue_write
_MISSING = object()
_RECEIVER_LOADS = {"LOAD_FAST", "LOAD_FAST_BORROW"}
_CONSTANT_LOADS = {"LOAD_CONST", "LOAD_SMALL_INT"}
_IGNORED_OPS = {"CACHE", "RESUME", "NOP", "NOT_TAKEN"}
_CLAMP_BUILTINS = {"min": min, "max": max}
_COMPARISONS = {
    "==": operator.eq, "!=": operator.ne, "<": operator.lt,
    "<=": operator.le, ">": operator.gt, ">=": operator.ge,
}


class _Condition(NamedTuple):
    field: str
    comparison: Optional[str]
    value: Optional[int]
    modulus: Optional[int]
    negate: bool


class _DrawCondition(NamedTuple):
    event: str
    draw_index: int
    comparison: str
    threshold: float
    negate: bool


class _BatchPlan(NamedTuple):
    condition: Optional[_Condition | _DrawCondition]
    adjustments: tuple[tuple[str, int], ...]


class _ClippedDrawUpdate(NamedTuple):
    field: str
    event: str
    draw_index: int
    lower: float
    upper: float
    scale: float
    offset: float


def _call_shape(code, instructions):
    shape = []
    for op in instructions:
        name, value = op.opname, op.argval
        if name in _IGNORED_OPS:
            continue
        if name in _RECEIVER_LOADS and value == code.co_varnames[0]:
            shape.append(("receiver", None))
        elif name == "KW_NAMES":
            shape.append(("keyword", code.co_consts[op.arg]))
        elif name in _CONSTANT_LOADS:
            shape.append(("constant", value))
        else:
            shape.append((name, value))
    return shape


def _draw_example(agent):
    if agent.draw("event") < 0.5:
        agent.x += 1


def _indexed_draw_example(agent):
    if agent.draw("event", index=1) < 0.5:
        agent.x += 1


_DRAW_SHAPES = tuple(
    _call_shape(fn.__code__, takewhile(
        lambda op: not op.opname.startswith("POP_JUMP"), dis.get_instructions(fn)
    ))
    for fn in (_draw_example, _indexed_draw_example)
)


def _clipped_draw_example(agent):
    agent.coord = max(-7.25, min(131.75, agent.coord + (agent.draw("event") * 2.75 - 1.25)))


def _indexed_clipped_draw_example(agent):
    agent.coord = max(-7.25, min(131.75, agent.coord + (agent.draw("event", index=7) * 2.75 - 1.25)))


_CLIPPED_DRAW_SHAPES = []
for _example in (_clipped_draw_example, _indexed_clipped_draw_example):
    _ops = list(dis.get_instructions(_example))
    _store = next(i for i, op in enumerate(_ops) if op.opname == "STORE_ATTR")
    _CLIPPED_DRAW_SHAPES.append(_call_shape(_example.__code__, _ops[:_store + 1]))


def _clipped_draw_update(code, instructions):
    shape = _call_shape(code, instructions)
    for template in _CLIPPED_DRAW_SHAPES:
        if len(shape) != len(template):
            continue
        constants = {7: 0}
        field = instructions[-1].argval
        for (name, value), (expected_name, expected_value) in zip(shape, template):
            if name != expected_name:
                break
            if name == "constant":
                if type(value) is not type(expected_value):
                    break
                constants[expected_value] = value
            elif expected_value == "coord":
                if value != field:
                    break
            elif value != expected_value:
                break
        else:
            lower, upper, scale, offset = (constants[key] for key in (-7.25, 131.75, 2.75, 1.25))
            if (field.startswith("_") or field in _AGENT_PROTECTED_ATTRS
                    or not constants["event"] or not 0 <= constants[7] <= _MAX_I64
                    or not all(map(isfinite, (lower, upper, scale, offset)))
                    or lower > upper or scale < 0 or offset < 0):
                return None
            return _ClippedDrawUpdate(field, constants["event"], constants[7], lower, upper, scale, offset)
    return None


@lru_cache(maxsize=256)
def _clipped_draw_plan(code: CodeType) -> tuple[_ClippedDrawUpdate, ...]:
    unsupported = CO_VARARGS | CO_VARKEYWORDS | CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR
    if (code.co_argcount != 1 or code.co_kwonlyargcount or code.co_flags & unsupported
            or getattr(code, "co_exceptiontable", b"")):
        return ()
    instructions = list(islice((op for op in dis.get_instructions(code) if op.opname not in _IGNORED_OPS), 161))
    end = _return_size(instructions)
    if not end or len(instructions) > 160:
        return ()
    updates = {}
    start = 0
    body = instructions[:-end]
    for index, instruction in enumerate(body):
        if instruction.opname != "STORE_ATTR":
            continue
        update = _clipped_draw_update(code, body[start:index + 1])
        if update is None or update.field in updates:
            return ()
        updates[update.field] = update
        start = index + 1
    return tuple(updates.values()) if start == len(body) else ()


def _draw_condition(code, instructions, negate):
    actual = _call_shape(code, instructions)
    for template in _DRAW_SHAPES:
        if len(actual) != len(template):
            continue
        index = 0
        for (name, value), (want_name, want_value) in zip(actual, template):
            if name != want_name:
                break
            if name == "constant" and want_value == "event":
                if type(value) is not str or not value:
                    break
                event = value
            elif name == "constant" and type(want_value) is int:
                if type(value) is not int or not 0 <= value <= _MAX_I64:
                    break
                index = value
            elif name == "constant" and type(want_value) is float:
                if type(value) not in (int, float) or not 0 <= value <= 1:
                    break
                threshold = value
            elif name == "COMPARE_OP":
                if value not in _COMPARISONS:
                    break
                comparison = value
            elif value != want_value:
                break
        else:
            return _DrawCondition(event, index, comparison, threshold, negate)
    return None


def _return_size(instructions) -> int:
    tail = [(op.opname, op.argval) for op in instructions[-2:]]
    if tail[-1:] == [("RETURN_CONST", None)]:
        return 1
    if tail == [("LOAD_CONST", None), ("RETURN_VALUE", None)]:
        return 2
    return 0


def _condition_from_instructions(instructions, receiver, negate):
    if (
        len(instructions) < 2 or instructions[0].opname not in _RECEIVER_LOADS
        or instructions[0].argval != receiver or instructions[1].opname != "LOAD_ATTR"
    ):
        return None
    field = instructions[1].argval
    if field.startswith("_") or (field in _AGENT_PROTECTED_ATTRS and field != "id"):
        return None
    tail = instructions[2:]
    if not tail or (len(tail) == 1 and tail[0].opname == "TO_BOOL"):
        return _Condition(field, None, None, None, negate)
    divisor = None
    if len(tail) == 4:
        constant, modulo, *tail = tail
        if (
            constant.opname not in _CONSTANT_LOADS or type(constant.argval) is not int
            or not 0 < constant.argval <= _MAX_I64
            or (modulo.opname != "BINARY_MODULO" and not (
                modulo.opname == "BINARY_OP" and modulo.argrepr == "%"
            ))
        ):
            return None
        divisor = constant.argval
    if len(tail) != 2:
        return None
    constant, comparison = tail
    if (
        constant.opname not in _CONSTANT_LOADS or type(constant.argval) is not int
        or not _MIN_I64 <= constant.argval <= _MAX_I64
        or comparison.opname != "COMPARE_OP" or comparison.argval not in _COMPARISONS
    ):
        return None
    return _Condition(field, comparison.argval, constant.argval, divisor, negate)


@lru_cache(maxsize=256)
def _agent_plan(code: CodeType) -> Optional[_BatchPlan]:
    """Recognize constant own-field updates behind at most one supported condition."""
    unsupported = CO_VARARGS | CO_VARKEYWORDS | CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR
    if (
        code.co_argcount != 1 or code.co_kwonlyargcount or code.co_flags & unsupported
        or getattr(code, "co_exceptiontable", b"")
    ):
        return None
    instructions = list(islice((op for op in dis.get_instructions(code) if op.opname not in _IGNORED_OPS), 41))
    end = _return_size(instructions)
    if not end or len(instructions) > 40:
        return None
    body = instructions[:-end]
    receiver = code.co_varnames[0]
    condition = None
    true_jumps = {"POP_JUMP_IF_TRUE", "POP_JUMP_FORWARD_IF_TRUE"}
    false_jumps = {"POP_JUMP_IF_FALSE", "POP_JUMP_FORWARD_IF_FALSE"}
    for index, op in enumerate(body):
        if op.opname in true_jumps or op.opname in false_jumps:
            if op.argval != instructions[-end].offset:
                return None
            negate = op.opname in true_jumps
            condition = _condition_from_instructions(body[:index], receiver, negate)
            if condition is None:
                condition = _draw_condition(code, body[:index], negate)
            if condition is None:
                return None
            body = body[index + 1:]
            early_end = _return_size(body)
            if early_end:
                body = body[:-early_end]
            break
    adjustments = {}
    while body:
        if len(body) < 6 or body[0].opname not in _RECEIVER_LOADS or body[0].argval != receiver:
            return None
        augmented = (body[1].opname, body[1].arg) in {("COPY", 1), ("DUP_TOP", None)}
        if augmented:
            if len(body) < 7:
                return None
            _, _, read, constant, operation, receiver_op, write = body[:7]
            receiver_matches = (receiver_op.opname, receiver_op.arg) in {
                ("SWAP", 2), ("ROT_TWO", None),
            }
            add = operation.opname == "INPLACE_ADD" or (
                operation.opname == "BINARY_OP" and operation.argrepr == "+="
            )
            subtract = operation.opname == "INPLACE_SUBTRACT" or (
                operation.opname == "BINARY_OP" and operation.argrepr == "-="
            )
            body = body[7:]
        else:
            _, read, constant, operation, receiver_op, write = body[:6]
            receiver_matches = receiver_op.opname in _RECEIVER_LOADS and receiver_op.argval == receiver
            add = operation.opname == "BINARY_ADD" or (
                operation.opname == "BINARY_OP" and operation.argrepr == "+"
            )
            subtract = operation.opname == "BINARY_SUBTRACT" or (
                operation.opname == "BINARY_OP" and operation.argrepr == "-"
            )
            body = body[6:]
        if not (add or subtract) or not receiver_matches:
            return None
        field = read.argval
        if (
            read.opname != "LOAD_ATTR" or write.opname != "STORE_ATTR"
            or write.argval != field or constant.opname not in _CONSTANT_LOADS
            or type(constant.argval) is not int
            or field.startswith("_") or field in _AGENT_PROTECTED_ATTRS or field in adjustments
        ):
            return None
        delta = constant.argval if add else -constant.argval
        if not (_MIN_I64 <= constant.argval <= _MAX_I64 and _MIN_I64 <= delta <= _MAX_I64):
            return None
        adjustments[field] = delta
        if len(adjustments) > 4:
            return None
    return _BatchPlan(condition, tuple(adjustments.items())) if adjustments else None


def _read_column(frame, pending, ids, field, *, allow_boolean=False):
    """Read the column as each agent would see it, including pending writes."""
    if field == "id":
        if any(not _MIN_I64 <= aid <= _MAX_I64 for aid in ids):
            return None
        return pl.Series(field, ids, dtype=pl.Int64)
    buffered = pending.get(field)
    if buffered and len(buffered) == len(ids):
        values = [buffered.get(aid) for aid in ids]
        if allow_boolean and all(type(value) is bool for value in values):
            return pl.Series(field, values, dtype=pl.Boolean)
        if all(type(value) is int and _MIN_I64 <= value <= _MAX_I64 for value in values):
            return pl.Series(field, values, dtype=pl.Int64)
        return None
    if field not in frame.columns:
        return None
    col = frame[field]
    if col.null_count() or (col.dtype != pl.Int64 and not (allow_boolean and col.dtype == pl.Boolean)):
        return None
    if not buffered:
        return col
    positions = {aid: index for index, aid in enumerate(ids)}
    values = col.to_list()
    for aid, value in buffered.items():
        if aid not in positions:
            return None
        if col.dtype == pl.Boolean:
            if type(value) is not bool:
                return None
        elif type(value) is not int or not _MIN_I64 <= value <= _MAX_I64:
            return None
        values[positions[aid]] = value
    return pl.Series(field, values, dtype=col.dtype)


def _condition_mask(condition: _Condition, frame, pending, ids):
    col = _read_column(frame, pending, ids, condition.field, allow_boolean=condition.comparison is None)
    if col is None:
        return None
    if condition.modulus is not None:
        if col.min() < 0:
            return None
        col = col % condition.modulus
    if condition.comparison is not None:
        selected = _COMPARISONS[condition.comparison](col, condition.value)
    else:
        selected = col if col.dtype == pl.Boolean else col != 0
    return ~selected if condition.negate else selected


def _draw_context(model_type, cls, state, frame, bindings):
    if (getattr_static(cls, "draw", None) is not _AGENT_DRAW
            or any(getattr_static(cls, name, _MISSING) is not _MISSING for name in ("id", "model"))
            or any("draw" in attrs for attrs in bindings)
            or "draw" in frame.columns
            or "draw" in state["_pending_writes"]
            or "draw" in state["_execution"].device_columns
            or any(getattr_static(model_type, name, _MISSING) is not _MISSING
                   for name in ("t", "_keyed_seed", "_contract"))):
        return None, "Certification requires the built-in keyed draw method and model bindings."
    seed, step = state["_keyed_seed"], state["t"]
    if type(seed) is not int or type(step) is not int:
        return None, "Certification requires an integer seed and step for keyed draws."
    return (seed, step), None


def _draw_mask(condition, values):
    compare = _COMPARISONS[condition.comparison]
    return [compare(value, condition.threshold) != condition.negate for value in values]


def _apply_device_plan(model, execution, plan, fields, frame, pending, ids, draw_mask):
    """Keep values and selection masks on the device until a host read."""
    if not execution.tracked_agents or "id" in execution.dirty_columns or fields & execution.borrowed_columns:
        return False

    touched = fields & pending.keys()
    condition = plan.condition if isinstance(plan.condition, _Condition) else None
    xp = execution.xp
    arrays = execution.device_columns
    candidates = arrays.copy() if touched else arrays
    for field in touched:
        allow_boolean = condition is not None and field == condition.field and condition.comparison is None
        host = _read_column(frame, pending, ids, field, allow_boolean=allow_boolean)
        if host is None:
            return False
        dtype = np.bool_ if host.dtype == pl.Boolean else np.int64
        if field in execution.dirty_columns:
            current = arrays[field].astype(dtype, copy=True)
            buffered = pending[field]
            row_positions = {aid: pos for pos, aid in enumerate(ids)}
            indices = [row_positions[aid] for aid in buffered]
            current[xp.asarray(indices, dtype=xp.int64)] = xp.asarray(list(buffered.values()), dtype=dtype)
            candidates[field] = current
        else:
            candidates[field] = xp.asarray(host.to_numpy().astype(dtype, copy=False))

    mask = xp.asarray(draw_mask, dtype=xp.bool_) if draw_mask is not None else None
    invalid = None
    if condition is not None:
        column = candidates.get(condition.field)
        if column is None:
            return False
        if condition.comparison is None:
            if column.dtype not in (np.int64, np.bool_):
                return False
            mask = column != 0
        else:
            if column.dtype != np.int64:
                return False
            if condition.modulus is not None:
                invalid = column < 0
                column = column % condition.modulus
            mask = _COMPARISONS[condition.comparison](column, condition.value)
        if condition.negate:
            mask = ~mask

    for field, delta in plan.adjustments:
        column = candidates.get(field)
        if column is None or column.dtype != np.int64:
            return False
        if not delta:
            continue
        overflows = column > _MAX_I64 - delta if delta > 0 else column < _MIN_I64 - delta
        if mask is not None:
            overflows &= mask
        invalid = overflows if invalid is None else invalid | overflows
    if invalid is not None and xp.any(invalid).item():
        return False

    if touched:
        model._flush_pending_writes(columns=touched)
    for field, delta in plan.adjustments:
        column = arrays[field]
        updated = column + delta
        column[:] = updated if mask is None else xp.where(mask, updated, column)
        if field not in execution.agent_writes:
            execution.agent_writes[field] = mask
        elif (previous := execution.agent_writes[field]) is not None:
            execution.agent_writes[field] = None if mask is None else previous | mask
        execution.dirty_columns.add(field)
    return True


def _tracked_sweep_method(agents, method_name):
    """Locate a plain method on the tracked root without invoking descriptors."""
    from .sequences import AgentList

    if type(agents) is not AgentList or not agents._agent_objects:
        return None
    objects = agents._agent_objects
    model = agents.model
    model_type = type(model)
    if (
        not isinstance(model, Model) or type(model_type) is not type
        or model_type.__getattribute__ is not _MODEL_GET
    ):
        return None
    state = vars(model)
    if (
        state.get("agents") is not agents
        or getattr_static(model_type, "_queue_write", None) is not _QUEUE_WRITE
        or "_queue_write" in state or type(state.get("population")) is not Population
    ):
        return None
    if len(agents._agents_by_id) != len(objects):
        return None
    cls = type(objects[0])
    if type(cls) is not type or cls.__getattribute__ is not _AGENT_GET or cls.__setattr__ is not _AGENT_SET:
        return None
    method = getattr_static(cls, method_name, None)
    if type(method) is not FunctionType:
        return None
    return state, cls, method


def _tracked_sweep_rows(agents, method_name, state, cls):
    """Check that the scheduling roster and the committed row index agree."""
    model = agents.model
    bindings = []
    ids = []
    for agent in agents._agent_objects:
        if type(agent) is not cls:
            return None
        attrs = object.__getattribute__(agent, "__dict__")
        if type(attrs) is not dict:
            return None
        aid = attrs.get("id")
        if (
            type(aid) is not int or attrs.get("model") is not model or method_name in attrs
            or agents._agents_by_id.get(aid) is not agent
        ):
            return None
        bindings.append(attrs)
        ids.append(aid)
    if len(set(ids)) != len(ids):
        return None

    frame = state["population"].data
    if type(frame) is not pl.DataFrame:
        return None
    if not frame.is_empty() and ("id" not in frame.columns or frame["id"].to_list() != ids):
        return None

    # A frame edited outside the normal write API can leave an old ID cache.
    # Keep the scalar behavior in that case.
    version = state["_id_version"]
    if state.get("_ids_arange_cache") == (version, True) and ids != list(range(len(ids))):
        return None
    positions = state.get("_id_pos_cache")
    if positions is not None and positions[0] == version:
        if any(positions[1].get(aid) != pos for pos, aid in enumerate(ids)):
            return None
    return frame, ids, bindings


def _interpreter_instrumented():
    """A tracer or profiler can execute Python between agent operations."""
    monitoring = getattr(sys, "monitoring", None)
    return (gettrace() is not None or getprofile() is not None
            or (monitoring is not None and any(monitoring.get_tool(i) for i in range(6))))


def _plan_fields(plan):
    fields = {field for field, _ in plan.adjustments}
    if isinstance(plan.condition, _Condition):
        fields.add(plan.condition.field)
    return fields


def _queue_batch_updates(pending, ids, bindings, updates):
    # Preserve the scalar write boundary and Python instance bindings.
    for field, values in updates:
        writes = pending.setdefault(field, {})
        writes.update(zip(ids, values))
        for attrs, value in zip(bindings, values):
            attrs[field] = value


def _apply_clipped_draw_plan(agents, method, plan, cls, state, frame, ids, bindings):
    if any(method.__globals__.get(name, method.__builtins__.get(name)) is not builtin
           for name, builtin in _CLAMP_BUILTINS.items()):
        return False
    context, _ = _draw_context(type(agents.model), cls, state, frame, bindings)
    if context is None:
        return False
    seed, step = context
    execution = state["_execution"]
    pending = state["_pending_writes"]
    device = execution.config.device == "gpu"
    fields = {update.field for update in plan}
    if (fields & pending.keys() or fields & execution.borrowed_columns
            or (device and (not execution.tracked_agents or "id" in execution.dirty_columns))):
        return False
    columns = {}
    for field in fields:
        if field not in frame.columns or frame[field].dtype != pl.Float64 or frame[field].null_count():
            return False
        array = execution.device_columns.get(field) if device else frame[field].to_numpy()
        if array is None or array.dtype != np.float64:
            return False
        columns[field] = array

    xp = execution.xp
    updates = []
    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for update in plan:
                draws = np.fromiter(
                    (keyed_uniform(seed, step, aid, update.event, update.draw_index) for aid in ids),
                    dtype=np.float64, count=len(ids),
                )
                moved = columns[update.field] + (xp.asarray(draws) * update.scale - update.offset)
                # Match Python min/max, including ties and unordered comparisons.
                bounded = xp.where(moved < update.upper, moved, update.upper)
                bounded = xp.where(update.lower < bounded, bounded, update.lower)
                updates.append((update.field, bounded if device else bounded.tolist()))
    except UnicodeEncodeError:
        return False
    if device:
        for field, values in updates:
            columns[field][:] = values
            execution.agent_writes[field] = None
            execution.dirty_columns.add(field)
    else:
        _queue_batch_updates(pending, ids, bindings, updates)
    return True


def try_batch(agents, method_name: str) -> int:
    """Apply an equivalent batch and return the number of calls replaced."""
    from .sequences import AgentList

    if type(agents) is not AgentList or len(agents._agent_objects) < _MIN_AGENTS:
        return 0
    tracked = _tracked_sweep_method(agents, method_name)
    if tracked is None:
        return 0
    state, cls, method = tracked
    execution = state.get("_execution")
    if (
        not state.get("_auto_batch_enabled") or execution is None
        or state["_contract"].mode != "off" or state["_contract"]._in_replay
        or state.get("_activation_override_order") is not None or _interpreter_instrumented()
        or any(getattr_static(cls, name, _MISSING) is not _MISSING for name in ("id", "model"))
    ):
        return 0
    plan = _agent_plan(method.__code__)
    clipped_plan = _clipped_draw_plan(method.__code__) if plan is None else ()
    if plan is None and not clipped_plan:
        return 0
    fields = _plan_fields(plan) if plan is not None else {update.field for update in clipped_plan}
    if any(getattr_static(cls, field, _MISSING) is not _MISSING for field in fields):
        return 0
    if execution.exposed_frame is not None:
        reconcile_host_frame(agents.model, execution)
    rows = _tracked_sweep_rows(agents, method_name, state, cls)
    if rows is None:
        return 0
    frame, ids, bindings = rows
    pending = state["_pending_writes"]
    if any(callable(attrs.get(field)) for field in fields for attrs in bindings):
        return 0
    if clipped_plan:
        if not _apply_clipped_draw_plan(agents, method, clipped_plan, cls, state, frame, ids, bindings):
            return 0
        state["_auto_batches"] += 1
        state["_auto_batched_agent_calls"] += len(ids)
        return len(ids)
    condition = plan.condition
    draw_mask = None
    if isinstance(condition, _DrawCondition):
        context, _ = _draw_context(type(agents.model), cls, state, frame, bindings)
        if context is None:
            return 0
        seed, step = context
        try:
            draw_mask = _draw_mask(condition, (
                keyed_uniform(seed, step, aid, condition.event, condition.draw_index) for aid in ids
            ))
        except UnicodeEncodeError:
            return 0
    if execution.config.device == "gpu":
        if not _apply_device_plan(agents.model, execution, plan, fields, frame, pending, ids, draw_mask):
            return 0
        state["_auto_batches"] += 1
        state["_auto_batched_agent_calls"] += len(ids)
        return len(ids)

    mask = None
    selected_ids = ids
    selected_bindings = bindings
    if draw_mask is not None:
        mask = pl.Series(draw_mask, dtype=pl.Boolean)
    elif condition is not None:
        mask = _condition_mask(condition, frame, pending, ids)
        if mask is None:
            return 0
    if mask is not None:
        selected_rows = mask.arg_true().to_list()
        selected_ids = [ids[pos] for pos in selected_rows]
        selected_bindings = [bindings[pos] for pos in selected_rows]

    updates = []
    if selected_ids:
        for field, delta in plan.adjustments:
            column = _read_column(frame, pending, ids, field)
            if column is None:
                return 0
            if mask is not None:
                column = column.filter(mask)
            low, high = column.min(), column.max()
            if low + delta < _MIN_I64 or high + delta > _MAX_I64:
                return 0
            updates.append((field, (column + delta).to_list()))

    _queue_batch_updates(pending, selected_ids, selected_bindings, updates)
    state["_auto_batches"] += 1
    state["_auto_batched_agent_calls"] += len(ids)
    return len(ids)
