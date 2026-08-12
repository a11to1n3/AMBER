from typing import Type, Dict, Any, List, Callable, Optional, Literal
from pathlib import Path
import json
import os
import pickle
import secrets
import shutil
import tempfile
import polars as pl
import numpy as np
import itertools
import traceback as _traceback_mod
from .model import Model

# SMAC is an optional dependency. Probe a real import path (not just the
# package name) so a broken smac/sklearn combo still reports HAS_SMAC=False.
HAS_SMAC = False
_SMAC_IMPORT_ERROR: Optional[BaseException] = None
try:
    import smac  # noqa: F401
    from smac import HyperparameterOptimizationFacade  # noqa: F401
    HAS_SMAC = True
except Exception as exc:  # ImportError, or sklearn.tree._tree.DTYPE breakage
    _SMAC_IMPORT_ERROR = exc

# Large finite penalty used when on_error='penalize' maps a failed evaluation.
_PENALTY_COST = 1e10

# Phrases SMAC / ConfigSpace may emit when the configuration space is
# exhausted mid-search. Only these (message-matched) failures are treated as
# non-fatal "stop and return partial history"; everything else re-raises.
_SEARCH_EXHAUSTED_MARKERS = (
    "configuration space exhausted",
    "no more configurations",
    "no configurations left",
    "cannot sample more",
    "exhausted the configuration space",
)


def _check_smac():
    """Check if SMAC is available, raise helpful error if not."""
    if not HAS_SMAC:
        hint = (
            "Install the advanced extra with a SMAC-compatible scikit-learn:\n"
            "  pip install 'ambr[advanced]'\n"
            "  # or: pip install 'smac>=2,<3' 'scikit-learn>=1.6.1,<1.9'\n"
            "SMAC 2.4 is incompatible with scikit-learn 1.9+ "
            "(missing sklearn.tree._tree.DTYPE; see automl/SMAC3#1314)."
        )
        detail = f"\nUnderlying error: {_SMAC_IMPORT_ERROR!r}" if _SMAC_IMPORT_ERROR else ""
        raise ImportError(
            "SMAC is required for advanced optimization features. " + hint + detail
        )


class RemoteObjectiveError(RuntimeError):
    """Raised in the parent when a worker objective failed under ``on_error='raise'``.

    Used when the original exception cannot be pickled across processes, or
    when only a structured type/message/traceback payload is available.
    """

    def __init__(
        self,
        message: str,
        *,
        exception_type: str = "Exception",
        traceback_text: str = "",
    ):
        super().__init__(message)
        self.exception_type = exception_type
        self.traceback_text = traceback_text or ""


def _is_search_exhausted(exc: BaseException) -> bool:
    """Return True only for SMAC configuration-space exhaustion.

    Matching rules (narrow — no substring matching on arbitrary type names):

    1. ``isinstance(exc, smac.main.exceptions.ConfigurationSpaceExhaustedException)``
       when SMAC is importable.
    2. Compatibility: exact class name
       ``type(exc).__name__ == "ConfigurationSpaceExhaustedException"``
       (covers re-exports / empty-message instances).
    3. Compatibility: plain ``Exception`` (or subclass that is *not* a custom
       misnamed type) whose message contains a known SMAC exhaustion phrase.

    Explicitly **not** matched: e.g. ``ConfigurationDataExhaustedError`` —
    names that merely contain both "configuration" and "exhausted".
    """
    # 1) Canonical SMAC type
    try:
        from smac.main.exceptions import ConfigurationSpaceExhaustedException

        if isinstance(exc, ConfigurationSpaceExhaustedException):
            return True
    except Exception:
        pass

    # 2) Exact class name only (no "configuration"+"exhausted" substring pair)
    if type(exc).__name__ == "ConfigurationSpaceExhaustedException":
        return True

    # 3) Message markers — only for generic Exception base (not custom
    #    domain errors that happen to mention "exhausted").
    if type(exc) is Exception or type(exc).__name__ in (
        "RuntimeError",
        "StopIteration",
    ):
        msg = str(exc).lower()
        return any(marker in msg for marker in _SEARCH_EXHAUSTED_MARKERS)
    return False


def _safe_exc_message(exc: BaseException) -> str:
    """Best-effort message even when ``str(exc)`` / ``repr(exc)`` raise."""
    try:
        return str(exc)
    except Exception:
        pass
    try:
        return repr(exc)
    except Exception:
        return f"<unprintable {type(exc).__name__}>"


def _safe_exc_traceback(exc: BaseException) -> str:
    try:
        return "".join(
            _traceback_mod.format_exception(type(exc), exc, exc.__traceback__)
        )
    except Exception:
        return ""


def _failure_record(
    configuration: Dict[str, Any],
    exc: BaseException,
) -> Dict[str, Any]:
    """Structured failure record for ``on_error='penalize'`` paths."""
    return {
        "configuration": dict(configuration),
        "exception_type": type(exc).__name__,
        "message": _safe_exc_message(exc),
        "traceback": _safe_exc_traceback(exc),
    }


def _coerce_fidelity_budget(budget: Any, fidelity_type: Optional[str]) -> Any:
    if budget is None:
        return None
    if fidelity_type == "int":
        return int(round(float(budget)))
    return float(budget)


def _error_payload(exc: BaseException) -> Dict[str, Any]:
    """JSON-serializable fallback when the exception itself cannot be pickled."""
    return {
        "kind": "amber_remote_objective_error",
        "exception_type": type(exc).__name__,
        "message": _safe_exc_message(exc),
        "traceback": _safe_exc_traceback(exc),
    }


def _error_json_path(path: str) -> Path:
    """Sibling structured-JSON path for ``first_error`` side-channel."""
    p = Path(path)
    # Support both ``foo.pkl`` → ``foo.json`` and bare paths → ``.json`` suffix.
    if p.suffix:
        return p.with_suffix(".json")
    return Path(str(p) + ".json")


def _write_first_error(path: Optional[str], exc: BaseException) -> None:
    """Side-channel for the first target failure (multi-process safe).

    Always writes a structured JSON sibling (type / message / traceback) so the
    parent can raise even when:

    * ``pickle.dumps(exc)`` fails, or
    * pickle succeeds but the parent cannot unpickle the class, or
    * ``str(exc)`` / ``repr(exc)`` fail.

    Pickle of the original exception is best-effort only.
    """
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        jp = _error_json_path(path)
        # Structured payload first — this is the reliable raise signal.
        if not jp.exists():
            try:
                jp.write_text(
                    json.dumps(_error_payload(exc), default=str),
                    encoding="utf-8",
                )
            except Exception:
                try:
                    jp.write_text(
                        json.dumps(
                            {
                                "kind": "amber_remote_objective_error",
                                "exception_type": type(exc).__name__,
                                "message": "<unprintable exception>",
                                "traceback": "",
                            }
                        ),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
        # Best-effort pickle of the original exception (same-process re-raise).
        if not p.exists():
            try:
                p.write_bytes(
                    pickle.dumps(exc, protocol=pickle.HIGHEST_PROTOCOL)
                )
            except Exception:
                pass
    except Exception:
        pass


def _payload_to_remote_error(payload: Dict[str, Any]) -> RemoteObjectiveError:
    et = str(payload.get("exception_type") or "Exception")
    msg = str(payload.get("message") or "")
    tb = str(payload.get("traceback") or "")
    text = f"{et}: {msg}" if msg else et
    if tb:
        text = f"{text}\n\nRemote traceback:\n{tb}"
    return RemoteObjectiveError(text, exception_type=et, traceback_text=tb)


def _load_error_side_channel(path: Optional[str]) -> Optional[BaseException]:
    """Load a pickled exception and/or structured remote-objective payload.

    Preference order:

    1. Structured JSON sibling (always raiseable, survives unpickleable classes)
    2. Pickle of the original exception (when loadable)
    3. Structured JSON written to the primary path
    4. ``RemoteObjectiveError`` if a side-channel file existed but was unreadable
    """
    if not path:
        return None
    p = Path(path)
    jp = _error_json_path(path)
    saw_file = p.is_file() or jp.is_file()

    # Prefer structured JSON — never depends on exception class pickling.
    if jp.is_file():
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("kind") == (
                "amber_remote_objective_error"
            ):
                return _payload_to_remote_error(payload)
            if isinstance(payload, dict):
                return _payload_to_remote_error(payload)
        except Exception:
            pass

    if p.is_file():
        raw = p.read_bytes()
        # Primary path may itself be JSON (older writers / last-resort marker).
        try:
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict):
                return _payload_to_remote_error(payload)
        except Exception:
            pass
        try:
            obj = pickle.loads(raw)
            if isinstance(obj, BaseException):
                return obj
        except Exception:
            # Pickle present but not loadable (local class in worker) — still
            # signal failure rather than returning None → silent inf.
            return RemoteObjectiveError(
                "Remote SMAC objective failed (exception not unpickleable in "
                "parent process; structured side-channel missing or corrupt)"
            )

    if saw_file:
        return RemoteObjectiveError(
            "Remote SMAC objective failed (unreadable error side-channel)"
        )
    return None


def _append_failure_jsonl(path: Optional[str], record: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def _smac_trial_target(
    config,
    seed: int = 0,
    budget=None,
    *,
    model_type: Type[Model],
    objective: Callable[[Model], float],
    fixed_params: Dict[str, Any],
    fidelity_name: Optional[str],
    fidelity_type: Optional[str],
    on_error: str,
    error_path: Optional[str] = None,
    failures_path: Optional[str] = None,
) -> float:
    """Module-level SMAC target body (pickle-safe context, not the public sig)."""
    params = dict(fixed_params or {})
    params.update(dict(config))
    if budget is not None and fidelity_name is not None:
        params[fidelity_name] = _coerce_fidelity_budget(budget, fidelity_type)
    params.setdefault("seed", seed)
    params.setdefault("show_progress", False)
    model = model_type(params)
    try:
        model.results = model.run()
        value = objective(model)
    except Exception as exc:
        if on_error == "raise":
            _write_first_error(error_path, exc)
            raise
        _append_failure_jsonl(failures_path, _failure_record(params, exc))
        return _PENALTY_COST
    if value is None or not np.isfinite(value):
        err = ValueError(f"Objective returned non-finite value: {value!r}")
        if on_error == "raise":
            _write_first_error(error_path, err)
            raise err
        _append_failure_jsonl(failures_path, _failure_record(params, err))
        return _PENALTY_COST
    return float(value)


class _SMACTrialEvaluator:
    """Pickle-safe SMAC target with a clean ``(config, seed[, budget])`` signature.

    ``functools.partial`` leaks keyword-only parameters into
    ``inspect.signature``, which triggers SMAC's "argument X is not set"
    warnings. This callable class exposes only the arguments SMAC actually
    injects. SMAC also fingerprints ``target.__code__`` (not
    ``target.__call__.__code__``), so we proxy that attribute.
    """

    __slots__ = (
        "model_type",
        "objective",
        "fixed_params",
        "fidelity_name",
        "fidelity_type",
        "on_error",
        "error_path",
        "failures_path",
        "use_budget",
    )

    def __init__(
        self,
        *,
        model_type: Type[Model],
        objective: Callable[[Model], float],
        fixed_params: Dict[str, Any],
        fidelity_name: Optional[str],
        fidelity_type: Optional[str],
        on_error: str,
        error_path: Optional[str],
        failures_path: Optional[str],
        use_budget: bool,
    ):
        self.model_type = model_type
        self.objective = objective
        self.fixed_params = fixed_params
        self.fidelity_name = fidelity_name
        self.fidelity_type = fidelity_type
        self.on_error = on_error
        self.error_path = error_path
        self.failures_path = failures_path
        self.use_budget = bool(use_budget)

    @property
    def __code__(self):
        # SMAC TargetFunctionRunner.meta reads target.__code__.co_code
        return type(self).__call__.__code__

    def __call__(self, config, seed: int = 0, budget=None) -> float:
        return _smac_trial_target(
            config,
            seed,
            budget,
            model_type=self.model_type,
            objective=self.objective,
            fixed_params=self.fixed_params,
            fidelity_name=self.fidelity_name,
            fidelity_type=self.fidelity_type,
            on_error=self.on_error,
            error_path=self.error_path,
            failures_path=self.failures_path,
        )


class _SMACTrialEvaluatorSF(_SMACTrialEvaluator):
    """Single-fidelity target: only ``(config, seed)`` — no budget warning."""

    def __call__(self, config, seed: int = 0) -> float:  # type: ignore[override]
        return _smac_trial_target(
            config,
            seed,
            None,
            model_type=self.model_type,
            objective=self.objective,
            fixed_params=self.fixed_params,
            fidelity_name=self.fidelity_name,
            fidelity_type=self.fidelity_type,
            on_error=self.on_error,
            error_path=self.error_path,
            failures_path=self.failures_path,
        )


def _extract_metric_value(model_data: Any, metric: str) -> float:
    """Return the last non-null finite numeric value of ``metric``.

    Raises
    ------
    KeyError
        If ``metric`` is not a column of ``model_data``.
    ValueError
        If the column is empty after dropping nulls, or the last value is
        non-numeric / non-finite.
    """
    if metric not in model_data.columns:
        raise KeyError(
            f"Metric {metric!r} was not recorded; available: {list(model_data.columns)}"
        )

    values = model_data[metric].drop_nulls()
    if values.is_empty():
        raise ValueError(f"Metric {metric!r} contains no values")

    value = values[-1]
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Metric {metric!r} is non-numeric: {value!r}"
        ) from exc
    if not np.isfinite(numeric):
        raise ValueError(f"Metric {metric!r} is non-finite: {value!r}")
    return numeric


# Simple ParameterSpace for basic optimization functions
class ParameterSpace:
    """Define the parameter space for optimization."""

    def __init__(self, parameters: Dict[str, Any]):
        """Initialize parameter space.

        Args:
            parameters: Dictionary mapping parameter names to values or ranges
        """
        self.parameters = parameters

    def sample(self, rng=None) -> Dict[str, Any]:
        """Sample a random parameter combination.

        Args:
            rng: optional ``numpy.random.Generator`` for reproducibility. When
                omitted, a fresh local generator is used (never the global
                ``np.random`` state).

        Returns:
            Dictionary with parameter values
        """
        from .experiment import IntRange

        if rng is None:
            rng = np.random.default_rng()
        result = {}
        for name, value in self.parameters.items():
            if isinstance(value, list):
                result[name] = value[int(rng.integers(0, len(value)))]
            elif isinstance(value, IntRange):  # IntRange objects (exclusive end)
                result[name] = int(rng.integers(value.start, value.end))
            else:  # Fixed value
                result[name] = value
        return result

    def grid_sample(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations in a grid.

        Returns:
            List of parameter dictionaries
        """
        from .experiment import IntRange

        param_lists = {}
        for name, value in self.parameters.items():
            if isinstance(value, list):
                param_lists[name] = value
            elif isinstance(value, IntRange):  # IntRange objects (exclusive end)
                param_lists[name] = list(range(value.start, value.end))
            else:  # Fixed value
                param_lists[name] = [value]

        # Generate all combinations
        names = list(param_lists.keys())
        combinations = list(itertools.product(*[param_lists[name] for name in names]))

        return [dict(zip(names, combo)) for combo in combinations]


def objective_function(model_class: Type[Model], parameters: Dict[str, Any],
                      metric: str, iterations: int = 1, minimize: bool = False) -> float:
    """Evaluate objective function for a model with given parameters.

    Args:
        model_class: Model class to instantiate
        parameters: Parameters to pass to model
        metric: Name of metric to optimize (must be present, non-empty, finite)
        iterations: Number of iterations to average over (must be ``>= 1``)
        minimize: Whether to minimize (True) or maximize (False). Sorting of
            search results uses this flag; the returned value is always the
            raw metric average (not negated).

    Returns:
        Objective value (mean of the last recorded metric over iterations)

    Raises:
        ValueError: If ``iterations < 1``, the metric column is empty, or the
            last value is non-numeric / non-finite.
        KeyError: If ``metric`` was never recorded on the model frame.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations!r}")

    total = 0.0

    for _ in range(iterations):
        # Disable progress reporting for optimization
        model_params = parameters.copy()
        model_params['show_progress'] = False
        model = model_class(model_params)
        results = model.run()

        model_data = results['model']
        total += _extract_metric_value(model_data, metric)

    return total / iterations


def grid_search(model_class: Type[Model], parameter_space: ParameterSpace,
                metric: str, iterations: int = 1, minimize: bool = False) -> List[Dict[str, Any]]:
    """Perform grid search optimization.

    Args:
        model_class: Model class to optimize
        parameter_space: Parameter space to search
        metric: Metric to optimize
        iterations: Number of iterations per parameter combination
        minimize: Whether to minimize the metric

    Returns:
        List of results sorted by objective value (best first)
    """
    results = []

    for params in parameter_space.grid_sample():
        obj_value = objective_function(model_class, params, metric, iterations, minimize)
        results.append({
            'parameters': params,
            'objective': obj_value
        })

    # Sort by objective value (descending for maximization, ascending for minimization)
    results.sort(key=lambda x: x['objective'], reverse=not minimize)

    return results


def random_search(model_class: Type[Model], parameter_space: ParameterSpace,
                  metric: str, n_samples: int = 10, iterations: int = 1,
                  minimize: bool = False, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Perform random search optimization.

    Args:
        model_class: Model class to optimize
        parameter_space: Parameter space to search
        metric: Metric to optimize
        n_samples: Number of random samples to evaluate
        iterations: Number of iterations per parameter combination
        minimize: Whether to minimize the metric
        seed: Random seed for reproducibility

    Returns:
        List of results sorted by objective value (best first)
    """
    rng = np.random.default_rng(seed)  # local generator; never touches global np.random

    results = []

    for _ in range(n_samples):
        params = parameter_space.sample(rng)
        obj_value = objective_function(model_class, params, metric, iterations, minimize)
        results.append({
            'parameters': params,
            'objective': obj_value
        })

    # Sort by objective value
    results.sort(key=lambda x: x['objective'], reverse=not minimize)

    return results


def bayesian_optimization(model_class: Type[Model], parameter_space: ParameterSpace,
                         metric: str, n_calls: int = 10, iterations: int = 1,
                         minimize: bool = False, random_state: Optional[int] = None,
                         n_initial_design: int = 5,
                         on_error: Literal["raise", "penalize"] = "raise",
                         ) -> List[Dict[str, Any]]:
    """Perform Bayesian optimisation using SMAC3's Gaussian Process facade.

    Converts the simple ``ParameterSpace`` to a SMAC3 ``ConfigurationSpace``
    internally and runs true Bayesian optimisation with Expected Improvement
    acquisition. Requires SMAC3 to be installed (``pip install smac``).

    Args:
        model_class: Model class to optimize.
        parameter_space: Parameter space to search.
        metric: Metric to optimize.
        n_calls: Total number of function evaluations.
        iterations: Number of iterations per parameter combination.
        minimize: Whether to minimize (True) or maximize (False).
        random_state: Random state for reproducibility.
        n_initial_design: Number of initial random designs before
            Bayesian search begins.
        on_error: How to handle target evaluation failures.

            * ``'raise'`` (default) — propagate the exception.
            * ``'penalize'`` — map the failure to a large finite cost and
              append a structured record to the returned result list under
              ``'failure'`` (configuration, exception type, message,
              traceback). Successful trials omit the ``failure`` key.

    Returns:
        List of results sorted by objective value (best first). Failed
        trials (when ``on_error='penalize'``) sort as worst.
    """
    _check_smac()
    if on_error not in ("raise", "penalize"):
        raise ValueError(
            f"on_error must be 'raise' or 'penalize', got {on_error!r}"
        )

    from ConfigSpace import (
        ConfigurationSpace,
        UniformIntegerHyperparameter,
        CategoricalHyperparameter,
    )
    from smac import HyperparameterOptimizationFacade, Scenario
    from smac.acquisition.function import EI
    from smac.model.random_forest import RandomForest
    from smac.initial_design import LatinHypercubeInitialDesign

    from .experiment import IntRange

    # --- build ConfigurationSpace from ParameterSpace -----------------------
    cs = ConfigurationSpace(seed=random_state or 0)
    cat_params: Dict[str, List[Any]] = {}
    fixed_params: Dict[str, Any] = {}

    for name, value in parameter_space.parameters.items():
        if isinstance(value, list):
            # Categorical — SMAC requires strings
            str_choices = [str(v) for v in value]
            cat_params[name] = value  # keep original mapping
            hp = CategoricalHyperparameter(
                name=name,
                choices=str_choices,
                default_value=str_choices[0],
            )
            cs.add(hp)
        elif isinstance(value, IntRange):
            hp = UniformIntegerHyperparameter(
                name=name,
                lower=value.start,
                upper=value.end - 1,  # IntRange.end is exclusive
                default_value=value.start,
            )
            cs.add(hp)
        elif isinstance(value, bool):
            # bool is a subclass of int — keep before the int branch.
            fixed_params[name] = value
        elif isinstance(value, (int, float, str)):
            # Fixed scalar (incl. float) — not optimised. Equal-bound
            # UniformFloatHyperparameter is rejected by ConfigSpace.
            fixed_params[name] = (
                float(value) if isinstance(value, float) else value
            )
        elif isinstance(value, (np.floating, np.integer)):
            # NumPy scalars — still fixed constants, not search dimensions.
            fixed_params[name] = (
                float(value) if isinstance(value, np.floating) else int(value)
            )
        else:
            raise TypeError(
                f"Unsupported parameter type for {name!r}: {type(value)}"
            )

    # --- scenario & SMAC facade --------------------------------------------
    # Use a temporary output directory so each call starts fresh.
    tmp_dir = tempfile.mkdtemp(prefix="amber_bayes_")
    error_path = str(Path(tmp_dir) / "first_error.pkl")
    failures_path = (
        str(Path(tmp_dir) / "failures.jsonl") if on_error == "penalize" else None
    )
    scenario = Scenario(
        cs,
        n_trials=n_calls,
        seed=random_state,
        deterministic=True,
        output_directory=tmp_dir,
    )

    # Side-channel for penalized failures (config id is not stable across
    # SMAC versions, so we key by a frozen parameter tuple).
    failure_by_params: Dict[tuple, Dict[str, Any]] = {}

    def _resolve_params(config: dict) -> Dict[str, Any]:
        params = dict(fixed_params)
        for k, v in dict(config).items():
            if k in cat_params:
                str_choices = [str(cv) for cv in cat_params[k]]
                try:
                    idx = str_choices.index(str(v))
                    params[k] = cat_params[k][idx]
                except ValueError:
                    params[k] = v
            else:
                params[k] = v
        return params

    # Target function for SMAC (always minimises)
    def _target(config: dict, seed: int = 0) -> float:
        # Merge SMAC config + fixed params + restore categorical types
        params = _resolve_params(config)
        try:
            obj = objective_function(
                model_class, params, metric, iterations, minimize
            )
        except Exception as exc:
            if on_error == "raise":
                _write_first_error(error_path, exc)
                raise
            key = tuple(sorted(params.items()))
            rec = _failure_record(params, exc)
            failure_by_params[key] = rec
            _append_failure_jsonl(failures_path, rec)
            # SMAC minimises; a large positive cost is "worst" for both
            # minimize and maximize (maximize path negates successful objs).
            return _PENALTY_COST
        if obj is None or not np.isfinite(obj):
            exc = ValueError(f"Objective returned non-finite value: {obj!r}")
            if on_error == "raise":
                _write_first_error(error_path, exc)
                raise exc
            key = tuple(sorted(params.items()))
            rec = _failure_record(params, exc)
            failure_by_params[key] = rec
            _append_failure_jsonl(failures_path, rec)
            return _PENALTY_COST
        # SMAC always minimises; if the user wants to maximise, negate
        return -obj if not minimize else obj

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=_target,
        model=RandomForest(configspace=cs),
        acquisition_function=EI(),
        initial_design=LatinHypercubeInitialDesign(
            scenario=scenario,
            n_configs=min(n_initial_design, n_calls),
        ),
    )

    try:
        try:
            smac.optimize()
        except Exception as exc:
            first = _load_error_side_channel(error_path)
            if on_error == "raise" and first is not None:
                raise first from exc
            # Only the documented "search exhausted" condition is non-fatal —
            # proceed with whatever SMAC evaluated so far (runhistory still has
            # the partial results). All other exceptions re-raise.
            if not _is_search_exhausted(exc):
                raise

        if on_error == "raise":
            first = _load_error_side_channel(error_path)
            if first is not None:
                raise first

        # --- collect history ---------------------------------------------------
        results: List[Dict[str, Any]] = []
        for config in smac.runhistory.get_configs():
            try:
                cost = smac.runhistory.get_cost(config)
            except KeyError:
                cost = float('inf')
            params = _resolve_params(config)
            entry: Dict[str, Any] = {
                'parameters': params,
                'objective': -cost if not minimize else cost,
            }
            fail = failure_by_params.get(tuple(sorted(params.items())))
            if fail is not None:
                entry['failure'] = fail
                # Keep penalized objectives as worst-sorted even after the
                # maximize-path negation of a large positive cost.
                entry['objective'] = -_PENALTY_COST if not minimize else _PENALTY_COST
            results.append(entry)

        results.sort(key=lambda x: x['objective'], reverse=not minimize)
        return results
    finally:
        if os.environ.get("AMBER_SMAC_KEEP_OUTPUT", "").strip() not in (
            "1",
            "true",
            "yes",
        ):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# Advanced SMAC-based ParameterSpace for complex optimization
class SMACParameterSpace:
    """Define the parameter space for SMAC optimization."""

    def __init__(self):
        """Initialize parameter space."""
        self.parameters = {}
        self.fidelity_parameters = {}

    def add_parameter(self, name: str, param_type: str,
                     bounds: Optional[tuple] = None,
                     choices: Optional[List[Any]] = None,
                     default: Any = None,
                     is_fidelity: bool = False):
        """Add a parameter to the space.

        Args:
            name: Parameter name
            param_type: Type of parameter ('float', 'int', 'categorical')
            bounds: Tuple of (min, max) for numeric parameters
            choices: List of possible values for categorical parameters
            default: Default value
            is_fidelity: Whether this is a fidelity parameter
        """
        if param_type not in ['float', 'int', 'categorical']:
            raise ValueError("param_type must be 'float', 'int', or 'categorical'")

        if param_type in ['float', 'int'] and bounds is None:
            raise ValueError(f"bounds must be provided for {param_type} parameters")

        if param_type == 'categorical' and choices is None:
            raise ValueError("choices must be provided for categorical parameters")

        param_dict = {
            'type': param_type,
            'bounds': bounds,
            'choices': choices,
            'default': default
        }

        if is_fidelity:
            self.fidelity_parameters[name] = param_dict
        else:
            self.parameters[name] = param_dict

    def get_configspace(self):
        """Get the SMAC configuration space."""
        from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, \
            UniformIntegerHyperparameter, CategoricalHyperparameter

        cs = ConfigurationSpace()

        # Add regular parameters
        for name, param in self.parameters.items():
            if param['type'] == 'float':
                hp = UniformFloatHyperparameter(
                    name=name,
                    lower=param['bounds'][0],
                    upper=param['bounds'][1],
                    default_value=param['default']
                )
            elif param['type'] == 'int':
                hp = UniformIntegerHyperparameter(
                    name=name,
                    lower=param['bounds'][0],
                    upper=param['bounds'][1],
                    default_value=param['default']
                )
            else:  # categorical
                hp = CategoricalHyperparameter(
                    name=name,
                    choices=param['choices'],
                    default_value=param['default']
                )
            cs.add_hyperparameter(hp)

        # Fidelity parameters are *not* searchable hyperparameters. Their
        # bounds define SMAC multi-fidelity min_budget/max_budget only; the
        # budget is injected into the trial via the target-function argument.
        # Adding them to the configspace caused independent samples that
        # disagreed with the Successive Halving budget actually evaluated.
        return cs

class SMACOptimizer:
    """Optimize model parameters using SMAC (supported options for AMBER 0.5.x).

    Supported strategies: ``bayesian`` (default), ``random``,
    ``algorithm_configuration``. ``strategy='random'`` and
    ``use_random_search=True`` both select SMAC's :class:`RandomFacade`.

    Supported acquisition functions: ``ei``, ``lcb``, ``pi``, ``eips``, ``ts``
    (Thompson sampling). Surrogate models: ``random_forest`` only.

    Non-search parameters (e.g. ``n_agents``, ``steps``) belong in
    ``fixed_params`` — they are merged into every trial.
    """

    def __init__(self, model_type: Type[Model],
                 param_space: SMACParameterSpace,
                 objective: Callable[[Model], float],
                 n_trials: int = 100,
                 n_workers: int = 1,
                 seed: Optional[int] = None,
                 strategy: str = 'bayesian',
                 acquisition_function: str = 'ei',
                 initial_design: str = 'latin_hypercube',
                 surrogate_model: str = 'random_forest',
                 use_multi_fidelity: bool = False,
                 use_random_search: bool = False,
                 fixed_params: Optional[Dict[str, Any]] = None,
                 on_error: Literal["raise", "penalize"] = "raise"):
        """Initialize the optimizer.

        Args:
            model_type: Class of model to optimize
            param_space: Parameter space definition
            objective: Function that takes a model and returns a score to minimize
            n_trials: Number of optimization trials
            n_workers: Number of parallel workers
            seed: Random seed
            strategy: ``'bayesian'``, ``'random'``, or ``'algorithm_configuration'``
            acquisition_function: ``'ei'``, ``'lcb'``, ``'pi'``, ``'eips'``, or
                ``'ts'`` (Thompson sampling). ``'log_ei'`` is **not** supported.
            initial_design: ``'latin_hypercube'``, ``'random'``, or ``'sobol'``
            surrogate_model: Only ``'random_forest'`` is supported. GP and
                ``random_forest_with_instances`` raise ``ValueError``.
            use_multi_fidelity: If True, requires fidelity parameters on
                ``param_space`` (``is_fidelity=True``) with numeric bounds used
                as SMAC ``min_budget`` / ``max_budget`` for Successive Halving.
            use_random_search: Alias for ``strategy='random'`` (RandomFacade)
            fixed_params: Merged into every trial (e.g. ``n_agents``, ``steps``,
                ``show_progress``). Overridden by search-space keys of the same name.
            on_error: ``'raise'`` (default) propagates evaluation failures;
                ``'penalize'`` maps them to a large finite cost and records a
                structured failure entry on ``self.failures``.
        """
        # Check SMAC availability and do lazy imports
        _check_smac()
        if on_error not in ("raise", "penalize"):
            raise ValueError(
                f"on_error must be 'raise' or 'penalize', got {on_error!r}"
            )
        from smac import (
            HyperparameterOptimizationFacade,
            Scenario,
            MultiFidelityFacade,
            RandomFacade,
            AlgorithmConfigurationFacade,
        )
        from smac.model.random_forest import RandomForest
        from smac.acquisition.function import EI, LCB, PI, EIPS, TS
        from smac.acquisition.maximizer import LocalAndSortedRandomSearch
        from smac.initial_design import (
            LatinHypercubeInitialDesign,
            RandomInitialDesign,
            SobolInitialDesign,
        )
        from smac.intensifier import SuccessiveHalving

        self.model_type = model_type
        self.param_space = param_space
        self.objective = objective
        self.n_trials = n_trials
        self.n_workers = n_workers
        self.seed = seed
        self.on_error = on_error
        self.fixed_params: Dict[str, Any] = dict(fixed_params or {})
        self.failures: List[Dict[str, Any]] = []
        self._fidelity_name: Optional[str] = None
        self._fidelity_type: Optional[str] = None

        # Initialize SMAC components
        self.configspace = param_space.get_configspace()

        # Select initial design class
        if initial_design == 'latin_hypercube':
            initial_design_cls = LatinHypercubeInitialDesign
        elif initial_design == 'random':
            initial_design_cls = RandomInitialDesign
        elif initial_design == 'sobol':
            initial_design_cls = SobolInitialDesign
        else:
            raise ValueError(f"Unknown initial design: {initial_design}")

        # Select acquisition function (supported set only)
        if acquisition_function == 'ei':
            acq_func = EI()
        elif acquisition_function == 'lcb':
            acq_func = LCB()
        elif acquisition_function == 'pi':
            acq_func = PI()
        elif acquisition_function == 'eips':
            acq_func = EIPS()
        elif acquisition_function == 'ts':
            acq_func = TS()
        elif acquisition_function == 'log_ei':
            raise ValueError(
                "acquisition_function='log_ei' is not supported (it previously "
                "mis-mapped to Thompson sampling). Use 'ei' or 'ts'."
            )
        else:
            raise ValueError(
                f"Unknown acquisition function: {acquisition_function!r}. "
                "Supported: 'ei', 'lcb', 'pi', 'eips', 'ts'."
            )

        # Surrogate model — only RandomForest is wired for SMAC 2.x here.
        if surrogate_model == 'random_forest':
            model = RandomForest(self.configspace)
        elif surrogate_model in (
            'gaussian_process',
            'random_forest_with_instances',
        ):
            raise ValueError(
                f"surrogate_model={surrogate_model!r} is not supported in this "
                "AMBER release. Use surrogate_model='random_forest' (default)."
            )
        else:
            raise ValueError(
                f"Unknown model type: {surrogate_model!r}. "
                "Supported: 'random_forest'."
            )

        want_random = bool(use_random_search) or strategy == "random"
        want_mf = bool(use_multi_fidelity)
        # Validate strategy before allocating temp dirs / starting Dask.
        if want_mf:
            facade_kind = "mf"
        elif want_random:
            facade_kind = "random"
        elif strategy == "algorithm_configuration":
            facade_kind = "ac"
        elif strategy == "bayesian":
            facade_kind = "bayesian"
        else:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                "Supported: 'bayesian', 'random', 'algorithm_configuration'."
            )

        if want_mf:
            if not param_space.fidelity_parameters:
                raise ValueError(
                    "use_multi_fidelity=True requires at least one parameter "
                    "added with is_fidelity=True (numeric bounds become "
                    "min_budget/max_budget for Successive Halving)."
                )
            fname, fparam = next(iter(param_space.fidelity_parameters.items()))
            if fparam.get("bounds") is None:
                raise ValueError(
                    f"Fidelity parameter {fname!r} must have numeric bounds"
                )
            lo, hi = fparam["bounds"]
            if lo is None or hi is None or float(lo) >= float(hi):
                raise ValueError(
                    f"Fidelity parameter {fname!r} bounds must satisfy min < max"
                )
            if fparam.get("type") == "int":
                lo, hi = int(lo), int(hi)
            self._fidelity_name = fname
            self._fidelity_type = fparam.get("type")
            mf_budgets = (lo, hi)
        else:
            mf_budgets = None

        # Unique SMAC run directory — created only after validation so failed
        # constructors do not leak amber_smac_* directories.
        self._output_dir = tempfile.mkdtemp(prefix="amber_smac_")
        # Error side-channel lives *outside* the SMAC output directory so it
        # survives SMAC/Dask touching that tree. Use a path that does **not**
        # pre-create an empty file (empty files would look like a failed
        # side-channel and spuriously raise after a successful run).
        self._error_path = str(
            Path(tempfile.gettempdir())
            / f"amber_smac_err_{os.getpid()}_{secrets.token_hex(8)}.pkl"
        )
        self._failures_path = (
            str(Path(self._output_dir) / "failures.jsonl")
            if on_error == "penalize"
            else None
        )
        try:
            scenario_kwargs: Dict[str, Any] = {
                "n_trials": n_trials,
                "n_workers": n_workers,
                "seed": seed,
                "output_directory": self._output_dir,
            }
            if mf_budgets is not None:
                scenario_kwargs["min_budget"] = mf_budgets[0]
                scenario_kwargs["max_budget"] = mf_budgets[1]

            self.scenario = Scenario(self.configspace, **scenario_kwargs)

            # Clean (config, seed[, budget]) signature — no partial kwargs.
            eval_kwargs = dict(
                model_type=self.model_type,
                objective=self.objective,
                fixed_params=self.fixed_params,
                fidelity_name=self._fidelity_name,
                fidelity_type=self._fidelity_type,
                on_error=self.on_error,
                error_path=self._error_path,
                failures_path=self._failures_path,
                use_budget=want_mf,
            )
            target: Any = (
                _SMACTrialEvaluator(**eval_kwargs)
                if want_mf
                else _SMACTrialEvaluatorSF(**eval_kwargs)
            )

            if facade_kind == "mf":
                self.smac = MultiFidelityFacade(
                    scenario=self.scenario,
                    target_function=target,
                    acquisition_function=acq_func,
                    model=model,
                    initial_design=initial_design_cls(
                        scenario=self.scenario,
                        n_configs=min(10, n_trials),
                    ),
                    intensifier=SuccessiveHalving(
                        scenario=self.scenario,
                        incumbent_selection="highest_budget",
                        max_incumbents=1,
                    ),
                )
            elif facade_kind == "random":
                self.smac = RandomFacade(
                    scenario=self.scenario,
                    target_function=target,
                )
            elif facade_kind == "ac":
                self.smac = AlgorithmConfigurationFacade(
                    scenario=self.scenario,
                    target_function=target,
                    acquisition_function=acq_func,
                    model=model,
                    initial_design=initial_design_cls(
                        scenario=self.scenario,
                        n_configs=min(10, n_trials),
                    ),
                )
            else:
                self.smac = HyperparameterOptimizationFacade(
                    scenario=self.scenario,
                    target_function=target,
                    acquisition_function=acq_func,
                    model=model,
                    initial_design=initial_design_cls(
                        scenario=self.scenario,
                        n_configs=min(10, n_trials),
                    ),
                    acquisition_maximizer=LocalAndSortedRandomSearch(
                        configspace=self.configspace,
                        acquisition_function=acq_func,
                        challengers=1000,
                        local_search_iterations=10,
                    ),
                )
        except Exception:
            # Constructor failed after temp dir allocation — do not leak.
            self._cleanup_output_dir()
            raise

    def _load_first_error(self) -> Optional[BaseException]:
        return _load_error_side_channel(getattr(self, "_error_path", None))

    def _load_failures_from_disk(self) -> List[Dict[str, Any]]:
        path = getattr(self, "_failures_path", None)
        if not path:
            return []
        p = Path(path)
        if not p.is_file():
            return []
        out: List[Dict[str, Any]] = []
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        except Exception:
            pass
        return out

    def _cleanup_output_dir(self) -> None:
        """Remove temp SMAC directory unless AMBER_SMAC_KEEP_OUTPUT is set."""
        if os.environ.get("AMBER_SMAC_KEEP_OUTPUT", "").strip() in (
            "1",
            "true",
            "yes",
        ):
            return
        out = getattr(self, "_output_dir", None)
        if out:
            shutil.rmtree(out, ignore_errors=True)
        # Error side-channel files (outside output dir)
        err = getattr(self, "_error_path", None)
        if err:
            try:
                Path(err).unlink(missing_ok=True)
            except TypeError:
                try:
                    if Path(err).is_file():
                        Path(err).unlink()
                except OSError:
                    pass
            except OSError:
                pass
            try:
                _error_json_path(err).unlink(missing_ok=True)
            except TypeError:
                try:
                    jp = _error_json_path(err)
                    if jp.is_file():
                        jp.unlink()
                except OSError:
                    pass
            except OSError:
                pass

    def _shutdown_parallel_resources(self) -> None:
        """Close SMAC's Dask client/workers when ``n_workers > 1``.

        SMAC's ``DaskParallelRunner`` only closes the client in ``__del__``,
        which leaves nannies/workers alive until GC. Explicitly close after
        optimize so process trees do not accumulate.
        """
        smac = getattr(self, "smac", None)
        if smac is None:
            return
        # SMAC stores the runner as ``_runner`` on the facade.
        runner = getattr(smac, "_runner", None) or getattr(smac, "runner", None)
        if runner is None:
            return
        # Prefer public close(force=True) on DaskParallelRunner.
        close = getattr(runner, "close", None)
        if callable(close):
            try:
                close(force=True)
            except TypeError:
                try:
                    close()
                except Exception:
                    pass
            except Exception:
                pass
        client = getattr(runner, "_client", None) or getattr(runner, "client", None)
        if client is not None:
            # Local clusters started by DaskParallelRunner need an explicit
            # cluster shutdown; client.close() alone may leave nannies.
            cluster = getattr(client, "cluster", None)
            try:
                if cluster is not None:
                    cluster.close(timeout=5)
            except TypeError:
                try:
                    cluster.close()
                except Exception:
                    pass
            except Exception:
                pass
            try:
                client.close(timeout=5)
            except TypeError:
                try:
                    client.close()
                except Exception:
                    pass
            except Exception:
                pass
            try:
                # Extra belt-and-suspenders for older dask versions.
                if hasattr(client, "shutdown"):
                    client.shutdown()
            except Exception:
                pass
    def optimize(self) -> Dict[str, Any]:
        """Run the optimization.

        Returns:
            Dict with:

            * ``best_config`` — incumbent hyperparameters (search space only)
            * ``best_cost`` / ``best_objective`` — minimized cost (aliases)
            * ``n_evaluations`` — number of history rows
            * ``history`` — Polars frame: search-space columns plus ``cost``,
              ``objective`` (same as cost), ``time``, ``trial``
            * ``failures`` — only when ``on_error='penalize'``

        When ``on_error='raise'`` (default), the first target/objective
        exception is re-raised after SMAC returns (SMAC itself swallows
        target crashes into CRASHED/inf trials). Works for ``n_workers>1`` via
        a pickle side-channel under the run directory.
        """
        try:
            # Run optimization; only documented search-exhaustion is non-fatal.
            try:
                incumbent = self.smac.optimize()
            except Exception as exc:
                first = self._load_first_error()
                if self.on_error == "raise" and first is not None:
                    raise first from exc
                if not _is_search_exhausted(exc):
                    raise
                incumbent = None

            if self.on_error == "raise":
                first = self._load_first_error()
                if first is not None:
                    raise first

            # Convert the run history to a DataFrame (SMAC 2.x RunHistory is a
            # Mapping of TrialKey -> TrialValue; access configs via get_config).
            history = self.smac.runhistory
            data = []
            crashed = False
            for trial_i, (trial_key, trial_value) in enumerate(history.items()):
                config = history.get_config(trial_key.config_id)
                cost = trial_value.cost
                status = getattr(trial_value, "status", None)
                status_name = (
                    getattr(status, "name", None) or str(status or "")
                ).upper()
                if status_name == "CRASHED" or (
                    isinstance(cost, (int, float)) and not np.isfinite(float(cost))
                ):
                    crashed = True
                row: Dict[str, Any] = {
                    **dict(config),
                    'cost': cost,
                    'objective': cost,  # alias — SMAC minimizes cost
                    'time': getattr(trial_value, 'time', 0.0),
                    'trial': trial_i,
                }
                # Multi-fidelity: report the budget actually used.
                if self._fidelity_name is not None:
                    budget = getattr(trial_key, "budget", None)
                    if budget is not None:
                        row[self._fidelity_name] = _coerce_fidelity_budget(
                            budget, self._fidelity_type
                        )
                data.append(row)

            # Last line of defense: never return silent inf under raise mode.
            if self.on_error == "raise":
                first = self._load_first_error()
                if first is not None:
                    raise first
                if crashed or (
                    incumbent is None
                    and data
                    and all(
                        not np.isfinite(float(r["cost"]))
                        if isinstance(r.get("cost"), (int, float))
                        else True
                        for r in data
                    )
                ):
                    raise RemoteObjectiveError(
                        "SMAC trial(s) crashed under on_error='raise' but the "
                        "original exception was not recoverable in the parent "
                        "(unpickleable worker exception or missing side-channel)."
                    )

            history_df = pl.DataFrame(data) if data else pl.DataFrame()

            best_cost = None
            if incumbent is not None:
                try:
                    best_cost = history.get_cost(incumbent)
                except KeyError:
                    best_cost = None
            out: Dict[str, Any] = {
                'best_config': dict(incumbent) if incumbent is not None else {},
                'best_cost': best_cost,
                'best_objective': best_cost,
                'n_evaluations': len(data),
                'history': history_df,
            }
            if self.on_error == "penalize":
                self.failures = self._load_failures_from_disk()
                out['failures'] = list(self.failures)
            return out
        finally:
            self._shutdown_parallel_resources()
            self._cleanup_output_dir()

class MultiObjectiveSMAC:
    """Multi-objective optimization by running one SMAC search per objective.

    Each named objective is optimized independently with a scalar SMAC facade
    (same evaluate path as :class:`SMACOptimizer`). Histories are merged and a
    simple non-dominated set is returned as ``pareto_front``. Suitable for
    small multi-objective calibration examples; not a full ParEGO/EHVI MOBO.
    """

    def __init__(
        self,
        model_type: Type[Model],
        param_space: SMACParameterSpace,
        objectives: Dict[str, Callable[[Model], float]],
        n_trials: int = 100,
        n_workers: int = 1,
        seed: Optional[int] = None,
        strategy: str = "pareto",
        use_multi_fidelity: bool = False,
    ):
        _check_smac()
        if use_multi_fidelity:
            raise NotImplementedError(
                "MultiObjectiveSMAC multi-fidelity is not supported; "
                "use SMACOptimizer(use_multi_fidelity=True) per objective."
            )
        if not objectives:
            raise ValueError("objectives must be a non-empty dict")

        self.model_type = model_type
        self.param_space = param_space
        self.objectives = dict(objectives)
        self.n_trials = int(n_trials)
        self.n_workers = n_workers
        self.seed = seed
        self.strategy = strategy
        # Built on first optimize() so construction stays cheap for smoke tests.
        self._optimizers: Optional[Dict[str, SMACOptimizer]] = None

    def _ensure_optimizers(self) -> Dict[str, "SMACOptimizer"]:
        if self._optimizers is not None:
            return self._optimizers
        opts: Dict[str, SMACOptimizer] = {}
        for i, (name, objective) in enumerate(self.objectives.items()):
            # Distinct seeds so independent searches explore differently.
            seed = None if self.seed is None else int(self.seed) + i * 17
            opts[name] = SMACOptimizer(
                model_type=self.model_type,
                param_space=self.param_space,
                objective=objective,
                n_trials=self.n_trials,
                n_workers=self.n_workers,
                seed=seed,
                strategy="bayesian",
            )
        self._optimizers = opts
        return opts

    def optimize(self) -> Dict[str, Any]:
        """Run per-objective SMAC and assemble a Pareto set.

        Returns:
            ``n_evaluations``, ``pareto_front`` (configs + objective costs),
            ``history`` (long-format rows), ``single_objective_results``.
        """
        optimizers = self._ensure_optimizers()
        single: Dict[str, Any] = {}
        history_frames: Dict[str, pl.DataFrame] = {}
        n_evals = 0

        for name, opt in optimizers.items():
            result = opt.optimize()
            single[name] = {
                "best_config": result.get("best_config", {}),
                "best_cost": result.get("best_cost"),
            }
            hist = result.get("history")
            if hist is not None and not hist.is_empty() and "cost" in hist.columns:
                history_frames[name] = hist.rename({"cost": name})
                n_evals += hist.height
            elif result.get("best_config"):
                # Degenerate history: still record the incumbent.
                history_frames[name] = pl.DataFrame(
                    [{**result["best_config"], name: result.get("best_cost")}]
                )
                n_evals += 1

        history_long = (
            pl.concat(list(history_frames.values()), how="diagonal_relaxed")
            if history_frames
            else pl.DataFrame()
        )
        pareto = self._pareto_from_incumbents(single)

        return {
            "n_evaluations": n_evals,
            "pareto_front": pareto,
            "history": history_long if not history_long.is_empty() else history_frames,
            "single_objective_results": single,
        }

    def _pareto_from_incumbents(self, single: Dict[str, Any]) -> pl.DataFrame:
        """Build a small Pareto table from per-objective incumbents.

        Each incumbent is re-scored on *all* objectives so the front carries
        comparable columns for plotting / example code.
        """
        rows = []
        seen = set()
        for _name, res in single.items():
            cfg = dict(res.get("best_config") or {})
            key = tuple(sorted(cfg.items()))
            if key in seen:
                continue
            seen.add(key)
            params = {**cfg, "show_progress": False}
            try:
                model = self.model_type(params)
                model.results = model.run()
                row = {**cfg}
                for obj_name, obj_fn in self.objectives.items():
                    try:
                        row[obj_name] = float(obj_fn(model))
                    except Exception:
                        row[obj_name] = float("inf")
                rows.append(row)
            except Exception:
                continue

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)
        obj_names = list(self.objectives.keys())
        costs = df.select(obj_names).to_numpy()
        # Non-dominated: minimize all objectives.
        n = costs.shape[0]
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                    keep[i] = False
                    break
        return df.filter(pl.Series(keep.tolist()))
