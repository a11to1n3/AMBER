"""Durable runtime tests for SMACOptimizer / network identity fixes.

Includes subprocess coverage for ``n_workers=2`` (pickle-safe target) and
``on_error='raise'`` under multi-process evaluation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import ambr as am
from ambr.optimization import SMACOptimizer, _is_search_exhausted

pytest.importorskip("smac")


class _TinyModel(am.Model):
    """Importable model for SMAC (and spawn-safe subprocess tests)."""

    def setup(self):
        n = int(self.p.get("n_agents", 8))
        self.add_agents(n, wealth=1)

    def step_vectorized(self):
        pass

    def update(self):
        self.record_model("s", float(self.agents.wealth.mean()))


def _const_obj(model: am.Model) -> float:
    return 0.25


def _boom_obj(model: am.Model) -> float:
    raise RuntimeError("smac runtime boom")


def _space() -> am.SMACParameterSpace:
    space = am.SMACParameterSpace()
    space.add_parameter("seed", param_type="int", bounds=(0, 3), default=0)
    return space


@pytest.mark.unit
def test_smac_optimizer_unique_output_and_cleanup():
    fixed = {"n_agents": 8, "steps": 2, "show_progress": False}
    before = {p for p in Path(tempfile.gettempdir()).glob("amber_smac_*")}
    opt = SMACOptimizer(
        _TinyModel,
        _space(),
        _const_obj,
        n_trials=3,
        seed=0,
        strategy="random",
        fixed_params=fixed,
    )
    out_dir = Path(opt._output_dir)
    assert out_dir.is_dir()
    result = opt.optimize()
    assert result["n_evaluations"] >= 1
    assert "cost" in result["history"].columns
    # Temp dir removed after optimize (unless keep env set)
    assert not out_dir.exists()
    after = {p for p in Path(tempfile.gettempdir()).glob("amber_smac_*")}
    # No new amber_smac_* dirs left behind by this run
    assert after <= before


@pytest.mark.unit
def test_constructor_validation_does_not_leak_temp_dir():
    before = {p for p in Path(tempfile.gettempdir()).glob("amber_smac_*")}
    with pytest.raises(ValueError, match="Unknown strategy"):
        SMACOptimizer(
            _TinyModel,
            _space(),
            _const_obj,
            n_trials=2,
            seed=0,
            strategy="not_a_real_strategy",
            fixed_params={"n_agents": 4, "steps": 1, "show_progress": False},
        )
    after = {p for p in Path(tempfile.gettempdir()).glob("amber_smac_*")}
    assert after <= before


@pytest.mark.unit
def test_n_workers_2_shuts_down_dask_and_no_target_arg_warnings(caplog):
    """n_workers>1 must close Dask and not spam partial-signature warnings."""
    import logging

    fixed = {"n_agents": 8, "steps": 1, "show_progress": False}
    with caplog.at_level(logging.WARNING, logger="smac"):
        opt = SMACOptimizer(
            _TinyModel,
            _space(),
            _const_obj,
            n_trials=4,
            seed=0,
            strategy="random",
            n_workers=2,
            fixed_params=fixed,
        )
        result = opt.optimize()
    assert result["n_evaluations"] >= 1
    # No leftover "argument X is not set by SMAC" warnings from partial kwargs
    arg_warns = [
        r.message
        for r in caplog.records
        if "is not set by SMAC" in str(r.message)
    ]
    assert arg_warns == [], arg_warns
    # Dask client must not remain as default after optimize
    try:
        from dask.distributed import get_client

        try:
            client = get_client(timeout="0.5s")
        except Exception:
            client = None
        assert client is None or getattr(client, "status", None) in (
            "closed",
            "closing",
            None,
        )
    except ImportError:
        pass
    # No runaway multiprocessing children from SMAC workers
    import multiprocessing as mp
    import time

    time.sleep(0.5)
    live = [p for p in mp.active_children() if p.is_alive()]
    # Allow unrelated children; SMAC dask workers should be gone
    assert live == [] or all(
        "dask" not in (p.name or "").lower() for p in live
    )


@pytest.mark.unit
def test_write_error_when_str_and_pickle_fail():
    """Even pathological exceptions must leave a raiseable side-channel."""
    from ambr.optimization import (
        RemoteObjectiveError,
        _load_error_side_channel,
        _write_first_error,
    )

    class _EvilError(Exception):
        def __getstate__(self):
            raise TypeError("no pickle")

        def __str__(self):
            raise RuntimeError("no str")

        def __repr__(self):
            raise RuntimeError("no repr")

    from ambr.optimization import _error_json_path

    path = Path(tempfile.mkdtemp(prefix="amber_err_")) / "first_error.pkl"
    try:
        _write_first_error(str(path), _EvilError())
        # Structured JSON sibling is always written; pickle may be absent.
        jp = _error_json_path(str(path))
        assert jp.is_file() or path.is_file()
        loaded = _load_error_side_channel(str(path))
        assert isinstance(loaded, RemoteObjectiveError)
        assert "EvilError" in str(loaded) or "unprintable" in str(loaded).lower()
    finally:
        import shutil

        shutil.rmtree(path.parent, ignore_errors=True)

@pytest.mark.unit
def test_smac_on_error_raise_after_smac_swallows():
    fixed = {"n_agents": 8, "steps": 1, "show_progress": False}
    opt = SMACOptimizer(
        _TinyModel,
        _space(),
        _boom_obj,
        n_trials=2,
        seed=0,
        strategy="random",
        fixed_params=fixed,
        on_error="raise",
    )
    with pytest.raises(RuntimeError, match="smac runtime boom"):
        opt.optimize()


@pytest.mark.unit
def test_search_exhausted_exact_type_only():
    class MyConfigurationExhaustedError(Exception):
        """Must NOT match — only substring overlap with SMAC type name."""

        pass

    class ConfigurationDataExhaustedError(Exception):
        pass

    assert not _is_search_exhausted(MyConfigurationExhaustedError())
    assert not _is_search_exhausted(
        ConfigurationDataExhaustedError("objective data missing")
    )
    try:
        from smac.main.exceptions import ConfigurationSpaceExhaustedException
    except ImportError:
        pytest.skip("SMAC exception type unavailable")
    assert _is_search_exhausted(ConfigurationSpaceExhaustedException())
    # isinstance path
    assert _is_search_exhausted(ConfigurationSpaceExhaustedException("x"))


@pytest.mark.unit
def test_error_side_channel_structured_fallback():
    """When pickle.dumps(exc) fails, structured JSON sibling still raises."""
    from ambr.optimization import (
        RemoteObjectiveError,
        _error_json_path,
        _load_error_side_channel,
        _write_first_error,
    )

    class _UnpicklableError(Exception):
        def __reduce__(self):
            raise TypeError("deliberately unpickleable")

    path = Path(tempfile.mkdtemp(prefix="amber_err_")) / "first_error.pkl"
    try:
        _write_first_error(str(path), _UnpicklableError("objective data missing"))
        # Structured sibling is the reliable signal
        jp = _error_json_path(str(path))
        assert jp.is_file(), "expected first_error.json structured payload"
        loaded = _load_error_side_channel(str(path))
        assert isinstance(loaded, RemoteObjectiveError)
        assert "objective data missing" in str(loaded)
        assert "UnpicklableError" in str(loaded) or loaded.exception_type.endswith(
            "UnpicklableError"
        )
        # Prefer JSON even when a stale/unreadable pickle exists
        path.write_bytes(b"not-a-valid-pickle")
        loaded2 = _load_error_side_channel(str(path))
        assert isinstance(loaded2, RemoteObjectiveError)
        assert "objective data missing" in str(loaded2)
    finally:
        import shutil

        shutil.rmtree(path.parent, ignore_errors=True)


@pytest.mark.unit
def test_on_error_raise_never_returns_inf_on_unpickleable_objective():
    """End-to-end: unpickleable objective error must raise, not best_cost=inf."""
    from ambr.optimization import RemoteObjectiveError

    class _UnpicklableError(Exception):
        def __reduce__(self):
            raise TypeError("deliberately unpickleable")

    def boom(m):
        raise _UnpicklableError("objective data missing")

    fixed = {"n_agents": 8, "steps": 1, "show_progress": False}
    opt = SMACOptimizer(
        _TinyModel,
        _space(),
        boom,
        n_trials=2,
        seed=0,
        strategy="random",
        fixed_params=fixed,
        on_error="raise",
    )
    with pytest.raises((RemoteObjectiveError, Exception)) as ei:
        out = opt.optimize()
        # Must not reach a successful return with non-finite cost
        raise AssertionError(f"optimize returned instead of raising: {out!r}")
    # Never a silent success
    assert ei.value is not None
    assert "objective data missing" in str(ei.value) or "Unpicklable" in str(
        ei.value
    ) or "crashed under on_error" in str(ei.value)


@pytest.mark.unit
def test_on_error_raise_unpickleable_exception_still_raises():
    """Local exception classes often fail pickle.dumps — must not return inf."""
    from ambr.optimization import RemoteObjectiveError

    class LocalUnpickleableError(Exception):
        def __getstate__(self):
            raise TypeError("deliberately unpickleable")

    def boom(m):
        raise LocalUnpickleableError("cannot cross process cleanly")

    fixed = {"n_agents": 8, "steps": 1, "show_progress": False}
    opt = SMACOptimizer(
        _TinyModel,
        _space(),
        boom,
        n_trials=2,
        seed=0,
        strategy="random",
        fixed_params=fixed,
        on_error="raise",
    )
    with pytest.raises((LocalUnpickleableError, RemoteObjectiveError)) as ei:
        opt.optimize()
    # Structured wrapper when pickle of the exception fails in the worker.
    assert "cannot cross process cleanly" in str(ei.value)


@pytest.mark.unit
def test_bayesian_optimization_fixed_float_parameter():
    """Scalar floats must be fixed_params, not degenerate float HPs."""
    from ambr import ParameterSpace, bayesian_optimization

    space = ParameterSpace(
        {
            "n_agents": [8, 12],
            "fixed_float": 0.05,  # must not crash ConfigSpace
            "steps": 2,
            "seed": 0,
            "show_progress": False,
        }
    )
    results = bayesian_optimization(
        _TinyModel,
        space,
        metric="s",
        n_calls=3,
        iterations=1,
        minimize=True,
        random_state=0,
    )
    assert results
    assert results[0]["parameters"].get("fixed_float") == 0.05


@pytest.mark.unit
def test_network_missing_node_id_column_no_crash():
    nx = pytest.importorskip("networkx")

    class NetModel(am.Model):
        def setup(self):
            G = nx.cycle_graph(5)
            # Network before agents — classic integration ordering
            self.network = am.NetworkEnvironment(self, G)
            self.add_agents(3)
            # Must not raise ColumnNotFoundError
            assert self.network.get_degree(0) == 0
            assert self.network.get_neighbors(0) == []
            assert self.network.get_clustering(0) == 0.0
            # Explicit node path still works
            assert self.network.get_degree(1, as_node=True) == 2

    NetModel({"steps": 1, "seed": 0, "show_progress": False}).run()


@pytest.mark.unit
def test_smac_n_workers_2_success_and_raise_subprocess():
    """Subprocess isolation: n_workers=2 must pickle target and run evaluations."""
    root = Path(__file__).resolve().parents[1]
    script = r'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(%r) / "src"))
import ambr as am
from ambr.optimization import SMACOptimizer

class Tiny(am.Model):
    def setup(self):
        self.add_agents(int(self.p.get("n_agents", 8)), wealth=1)
    def step_vectorized(self):
        pass
    def update(self):
        self.record_model("s", 1.0)

def ok(m):
    return 0.5

def boom(m):
    raise RuntimeError("parallel boom")

space = am.SMACParameterSpace()
space.add_parameter("seed", param_type="int", bounds=(0, 2), default=0)
fixed = {"n_agents": 8, "steps": 1, "show_progress": False}

opt = SMACOptimizer(
    Tiny, space, ok, n_trials=4, seed=0, strategy="random",
    n_workers=2, fixed_params=fixed,
)
out = opt.optimize()
assert out["n_evaluations"] >= 1, out
assert out["best_cost"] == 0.5 or abs(float(out["best_cost"]) - 0.5) < 1e-9
print("SUCCESS_OK", out["n_evaluations"])

opt2 = SMACOptimizer(
    Tiny, space, boom, n_trials=2, seed=1, strategy="random",
    n_workers=2, fixed_params=fixed, on_error="raise",
)
try:
    opt2.optimize()
    raise SystemExit("expected raise")
except RuntimeError as e:
    assert "parallel boom" in str(e)
    print("RAISE_OK")
''' % str(root)
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "AMBER_SUPPRESS_DEPRECATIONS": "1",
        "MPLBACKEND": "Agg",
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=str(root),
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"exit={proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-3000:]}"
        )
    assert "SUCCESS_OK" in proc.stdout
    assert "RAISE_OK" in proc.stdout
    assert "cannot pickle" not in (proc.stderr or "").lower()
