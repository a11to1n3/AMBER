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
        pass

    assert not _is_search_exhausted(MyConfigurationExhaustedError())
    try:
        from smac.main.exceptions import ConfigurationSpaceExhaustedException
    except ImportError:
        pytest.skip("SMAC exception type unavailable")
    assert _is_search_exhausted(ConfigurationSpaceExhaustedException())


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
