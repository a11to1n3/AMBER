"""Model.gpu() / Model.cpu() on the vectorized view API."""

import numpy as np
import pytest

import ambr as am
from ambr.execution import ExecutionConfig, begin_execution, end_execution, get_device_column
from ambr.gpu import GPU_AVAILABLE, to_host


class _Wealth(am.Model):
    """Canonical quickstart idiom — same step body as docs/quickstart.rst."""

    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, wealth=np.ones(n, dtype=np.int64))

    def step(self):
        donors = self.agents.where(self.agents.wealth > 0)
        if len(donors) == 0:
            return
        donors.wealth -= 1
        ids = self.agents.ids.to_numpy()
        recipients = self.rng.choice(ids, size=len(donors))
        self.agents.at[recipients].scatter_add(wealth=1)


def test_run_cpu():
    res = _Wealth({"n": 200, "steps": 10, "seed": 0}).cpu().run()
    assert res.agents.height == 200
    assert int(res.agents["wealth"].sum()) == 200


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA/CuPy not available")
def test_run_gpu_same_step():
    res = _Wealth({"n": 200, "steps": 10, "seed": 0}).gpu().run()
    assert res.agents.height == 200
    assert int(res.agents["wealth"].sum()) == 200


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA/CuPy not available")
def test_cpu_gpu_conserves_wealth():
    """Device RNG differs from NumPy; both backends preserve total wealth."""
    params = {"n": 500, "steps": 25, "seed": 42}
    cpu = _Wealth(params).cpu().run()
    gpu = _Wealth(params).gpu().run()
    assert int(cpu.agents["wealth"].sum()) == 500
    assert int(gpu.agents["wealth"].sum()) == 500


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA/CuPy not available")
def test_recipient_scatter_avoids_host_round_trip():
    """Recipient ``scatter_add`` stays on device (fused debit/credit tested separately)."""
    from unittest.mock import patch

    m = _Wealth({"n": 100, "steps": 1, "seed": 0})
    m.setup()
    begin_execution(m, ExecutionConfig(device="gpu", mode="vectorized"))
    donors = m.agents.where(m.agents.wealth > 0)
    donors.wealth -= 1
    ids = m.agents.ids.to_numpy()
    recipients = m.rng.choice(ids, size=len(donors))

    def _forbid_export(arr):
        raise AssertionError("unexpected host export during GPU scatter_add")

    try:
        with patch("ambr.device_columns.to_host", side_effect=_forbid_export):
            with patch("ambr.gpu.to_host", side_effect=_forbid_export):
                m.agents.at[recipients].scatter_add(wealth=1)
    finally:
        end_execution(m)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA/CuPy not available")
def test_device_column_array_is_device_resident():
    m = _Wealth({"n": 50, "steps": 1, "seed": 0})
    m.setup()
    begin_execution(m, ExecutionConfig(device="gpu", mode="vectorized"))
    try:
        wealth = m.agents.wealth
        assert hasattr(wealth, "array")
        arr = wealth.array
        assert arr is get_device_column(m, "wealth")
        m.step()
        dev = to_host(get_device_column(m, "wealth"))
        stale = m.population.data["wealth"].to_numpy()
        assert int(dev.sum()) == 50
        assert np.array_equal(stale, np.ones(50))
        assert not np.array_equal(dev, stale)
    finally:
        end_execution(m)


def test_run_gpu_requires_cuda():
    if GPU_AVAILABLE:
        pytest.skip("GPU present")
    with pytest.raises(RuntimeError, match="GPU requested"):
        _Wealth({"n": 10, "steps": 1}).gpu().run()


def _vectorized_benchmark_models():
    """Same classes the AMBER (GPU) benchmark row uses."""
    import sys
    from pathlib import Path

    bench = Path(__file__).resolve().parents[1] / "benchmarks"
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    from models.amber_models import AMBER_VECTORIZED_MODELS

    return AMBER_VECTORIZED_MODELS


@pytest.mark.parametrize(
    "name",
    ["wealth_transfer", "random_walk", "sir_epidemic", "schelling"],
)
def test_vectorized_models_run_cpu(name):
    cls = _vectorized_benchmark_models()[name]
    cfg = {
        "n": 40,
        "steps": 3,
        "show_progress": False,
        "seed": 0,
        "initial_wealth": 1,
        "initial_infected": 3,
        "world_size": 50,
        "speed": 1.0,
        "movement_speed": 2.0,
        "infection_radius": 5.0,
        "transmission_rate": 0.1,
        "recovery_time": 14,
        "density": 0.8,
        "fraction_a": 0.5,
        "tolerance": 0.3,
    }
    res = cls(cfg).cpu().run()
    assert res.agents.height == 40


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA/CuPy not available")
@pytest.mark.parametrize(
    "name",
    ["wealth_transfer", "random_walk", "sir_epidemic", "schelling"],
)
def test_vectorized_models_run_gpu_same_api(name):
    """AMBER (GPU) must be model.gpu().run() on the same vectorized classes."""
    cls = _vectorized_benchmark_models()[name]
    cfg = {
        "n": 40,
        "steps": 3,
        "show_progress": False,
        "seed": 0,
        "initial_wealth": 1,
        "initial_infected": 3,
        "world_size": 50,
        "speed": 1.0,
        "movement_speed": 2.0,
        "infection_radius": 5.0,
        "transmission_rate": 0.1,
        "recovery_time": 14,
        "density": 0.8,
        "fraction_a": 0.5,
        "tolerance": 0.3,
    }
    res = cls(cfg).gpu().run()
    assert res.agents.height == 40
    assert res.info.get("device") == "gpu"