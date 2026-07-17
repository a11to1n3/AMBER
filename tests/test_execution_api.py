"""Keras-style device placement: model.cpu(mode=...) / model.gpu(mode=...)."""

import pytest

import ambr as am
from ambr.gpu import GPU_AVAILABLE
from ambr.execution import EXECUTION_DEVICES, EXECUTION_MODES


def test_cpu_gpu_fluent_setters_chain():
    m = am.Model({"steps": 1})
    assert m.cpu() is m
    assert m.device == "cpu"
    assert m.gpu() is m
    assert m.device == "gpu"
    assert m.cpu().device == "cpu"


def test_cpu_mode_fluent_setter():
    m = am.Model({"steps": 1})
    assert m.cpu(mode="vectorized") is m
    assert m.device == "cpu"
    assert m.mode == "vectorized"
    m.cpu(mode="oop")
    assert m.device == "cpu"
    assert m.mode == "oop"


def test_gpu_mode_fluent_setter():
    m = am.Model({"steps": 1})
    m.gpu(mode="vectorized")
    assert m.device == "gpu"
    assert m.mode == "vectorized"


def test_cpu_mode_run_records_mode():
    class M(am.Model):
        def step(self):
            pass

    r = M({"steps": 2}).cpu(mode="vectorized").run()
    assert r.info["device"] == "cpu"
    assert r.info["mode"] == "vectorized"


def test_run_mode_overrides_fluent_mode():
    class M(am.Model):
        def step(self):
            pass

    r = M({"steps": 1}).cpu(mode="oop").run(mode="vectorized")
    assert r.info["mode"] == "vectorized"


def test_run_records_device_and_mode():
    class M(am.Model):
        def step(self):
            pass

    r = M({"steps": 2}).cpu().run(mode="vectorized")
    assert r.info["device"] == "cpu"
    assert r.info["mode"] == "vectorized"


def test_run_device_param_overrides_fluent():
    class M(am.Model):
        def step(self):
            pass

    r = M({"steps": 1}).gpu().run(device="cpu")
    assert r.info["device"] == "cpu"


def test_run_backend_legacy_alias():
    class M(am.Model):
        def step(self):
            pass

    r = M({"steps": 1}).run(backend="cpu", mode="vectorized")
    assert r.info["device"] == "cpu"


def test_invalid_device_raises():
    class M(am.Model):
        def step(self):
            pass

    with pytest.raises(ValueError, match="device must be"):
        M({"steps": 1}).run(device="tpu")


def test_invalid_mode_raises():
    class M(am.Model):
        def step(self):
            pass

    with pytest.raises(ValueError, match="mode must be"):
        M({"steps": 1}).run(mode="quantum")


def test_invalid_fluent_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        am.Model({"steps": 1}).cpu(mode="quantum")
    with pytest.raises(ValueError, match="mode must be"):
        am.Model({"steps": 1}).gpu(mode="quantum")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA/CuPy not available")
def test_gpu_fluent_run():
    class M(am.Model):
        def setup(self):
            self.add_agents(10, wealth=1)

        def step(self):
            donors = self.agents.where(self.agents.wealth > 0)
            if len(donors) == 0:
                return
            donors.wealth -= 1
            ids = self.agents.ids.to_numpy()
            self.agents.at[self.rng.choice(ids, size=len(donors))].scatter_add(wealth=1)

    r = M({"steps": 3, "seed": 0}).gpu().run()
    assert r.info["device"] == "gpu"
    assert int(r.agents["wealth"].sum()) == 10


def test_execution_constants():
    assert "cpu" in EXECUTION_DEVICES
    assert "gpu" in EXECUTION_DEVICES
    assert "vectorized" in EXECUTION_MODES
    assert "oop" in EXECUTION_MODES


def test_execution_state_cleared_after_run():
    class M(am.Model):
        def step(self):
            pass

    m = M({"steps": 1})
    m.cpu().run()
    assert m._execution is None