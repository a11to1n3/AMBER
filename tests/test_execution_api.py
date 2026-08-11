"""Keras-style device placement: model.cpu(mode=...) / model.gpu(mode=...)."""

import pytest
import numpy as np

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


def test_private_fast_path_requires_explicit_instance_approval():
    m = am.Model({"steps": 1})
    config = am.ExecutionConfig(device="gpu", mode="vectorized")
    def hook():
        pass

    assert m.fast_path_approval is None
    assert not m._fast_path_is_eligible(config, "off", hook, hook)
    assert m.approve_fast_path("gpu-smoke-report-2026-07") is m
    assert m.fast_path_approval == "gpu-smoke-report-2026-07"
    assert m._fast_path_is_eligible(config, "off", hook, hook)
    assert not m._fast_path_is_eligible(config, "check", hook, hook)
    assert m.revoke_fast_path_approval() is m
    assert m.fast_path_approval is None
    assert not m._fast_path_is_eligible(config, "off", hook, hook)

    with pytest.raises(ValueError, match="non-empty evidence label"):
        m.approve_fast_path("   ")


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


def test_execution_mode_dispatches_native_step_hooks():
    calls = []

    class A(am.Agent):
        pass

    class M(am.Model):
        def setup(self):
            if self.mode == "oop":
                self.add_agents(1, agent_class=A)
            else:
                self.add_agents(1, x=0)

        def step_vectorized(self):
            calls.append("vectorized")

        def step_oop(self):
            calls.append("oop")

    M({"steps": 1}).cpu(mode="vectorized").run()
    M({"steps": 1}).cpu(mode="oop").run()
    assert calls == ["vectorized", "oop"]


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


def test_gpu_oop_mode_is_rejected():
    with pytest.raises(ValueError, match="GPU execution supports mode='vectorized'"):
        am.Model({"steps": 1}).gpu(mode="oop").run()


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


def test_end_execution_clears_state_even_when_sync_fails(monkeypatch):
    """GPU sync failure must not leave model._execution active."""
    from ambr.execution import (
        ActiveExecution,
        ExecutionConfig,
        end_execution,
    )

    class M(am.Model):
        def step(self):
            pass

    m = M({"steps": 1, "show_progress": False})
    m._execution = ActiveExecution(
        config=ExecutionConfig(device="gpu", mode="vectorized"),
        xp=None,
        device_columns={},
        dirty_columns=set(),
        device_rng=None,
        ids_are_arange=True,
    )

    def boom_sync(*_a, **_k):
        raise RuntimeError("simulated sync failure")

    monkeypatch.setattr("ambr.execution.sync_all_device_columns", boom_sync)
    with pytest.raises(RuntimeError, match="simulated sync failure"):
        end_execution(m)
    assert m._execution is None


def test_run_does_not_mask_simulation_error_with_teardown_error(monkeypatch):
    """If step fails and teardown also fails, the step error remains visible."""
    from ambr.execution import end_execution as real_end

    class Boom(am.Model):
        def step(self):
            raise ValueError("step exploded")

    m = Boom({"steps": 1, "show_progress": False})

    def bad_teardown(model):
        # Clear like the real end_execution, then raise.
        model._execution = None
        raise RuntimeError("teardown also failed")

    monkeypatch.setattr("ambr.model.end_execution", bad_teardown)
    with pytest.raises(ValueError, match="step exploded"):
        m.run()
    assert m._execution is None


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA/CuPy not available")
def test_native_gpu_step_does_not_export_boolean_masks():
    from unittest.mock import patch

    class NativeWealth(am.Model):
        def setup(self):
            self.add_agents(100, wealth=np.ones(100, dtype=np.int64))

        def step_vectorized(self):
            xp = self.xp
            wealth = self.agents.array("wealth")
            donor_positions = xp.nonzero(wealth > 0)[0]
            if int(donor_positions.size) == 0:
                return
            wealth[donor_positions] -= 1
            recipients = self.rng.choice(
                self.agents.array("id"), size=int(donor_positions.size)
            )
            self.agents.at[recipients].scatter_add(wealth=1)

        def update(self):
            pass

    with patch(
        "ambr.gpu.to_host",
        side_effect=AssertionError("unexpected per-step host export"),
    ):
        result = NativeWealth({"steps": 2, "seed": 0}).gpu().run()
    assert int(result.agents["wealth"].sum()) == 100
