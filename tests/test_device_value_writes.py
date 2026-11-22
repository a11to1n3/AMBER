"""Column writes accept device (CuPy) arrays on every write route.

Host write paths used to accept lists and NumPy arrays only, so a CuPy array
was treated as one scalar: every row received the whole array (an Object
column) or the write failed. The same model runs on CPU with NumPy arrays,
which checks that host behaviour is unchanged.
"""

import numpy as np
import pytest

import ambr as am
from ambr import GPU_AVAILABLE, get_array_module, to_host
from ambr.gpu import host_value

ROUTES = ("assign", "set", "subview", "subview_half", "commit", "scalar")
KINDS = ("bool", "int64", "float64")
DEVICES = [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.skipif(not GPU_AVAILABLE, reason="needs CuPy and a CUDA device")),
]
N = 6


def _value(w, step, kind):
    """The value written at ``step`` (0-based): differs between steps."""
    if kind == "bool":
        return w > step + 1
    return (w * (step + 2)).astype(kind)


class Writer(am.Model):
    """Writes column ``c`` through one route each step and reads it back."""

    def setup(self):
        kind = self.p.kind
        self.seen = []
        self.add_agents(N, c=np.zeros(N, dtype=kind), w=np.arange(N, dtype=np.int64))

    def step_vectorized(self):
        route, kind = self.p.route, self.p.kind
        self.seen.append(np.asarray(to_host(self.agents.array("c"))).copy())
        w = self.agents.array("w")
        v = _value(w, self.t, kind)
        if route == "assign":
            self.agents.c = v
        elif route == "set":
            self.agents.set(c=v)
        elif route == "subview":
            self.agents.where(self.agents.w >= 0).c = v
        elif route == "subview_half":
            self.agents.where(self.agents.w >= 3).c = v[3:]
        elif route == "commit":
            self.agents.commit(c=v)
        elif route == "scalar":
            self.agents.c = v.any() if kind == "bool" else v.sum()


def _expected(route, kind, step):
    w = np.arange(N, dtype=np.int64)
    v = _value(w, step, kind)
    if route == "scalar":
        return np.full(N, v.any() if kind == "bool" else v.sum(), dtype=kind)
    if route == "subview_half":
        out = np.zeros(N, dtype=kind)  # rows outside the view keep their initial value
        out[3:] = v[3:]
        return out
    return v


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("route", ROUTES)
def test_column_writes_keep_one_value_per_agent(route, kind, device):
    model = Writer({"seed": 1, "route": route, "kind": kind})
    (model.gpu() if device == "gpu" else model.cpu(mode="vectorized")).run(steps=2)
    frame = model.agents_df
    expected_dtype = {"bool": "Boolean", "int64": "Int64", "float64": "Float64"}[kind]
    assert str(frame.schema["c"]) == expected_dtype
    np.testing.assert_array_equal(frame["c"].to_numpy(), _expected(route, kind, 1))
    # The step-2 read sees the step-1 write on the device as well as on the host.
    np.testing.assert_array_equal(model.seen[1], _expected(route, kind, 0))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_add_agents_accepts_device_arrays(kind, device):
    xp = get_array_module(prefer_gpu=(device == "gpu"))

    class Adder(am.Model):
        def setup(self):
            self.add_agents(N, c=xp.arange(N).astype(kind), w=np.arange(N, dtype=np.int64))

    model = Adder({"seed": 1})
    (model.gpu() if device == "gpu" else model.cpu(mode="vectorized")).run(steps=0)
    frame = model.agents_df
    assert str(frame.schema["c"]) == {"bool": "Boolean", "int64": "Int64", "float64": "Float64"}[kind]
    np.testing.assert_array_equal(frame["c"].to_numpy(), np.arange(N).astype(kind))


def test_host_value_leaves_host_values_unchanged():
    array = np.arange(3)
    assert host_value(array) is array
    assert host_value([1, 2]) == [1, 2]
    assert host_value(5) == 5
    assert host_value(None) is None


@pytest.mark.skipif(not GPU_AVAILABLE, reason="needs CuPy and a CUDA device")
def test_host_value_copies_device_arrays():
    import cupy as cp

    out = host_value(cp.arange(4))
    assert isinstance(out, np.ndarray) and out.tolist() == [0, 1, 2, 3]
    scalar = host_value(cp.asarray(7))
    assert scalar == 7 and not isinstance(scalar, np.ndarray)
    assert host_value(cp.asarray(True)) is True
