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
from ambr.tensor_lane import TensorLane

ROUTES = ("assign", "set", "subview", "subview_half", "commit", "scalar", "commit0d", "lane_result")
KINDS = ("bool", "int64", "float64")
# Narrow dtypes: a 0-d value of these kinds must not widen to Int64/Float64.
NARROW_KINDS = ("int32", "float32", "int16", "uint8")
DTYPE_NAMES = {
    "bool": "Boolean",
    "int64": "Int64",
    "float64": "Float64",
    "int32": "Int32",
    "float32": "Float32",
    "int16": "Int16",
    "uint8": "UInt8",
}
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


def _reduce(v):
    """A 0-d value of ``v``'s dtype (NumPy scalar on CPU, 0-d array on GPU)."""
    return v.any() if v.dtype == bool else v.sum(dtype=v.dtype)


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
            self.agents.c = _reduce(v)
        elif route == "commit0d":
            self.agents.commit(c=_reduce(v))
        elif route == "lane_result":
            TensorLane(self, ["c"], prefer_array=False).write_result("c", v)


def _expected(route, kind, step):
    w = np.arange(N, dtype=np.int64)
    v = _value(w, step, kind)
    if route in ("scalar", "commit0d"):
        return np.full(N, _reduce(v), dtype=kind)
    if route == "subview_half":
        out = np.zeros(N, dtype=kind)  # rows outside the view keep their initial value
        out[3:] = v[3:]
        return out
    return v


ROUTE_KINDS = [(route, kind) for route in ROUTES for kind in KINDS] + [
    (route, kind) for route in ("scalar", "commit0d") for kind in NARROW_KINDS
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("route", "kind"), ROUTE_KINDS)
def test_column_writes_keep_one_value_per_agent(route, kind, device):
    model = Writer({"seed": 1, "route": route, "kind": kind})
    (model.gpu() if device == "gpu" else model.cpu(mode="vectorized")).run(steps=2)
    frame = model.agents_df
    assert str(frame.schema["c"]) == DTYPE_NAMES[kind]
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
    assert str(frame.schema["c"]) == DTYPE_NAMES[kind]
    np.testing.assert_array_equal(frame["c"].to_numpy(), np.arange(N).astype(kind))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS + NARROW_KINDS)
def test_agent_object_scalar_write_keeps_column_dtype(kind, device):
    xp = get_array_module(prefer_gpu=(device == "gpu"))

    class Obj(am.Model):
        def setup(self):
            self.add_agents(N, agent_class=am.Agent, c=np.zeros(N, dtype=kind))

        def step(self):
            total = _reduce(xp.arange(N).astype(kind))
            for agent in self.agents:
                agent.c = total

    model = Obj({"seed": 1})
    (model.gpu() if device == "gpu" else model.cpu()).run(steps=1)
    frame = model.agents_df
    assert str(frame.schema["c"]) == DTYPE_NAMES[kind]
    expected = _reduce(np.arange(N).astype(kind))
    np.testing.assert_array_equal(frame["c"].to_numpy(), np.full(N, expected, dtype=kind))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", NARROW_KINDS)
@pytest.mark.parametrize("route", ["assign", "commit"])
def test_0d_value_to_new_column_keeps_dtype(route, kind, device):
    xp = get_array_module(prefer_gpu=(device == "gpu"))

    class NewColumn(am.Model):
        def setup(self):
            self.add_agents(N, w=np.arange(N))

        def step_vectorized(self):
            total = _reduce(xp.arange(N).astype(kind))
            if route == "assign":
                self.agents.c = total
            else:
                self.agents.commit(c=total)

    model = NewColumn({"seed": 1})
    (model.gpu() if device == "gpu" else model.cpu(mode="vectorized")).run(steps=1)
    frame = model.agents_df
    assert str(frame.schema["c"]) == DTYPE_NAMES[kind]
    expected = _reduce(np.arange(N).astype(kind))
    np.testing.assert_array_equal(frame["c"].to_numpy(), np.full(N, expected, dtype=kind))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prefer_array", [True, False])
def test_tensor_lane_commit_accepts_device_arrays(prefer_array, device):
    xp = get_array_module(prefer_gpu=(device == "gpu"))

    class Lane(am.Model):
        def setup(self):
            self.add_agents(N, a=np.zeros(N), b=np.zeros(N))

        def step_vectorized(self):
            lane = TensorLane(self, ["a", "b"], prefer_array=prefer_array)
            lane.commit(xp.stack([xp.arange(N) * 1.0, xp.arange(N) * 2.0], axis=1))
            lane.unpack()

    model = Lane({"seed": 1})
    (model.gpu() if device == "gpu" else model.cpu(mode="vectorized")).run(steps=1)
    frame = model.agents_df
    np.testing.assert_array_equal(frame["a"].to_numpy(), np.arange(N) * 1.0)
    np.testing.assert_array_equal(frame["b"].to_numpy(), np.arange(N) * 2.0)


def test_commit_broadcasts_scalars_and_keeps_arrays_and_lists():
    class Committer(am.Model):
        def setup(self):
            self.add_agents(N, s=np.zeros(N), z=np.zeros(N), a=np.zeros(N), li=np.zeros(N))

        def step_vectorized(self):
            self.agents.commit(
                s=3.5, z=np.asarray(2.0), a=np.arange(N) * 1.0, li=[float(i) for i in range(N)]
            )

    model = Committer({"seed": 1})
    model.cpu(mode="vectorized").run(steps=1)
    frame = model.agents_df
    np.testing.assert_array_equal(frame["s"].to_numpy(), np.full(N, 3.5))
    np.testing.assert_array_equal(frame["z"].to_numpy(), np.full(N, 2.0))
    np.testing.assert_array_equal(frame["a"].to_numpy(), np.arange(N) * 1.0)
    np.testing.assert_array_equal(frame["li"].to_numpy(), np.arange(N) * 1.0)


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
    # A 0-d array becomes a NumPy scalar of the same dtype, as under cpu().
    for kind, value in (("int64", 7), ("int32", 7), ("float32", 1.5), ("uint8", 3), ("bool", True)):
        scalar = host_value(cp.asarray(value, dtype=kind))
        assert isinstance(scalar, np.generic) and not isinstance(scalar, np.ndarray)
        assert scalar.dtype == np.dtype(kind) and scalar == value
