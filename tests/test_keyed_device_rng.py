"""Keyed (counter-based) draws used by the timed GPU models.

Each draw is a pure function of (seed, step, event, key, partner, draw index),
so its value cannot depend on the CUDA device or on thread order.  The NumPy
path (``xp=numpy``) is the reference; the GPU tests require the CUDA kernels to
reproduce it bit for bit.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ambr import Model
from ambr import gpu as gpu_backend
from ambr.gpu import (
    EVT_INIT,
    EVT_PRIORITY,
    GPU_AVAILABLE,
    DeviceRNG,
    KeyedDeviceRNG,
    keyed_rng64,
    keyed_u01,
    make_keyed_device_rng,
)

SEED = 42
gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="requires CuPy and a CUDA device")


@pytest.fixture(params=["numpy", pytest.param("cupy", marks=gpu)])
def xp(request):
    if request.param == "cupy":
        import cupy

        return cupy
    return np


def host(seed=SEED):
    return make_keyed_device_rng(seed, xp=np)


def test_reference_matches_sir_counter_tape_spec():
    # Frozen spec-2 anchor shared with tests/test_sir_counter_tape.py.
    assert keyed_u01(0, 0, 4, 0, 1, 0) == pytest.approx(0.7682208223562327, abs=1e-15)
    assert keyed_u01(0, 1, 4, 0, 1, 0) != keyed_u01(1, 0, 4, 0, 1, 0)


def test_bulk_draws_equal_scalar_reference():
    rng = host()
    rng.begin_step(3, 2)
    u = rng.random(7)
    v = rng.random(7)
    assert u.tolist() == [keyed_u01(SEED, 3, 2, i, 0, 0) for i in range(7)]
    assert v.tolist() == [keyed_u01(SEED, 3, 2, i, 0, 1) for i in range(7)]


def test_draws_before_first_step_use_init_event():
    rng = host()
    assert rng.random(3).tolist() == [keyed_u01(SEED, 0, EVT_INIT, i) for i in range(3)]


def test_begin_step_resets_stream_and_keys():
    rng = host()
    rng.begin_step(5, 0)
    a = rng.random(4)
    rng.set_agent_keys([9, 8, 7, 6])
    rng.random(4)
    rng.begin_step(5, 0)
    np.testing.assert_array_equal(rng.random(4), a)


def test_agent_keys_make_draws_follow_identity_not_position():
    rng = host()
    rng.begin_step(1, 1)
    rng.set_agent_keys([10, 20, 30])
    forward = rng.random(3)
    rng.begin_step(1, 1)
    rng.set_agent_keys([30, 10, 20])
    shuffled = rng.random(3)
    np.testing.assert_array_equal(shuffled, forward[[2, 0, 1]])


@pytest.mark.parametrize("keys", [[], [5], [0, 1, 2, 3, 4, 5, 6]])
def test_bulk_key_count_must_match_draw_size(xp, keys):
    rng = make_keyed_device_rng(7, xp=xp)
    rng.set_agent_keys(keys)
    for draw in (rng.random, rng.uniform, lambda size: rng.integers(10, size=size)):
        with pytest.raises(ValueError, match="agent keys.*draw size"):
            draw(size=(2, 3))
    rng.set_agent_keys(None)
    np.testing.assert_array_equal(rng.random(6).tolist(), host(7).random(6))


def test_scalar_draws_require_one_key_when_bound(xp):
    rng = make_keyed_device_rng(7, xp=xp)
    rng.set_agent_keys([5, 6])
    with pytest.raises(ValueError, match="agent keys.*draw size"):
        rng.random()
    rng.set_agent_keys([5])
    assert rng.random() == keyed_u01(7, 0, EVT_INIT, 5, draw=0)
    assert rng.random() == keyed_u01(7, 0, EVT_INIT, 5, draw=1)


def test_uniform_is_affine_map_of_u01():
    rng = host()
    rng.begin_step(2, 0)
    dx = rng.uniform(-1.5, 1.5, 6)
    u = np.array([keyed_u01(SEED, 2, 0, i) for i in range(6)])
    np.testing.assert_array_equal(dx, -1.5 + 3.0 * u)
    assert dx.dtype == np.float64
    assert rng.uniform(0, 1, 4, dtype=np.float32).dtype == np.float32


def test_float32_is_rounded_float64_draw():
    rng = host()
    a = rng.random(5, dtype=np.float32)
    rng2 = host()
    np.testing.assert_array_equal(a, rng2.random(5).astype(np.float32))


@pytest.mark.parametrize("size", [None, 1])
@pytest.mark.parametrize("method, args", [("random", ()), ("uniform", ()), ("uniform", (0.5, 1))])
def test_float32_clamps_rounded_unit_endpoint(xp, size, method, args):
    rng = make_keyed_device_rng(5042506492838628837, xp=xp)
    actual = getattr(rng, method)(*args, size=size, dtype=xp.float32)
    expected = np.nextafter(np.float32(1), np.float32(0))
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.tolist(), expected if size is None else [expected])


def test_float32_clamp_preserves_numpy_byte_order():
    dtype = np.dtype(np.float32).newbyteorder("S")
    actual = host(5042506492838628837).random(1, dtype=dtype)
    assert actual.dtype == dtype
    assert actual[0] == np.nextafter(np.float32(1), np.float32(0))


def test_integers_and_choice_reduce_modulo_range():
    rng = host()
    rng.begin_step(4, 1)
    rng.set_agent_keys([3, 1, 4, 1, 5])
    got = rng.integers(0, 10, size=5)
    assert got.tolist() == [keyed_rng64(SEED, 4, 1, k) % 10 for k in (3, 1, 4, 1, 5)]
    rng.begin_step(4, 1)
    rng.set_agent_keys([3, 1, 4, 1, 5])
    pool = np.arange(100, 110)
    np.testing.assert_array_equal(rng.choice(pool, size=5), pool[got])
    rng.set_agent_keys(None)
    assert 0 <= rng.integers(7) < 7
    assert rng.integers(2, 3, size=4).tolist() == [2, 2, 2, 2]
    assert rng.integers(2, 2, size=3, endpoint=True).tolist() == [2, 2, 2]
    with pytest.raises(ValueError):
        rng.integers(3, 3)


def test_choice_without_replacement_and_errors():
    rng = host()
    picked = rng.choice(np.arange(50), size=50, replace=False)
    assert sorted(picked.tolist()) == list(range(50))
    assert int(rng.choice(np.arange(5, 9))) in range(5, 9)
    with pytest.raises(ValueError):
        rng.choice(np.arange(3), size=4, replace=False)
    with pytest.raises(ValueError):
        rng.choice(np.array([]), size=1)
    with pytest.raises(NotImplementedError):
        rng.choice(np.arange(3), size=1, p=[0.2, 0.3, 0.5])


def test_permutation_is_keyed_and_complete():
    rng = host()
    rng.begin_step(6, 0)
    order = rng.permutation(20)
    expected = np.argsort([keyed_u01(SEED, 6, EVT_PRIORITY, i) for i in range(20)])
    np.testing.assert_array_equal(order, expected)
    shuffled = rng.permutation(np.arange(100, 120))
    assert sorted(shuffled.tolist()) == list(range(100, 120))


def test_scalar_draws_use_successive_positions_in_one_stream():
    rng = host()
    rng.begin_step(1, 3)
    a, b = rng.random(), rng.random()
    assert (a, b) == (keyed_u01(SEED, 1, 3, 0), keyed_u01(SEED, 1, 3, 1))
    assert rng.uniform(2.0, 4.0) == 2.0 + 2.0 * keyed_u01(SEED, 1, 3, 2)
    assert rng.random(dtype=np.float32).dtype == np.float32


@pytest.mark.parametrize("method, args", [("random", ()), ("uniform", ()), ("integers", (1 << 62,))])
@pytest.mark.parametrize("first_bulk", [None, 5])
def test_scalar_followed_by_bulk_uses_distinct_counters(xp, method, args, first_bulk):
    rng = make_keyed_device_rng(7, xp=xp)
    draw = getattr(rng, method)
    prior = [] if first_bulk is None else draw(*args, size=first_bulk).tolist()
    values = prior + [draw(*args)] + draw(*args, size=8).tolist()
    assert len(set(values)) == len(values)


def test_rebinding_keys_after_a_scalar_reserves_its_stream(xp):
    rng = make_keyed_device_rng(7, xp=xp)
    rng.set_agent_keys([5])
    scalar = rng.random()
    rng.set_agent_keys([5])
    assert rng.random() != scalar


def test_2d_draws_are_keyed_by_flat_index():
    rng = host()
    grid = rng.random((3, 4))
    assert grid.shape == (3, 4)
    np.testing.assert_array_equal(grid.ravel(), host().random(12))


def test_snapshot_restore_replays_draws(xp):
    rng = make_keyed_device_rng(SEED, xp=xp)
    model = Model({"seed": SEED})
    model._execution = SimpleNamespace(device_rng=rng)
    rng.begin_step(2, 1)
    keys = xp.array([5, 6, 7], dtype=xp.int64)
    rng.set_agent_keys(keys)
    rng.random(3)
    snapshot = model._capture_rng_state()
    saved = next(state for kind, _, _, state in snapshot.streams if kind == "device_rng")
    assert isinstance(saved[-1], np.ndarray)
    assert model._replay_values_equal(saved, rng._snapshot_state())
    first = rng.random(3).tolist()
    keys[:] = -1
    rng.set_agent_keys(None)
    rng.random()
    assert model._restore_rng_state(snapshot)
    assert isinstance(rng._keys, xp.ndarray)
    assert rng.random(3).tolist() == first
    rng._keys[:] = -2
    assert model._restore_rng_state(snapshot)
    assert rng.random(3).tolist() == first


def test_replay_compares_device_arrays_by_shape_dtype_and_value(xp, monkeypatch):
    model = Model({})
    asarray = xp.asarray
    if xp is np:
        class SimulatedCuPyArray:
            def __init__(self, values):
                self.values = values

            def __array__(self, *args, **kwargs):
                raise TypeError("implicit host conversion is unsupported")

            def __eq__(self, other):
                return SimulatedCuPyArray(self.values == other.values)

            def __bool__(self):
                raise ValueError("array truth value is ambiguous")

        fake_cupy = SimpleNamespace(ndarray=SimulatedCuPyArray, asnumpy=lambda a: a.values.copy())
        monkeypatch.setattr(gpu_backend, "_cp", fake_cupy)
        monkeypatch.setattr(gpu_backend, "GPU_AVAILABLE", True)

        def asarray(values):
            return SimulatedCuPyArray(np.asarray(values))

    values = np.array([1.0, np.nan])
    original = {"state": (0, asarray(values))}
    assert model._replay_values_equal(original, {"state": (0, asarray(values.copy()))})
    assert model._replay_values_equal(original, {"state": (0, values)})
    for other in (values.astype(np.float32), values.reshape(1, 2), [0.0, np.nan]):
        assert not model._replay_values_equal(original, {"state": (0, asarray(other))})


def test_is_a_device_rng_and_rejects_unknown_attributes():
    rng = host()
    assert isinstance(rng, DeviceRNG)
    assert rng.device_keys is True
    with pytest.raises(AttributeError):
        rng.standard_normal
    assert isinstance(make_keyed_device_rng(None, xp=np).global_seed, int)


def test_timed_models_use_keyed_draws():
    src = (Path(__file__).resolve().parents[1] / "benchmarks" / "models" / "amber_models.py").read_text(encoding="utf-8")
    assert src.count("rng = make_keyed_device_rng(self.p.get('seed'))") == 4
    assert "make_device_rng(" not in src


@gpu
@pytest.mark.parametrize("keyed", [False, True])
def test_device_kernels_match_host_reference(keyed):
    import cupy as cp

    dev = make_keyed_device_rng(SEED)
    ref = host()
    for rng in (dev, ref):
        rng.begin_step(12, 1)
        if keyed:
            rng.set_agent_keys(np.arange(1000)[::-1])

    def same(a, b):
        np.testing.assert_array_equal(cp.asnumpy(a), b)
        assert a.dtype == b.dtype

    same(dev.random(1000), ref.random(1000))
    same(dev.random(1000, dtype=cp.float32), ref.random(1000, dtype=np.float32))
    same(dev.uniform(-2.0, 2.0, 1000), ref.uniform(-2.0, 2.0, 1000))
    same(dev.uniform(0.0, 100.0, 1000, dtype=cp.float32),
         ref.uniform(0.0, 100.0, 1000, dtype=np.float32))
    same(dev.integers(-5, 1_000_003, size=1000), ref.integers(-5, 1_000_003, size=1000))
    same(dev.choice(cp.arange(10**6), size=1000), ref.choice(np.arange(10**6), size=1000))
    same(dev.random((40, 25)), ref.random((40, 25)))
    same(dev.permutation(300), ref.permutation(300))


@gpu
def test_device_keys_stay_on_device():
    import cupy as cp

    dev = make_keyed_device_rng(SEED)
    dev.begin_step(0, 1)
    dev.set_agent_keys(cp.arange(10, 20, dtype=cp.int64))
    assert isinstance(dev._keys, cp.ndarray)
    ref = host()
    ref.begin_step(0, 1)
    ref.set_agent_keys(np.arange(10, 20))
    np.testing.assert_array_equal(cp.asnumpy(dev.random(10)), ref.random(10))
