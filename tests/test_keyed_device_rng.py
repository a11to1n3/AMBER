"""Keyed (counter-based) draws used by the timed GPU models.

Each draw is a pure function of (seed, step, event, key, partner, draw index),
so its value cannot depend on the CUDA device or on thread order.  The NumPy
path (``xp=numpy``) is the reference; the GPU tests require the CUDA kernels to
reproduce it bit for bit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
    # Positions beyond the bound keys fall back to the position index.
    rng.begin_step(1, 1)
    rng.set_agent_keys([10])
    assert rng.random(2).tolist() == [keyed_u01(SEED, 1, 1, 10), keyed_u01(SEED, 1, 1, 1)]


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


def test_scalar_draws_advance_cursor_not_stream():
    rng = host()
    rng.begin_step(1, 3)
    a, b = rng.random(), rng.random()
    assert (a, b) == (keyed_u01(SEED, 1, 3, 0), keyed_u01(SEED, 1, 3, 1))
    assert rng.uniform(2.0, 4.0) == 2.0 + 2.0 * keyed_u01(SEED, 1, 3, 2)
    assert rng.random(dtype=np.float32).dtype == np.float32


def test_2d_draws_are_keyed_by_flat_index():
    rng = host()
    grid = rng.random((3, 4))
    assert grid.shape == (3, 4)
    np.testing.assert_array_equal(grid.ravel(), host().random(12))


def test_snapshot_restore_replays_draws():
    rng = host()
    rng.begin_step(2, 1)
    rng.set_agent_keys([5, 6, 7])
    rng.random(3)
    snap = rng._snapshot_state()
    first = rng.random(3)
    rng.random(9)
    rng._restore_state(snap)
    np.testing.assert_array_equal(rng.random(3), first)


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


gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="requires CuPy and a CUDA device")


@gpu
@pytest.mark.parametrize("keys", [None, [7, 3, 11, 2, 9]])
def test_device_kernels_match_host_reference(keys):
    import cupy as cp

    dev = make_keyed_device_rng(SEED)
    ref = host()
    for rng in (dev, ref):
        rng.begin_step(12, 1)
        if keys is not None:
            rng.set_agent_keys(keys)

    def same(a, b):
        np.testing.assert_array_equal(cp.asnumpy(a), b)
        assert a.dtype == b.dtype

    same(dev.random(1000), ref.random(1000))
    same(dev.random(1000, dtype=cp.float32), ref.random(1000, dtype=np.float32))
    same(dev.uniform(-2.0, 2.0, 1000), ref.uniform(-2.0, 2.0, 1000))
    same(dev.uniform(0.0, 100.0, 1000, dtype=cp.float32),
         ref.uniform(0.0, 100.0, 1000, dtype=np.float32))
    same(dev.integers(-5, 1_000_003, size=1000), ref.integers(-5, 1_000_003, size=1000))
    same(dev.choice(cp.arange(10**6), size=500), ref.choice(np.arange(10**6), size=500))
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
