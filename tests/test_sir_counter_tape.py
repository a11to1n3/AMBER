"""Counter-tape SplitMix64 used by production GPU SIR infection draws.

The CUDA kernel in ``benchmarks/models/amber_gpu_scale_models.py`` keys each
pair infection Bernoulli by
``(global_seed, step, EVT_INFECTION=4, min(i,j), max(i,j), draw_index=0)``.
These tests lock the pure-Python reference (must match device ``mix64`` /
``counter_u01`` bit-for-bit) and the Python call-path wiring for ``global_seed``.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Pure-Python reference — keep in lockstep with CUDA _MODULE_SRC counter_u01
# ---------------------------------------------------------------------------

_GOLDEN = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_MASK64 = (1 << 64) - 1
EVT_INFECTION = 4


def mix64(z: int) -> int:
    z = (z + _GOLDEN) & _MASK64
    z = ((z ^ (z >> 30)) * _M1) & _MASK64
    z = ((z ^ (z >> 27)) * _M2) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def counter_u01(
    global_seed: int,
    step: int,
    event_type: int,
    agent_id: int,
    partner_id: int = 0,
    draw_index: int = 0,
) -> float:
    x = int(global_seed) & _MASK64
    for v in (step, event_type, agent_id, partner_id, draw_index):
        x = mix64(x ^ (int(v) & _MASK64))
    u = mix64(x)
    return (u >> 11) * (1.0 / (1 << 53))


def pair_infection_u01(global_seed: int, step: int, i: int, j: int) -> float:
    lo, hi = (i, j) if i < j else (j, i)
    return counter_u01(global_seed, step, EVT_INFECTION, lo, hi, 0)


def test_mix64_known_constants():
    """SplitMix64 golden path: seed 0 mixes to a stable non-zero value."""
    assert mix64(0) == 0xE220A8397B1DCDAF
    assert mix64(1) != mix64(0)


def test_counter_u01_in_unit_interval_and_deterministic():
    keys = [
        (0, 0, 4, 0, 1, 0),
        (42, 3, 4, 10, 99, 0),
        (2**63 - 1, 100, 4, 5, 7, 0),
        (123456789, 0, 4, 1000, 1001, 1),
    ]
    for key in keys:
        a = counter_u01(*key)
        b = counter_u01(*key)
        assert a == b
        assert 0.0 <= a < 1.0


def test_counter_u01_exact_reference_values():
    """Bit-stable reference outputs (float from top 53 mantissa bits)."""

    def expand(global_seed, step, event_type, agent_id, partner_id, draw_index):
        x = int(global_seed) & _MASK64
        for v in (step, event_type, agent_id, partner_id, draw_index):
            x = mix64(x ^ (int(v) & _MASK64))
        return (mix64(x) >> 11) * (1.0 / (1 << 53))

    keys = [
        (0, 0, 4, 0, 1, 0),
        (1, 2, 4, 3, 5, 0),
        (42, 7, 4, 100, 101, 0),
    ]
    for key in keys:
        assert counter_u01(*key) == expand(*key)

    # Frozen numeric anchors (fail if constants or folding change)
    assert abs(counter_u01(0, 0, 4, 0, 1, 0) - 0.9540213506737841) < 1e-15
    assert abs(counter_u01(1, 2, 4, 3, 5, 0) - expand(1, 2, 4, 3, 5, 0)) < 1e-15


def test_pair_key_order_invariant():
    """Visit order must not change the RV assigned to unordered pair (i, j)."""
    for seed, step, i, j in [(0, 0, 1, 9), (7, 3, 100, 2), (99, 12, 5, 5)]:
        a = pair_infection_u01(seed, step, i, j)
        b = pair_infection_u01(seed, step, j, i)
        assert a == b


def test_different_pairs_or_steps_differ():
    a = pair_infection_u01(0, 0, 1, 2)
    b = pair_infection_u01(0, 0, 1, 3)
    c = pair_infection_u01(0, 1, 1, 2)
    d = pair_infection_u01(1, 0, 1, 2)
    assert len({a, b, c, d}) >= 3


def test_cuda_source_documents_counter_tape_and_pair_key():
    """Regression guard: production module source must keep pair-keyed SplitMix."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "benchmarks" / "models" / "amber_gpu_scale_models.py").read_text()
    assert "counter_u01" in src
    assert "SplitMix64" in src or "mix64" in src
    assert "min(i,j)" in src or "id < jid" in src
    assert "global_seed" in src
    assert "EVT_INFECTION" in src or "4u" in src


def test_sir_kernel_step_accepts_global_seed():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "benchmarks" / "models"))
    # Import may pull cupy only inside functions; module import should work.
    import amber_gpu_scale_models as m  # type: ignore

    sig = inspect.signature(m.sir_kernel_step)
    assert "global_seed" in sig.parameters
    # Default present so callers without seed stay backward-compatible
    assert sig.parameters["global_seed"].default == 0


def test_vectorized_sir_model_wires_global_seed_in_source():
    """AMBERVectorizedSIRModel must pass p.seed into sir_kernel_step."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "benchmarks" / "models" / "amber_models.py").read_text()
    assert "global_seed" in src
    assert "sir_kernel_step" in src
    # seed from parameters (either quote style)
    assert 'self.p.get("seed")' in src or "self.p.get('seed')" in src


def test_sir_kernel_step_seed_deterministic_on_gpu():
    """Same seed → identical status after one step (CuPy + GPU only)."""
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("No usable CUDA device")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "benchmarks" / "models"))
    from amber_gpu_scale_models import sir_kernel_step  # type: ignore

    import numpy as np

    n = 64
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 50, size=n).astype(np.float32)
    y = rng.uniform(0, 50, size=n).astype(np.float32)
    status = np.zeros(n, dtype=np.int8)
    status[:4] = 1
    infection_time = np.zeros(n, dtype=np.int32)

    def run(seed: int):
        return sir_kernel_step(
            x.copy(),
            y.copy(),
            status.copy(),
            infection_time.copy(),
            step=0,
            world_size=50.0,
            radius=8.0,
            transmission=0.9,
            recovery_time=14,
            global_seed=seed,
        )

    out_a = run(123)
    out_b = run(123)
    assert np.array_equal(np.asarray(out_a[2]), np.asarray(out_b[2]))


def test_flame_runtime_configure_does_not_raise():
    """FLAME NVRTC preload path must be safe when pyflamegpu is absent."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "benchmarks"))
    import run_all_frameworks as raf  # type: ignore

    # Should not raise regardless of whether CUDA libs exist
    raf._configure_flamegpu_runtime()
