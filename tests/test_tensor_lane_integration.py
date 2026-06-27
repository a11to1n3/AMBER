"""End-to-end: the Boids flocking model on the tensor lane, under the contract.

Verifies that (1) the vectorized tensor interaction kernel matches an explicit
double-loop reference (including the coincident-agent edge case), (2) running
the model under ``contract='check'`` produces clean certificates -- and that the
certificate is *non-vacuous* for lane writes: it flags same-step double-commits
and read-after-write, not just schema drift, (3) the run is deterministic for a
fixed seed and touches no global RNG, and (4) the contract does not silently
miss dtype changes.
"""

import os
import sys

import numpy as np
import pytest

import ambr as am
from ambr.tensor_lane import borrow_numeric, commit_columns, SINGLE_COL_ZERO_COPY

# Import the example model (single source of truth for the physics).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
try:
    from flocking_tensor import (  # noqa: E402
        FlockingModel,
        boids_forces_vectorized,
        boids_forces_loop,
    )
except ImportError:
    pytest.skip(
        "examples/flocking_tensor.py is not present in this tree (it lands in the "
        "examples commit); skipping the tensor-lane integration test. The tensor "
        "lane itself is covered by tests/test_tensor_lane.py.",
        allow_module_level=True,
    )


PARAMS = dict(n_agents=40, width=50.0, radius=6.0, w_cohesion=0.03,
              w_align=0.07, w_separation=1.5, vmax=2.0, show_progress=False)


def _state(m):
    df = m.agents_df.sort("id")
    return tuple(df[c].to_numpy().copy() for c in ("x", "y", "vx", "vy"))


# --- (1) the tensor kernel is correct --------------------------------------

def test_vectorized_forces_match_loop_reference():
    m = FlockingModel({**PARAMS, "n_agents": 30, "seed": 5})
    m.setup()
    x, y, vx, vy = _state(m)
    a = boids_forces_vectorized(x, y, vx, vy, m.L, m.r, m.w_c, m.w_a, m.w_s)
    b = boids_forces_loop(x, y, vx, vy, m.L, m.r, m.w_c, m.w_a, m.w_s)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-10, atol=1e-12)


def test_kernel_handles_degenerate_sizes():
    # N in {0, 1, 2} must not divide by zero or produce NaN.
    for N in (0, 1, 2):
        z = np.zeros(N)  # for N=2 these are coincident agents (dist2 == 0)
        ax, ay = boids_forces_vectorized(z, z, z, z, 10.0, 2.0, 1.0, 1.0, 1.0)
        assert ax.shape == (N,) and ay.shape == (N,)
        assert np.all(np.isfinite(ax)) and np.all(np.isfinite(ay))


def test_coincident_agents_do_not_produce_nan():
    # Two distinct agents at the same position give an off-diagonal dist2 == 0;
    # the separation term must exclude it (not 0 * inf = NaN), matching the loop.
    x = np.array([1.0, 1.0, 5.0])
    y = np.array([1.0, 1.0, 5.0])
    vx = np.array([0.1, -0.2, 0.3])
    vy = np.array([0.0, 0.1, -0.1])
    args = (x, y, vx, vy, 100.0, 3.0, 0.03, 0.07, 1.5)
    a = boids_forces_vectorized(*args)
    b = boids_forces_loop(*args)
    assert np.all(np.isfinite(a[0])) and np.all(np.isfinite(a[1]))
    np.testing.assert_allclose(a[0], b[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-10, atol=1e-12)


# --- (2) the run is snapshot-conforming ------------------------------------

def test_flocking_run_is_contract_conforming():
    m = FlockingModel({**PARAMS, "seed": 1, "steps": 15})
    res = m.run(contract="check")
    assert "contract" in res
    certs = res["contract"]
    assert len(certs) == 15
    # Every step writes only existing columns, no births/deaths -> clean.
    assert all(c.clean for c in certs), [c for c in certs if not c.clean]


def test_flocking_raise_mode_completes():
    m = FlockingModel({**PARAMS, "seed": 1, "steps": 10})
    res = m.run(contract="raise")  # would raise on any error-severity violation
    assert len(res["contract"]) == 10


# --- (3) determinism + state sanity ----------------------------------------

def test_flocking_is_deterministic_for_fixed_seed():
    a = FlockingModel({**PARAMS, "seed": 99, "steps": 12}).run()
    b = FlockingModel({**PARAMS, "seed": 99, "steps": 12}).run()
    for ca, cb in zip(("x", "y", "vx", "vy"), ("x", "y", "vx", "vy")):
        np.testing.assert_array_equal(
            a["agents"].sort("id")[ca].to_numpy(),
            b["agents"].sort("id")[cb].to_numpy(),
        )


def test_state_stays_finite_and_in_bounds():
    m = FlockingModel({**PARAMS, "seed": 3, "steps": 30})
    res = m.run()
    df = res["agents"]
    for c in ("x", "y", "vx", "vy"):
        arr = df[c].to_numpy()
        assert np.all(np.isfinite(arr)), f"{c} went non-finite"
    assert df["x"].min() >= 0 and df["x"].max() < m.L
    assert df["y"].min() >= 0 and df["y"].max() < m.L
    speed = np.sqrt(df["vx"].to_numpy() ** 2 + df["vy"].to_numpy() ** 2)
    assert np.all(speed <= m.vmax + 1e-9)


def test_population_and_schema_are_stable():
    m = FlockingModel({**PARAMS, "seed": 1, "steps": 5})
    res = m.run(contract="check")
    # No certificate reports schema or population mutation.
    kinds = {v.kind for c in res["contract"] for v in c.violations}
    assert "population_mutation" not in kinds
    assert "schema_mutation" not in kinds


# --- (4) the contract is non-vacuous for this model style ------------------

def test_contract_flags_schema_drift_variant():
    """A flocking variant that adds a column mid-step must be flagged."""
    class DriftingFlock(FlockingModel):
        def step(self):
            super().step()
            # Introduce a brand-new column mid-step (only on step 0).
            x, _ = borrow_numeric(self, "x")
            self.population.data = self.agents_df.with_columns(
                __scratch__=__import__("polars").Series("__scratch__", x * 2)
            )
    m = DriftingFlock({**PARAMS, "seed": 1, "steps": 3})
    res = m.run(contract="check")
    kinds = {v.kind for c in res["contract"] for v in c.violations}
    assert "schema_mutation" in kinds


def test_contract_flags_same_step_double_commit():
    """A column committed twice in one step (lane path) must be flagged."""
    class DoubleCommit(FlockingModel):
        def step(self):
            x, _ = borrow_numeric(self, "x")
            commit_columns(self, x=x + 1.0)
            commit_columns(self, x=x + 2.0)  # clobbers the first commit
    m = DoubleCommit({**PARAMS, "seed": 1, "steps": 2})
    res = m.run(contract="check")
    certs = res["contract"]
    assert any(not c.ok for c in certs)
    assert "duplicate_write" in {v.kind for c in certs for v in c.violations}


def test_contract_flags_read_after_write():
    """Borrowing a column after committing it in the same step must be flagged."""
    class ReadAfterWrite(FlockingModel):
        def step(self):
            x, _ = borrow_numeric(self, "x")
            commit_columns(self, x=x + 1.0)
            x2, _ = borrow_numeric(self, "x")  # reads post-write state
            commit_columns(self, y=x2)         # commit a different column (no dup)
    m = ReadAfterWrite({**PARAMS, "seed": 1, "steps": 2})
    res = m.run(contract="check")
    certs = res["contract"]
    kinds = {v.kind for c in certs for v in c.violations}
    assert "read_after_write" in kinds
    assert any(not c.ok for c in certs)


def test_contract_flags_dtype_change_when_not_preserved():
    """A raw with_columns that re-types a column is caught (dtype in snapshot)."""
    import polars as pl

    class Retype(FlockingModel):
        def step(self):
            super().step()
            # Bypass commit_columns' dtype preservation: force x to Float32.
            df = self.agents_df
            self.population.data = df.with_columns(
                pl.col("x").cast(pl.Float32)
            )
    m = Retype({**PARAMS, "seed": 1, "steps": 2})
    res = m.run(contract="check")
    kinds = {v.kind for c in res["contract"] for v in c.violations}
    assert "schema_mutation" in kinds


def test_commit_columns_preserves_dtype():
    """commit_columns must not silently widen/narrow an existing column dtype."""
    import polars as pl
    import numpy as np

    class M(am.Model):
        def setup(self):
            self.add_agents(5, energy=np.arange(5, dtype=np.int64))
        def step(self):
            e, _ = borrow_numeric(self, "energy")
            commit_columns(self, energy=e.astype(np.float64) + 0.0)
    m = M({"show_progress": False})
    m.setup()
    m.step()
    assert m.agents_df["energy"].dtype == pl.Int64


def test_run_does_not_touch_global_numpy_rng():
    """Determinism robustness: the model must use only its per-instance RNG."""
    import numpy as np
    before = np.random.get_state()
    FlockingModel({**PARAMS, "seed": 7, "steps": 8}).run()
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])  # MT19937 state array unchanged


# --- zero-copy borrow is actually used -------------------------------------

def test_borrow_is_zero_copy_when_supported():
    m = FlockingModel({**PARAMS, "seed": 1})
    m.setup()
    _, is_view = borrow_numeric(m, "x")
    assert is_view == SINGLE_COL_ZERO_COPY
