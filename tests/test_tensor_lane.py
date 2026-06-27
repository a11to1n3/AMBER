"""Tests for the Polars-backed numeric tensor lane.

These tests are written to be *invariant to the Polars version*: they assert
correctness unconditionally, and only ever assert that the zero-copy status is
*consistent with the runtime capability probe* -- never that zero-copy is
achieved on a particular build. They therefore pass whether or not this Polars
build supports Array-dtype zero-copy round-tripping.
"""

import numpy as np
import polars as pl
import pytest

import ambr as am
from ambr.tensor_lane import (
    TensorLane,
    borrow_numeric,
    ARRAY_BACKING_AVAILABLE,
    SINGLE_COL_ZERO_COPY,
)


def _model(n=50, F=4, seed=0):
    rng = np.random.default_rng(seed)
    cols = {f"a{i}": rng.random(n) for i in range(F)}

    class M(am.Model):
        def setup(self):
            self.add_agents(n, **cols)

        def step(self):
            pass

    m = M({"show_progress": False})
    m.setup()
    return m, cols


# --- single-column borrow --------------------------------------------------

def test_borrow_numeric_is_correct():
    m, cols = _model(F=1)
    arr, is_view = borrow_numeric(m, "a0")
    np.testing.assert_allclose(arr, m.agents_df["a0"].to_numpy())
    # is_view must be consistent with the probe (never asserted absolutely).
    if not SINGLE_COL_ZERO_COPY:
        assert is_view is False


def test_borrow_numeric_handles_chunked_column_without_mutating_model():
    m, _ = _model(n=10, F=1)
    # Force a multi-chunk column the way AMBER's append/concat path would.
    df = m.agents_df
    m.population.data = pl.concat([df, df])  # vstack -> >1 chunk
    chunks_before = m.agents_df["a0"].n_chunks()
    arr, is_view = borrow_numeric(m, "a0")
    col = m.agents_df["a0"]
    # The borrow returns a correct, contiguous, read-only array...
    assert len(arr) == col.len()
    np.testing.assert_allclose(arr, col.to_numpy())
    assert arr.flags.writeable is False
    # ...and a "borrow" must not have mutated the model (no persisted rechunk).
    assert m.agents_df["a0"].n_chunks() == chunks_before


# --- packed (N, F) lane ----------------------------------------------------

def test_lane_borrow_matches_source():
    m, _ = _model(n=40, F=5)
    expected = m.agents_df.select([f"a{i}" for i in range(5)]).to_numpy()
    lane = TensorLane(m, [f"a{i}" for i in range(5)])
    arr, is_view = lane.borrow()
    assert arr.shape == (40, 5)
    np.testing.assert_allclose(arr, expected)
    # zero-copy status is consistent with the probe, not asserted absolutely
    assert is_view == lane.is_zero_copy


def test_lane_commit_roundtrip():
    m, _ = _model(n=30, F=4)
    lane = TensorLane(m, [f"a{i}" for i in range(4)])
    arr, _ = lane.borrow()
    lane.commit(arr + 1.0)
    after, _ = lane.borrow()
    np.testing.assert_allclose(after, arr + 1.0)


def test_lane_unpack_restores_named_columns():
    m, _ = _model(n=25, F=3)
    names = [f"a{i}" for i in range(3)]
    before = m.agents_df.select(names).to_numpy()
    lane = TensorLane(m, names)
    arr, _ = lane.borrow()
    lane.commit(arr * 2.0)
    lane.unpack()
    df = m.agents_df
    for i, name in enumerate(names):
        assert name in df.columns
        np.testing.assert_allclose(df[name].to_numpy(), before[:, i] * 2.0)


def test_lane_write_result_attaches_column():
    m, _ = _model(n=20, F=2)
    lane = TensorLane(m, ["a0", "a1"])
    arr, _ = lane.borrow()
    lane.write_result("rowsum", arr.sum(axis=1))
    np.testing.assert_allclose(
        m.agents_df["rowsum"].to_numpy(), arr.sum(axis=1)
    )


def test_stacked_fallback_is_correct_regardless_of_probe():
    """Force the copy-based path and prove it is identical to the source."""
    m, _ = _model(n=35, F=6)
    names = [f"a{i}" for i in range(6)]
    expected = m.agents_df.select(names).to_numpy()
    lane = TensorLane(m, names, prefer_array=False)  # force 'stacked'
    assert lane.mode == "stacked"
    arr, is_view = lane.borrow()
    assert is_view is False  # copy path never claims to be a view
    np.testing.assert_allclose(arr, expected)
    lane.commit(arr + 3.0)
    for i, name in enumerate(names):
        np.testing.assert_allclose(m.agents_df[name].to_numpy(), expected[:, i] + 3.0)


# --- interaction kernel through the lane (the headline use case) -----------

def test_interaction_kernel_matches_reference():
    m, _ = _model(n=64, F=1, seed=7)
    x_ref = m.agents_df["a0"].to_numpy().copy()
    reference = (x_ref[None, :] - x_ref[:, None]).sum(axis=1)

    x, _ = borrow_numeric(m, "a0")
    force = (x[None, :] - x[:, None]).sum(axis=1)  # (N, N) contraction
    m.population.data = m.agents_df.with_columns(pl.Series("force", force))

    np.testing.assert_allclose(m.agents_df["force"].to_numpy(), reference)


def test_probe_flags_are_boolean():
    assert isinstance(ARRAY_BACKING_AVAILABLE, bool)
    assert isinstance(SINGLE_COL_ZERO_COPY, bool)
