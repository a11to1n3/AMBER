"""
Tests for ambr.performance module.
"""

import json
import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
import ambr as am
from ambr.performance import (
    SpatialIndex,
    ParallelRunner,
    RunOutcome,
    fast_distance_matrix,
    fast_neighbors_within_radius,
    fast_all_neighbors_within_radius,
    fast_random_walk_step,
    vectorized_wealth_transfer,
    vectorized_move,
    vectorized_random_velocities,
    vectorized_sir_infections,
    check_performance_deps,
    install_performance_deps,
    HAS_SCIPY,
    HAS_NUMBA,
    _run_single_simulation,  # private worker for direct testing
)

# Mock model for ParallelRunner testing (must be top-level for spawn pickling)
class MockModel(am.Model):
    def setup(self):
        pass
    def step(self):
        self.record_model("step_val", self.t)


class BoomModel(am.Model):
    """Always fails — used to assert structured ParallelRunner errors."""

    def setup(self):
        pass

    def step(self):
        raise RuntimeError("intentional boom")


class SlowModel(am.Model):
    """Sleeps in step — for hard-timeout tests."""

    def setup(self):
        pass

    def step(self):
        time.sleep(float(self.p.get("sleep", 2.0)))


class SideEffectModel(am.Model):
    """Either fails immediately or writes a marker after a delay.

    Uses param ``behavior`` (not ``mode``) — ``mode`` is reserved for AMBER
    execution lanes (``vectorized`` / ``oop``).
    """

    def setup(self):
        pass

    def step(self):
        from pathlib import Path as _Path

        behavior = self.p.get("behavior", "boom")
        if behavior == "boom":
            raise RuntimeError("boom for fail_fast")
        delay = float(self.p.get("delay", 1.5))
        time.sleep(delay)
        marker = self.p.get("marker")
        if marker:
            _Path(marker).write_text("written\n", encoding="utf-8")


class RetryThenSlowModel(am.Model):
    """First attempt fails (for retry); retry sleeps then writes a marker.

    Used to prove a retry started mid-scan cannot escape fail_fast cleanup.
    Process-local attempt counter via a file next to the marker.
    """

    def setup(self):
        pass

    def step(self):
        from pathlib import Path as _Path

        role = self.p.get("role", "fail_fast_trigger")
        if role == "fail_fast_trigger":
            time.sleep(float(self.p.get("delay", 0.05)))
            raise RuntimeError("trigger fail_fast")

        # retry_then_slow: count attempts in a sibling file
        marker = self.p.get("marker")
        counter = _Path(str(marker) + ".attempts")
        n = 0
        if counter.is_file():
            try:
                n = int(counter.read_text(encoding="utf-8").strip() or "0")
            except ValueError:
                n = 0
        n += 1
        counter.write_text(str(n), encoding="utf-8")
        if n == 1:
            raise RuntimeError("first attempt fails — schedule retry")
        # Second attempt (retry): long delay then side effect
        time.sleep(float(self.p.get("delay", 3.0)))
        if marker:
            _Path(marker).write_text("retry completed\n", encoding="utf-8")
        self.record_model("param_val", self.p.get("param", 0))


class Cancelled(Exception):
    """User exception whose *name* collides with fail_fast cancellation."""


class RaisesCancelled(am.Model):
    """Raises an exception class named Cancelled — a real failure."""

    def setup(self):
        pass

    def step(self):
        raise Cancelled("user failure named Cancelled")


@pytest.fixture
def sample_positions():
    """Create sample 2D positions for testing."""
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 2.0],
        [10.0, 10.0]  # Far away point
    ])

class TestSpatialIndex:
    """Test cases for SpatialIndex class."""

    def test_spatial_index_initialization(self):
        """Test initialization."""
        index = SpatialIndex()
        assert index.tree is None
        assert index.positions is None

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_spatial_index_build_and_query(self, sample_positions):
        """Test building index and querying radius."""
        index = SpatialIndex().build(sample_positions)

        # Query radius 1.5 around (0,0) -> should get (0,0), (1,0), (0,1)
        neighbors = index.query_radius(np.array([0.0, 0.0]), 1.5)
        neighbors.sort()
        assert neighbors == [0, 1, 2] # Indices

        # Query radius 0.5 around (0,0) -> only (0,0)
        neighbors = index.query_radius(np.array([0.0, 0.0]), 0.5)
        assert neighbors == [0]

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_spatial_index_knn(self, sample_positions):
        """Test KNN query."""
        index = SpatialIndex().build(sample_positions)

        # 3 Nearest to (0.1, 0.1) -> should be 0, 1, 2
        dists, indices = index.query_knn(np.array([0.1, 0.1]), k=3)

        # Check indices (order might vary slightly, but set should be {0,1,2})
        assert set(indices) == {0, 1, 2}
        assert len(dists) == 3

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_spatial_index_query_pairs(self, sample_positions):
        """Test querying pairs."""
        index = SpatialIndex().build(sample_positions)

        # Pairs within 1.1 distance
        # (0,0)-(1,0) dist 1.0
        # (0,0)-(0,1) dist 1.0
        # (1,0)-(2,2) dist sqrt(1+4) = 2.23 > 1.1
        pairs = index.query_pairs(1.1)

        # Should contain (0,1) and (0,2) pairs (indices)
        # scipy returns a set of tuples

        # Verify 0-1 and 0-2 are close enough
        has_01 = (0, 1) in pairs or (1, 0) in pairs
        has_02 = (0, 2) in pairs or (2, 0) in pairs
        assert has_01
        assert has_02

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_spatial_index_batch_query(self, sample_positions):
        """Test batch radius query."""
        index = SpatialIndex().build(sample_positions)

        points = np.array([[0.0, 0.0], [10.0, 10.0]])
        results = index.batch_query_radius(points, 1.5)

        assert len(results) == 2

        # First point neighbors [0, 1, 2]
        r1 = sorted(results[0])
        assert r1 == [0, 1, 2]

        # Second point neighbors [4]
        r2 = results[1]
        assert r2 == [4]

    def test_spatial_index_errors(self):
        """Test error handling when not built."""
        index = SpatialIndex()
        with pytest.raises(ValueError):
            index.query_radius([0,0], 1.0)
        with pytest.raises(ValueError):
            index.query_knn([0,0], 1)
        with pytest.raises(ValueError):
            index.query_pairs(1.0)
        with pytest.raises(ValueError):
            index.batch_query_radius([[0,0]], 1.0)


class TestNumbaFunctions:
    """Test Numba-accelerated functions."""

    def test_fast_distance_matrix(self, sample_positions):
        """Test distance matrix calculation."""
        # Calculate manually first
        # 0: (0,0), 1: (1,0) -> dist 1.0
        # 0: (0,0), 2: (0,1) -> dist 1.0
        # 1: (1,0), 2: (0,1) -> dist sqrt(2) ~ 1.414

        dist_mat = fast_distance_matrix(sample_positions)

        assert dist_mat.shape == (5, 5)
        assert dist_mat[0, 0] == 0.0
        assert dist_mat[0, 1] == 1.0
        assert dist_mat[1, 0] == 1.0
        assert np.isclose(dist_mat[1, 2], np.sqrt(2))

    def test_fast_neighbors_within_radius(self, sample_positions):
        """Test finding neighbors for single agent."""
        # Agent 0 at (0,0), Radius 1.5 -> Neighbors 1, 2
        neighbors = fast_neighbors_within_radius(sample_positions, 0, 1.5)
        neighbors.sort()
        assert neighbors == [1, 2]

        # Agent 4 at (10,10), Radius 1.0 -> No neighbors
        neighbors = fast_neighbors_within_radius(sample_positions, 4, 1.0)
        assert neighbors == []

    def test_fast_all_neighbors_within_radius(self, sample_positions):
        """Test finding all neighbors."""
        # Radius 1.5
        neighbors_matrix = fast_all_neighbors_within_radius(sample_positions, 1.5)

        # Agent 0 should have 1 and 2
        row0 = neighbors_matrix[0]
        valid0 = sorted([x for x in row0 if x != -1])
        assert valid0 == [1, 2]

        # Agent 4 should have none
        row4 = neighbors_matrix[4]
        valid4 = [x for x in row4 if x != -1]
        assert valid4 == []

    def test_fast_random_walk_step(self):
        """Test random walk movement."""
        positions = np.zeros((10, 2))
        velocities = np.ones((10, 2)) # Move by (1, 1)
        bounds = np.array([[0, 10], [0, 10]])

        # Simple move
        new_pos = fast_random_walk_step(positions, velocities, bounds, wrap=False)
        assert np.allclose(new_pos, 1.0)

        # Test clipping (bounds 0-10)
        pos_edge = np.array([[9.5, 9.5]])
        vel_large = np.array([[2.0, 2.0]])
        new_pos_clip = fast_random_walk_step(pos_edge, vel_large, bounds, wrap=False)
        # Should be clipped to 9.999 (bounds[1] - 0.001)
        assert new_pos_clip[0, 0] < 10.0
        assert new_pos_clip[0, 0] > 9.9

        # Test wrapping
        # 9.5 + 2.0 = 11.5. Wrap len 10 -> 1.5
        new_pos_wrap = fast_random_walk_step(pos_edge, vel_large, bounds, wrap=True)
        assert np.isclose(new_pos_wrap[0, 0], 1.5)


class TestVectorizedOperations:
    """Test vectorized utility functions."""

    def test_vectorized_wealth_transfer(self):
        """Test wealth transfer."""
        wealths = np.array([100.0, 100.0, 100.0])
        sources = np.array([0, 1])
        targets = np.array([1, 2])
        amounts = np.array([10.0, 20.0])

        # 0 -> 1: 10
        # 1 -> 2: 20
        # Expected:
        # 0: 100 - 10 = 90
        # 1: 100 + 10 - 20 = 90
        # 2: 100 + 20 = 120

        new_wealths = vectorized_wealth_transfer(wealths, amounts, sources, targets)

        assert new_wealths[0] == 90.0
        assert new_wealths[1] == 90.0
        assert new_wealths[2] == 120.0

        # Original should not be modified
        assert wealths[0] == 100.0

    def test_vectorized_move(self):
        """Test vectorized movement."""
        positions = np.zeros((5, 2))
        velocities = np.ones((5, 2))

        # Unbounded
        new_pos = vectorized_move(positions, velocities)
        assert np.all(new_pos == 1.0)

        # Bounded clip
        bounds = (0, 0.5)
        new_pos_clip = vectorized_move(positions, velocities, bounds, wrap=False)
        assert np.all(new_pos_clip == 0.5)

        # Bounded wrap
        # 0 + 1 = 1. Range 0.5. 1 % 0.5 = 0.
        new_pos_wrap = vectorized_move(positions, velocities, bounds, wrap=True)
        assert np.all(new_pos_wrap == 0.0)

    def test_vectorized_random_velocities(self):
        """Test velocity generation."""
        rng = np.random.default_rng(42)
        vels = vectorized_random_velocities(100, 1.0, rng=rng)

        assert vels.shape == (100, 2)
        assert np.all(vels >= -1.0)
        assert np.all(vels <= 1.0)
        assert not np.all(vels == vels[0]) # random values

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_vectorized_sir_infections(self, sample_positions):
        """Test SIR infection logic."""
        # 0=S, 1=I, 2=R
        statuses = np.zeros(5, dtype=int)
        statuses[0] = 1 # Patient zero at (0,0)

        index = SpatialIndex().build(sample_positions)

        # Radius 1.5 includes agents 1 (1,0) and 2 (0,1)
        # Transmission rate 1.0 -> definite infection
        new_statuses = vectorized_sir_infections(
            sample_positions, statuses, index,
            infection_radius=1.5, transmission_rate=1.0
        )

        assert new_statuses[0] == 1 # Still infected
        assert new_statuses[1] == 1 # Infected
        assert new_statuses[2] == 1 # Infected
        assert new_statuses[3] == 0 # Too far
        assert new_statuses[4] == 0 # Too far

        # Test with rate 0.0 -> no new infections
        new_statuses_safe = vectorized_sir_infections(
            sample_positions, statuses, index,
            infection_radius=1.5, transmission_rate=0.0
        )
        t = list(new_statuses_safe)
        assert t == [1, 0, 0, 0, 0]


class TestParallelRunner:
    """Test ParallelRunner class."""

    def test_parallel_runner_initialization(self):
        """Test initialization."""
        runner = ParallelRunner(MockModel, n_workers=2)
        assert runner.model_class == MockModel
        assert runner.n_workers == 2

    def test_run_single_simulation_helper(self):
        """Directly test the worker function."""
        params = {"param": 123, "steps": 5, "show_progress": False}
        result = _run_single_simulation(0, params, MockModel)

        assert result["index"] == 0
        assert result["status"] == "success"
        assert result["params"] == params
        assert result["result"] is not None
        assert result["result"]["model"] is not None

    def test_parallel_runner_execution(self):
        """Test full parallel execution — outcomes in input order."""
        runner = ParallelRunner(MockModel, n_workers=2)

        params_list = [
            {"steps": 2, "param": 10, "show_progress": False},
            {"steps": 2, "param": 20, "show_progress": False},
        ]

        with patch("builtins.print"):
            results = runner.run(params_list, show_progress=False)

        assert len(results) == 2
        assert all(isinstance(r, RunOutcome) for r in results)
        assert [r.index for r in results] == [0, 1]
        assert results[0].status == "success"
        assert results[0].params["param"] == 10
        assert results[1].params["param"] == 20
        assert results[0].result is not None
        assert "info" in results[0].result

    def test_parallel_runner_with_seeds(self):
        """Test run_with_seeds."""
        runner = ParallelRunner(MockModel, n_workers=2)
        base_params = {"steps": 1, "show_progress": False}
        seeds = [42, 43]

        with patch("builtins.print"):
            results = runner.run_with_seeds(base_params, seeds, show_progress=False)

        assert len(results) == 2
        assert {r.params["seed"] for r in results} == {42, 43}
        assert [r.index for r in results] == [0, 1]

    def test_parallel_runner_failed_outcome_visible(self, tmp_path):
        """Intentional model failures remain structured and diagnosable."""
        # Model class must be top-level (spawn pickling).
        runner = ParallelRunner(BoomModel, n_workers=1)
        outcomes = runner.run(
            [{"steps": 1, "show_progress": False}],
            show_progress=False,
            retry=0,
        )
        assert len(outcomes) == 1
        assert outcomes[0].status == "failed"
        assert outcomes[0].error_type == "RuntimeError"
        assert "boom" in (outcomes[0].error_message or "")
        assert outcomes[0].traceback

    def test_parallel_runner_checkpoint_resume(self, tmp_path):
        runner = ParallelRunner(MockModel, n_workers=1)
        ckpt = tmp_path / "runs.json"
        params_list = [
            {"steps": 1, "param": 1, "show_progress": False},
            {"steps": 1, "param": 2, "show_progress": False},
        ]
        with patch("builtins.print"):
            first = runner.run(
                params_list, show_progress=False, checkpoint_path=ckpt
            )
        assert ckpt.is_file()
        assert all(o.status == "success" for o in first)
        # JSON only — never pickle
        text = ckpt.read_text(encoding="utf-8")
        assert "schema_version" in text
        assert "ambr.ParallelRunner.checkpoint+json" in text

        with patch("builtins.print"):
            second = runner.run(
                params_list,
                show_progress=False,
                checkpoint_path=ckpt,
                resume=True,
                trust_checkpoint=True,
            )
        assert [o.params["param"] for o in second] == [1, 2]

    def test_checkpoint_resume_requires_trust_flag(self, tmp_path):
        runner = ParallelRunner(MockModel, n_workers=1)
        ckpt = tmp_path / "runs.json"
        with patch("builtins.print"):
            runner.run(
                [{"steps": 1, "show_progress": False}],
                show_progress=False,
                checkpoint_path=ckpt,
            )
        with pytest.raises(ValueError, match="trust_checkpoint"):
            runner.run(
                [{"steps": 1, "show_progress": False}],
                show_progress=False,
                checkpoint_path=ckpt,
                resume=True,
            )

    def test_checkpoint_rejects_pickle_bytes(self, tmp_path):
        ckpt = tmp_path / "evil.pkl"
        # Binary payload must not be loaded as a checkpoint
        ckpt.write_bytes(b"\x80\x04\x95pickle-payload")
        with pytest.raises(ValueError, match="JSON|pickle"):
            ParallelRunner._load_checkpoint(ckpt)

    def test_resume_rejects_different_workload(self, tmp_path):
        """Resume must not silently return results for different params."""
        runner = ParallelRunner(MockModel, n_workers=1)
        ckpt = tmp_path / "wl.json"
        with patch("builtins.print"):
            runner.run(
                [{"steps": 1, "param": 1, "show_progress": False}],
                show_progress=False,
                checkpoint_path=ckpt,
            )
        with patch("builtins.print"):
            with pytest.raises(ValueError, match="fingerprint|params|mismatch"):
                runner.run(
                    [{"steps": 1, "param": 999, "show_progress": False}],
                    show_progress=False,
                    checkpoint_path=ckpt,
                    resume=True,
                    trust_checkpoint=True,
                )

    def test_cancelled_slots_remain_pending_on_resume(self, tmp_path):
        """fail_fast Cancelled indices must re-run on resume, not stay finished."""
        runner = ParallelRunner(BoomModel, n_workers=1)
        ckpt = tmp_path / "cancel.json"
        params = [
            {"steps": 1, "show_progress": False, "tag": 0},
            {"steps": 1, "show_progress": False, "tag": 1},
            {"steps": 1, "show_progress": False, "tag": 2},
        ]
        with patch("builtins.print"):
            first = runner.run(
                params,
                show_progress=False,
                fail_fast=True,
                max_in_flight=1,
                checkpoint_path=ckpt,
            )
        assert first[0].status == "failed"
        assert first[1].status == "cancelled"
        assert first[2].status == "cancelled"
        # Later slots cancelled in-memory but must not be in checkpoint
        text = ckpt.read_text(encoding="utf-8")
        payload = json.loads(text)
        assert "1" not in payload["outcomes"]
        assert "2" not in payload["outcomes"]
        # Cancelled is a status, not an error_type string baked into the file
        for entry in payload["outcomes"].values():
            assert entry.get("status") != "cancelled"

        # Resume with a successful model for remaining work
        runner2 = ParallelRunner(MockModel, n_workers=1)
        # Workload fingerprint includes model_class — different model must fail
        with pytest.raises(ValueError, match="fingerprint|model_class"):
            runner2.run(
                params,
                show_progress=False,
                checkpoint_path=ckpt,
                resume=True,
                trust_checkpoint=True,
            )

        # Same model: cancelled indices are pending and re-executed (still Boom)
        with patch("builtins.print"):
            second = runner.run(
                params,
                show_progress=False,
                fail_fast=True,
                max_in_flight=1,
                checkpoint_path=ckpt,
                resume=True,
                trust_checkpoint=True,
            )
        # Index 0 was already failed in checkpoint — restored without re-run;
        # 1 and 2 were pending (omitted cancelled) so they execute again.
        assert second[0].status == "failed"
        assert second[0].error_type == "RuntimeError"

    def test_parallel_runner_hard_timeout_wall_clock(self):
        """timeout must terminate the worker, not wait for it to finish."""
        runner = ParallelRunner(SlowModel, n_workers=1)
        sleep_s = 5.0
        t0 = time.monotonic()
        with patch("builtins.print"):
            outcomes = runner.run(
                [{"steps": 1, "show_progress": False, "sleep": sleep_s}],
                show_progress=False,
                timeout=0.3,
            )
        elapsed = time.monotonic() - t0
        assert len(outcomes) == 1
        assert outcomes[0].status == "timeout"
        # Spawn/import overhead varies by platform; only require we did not
        # wait out the full worker sleep.
        assert elapsed < sleep_s * 0.8, (
            f"timeout did not kill worker: elapsed={elapsed:.2f}s "
            f"(worker sleep={sleep_s}s)"
        )
        import multiprocessing as mp

        live = [p for p in mp.active_children() if p.is_alive()]
        assert live == [], f"live children remain: {live}"

    def test_parallel_runner_fail_fast_cancels_rest(self):
        runner = ParallelRunner(BoomModel, n_workers=2)
        with patch("builtins.print"):
            outcomes = runner.run(
                [
                    {"steps": 1, "show_progress": False},
                    {"steps": 1, "show_progress": False},
                    {"steps": 1, "show_progress": False},
                ],
                show_progress=False,
                fail_fast=True,
                max_in_flight=1,
            )
        assert len(outcomes) == 3
        assert outcomes[0].status == "failed"
        # Remaining slots marked cancelled, not silent success
        assert all(o.status in {"failed", "timeout", "cancelled"} for o in outcomes)
        assert any(o.status == "cancelled" for o in outcomes)

    def test_retry_cannot_escape_fail_fast_cleanup(self, tmp_path):
        """A mid-scan retry must be in the live registry for fail_fast kill.

        Reproduction shape: worker A fails (retry starts), then sibling B
        fails with fail_fast — the retry must not stay alive and write a
        delayed side effect after run() returns.
        """
        marker = tmp_path / "retry_side_effect.txt"
        runner = ParallelRunner(RetryThenSlowModel, n_workers=2)
        params = [
            {
                "steps": 1,
                "show_progress": False,
                "marker": str(marker),
                "role": "retry_then_slow",
                "delay": 3.0,
            },
            {
                "steps": 1,
                "show_progress": False,
                "marker": str(marker),
                "role": "fail_fast_trigger",
                "delay": 0.05,
            },
        ]
        with patch("builtins.print"):
            outcomes = runner.run(
                params,
                show_progress=False,
                fail_fast=True,
                retry=1,
                max_in_flight=2,
            )
        assert len(outcomes) == 2
        time.sleep(0.5)
        assert not marker.exists(), "retry worker wrote after fail_fast return"
        import multiprocessing as mp

        live = [p for p in mp.active_children() if p.is_alive()]
        assert live == [], f"live children remain: {live}"

    def test_fail_fast_terminates_in_flight_siblings(self, tmp_path):
        """With two in-flight workers, fail_fast must kill the sibling.

        Regression: breaking mid-loop left the second process alive so it
        could still write side effects after run() returned.

        Uses param ``behavior`` (not reserved ``mode``) so SideEffectModel.step
        actually runs. Assertions prefer side-effect / live-child checks over
        tight wall-clock bounds (spawn+import overhead varies by platform).
        """
        marker = tmp_path / "side_effect.txt"
        slow_delay = 8.0
        # Slow write first so it is almost certainly running when boom fails;
        # both must be in flight (max_in_flight=2).
        runner = ParallelRunner(SideEffectModel, n_workers=2)
        params = [
            {
                "steps": 1,
                "show_progress": False,
                "marker": str(marker),
                "behavior": "slow_write",
                "delay": slow_delay,
            },
            {
                "steps": 1,
                "show_progress": False,
                "marker": str(marker),
                "behavior": "boom",
                "delay": 0.0,
            },
        ]
        t0 = time.monotonic()
        with patch("builtins.print"):
            outcomes = runner.run(
                params,
                show_progress=False,
                fail_fast=True,
                max_in_flight=2,  # both in flight — critical for the bug
            )
        elapsed = time.monotonic() - t0
        assert len(outcomes) == 2
        # Boom must actually run step() and fail as RuntimeError (not mode validation)
        assert any(
            o.status == "failed" and o.error_type == "RuntimeError" for o in outcomes
        ), outcomes
        # Primary correctness: sibling must not complete its delayed write, and
        # no worker children remain. Allow generous headroom for spawn/import.
        time.sleep(0.75)
        assert not marker.exists(), "sibling worker still wrote after fail_fast"
        import multiprocessing as mp

        live = [p for p in mp.active_children() if p.is_alive()]
        assert live == [], f"live children remain: {live}"
        # Soft timing bound only: must not wait out the full slow delay.
        assert elapsed < slow_delay * 0.85, (
            f"fail_fast waited for sibling: {elapsed:.2f}s "
            f"(slow_delay={slow_delay}s)"
        )

    def test_max_in_flight_never_exceeds_n_workers(self):
        runner = ParallelRunner(MockModel, n_workers=2)
        with pytest.raises(ValueError, match="max_in_flight"):
            runner.run(
                [{"steps": 1, "show_progress": False}],
                show_progress=False,
                max_in_flight=0,
            )
        # max_in_flight=8 with n_workers=2 must not start 8 processes
        with patch("builtins.print"):
            outcomes = runner.run(
                [{"steps": 1, "param": i, "show_progress": False} for i in range(4)],
                show_progress=False,
                max_in_flight=8,
            )
        assert len(outcomes) == 4
        assert all(o.status == "success" for o in outcomes)

    def test_checkpoint_preserves_dtypes(self):
        """Arrow IPC payload keeps UInt8 / Categorical / Datetime."""
        import polars as pl
        from datetime import datetime, timezone
        from ambr.performance import _deserialize_frame, _serialize_frame

        df = pl.DataFrame(
            {
                "flag": pl.Series("flag", [1, 2], dtype=pl.UInt8),
                "label": pl.Series("label", ["A", "B"], dtype=pl.Categorical),
                "ts": pl.Series(
                    "ts",
                    [
                        datetime(2024, 1, 1, tzinfo=timezone.utc),
                        datetime(2024, 1, 2, tzinfo=timezone.utc),
                    ],
                ),
            }
        )
        payload = _serialize_frame(df)
        assert payload is not None
        assert payload["_kind"] == "polars_ipc_b64"
        restored = _deserialize_frame(payload, schema_version=2)
        assert restored.schema["flag"] == pl.UInt8
        assert str(restored.schema["label"]).startswith("Categorical")
        assert restored.schema["ts"] == df.schema["ts"]
        assert restored["flag"].to_list() == [1, 2]

    def test_checkpoint_writer_emits_schema_4(self, tmp_path):
        runner = ParallelRunner(MockModel, n_workers=1)
        ckpt = tmp_path / "s4.json"
        with patch("builtins.print"):
            runner.run(
                [{"steps": 1, "show_progress": False}],
                show_progress=False,
                checkpoint_path=ckpt,
            )
        payload = json.loads(ckpt.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 4
        assert "workload_fingerprint" in payload
        assert payload["model_class"].endswith("MockModel")
        assert "model_source_digest" in payload
        assert payload["model_source_digest"]  # source available for test models
        # Frames use IPC kind under schema 2+ (never under schema 1)
        model = payload["outcomes"]["0"]["result"]["model"]
        if model is not None:
            assert model.get("_kind") == "polars_ipc_b64"

    def test_legacy_schema_1_records_still_load(self, tmp_path):
        """Explicit legacy schema 1 (lossy records) remains readable."""
        import polars as pl
        from ambr.performance import ParallelRunner

        ckpt = tmp_path / "legacy.json"
        legacy = {
            "schema_version": 1,
            "format": "ambr.ParallelRunner.checkpoint+json",
            "outcomes": {
                "0": {
                    "index": 0,
                    "status": "success",
                    "params": {"steps": 1},
                    "result": {
                        "params": {"steps": 1},
                        "info": {"steps": 1},
                        "model": {
                            "_kind": "polars_records",
                            "columns": ["t", "x"],
                            "rows": [{"t": 1, "x": 2}],
                        },
                        "agents": None,
                    },
                    "error_type": None,
                    "error_message": None,
                    "traceback": None,
                    "attempts": 1,
                }
            },
        }
        ckpt.write_text(json.dumps(legacy), encoding="utf-8")
        _meta, loaded = ParallelRunner._load_checkpoint(ckpt)
        assert 0 in loaded
        assert loaded[0].status == "success"
        assert isinstance(loaded[0].result["model"], pl.DataFrame)
        assert loaded[0].result["model"]["x"].to_list() == [2]

    def test_ipc_failure_raises_clearly(self):
        """Object columns that cannot IPC-encode must not silently become str."""
        import polars as pl
        from ambr.performance import (
            CheckpointSerializationError,
            _serialize_frame,
        )

        df = pl.DataFrame({"obj": pl.Series("obj", [{"a": 1}], dtype=pl.Object)})
        with pytest.raises(CheckpointSerializationError, match="Arrow IPC|Object"):
            _serialize_frame(df)

    def test_schema_1_rejects_ipc_kind(self):
        """IPC payloads must not load under pure schema-1 (records-only)."""
        from ambr.performance import _deserialize_frame

        with pytest.raises(ValueError, match="polars_ipc_b64|schema_version"):
            _deserialize_frame(
                {
                    "_kind": "polars_ipc_b64",
                    "columns": ["x"],
                    "dtypes": ["Int64"],
                    "data": "AAAA",
                },
                schema_version=1,
            )

    def test_schema_2_rejects_lossy_records_kind(self):
        from ambr.performance import _deserialize_frame

        with pytest.raises(ValueError, match="polars_records"):
            _deserialize_frame(
                {"_kind": "polars_records", "columns": ["x"], "rows": [{"x": 1}]},
                schema_version=2,
            )

    def test_user_cancelled_exception_is_persisted(self, tmp_path):
        """A model raising class Cancelled must not be treated as never-run."""
        runner = ParallelRunner(RaisesCancelled, n_workers=1)
        ckpt = tmp_path / "user_cancel.json"
        with patch("builtins.print"):
            first = runner.run(
                [{"steps": 1, "show_progress": False}],
                show_progress=False,
                checkpoint_path=ckpt,
            )
        assert first[0].status == "failed"
        assert first[0].error_type == "Cancelled"
        payload = json.loads(ckpt.read_text(encoding="utf-8"))
        assert "0" in payload["outcomes"]
        assert payload["outcomes"]["0"]["status"] == "failed"
        assert payload["outcomes"]["0"]["error_type"] == "Cancelled"

        # Resume must restore the failed outcome, not re-run it away
        with patch("builtins.print"):
            second = runner.run(
                [{"steps": 1, "show_progress": False}],
                show_progress=False,
                checkpoint_path=ckpt,
                resume=True,
                trust_checkpoint=True,
            )
        assert second[0].status == "failed"
        assert second[0].error_type == "Cancelled"

    def test_fingerprint_includes_model_source_digest(self):
        """Different class bodies produce different source digests / fingerprints.

        Reproduces the review case where two model definitions share a name
        string: we force identical ``model_class`` in the identity payload and
        still require the digest to diverge.
        """
        from ambr.performance import (
            _model_source_digest,
            _workload_fingerprint,
            _workload_identity,
        )

        params = [{"steps": 1, "show_progress": False}]

        class BodyA(am.Model):
            def setup(self):
                pass

            def step(self):
                self.record_model("v", 1)

        class BodyB(am.Model):
            def setup(self):
                pass

            def step(self):
                self.record_model("v", 999)

        d1 = _model_source_digest(BodyA)
        d2 = _model_source_digest(BodyB)
        assert d1 and d2 and d1 != d2

        id_a = _workload_identity(BodyA, params)
        id_b = _workload_identity(BodyB, params)
        # Simulate same qualified name (review reproduction) — digests still differ.
        id_a["model_class"] = "pkg.SameModel"
        id_b["model_class"] = "pkg.SameModel"
        assert id_a["model_source_digest"] != id_b["model_source_digest"]
        from ambr.performance import _stable_json
        import hashlib

        fp_a = hashlib.sha256(_stable_json(id_a).encode("utf-8")).hexdigest()
        fp_b = hashlib.sha256(_stable_json(id_b).encode("utf-8")).hexdigest()
        assert fp_a != fp_b
        # Sanity: normal fingerprints (with real class names) also differ.
        assert _workload_fingerprint(BodyA, params) != _workload_fingerprint(
            BodyB, params
        )

    def test_workload_revision_invalidates_resume(self, tmp_path):
        """Caller-supplied workload_revision is part of the fingerprint."""
        from ambr.performance import _workload_fingerprint

        params = [{"steps": 1, "show_progress": False}]
        a = _workload_fingerprint(MockModel, params, workload_revision="v1")
        b = _workload_fingerprint(MockModel, params, workload_revision="v2")
        assert a != b

        ckpt = tmp_path / "rev.json"
        runner_v1 = ParallelRunner(MockModel, n_workers=1, workload_revision="v1")
        with patch("builtins.print"):
            runner_v1.run(params, show_progress=False, checkpoint_path=ckpt)
        runner_v2 = ParallelRunner(MockModel, n_workers=1, workload_revision="v2")
        with patch("builtins.print"):
            with pytest.raises(ValueError, match="fingerprint"):
                runner_v2.run(
                    params,
                    show_progress=False,
                    checkpoint_path=ckpt,
                    resume=True,
                    trust_checkpoint=True,
                )

    def test_checkpoint_symlink_tmp_cannot_escape(self, tmp_path):
        """Predictable *.json.tmp plant must not redirect checkpoint writes."""
        dest = tmp_path / "ckpt.json"
        outside = tmp_path / "escaped.txt"
        outside.write_text("ORIGINAL\n", encoding="utf-8")
        planted = tmp_path / "ckpt.json.tmp"
        planted.symlink_to(outside)

        runner = ParallelRunner(MockModel, n_workers=1)
        with patch("builtins.print"):
            runner.run(
                [{"steps": 1, "show_progress": False}],
                show_progress=False,
                checkpoint_path=dest,
            )
        assert outside.read_text(encoding="utf-8") == "ORIGINAL\n"
        assert dest.is_file() and not dest.is_symlink()
        assert "schema_version" in dest.read_text(encoding="utf-8")

    def test_parallel_runner_ordered_indices(self):
        runner = ParallelRunner(MockModel, n_workers=2)
        params = [
            {"steps": 1, "param": i, "show_progress": False} for i in range(5)
        ]
        with patch("builtins.print"):
            outcomes = runner.run(params, show_progress=False, max_in_flight=2)
        assert [o.index for o in outcomes] == list(range(5))
        assert [o.params["param"] for o in outcomes] == list(range(5))


class TestDependencyUtilities:
    """Test dependency checking functions."""

    def test_check_performance_deps(self):
        """Test check_performance_deps."""
        deps = check_performance_deps()
        assert "scipy" in deps
        assert "numba" in deps
        assert "multiprocessing" in deps

        if HAS_SCIPY:
            assert deps["scipy"] is True

    def test_install_performance_deps(self):
        """Test install print output."""
        with patch("builtins.print") as mock_print:
            install_performance_deps()
            mock_print.assert_called()
