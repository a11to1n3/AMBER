"""Import hygiene: no matplotlib backend clobber; cold-start smoke."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time

import pytest


@pytest.mark.unit
def test_import_ambr_does_not_change_matplotlib_backend():
    """``import ambr`` must not call matplotlib.use or alter the active backend."""
    matplotlib = pytest.importorskip("matplotlib")

    # Establish a known backend before importing ambr (fresh check via name).
    backend = matplotlib.get_backend()
    import ambr  # noqa: F401

    assert matplotlib.get_backend() == backend


@pytest.mark.unit
def test_lazy_viz_exports_resolvable():
    import ambr as am

    # Attribute access should resolve without ImportError when viz deps missing
    # only if matplotlib is absent — HAS_MATPLOTLIB becomes False. Accessing
    # the names must not raise AttributeError.
    assert "HAS_MATPLOTLIB" in dir(am)
    _ = am.HAS_MATPLOTLIB
    assert callable(am.plot_timeseries) or am.HAS_MATPLOTLIB is False
    assert callable(am.plot_grid) or am.HAS_MATPLOTLIB is False


@pytest.mark.unit
def test_import_smoke_subprocess_cold_start():
    """Catch large cold-start regressions in a clean subprocess."""
    # Budget is generous (CI VMs vary); the point is "seconds not tens of seconds".
    budget_s = 8.0
    code = textwrap.dedent(
        f"""
        import time
        t0 = time.perf_counter()
        import ambr
        elapsed = time.perf_counter() - t0
        # Touch a few core symbols so the package is actually usable.
        assert ambr.Model is not None
        assert ambr.Agent is not None
        assert ambr.RunResults is not None
        print(f"import_seconds={{elapsed:.4f}}")
        if elapsed > {budget_s}:
            raise SystemExit(
                f"cold import too slow: {{elapsed:.3f}}s > {budget_s}s"
            )
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"import smoke failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "import_seconds=" in proc.stdout
