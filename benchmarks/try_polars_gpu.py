#!/usr/bin/env python3
"""Head-to-head: Polars Lazy GPU vs AMBER model.gpu() vs fused CuPy vs FLAME."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

_site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def _rapids_library_dirs() -> list[str]:
    """Return bundled RAPIDS/CUDA library directories available on this host."""
    roots = (
        _site,
        Path("/usr/local/cuda"),
        Path("/usr/local/lib") / f"python{sys.version_info.major}.{sys.version_info.minor}" / "dist-packages" / "nvidia",
    )
    candidates = []
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("*/lib64", "*/lib", "*/*/lib64", "*/*/lib"):
            candidates.extend(root.glob(pattern))
        if root.name in {"cuda", "lib64", "lib"}:
            candidates.append(root)
    return list(dict.fromkeys(str(path) for path in candidates if path.is_dir()))


def _reexec_with_rapids_library_path() -> None:
    """Restart the CLI after setting library paths early enough for ``dlopen``.

    The dynamic loader reads ``LD_LIBRARY_PATH`` when Python starts, so changing
    it after startup does not make ``libcudf.so`` discoverable.  Re-execing the
    CLI gives the loader the path at process startup while preserving the user's
    existing entries.
    """
    if os.environ.get("AMBR_RAPIDS_PATH_READY"):
        return
    paths = _rapids_library_dirs()
    existing = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if path]
    merged = list(dict.fromkeys(paths + existing))
    if not merged or all(path in existing for path in paths):
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(merged)
    env["AMBR_RAPIDS_PATH_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]

STEPS = 50
SEED = 42
NS = [10_000, 100_000, 1_000_000]


def _trim_mean(samples: list[float]) -> float:
    s = sorted(samples)
    if len(s) >= 3:
        s = s[:-1]
    return float(sum(s) / len(s))


def time_fn(fn, runs=3, warmup=1) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return _trim_mean(samples)


def wealth_polars_lazy(n: int, steps: int, engine) -> None:
    """Wealth as Lazy filter + debit/credit joins using seeded random draws."""
    df = pl.DataFrame(
        {
            "id": np.arange(n, dtype=np.int64),
            "wealth": np.ones(n, dtype=np.int64),
        }
    )
    rng = np.random.default_rng(SEED)
    donor_ids = np.arange(n, dtype=np.int64)
    for _ in range(steps):
        # cudf-polars does not support the random expression needed here, so
        # materialise one seeded recipient per donor and join it into the lazy
        # plan. This preserves the same active-donor sampling semantics as the
        # CuPy and AMBER implementations.
        recipient_map = pl.DataFrame(
            {
                "donor": donor_ids,
                "recipient": rng.integers(0, n, size=n, dtype=np.int64),
            }
        )
        transfers = (
            df.lazy()
            .filter(pl.col("wealth") > 0)
            .select(pl.col("id").alias("donor"))
            .join(recipient_map.lazy(), on="donor", how="left")
        )
        credits = transfers.group_by("recipient").agg(pl.len().alias("credit"))
        debits = transfers.group_by("donor").agg(pl.len().alias("debit"))
        df = (
            df.lazy()
            .join(debits, left_on="id", right_on="donor", how="left")
            .join(credits, left_on="id", right_on="recipient", how="left")
            .with_columns(
                (
                    pl.col("wealth")
                    - pl.col("debit").fill_null(0)
                    + pl.col("credit").fill_null(0)
                ).alias("wealth")
            )
            .select("id", "wealth")
            .collect(engine=engine)
        )
    total = int(df["wealth"].sum())
    assert total == n, (total, n)


def wealth_amber_cupy_gpu(n: int, steps: int) -> None:
    from models.amber_models import AMBERVectorizedWealthTransfer
    from ambr.gpu import synchronize

    class RawWealthTransfer(AMBERVectorizedWealthTransfer):
        """Run the same kernel without per-step benchmark reporters."""

        def update(self):
            pass

    m = RawWealthTransfer(
        {"n": n, "steps": steps, "show_progress": False, "seed": SEED, "initial_wealth": 1}
    )
    results = m.gpu().run()
    synchronize()
    total = int(results["agents"]["wealth"].sum())
    assert total == n, (total, n)


def wealth_fused_cupy(n: int, steps: int) -> None:
    import cupy as cp

    rng = cp.random.default_rng(SEED)
    wealth = cp.ones(n, dtype=cp.int64)
    for _ in range(steps):
        idx = cp.nonzero(wealth > 0)[0]
        if idx.size == 0:
            continue
        wealth[idx] -= 1
        rec = rng.integers(0, n, size=int(idx.size))
        wealth += cp.bincount(rec, minlength=n).astype(cp.int64)
    cp.cuda.Stream.null.synchronize()
    assert int(wealth.sum()) == n


def wealth_flame(n: int, steps: int) -> None:
    from models import flamegpu_models as fg

    fg.WealthModel(n, steps, {"initial_wealth": 1}).run()


def bulk_groupby_polars(n: int, engine) -> None:
    rng = np.random.default_rng(SEED)
    (
        pl.DataFrame(
            {
                "cell": rng.integers(0, max(n // 100, 1), size=n, dtype=np.int64),
                "val": rng.random(n),
            }
        )
        .lazy()
        .group_by("cell")
        .agg(pl.col("val").sum(), pl.len())
        .collect(engine=engine)
    )


def main():
    _reexec_with_rapids_library_path()
    print("=== Polars GPU tryout (RTX 5090) ===")
    import cudf_polars  # noqa: F401

    print("cudf_polars imported OK")
    print(f"polars {pl.__version__}")
    print(f"steps={STEPS} runs=3 (trim slowest) warmup=1")
    print()

    pl.LazyFrame({"x": [1, 2, 3]}).select(pl.col("x").sum()).collect(
        engine=pl.GPUEngine(raise_on_fail=True)
    )
    print("GPUEngine(raise_on_fail=True) smoke OK")

    # FLAME RTC warm (compile once at small n)
    try:
        wealth_flame(1000, 2)
        print("FLAME RTC warm OK")
    except Exception as e:
        print(f"FLAME warm FAIL {e}")
    print()

    rows = []
    failures = []
    for n in NS:
        print(f"[n={n:,}]")
        for label, fn in [
            ("polars groupby GPU", lambda: bulk_groupby_polars(n, pl.GPUEngine(raise_on_fail=True))),
            ("polars groupby CPU", lambda: bulk_groupby_polars(n, "cpu")),
            ("wealth polars GPU", lambda: wealth_polars_lazy(n, STEPS, pl.GPUEngine(raise_on_fail=True))),
            ("wealth polars CPU", lambda: wealth_polars_lazy(n, STEPS, "cpu")),
            ("wealth AMBER gpu()", lambda: wealth_amber_cupy_gpu(n, STEPS)),
            ("wealth fused CuPy", lambda: wealth_fused_cupy(n, STEPS)),
            ("wealth FLAME GPU 2", lambda: wealth_flame(n, STEPS)),
        ]:
            try:
                t = time_fn(fn)
                print(f"  {label:22s} {t*1000:10.1f} ms")
                rows.append((label, n, t))
            except Exception as e:
                msg = str(e).split("\n")[0][:120]
                print(f"  {label:22s} FAIL {type(e).__name__}: {msg}")
                failures.append((label, n, type(e).__name__, msg))
        print()

    print("=== summary (seconds) ===")
    print(f"{'backend':24s} {'n':>10s} {'sec':>10s}")
    for label, n, t in rows:
        print(f"{label:24s} {n:10d} {t:10.4f}")
    if failures:
        print("\n=== failures ===")
        for label, n, error_type, message in failures:
            print(f"{label} @ n={n}: {error_type}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
