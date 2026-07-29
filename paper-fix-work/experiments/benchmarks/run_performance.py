#!/usr/bin/env python3
"""E4 — Unified performance campaign with cold / warm / steady separation.

Semantics-matched track uses counter-RNG backends from this package.
Native-idiom track uses AMBER public API + FLAME GPU 2 where available.
No sample trimming; every run retained with status code.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ROOT / "semantic"))

from backends import (  # noqa: E402
    schelling_private_gpu_style,
    schelling_reference,
    sir_private_gpu_style,
    sir_reference,
    walk_private_gpu_style,
    walk_reference,
    wealth_private_gpu_style,
    wealth_reference,
    wealth_vectorized_numpy,
)


def _try_cupy():
    try:
        import cupy as cp
        return cp
    except Exception:
        return None


def _try_ambr():
    try:
        import ambr as am
        return am
    except Exception:
        return None


def _try_flame():
    try:
        # Reuse benchmark harness path setup (LD preload for cuda130 NVRTC).
        import sys
        from pathlib import Path as _P
        bench = _P(__file__).resolve().parents[2] / "benchmarks"
        if str(bench) not in sys.path:
            sys.path.insert(0, str(bench))
        from run_all_frameworks import _configure_flamegpu_runtime
        _configure_flamegpu_runtime()
        import pyflamegpu
        return pyflamegpu
    except Exception:
        return None


def record_base(**kw):
    return {
        "timestamp": time.time(),
        "host": platform.node(),
        "git_note": "AMBER_aamas_exp campaign",
        **kw,
    }


def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    # sync GPU if cupy
    cp = _try_cupy()
    if cp is not None:
        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass
    return time.perf_counter() - t0


def run_block(name, fn, runs, cold=True):
    """Return list of sample dicts: first is cold if cold=True."""
    samples = []
    for i in range(runs):
        try:
            # steady: after warm, run again
            elapsed = time_call(fn)
            samples.append({
                "run": i,
                "scope": "cold" if (cold and i == 0) else ("warm" if i == 1 else "steady"),
                "total_s": elapsed,
                "status": "success",
            })
        except Exception as exc:
            samples.append({
                "run": i,
                "scope": "cold" if (cold and i == 0) else "warm",
                "total_s": None,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return samples


def semantics_matched_suite(quick=False):
    if quick:
        configs = [
            ("wealth", 10_000, 20),
            ("wealth", 100_000, 20),
            ("random_walk", 10_000, 20),
            ("sir", 2_000, 20),
            ("schelling", 16, 20),  # side
        ]
        runs = 3
    else:
        # Keep pure-Python reference tractable; native AMBER track covers larger N.
        configs = [
            ("wealth", 10_000, 50),
            ("wealth", 100_000, 50),
            ("random_walk", 5_000, 50),
            ("random_walk", 20_000, 50),
            ("sir", 5_000, 50),
            ("sir", 10_000, 50),
            ("schelling", 32, 50),
            ("schelling", 48, 30),
        ]
        runs = 5

    cp = _try_cupy()
    rows = []
    for workload, n, steps in configs:
        backends = {
            "reference": None,
            "vectorized_numpy": None,
        }
        if workload == "wealth":
            backends["reference"] = lambda n=n, steps=steps: wealth_reference(n, steps, 0)
            backends["vectorized_numpy"] = lambda n=n, steps=steps: wealth_vectorized_numpy(n, steps, 0)
            if cp is not None:
                backends["private_gpu_style"] = lambda n=n, steps=steps: wealth_private_gpu_style(n, steps, 0)
        elif workload == "random_walk":
            backends["reference"] = lambda n=n, steps=steps: walk_reference(n, steps, 0)
            backends["vectorized_numpy"] = lambda n=n, steps=steps: walk_reference(n, steps, 0)
            if cp is not None:
                backends["private_gpu_style"] = lambda n=n, steps=steps: walk_private_gpu_style(n, steps, 0)
        elif workload == "sir":
            backends["reference"] = lambda n=n, steps=steps: sir_reference(n, steps, 0)
            backends["vectorized_numpy"] = lambda n=n, steps=steps: sir_reference(n, steps, 0)
            if cp is not None:
                backends["private_gpu_style"] = lambda n=n, steps=steps: sir_private_gpu_style(n, steps, 0)
        elif workload == "schelling":
            side = n
            backends["reference"] = lambda side=side, steps=steps: schelling_reference(side, steps, 0)
            backends["vectorized_numpy"] = lambda side=side, steps=steps: schelling_reference(side, steps, 0)
            if cp is not None:
                backends["private_gpu_style"] = lambda side=side, steps=steps: schelling_private_gpu_style(side, steps, 0)

        for bname, fn in backends.items():
            if fn is None:
                continue
            print(f"[matched] {workload} N={n} steps={steps} backend={bname}", flush=True)
            samples = run_block(bname, fn, runs=runs)
            ok = [s["total_s"] for s in samples if s["status"] == "success" and s["scope"] != "cold"]
            rows.append(record_base(
                track="semantics_matched",
                workload=workload,
                population=n,
                steps=steps,
                framework=bname,
                samples=samples,
                warm_median_s=statistics.median(ok) if ok else None,
                status="success" if ok else "error",
            ))
    return rows


def amber_native_suite(quick=False):
    am = _try_ambr()
    if am is None:
        return [record_base(track="native_idiom", framework="AMBER", status="skipped", error="ambr missing")]

    # Import benchmark models
    sys.path.insert(0, str(REPO / "benchmarks" / "models"))
    try:
        from amber_models import AMBERVectorizedWealthTransfer, AMBERVectorizedRandomWalk
    except Exception as exc:
        return [record_base(track="native_idiom", framework="AMBER", status="error", error=str(exc))]

    if quick:
        sizes = [10_000, 100_000]
        steps = 20
        runs = 3
    else:
        sizes = [10_000, 100_000, 1_000_000]
        steps = 50
        runs = 5  # native AMBER path; 1M is OK via fused kernels

    rows = []
    cp = _try_cupy()
    model_specs = [
        ("wealth", AMBERVectorizedWealthTransfer, {"initial_wealth": 1}),
        ("random_walk", AMBERVectorizedRandomWalk, {"world_size": 100, "speed": 1.0}),
    ]
    for workload, model_cls, extra in model_specs:
        for n in sizes:
            for device, mode, label in [
                ("cpu", "vectorized", "AMBER_vectorized_cpu"),
                ("gpu", "vectorized", "AMBER_gpu"),
            ]:
                if device == "gpu" and cp is None:
                    continue

                def run_once(
                    n=n, device=device, mode=mode, model_cls=model_cls, extra=extra, workload=workload
                ):
                    params = {
                        "n": n,
                        "steps": steps,
                        "seed": 0,
                        "show_progress": False,
                        **extra,
                    }
                    model = model_cls(params)
                    if device == "gpu":
                        if hasattr(model, "approve_fast_path"):
                            model.approve_fast_path(f"exp_perf_campaign_{workload}")
                        model.gpu().run(contract="off")
                    else:
                        model.cpu(mode=mode).run(contract="off")

                print(f"[native] {workload} N={n} {label}", flush=True)
                samples = run_block(label, run_once, runs=runs)
                ok = [
                    s["total_s"]
                    for s in samples
                    if s["status"] == "success" and s["scope"] != "cold"
                ]
                rows.append(record_base(
                    track="native_idiom",
                    workload=workload,
                    population=n,
                    steps=steps,
                    framework=label,
                    samples=samples,
                    warm_median_s=statistics.median(ok) if ok else None,
                    status="success" if ok else samples[-1].get("status", "error"),
                ))
    return rows


def _ensure_cuda_path():
    """FLAME RTC needs CUDA include dir (InvalidFilePath without CUDA_PATH)."""
    import os
    if os.environ.get("CUDA_PATH") and Path(os.environ["CUDA_PATH"], "include", "cuda_runtime.h").exists():
        return os.environ["CUDA_PATH"]
    candidates = [
        Path("/usr/local/cuda-12.9"),
        Path("/usr/local/cuda-13.0"),
        Path("/usr/local/cuda"),
        Path.home() / "cuda-12.0",
        Path("/usr/local/cuda-12.6"),
        Path("/usr/local/cuda-12.3"),
    ]
    for c in candidates:
        if (c / "include" / "cuda_runtime.h").exists():
            os.environ["CUDA_PATH"] = str(c)
            os.environ.setdefault("CUDA_HOME", str(c))
            return str(c)
    return os.environ.get("CUDA_PATH")


def flame_native_suite(quick=False):
    """Native-idiom FLAME GPU 2 timings (not counter-RNG semantics-matched)."""
    if _try_flame() is None:
        return [record_base(
            track="native_idiom", framework="FLAME_GPU_2",
            status="skipped", error="pyflamegpu missing",
        )]
    cuda = _ensure_cuda_path()
    if not cuda:
        return [record_base(
            track="native_idiom", framework="FLAME_GPU_2",
            status="error", error="CUDA_PATH not found for RTC",
        )]
    print(f"[flame] CUDA_PATH={cuda}", flush=True)

    sys.path.insert(0, str(REPO / "benchmarks" / "models"))
    try:
        from flamegpu_models import WalkModel, WealthModel
    except Exception as exc:
        # wealth may be named differently
        try:
            import flamegpu_models as fm
            WalkModel = getattr(fm, "WalkModel", None)
            WealthModel = getattr(fm, "WealthModel", None) or getattr(fm, "WealthTransferModel", None)
            if WalkModel is None or WealthModel is None:
                # discover class names
                names = [n for n in dir(fm) if "Wealth" in n or "Walk" in n or "Model" in n]
                return [record_base(
                    track="native_idiom", framework="FLAME_GPU_2",
                    status="error", error=f"import classes failed: {exc}; available={names}",
                )]
        except Exception as exc2:
            return [record_base(
                track="native_idiom", framework="FLAME_GPU_2",
                status="error", error=str(exc2),
            )]

    if quick:
        sizes = [10_000, 100_000]
        steps = 20
        runs = 3
    else:
        sizes = [10_000, 100_000, 1_000_000]
        steps = 50
        runs = 5

    rows = []
    workloads = []
    if WealthModel is not None:
        workloads.append(("wealth", WealthModel, {"initial_wealth": 1}))
    if WalkModel is not None:
        workloads.append(("random_walk", WalkModel, {"world_size": 100, "speed": 1.0}))

    for workload, cls, cfg in workloads:
        for n in sizes:
            def run_once(cls=cls, n=n, cfg=cfg, steps=steps):
                cls(n, steps, cfg).run()

            print(f"[flame] {workload} N={n}", flush=True)
            samples = run_block("FLAME_GPU_2", run_once, runs=runs)
            ok = [s["total_s"] for s in samples if s["status"] == "success" and s["scope"] != "cold"]
            err = next((s.get("error") for s in samples if s["status"] != "success"), "")
            rows.append(record_base(
                track="native_idiom",
                workload=workload,
                population=n,
                steps=steps,
                framework="FLAME_GPU_2",
                samples=samples,
                warm_median_s=statistics.median(ok) if ok else None,
                status="success" if ok else "error",
                error=err,
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "raw" / "performance")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-matched", action="store_true")
    ap.add_argument("--only-flame", action="store_true")
    ap.add_argument("--only-activation-support", action="store_true",
                    help="unused placeholder for orchestration")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = []
    if args.only_flame:
        rows += flame_native_suite(quick=args.quick)
    else:
        if not args.skip_matched:
            rows += semantics_matched_suite(quick=args.quick)
        rows += amber_native_suite(quick=args.quick)
        rows += flame_native_suite(quick=args.quick)

    # Summarize speedups gpu vs reference where possible
    by = {}
    for r in rows:
        if r.get("status") != "success":
            continue
        key = (r.get("track"), r.get("workload"), r.get("population"))
        by.setdefault(key, {})[r["framework"]] = r.get("warm_median_s")

    speedups = []
    for key, d in by.items():
        if "reference" in d and "private_gpu_style" in d and d["reference"] and d["private_gpu_style"]:
            speedups.append({
                "track": key[0],
                "workload": key[1],
                "population": key[2],
                "reference_s": d["reference"],
                "private_gpu_style_s": d["private_gpu_style"],
                "speedup": d["reference"] / d["private_gpu_style"] if d["private_gpu_style"] else None,
            })

    # AMBER GPU vs FLAME where both present
    amber_vs_flame = []
    for key, d in by.items():
        if key[0] != "native_idiom":
            continue
        if "AMBER_gpu" in d and "FLAME_GPU_2" in d and d["AMBER_gpu"] and d["FLAME_GPU_2"]:
            amber_vs_flame.append({
                "workload": key[1],
                "population": key[2],
                "AMBER_gpu_s": d["AMBER_gpu"],
                "FLAME_GPU_2_s": d["FLAME_GPU_2"],
                "flame_over_amber": d["FLAME_GPU_2"] / d["AMBER_gpu"] if d["AMBER_gpu"] else None,
                "amber_over_flame": d["AMBER_gpu"] / d["FLAME_GPU_2"] if d["FLAME_GPU_2"] else None,
            })

    report = {
        "tag": args.tag,
        "host": platform.node(),
        "platform": platform.platform(),
        "elapsed_s": time.time() - t0,
        "cupy": _try_cupy() is not None,
        "ambr": _try_ambr() is not None,
        "flame": _try_flame() is not None,
        "rows": rows,
        "speedups_gpu_vs_reference": speedups,
        "amber_vs_flame": amber_vs_flame,
        "notes": (
            "Cold = first timed run after process start for that config; "
            "warm/steady = subsequent runs. No samples trimmed. "
            "FLAME track is native-idiom (framework RNG), not counter-RNG matched."
        ),
    }
    path = args.out / f"performance_{args.tag}.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"speedups": speedups, "n_rows": len(rows)}, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
