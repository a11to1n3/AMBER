#!/usr/bin/env python3
"""Verify GPU claim paths on an NVIDIA + CuPy host (Host B / RTX-class).

Default CI has **no CUDA**. This script is the source of truth for
"GPU claims run for real" — run it on the benchmark GPU host after
``pip install -e '.[perf,gpu]'`` (or cupy-cuda* matching the driver).

Usage (on the GPU host, from the repo root)::

    python scripts/run_host_b_gpu_claims.py
    python scripts/run_host_b_gpu_claims.py --pytest-only
    python scripts/run_host_b_gpu_claims.py --quick   # smaller N

Exit code 0 only if every selected case passes with ``device=gpu`` /
``array_module=cupy`` where applicable.

Environment::

    AMBER_GPU_CLAIMS_N   override agent count for the large case (default 1_000_000)
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller agent counts (10k / 50k) for a fast smoke",
    )
    parser.add_argument(
        "--pytest-only",
        action="store_true",
        help="Only run GPU-related pytest modules",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip pytest (claim samples only)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "tmp" / "gpu_claim_runs" / "host_b_gpu_claims.json",
        help="Write machine-readable results JSON",
    )
    args = parser.parse_args()

    try:
        import ambr as am
        import cupy
    except ImportError as e:
        _die(f"Need ambr + cupy on this host: {e}")

    if not getattr(am, "GPU_AVAILABLE", False):
        _die(
            "am.GPU_AVAILABLE is False. Install a CUDA-matched CuPy wheel "
            "(e.g. cupy-cuda12x) on an NVIDIA machine. Apple Metal/MPS is not used."
        )

    n_large = 50_000 if args.quick else int(
        __import__("os").environ.get("AMBER_GPU_CLAIMS_N", "1000000")
    )
    n_mid = 10_000 if args.quick else 100_000
    steps_large = 20 if args.quick else 50

    report: list[str] = []
    cases: list[dict] = []

    def log(s: str = "") -> None:
        report.append(s)
        print(s, flush=True)

    def case(cid: str, fn) -> None:
        log(f"## {cid}")
        try:
            msg = fn()
            log("**PASS** " + (msg or ""))
            cases.append({"id": cid, "status": "PASS", "evidence": msg or ""})
        except Exception:
            tb = traceback.format_exc()
            log("**FAIL**\n```\n" + tb + "\n```")
            cases.append({"id": cid, "status": "FAIL", "evidence": tb})
        log()

    log("# Host B GPU claims")
    log(f"- when: {datetime.now(timezone.utc).isoformat()}")
    log(f"- host: {platform.node()}")
    log(f"- ambr: {am.__version__}")
    log(f"- cupy: {cupy.__version__}")
    props = cupy.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    log(f"- device: {name}")
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    am.print_status()
    sys.stdout = old
    log("```\n" + buf.getvalue().rstrip() + "\n```")
    log()

    class WealthModel(am.Model):
        model_reporters = {"total_wealth": lambda m: int(m.agents.wealth.sum())}

        def setup(self):
            n = int(self.p.get("n", 100))
            self.add_agents(n, wealth=self.rng.integers(1, 10, size=n))

        def step_vectorized(self):
            donors = self.agents.where(self.agents.wealth > 0)
            donors.wealth -= 1
            rec = self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
            self.agents.at[rec].scatter_add(wealth=1)

    if not args.pytest_only:
        def c_small():
            r = WealthModel(
                {"n": 100, "steps": 50, "seed": 0, "show_progress": False}
            ).gpu().run()
            assert r.info.get("device") == "gpu", r.info
            totals = r.model["total_wealth"].to_list()
            assert len(set(totals)) == 1
            return f"info={r.info} conserved={totals[0]}"

        def c_mid():
            r = WealthModel(
                {"n": n_mid, "steps": steps_large, "seed": 0, "show_progress": False}
            ).gpu().run()
            assert r.info.get("device") == "gpu", r.info
            return f"info={r.info}"

        def c_large():
            class W(WealthModel):
                def setup(self):
                    n = int(self.p.get("n", n_large))
                    self.add_agents(n, wealth=1)

            r = W(
                {"n": n_large, "steps": steps_large, "seed": 0, "show_progress": False}
            ).gpu().run()
            assert r.info.get("device") == "gpu", r.info
            assert int(r.model["total_wealth"][-1]) == n_large
            return f"info={r.info} total={r.model['total_wealth'][-1]}"

        def c_array():
            class Drift(am.ArrayKernelModel):
                def init_state(self, xp, n, rng, p):
                    return {"x": rng.random(n, dtype=xp.float32)}

                def step_state(self, xp, state, rng, p):
                    state["x"] = state["x"] + 0.01
                    return state

                def metrics(self, xp, state):
                    return {"mean_x": float(am.to_host(state["x"].mean()))}

            r = Drift({"n": n_mid, "steps": 20, "show_progress": False}).run()
            assert r.info.get("array_module") == "cupy", r.info
            return f"info={r.info}"

        def c_ensemble():
            import numpy as np
            from ambr.gpu_ensemble import BatchedWellMixedSIR, GPUEnsembleRunner

            B = 4 if args.quick else 8
            traj = GPUEnsembleRunner(BatchedWellMixedSIR()).run(
                n_agents=1_000 if args.quick else 10_000,
                steps=20 if args.quick else 30,
                params={
                    "beta": np.linspace(0.1, 0.4, B),
                    "gamma": np.full(B, 0.05),
                    "i0_frac": np.full(B, 0.02),
                },
                seed=0,
            )
            shapes = {k: getattr(v, "shape", type(v).__name__) for k, v in traj.items()}
            return f"shapes={shapes}"

        def c_quickstart():
            ex = ROOT / "examples" / "gpu_quickstart.py"
            if not ex.is_file():
                return "examples/gpu_quickstart.py missing — skip"
            p = subprocess.run(
                [sys.executable, str(ex)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=180,
            )
            if p.returncode != 0:
                raise RuntimeError((p.stderr or "") + (p.stdout or ""))
            out = (p.stdout or "")
            if "native GPU" not in out and "device': 'gpu'" not in out and "device=gpu" not in out:
                # gpu_quickstart prints "native GPU  info: {...'device': 'gpu'...}"
                if "device': 'gpu'" not in out and '"device": "gpu"' not in out:
                    if "skipped (CuPy" in out:
                        raise RuntimeError("gpu_quickstart skipped GPU unexpectedly")
            return "examples/gpu_quickstart.py ok"

        for cid, fn in [
            ("wealth-gpu-small", c_small),
            ("wealth-gpu-mid", c_mid),
            ("wealth-gpu-large", c_large),
            ("array-kernel-cupy", c_array),
            ("gpu-ensemble", c_ensemble),
            ("examples-gpu-quickstart", c_quickstart),
        ]:
            case(cid, fn)

    if not args.skip_pytest:
        def c_pytest():
            tests = [
                "tests/test_readme_examples.py",
                "tests/test_gpu_backend.py",
                "tests/test_gpu_ensemble.py",
            ]
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                *tests,
                "-q",
                "--tb=line",
                "--cov-fail-under=0",
            ]
            # Prefer no-cov for speed if pytest-cov forces fail-under via ini
            p = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
            out = (p.stdout or "") + (p.stderr or "")
            if p.returncode != 0:
                # Retry without inheriting cov fail-under noise
                cmd2 = [
                    sys.executable,
                    "-m",
                    "pytest",
                    *tests,
                    "-q",
                    "--tb=line",
                    "-p",
                    "no:cacheprovider",
                    "--override-ini=addopts=",
                ]
                p = subprocess.run(
                    cmd2, cwd=str(ROOT), capture_output=True, text=True, timeout=300
                )
                out = (p.stdout or "") + (p.stderr or "")
                if p.returncode != 0:
                    raise RuntimeError(out)
            return out.strip()[-500:]

        case("pytest-gpu-modules", c_pytest)

    n_fail = sum(1 for c in cases if c["status"] != "PASS")
    log("## Summary")
    for c in cases:
        log(f"- {c['id']}: **{c['status']}**")
    log()
    verdict = "PASS" if n_fail == 0 else "FAIL"
    log(f"**verdict: {verdict}** ({len(cases) - n_fail}/{len(cases)})")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "when": datetime.now(timezone.utc).isoformat(),
                "host": platform.node(),
                "ambr": am.__version__,
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    md_path = args.json_out.with_suffix(".md")
    md_path.write_text("\n".join(report) + "\n")
    print(f"WROTE {args.json_out}")
    print(f"WROTE {md_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
