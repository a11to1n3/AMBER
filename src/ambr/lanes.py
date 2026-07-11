"""Progressive speed lanes — make “how do I go faster?” answerable in one place.

Lanes (same mental model as the docs):

1. **OOP** — ``AgentList`` + agent methods (AgentPy-shaped). Always available.
2. **Vectorized** — ``agents.where`` / ``.at`` / ``.set`` / ``scatter_add``.
   No switch: writing this style *is* the fast path on CPU/Polars.
3. **Tensor** — ``agents.borrow`` / ``agents.commit`` for dense NumPy kernels.
4. **GPU** — ``ArrayKernelModel`` (single run) or ``GPUEnsembleRunner`` (many
   parameter sets). Uses CuPy when installed, else NumPy.

Call :func:`status` / :func:`print_status` to see what this machine can run, and
:func:`recommend` for a one-line suggestion by population size.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .gpu import GPU_AVAILABLE, device_name, get_array_module, to_host
from .results import RunResults


def status() -> Dict[str, Any]:
    """Return a machine/lane status dict (safe to print or log)."""
    gpu = bool(GPU_AVAILABLE)
    return {
        "gpu_available": gpu,
        "device": device_name(),
        "array_module": "cupy" if gpu else "numpy",
        "lanes": {
            "oop": "AgentList + agent methods — always on",
            "vectorized": "agents.where / .at / .set / scatter_add — write this style",
            "tensor": "agents.borrow / .commit — dense NumPy interaction kernels",
            "gpu_single": "ArrayKernelModel — one large run on CuPy/NumPy arrays",
            "gpu_ensemble": "GPUEnsembleRunner — B parameter sets in one batch",
        },
        "enable_gpu": (
            "CuPy is active"
            if gpu
            else "pip install cupy-cuda12x   # pick the wheel matching your CUDA"
        ),
    }


def print_status() -> None:
    """Human-readable lane status (for notebooks / CLIs)."""
    s = status()
    if s["gpu_available"]:
        print(f"GPU: yes ({s['device']})")
    else:
        print("GPU: no — using NumPy/CPU")
        print(f"  enable: {s['enable_gpu']}")
    print("Lanes:")
    for name, desc in s["lanes"].items():
        print(f"  {name:12}  {desc}")
    print(f"Tip: {recommend(10_000)}")


def recommend(n_agents: int, *, ensemble: bool = False) -> str:
    """One-line backend suggestion for a typical step-based model.

    Heuristic only (device cost and kernel shape matter). Safe defaults that
    match the public benchmarks' qualitative breakpoints.
    """
    n = int(n_agents)
    if ensemble:
        if GPU_AVAILABLE:
            return (
                "Use GPUEnsembleRunner + a batched setup/step/record model "
                "(best for calibration / many short runs)."
            )
        return (
            "No GPU: use Experiment / ParallelRunner on CPU, or install CuPy "
            "for GPUEnsembleRunner."
        )
    if n < 2_000:
        return (
            "OOP (AgentList + methods) or light vectorized is fine; "
            "GPU will usually be slower (transfer overhead)."
        )
    if n < 50_000:
        return (
            "Prefer vectorized: agents.where / .at / scatter_add "
            "(this is AMBER's default fast path)."
        )
    if n < 500_000:
        return (
            "Vectorized or tensor (borrow/commit) if you have dense interactions; "
            "try ArrayKernelModel if the whole step is array math."
        )
    if GPU_AVAILABLE:
        return (
            "Large N: use ArrayKernelModel (single run) so state stays on device; "
            "vectorized Polars is still fine if the step is simple filters/updates."
        )
    return (
        "Large N on CPU: stay vectorized; install CuPy and use ArrayKernelModel "
        "when the step is mostly array kernels."
    )


class ArrayKernelModel:
    """Easiest single-run CPU/GPU array model (no Polars step required).

    Subclass and implement:

    * ``init_state(xp, n, rng, p) -> dict`` of arrays
    * ``step_state(xp, state, rng, p) -> state``
    * optional ``metrics(xp, state) -> dict`` of Python scalars each step

    ``xp`` is CuPy when a GPU is available, else NumPy — same code path.

    Example::

        class Drift(am.ArrayKernelModel):
            def init_state(self, xp, n, rng, p):
                return {"x": rng.random(n, dtype=xp.float32)}
            def step_state(self, xp, state, rng, p):
                state["x"] = state["x"] + 0.01
                return state
            def metrics(self, xp, state):
                return {"mean_x": float(am.to_host(state["x"].mean()))}

        res = Drift({"n": 100_000, "steps": 50, "seed": 0}).run()
        print(res.model)  # per-step metrics if metrics() defined
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.p = dict(parameters or {})
        self.t = 0
        self._prefer_gpu = bool(self.p.get("prefer_gpu", True))

    def init_state(self, xp, n: int, rng, p: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def step_state(self, xp, state: Dict[str, Any], rng, p: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def metrics(self, xp, state: Dict[str, Any]) -> Dict[str, Any]:
        """Optional per-step scalars (host floats). Default: none."""
        return {}

    def run(
        self,
        steps: Optional[int] = None,
        *,
        prefer_gpu: Optional[bool] = None,
    ) -> RunResults:
        """Run ``steps`` device-resident updates; return :class:`RunResults`."""
        import numpy as np
        import polars as pl

        prefer = self._prefer_gpu if prefer_gpu is None else bool(prefer_gpu)
        xp = get_array_module(prefer)
        n = int(self.p.get("n", self.p.get("n_agents", 1000)))
        max_steps = int(steps if steps is not None else self.p.get("steps", 100))
        seed = self.p.get("seed", None)
        rng = xp.random.default_rng(seed)

        state = self.init_state(xp, n, rng, self.p)
        rows: List[Dict[str, Any]] = []
        for t in range(1, max_steps + 1):
            self.t = t
            state = self.step_state(xp, state, rng, self.p)
            m = self.metrics(xp, state) or {}
            if m:
                row = {"t": t}
                for k, v in m.items():
                    row[k] = float(to_host(v)) if hasattr(v, "shape") else float(v)
                rows.append(row)

        model_df = pl.DataFrame(rows) if rows else pl.DataFrame({"t": []})
        # Snapshot final 1-D numeric arrays as a thin agents table (optional UX).
        agent_cols: Dict[str, Any] = {"id": np.arange(n, dtype=np.int64)}
        for k, v in state.items():
            try:
                arr = to_host(v)
                if getattr(arr, "ndim", 0) == 1 and len(arr) == n:
                    agent_cols[k] = arr
            except Exception:
                continue
        agents_df = pl.DataFrame(agent_cols)
        return RunResults(
            info={
                "steps": max_steps,
                "n": n,
                "array_module": "cupy" if (prefer and GPU_AVAILABLE) else "numpy",
                "device": device_name(),
            },
            agents=agents_df,
            model=model_df,
        )
