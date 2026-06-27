"""GPU-batched ensemble runner for AMBER (CuPy, NumPy fallback).

Runs ``B`` independent simulations -- different parameter sets and/or seeds --
as a single ``(B, N)`` tensor batch. This is the natural GPU upgrade for
calibration / sensitivity analysis / parameter sweeps (``Experiment``,
``SMACOptimizer``, ``ParallelRunner``), where you evaluate thousands of small
replicate runs: batching collapses ``B*steps`` kernel launches into ``steps``,
saturates the device, and runs ~100-600x faster than looping (the speedup is
largest when each individual run is small -- exactly the calibration regime).

A *batched model* implements three methods over ``(B, N)`` arrays from the
active array module (CuPy on GPU, else NumPy):

    setup(B, N, params, rng)  -> state   (dict of (B, N) arrays)
    step(state, params, rng)  -> state
    record(state)             -> dict of per-run metrics, each shape (B,)

where ``params`` maps ``name -> (B, 1)`` array (one value per run, broadcast
over the N agents). The runner stacks each recorded metric into a ``(B, steps)``
trajectory -- so a calibration loss against a target curve is one more batched
tensor op over all B candidates at once.
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as _np

from .gpu import get_array_module, to_host


class GPUEnsembleRunner:
    """Run a batched model's ``B`` simulations together as a ``(B, N)`` batch."""

    def __init__(self, model):
        self.model = model

    def run(self, n_agents: int, steps: int, params: Dict, seed: int = 0) -> Dict:
        """Run ``B`` simulations (``B = len`` of each param array) for ``steps``.

        Returns a dict ``metric -> (B, steps)`` array of per-run trajectories.
        """
        xp = get_array_module()
        names = list(params)
        B = len(params[names[0]])
        pb = {k: xp.asarray(v, dtype=xp.float32).reshape(B, 1) for k, v in params.items()}
        rng = xp.random.default_rng(seed)

        state = self.model.setup(B, n_agents, pb, rng)
        traj = None
        for _ in range(steps):
            state = self.model.step(state, pb, rng)
            rec = self.model.record(state)
            if traj is None:
                traj = {k: [] for k in rec}
            for k, v in rec.items():
                traj[k].append(v)
        return {k: xp.stack(v, axis=1) for k, v in traj.items()}


class BatchedWellMixedSIR:
    """Reference batched model: well-mixed (mean-field) SIR.

    Per agent, per run: a susceptible becomes infected with probability
    ``beta * I_fraction`` (force of infection), an infected recovers with
    probability ``gamma``. Per-run parameters ``beta``, ``gamma``,
    ``i0_frac`` are ``(B, 1)`` arrays. This is the canonical curve-fitting
    target in epidemiology and batches perfectly (O(B*N) per step).
    """

    S, I, R = 0, 1, 2

    def setup(self, B, N, params, rng):
        xp = get_array_module()
        idx = xp.arange(N, dtype=xp.float32).reshape(1, N)
        seeded = idx < (params["i0_frac"] * N)            # (B, N) bool
        return {"status": xp.where(seeded, self.I, self.S).astype(xp.int8), "N": N}

    def step(self, state, params, rng):
        xp = get_array_module()
        status = state["status"]
        i_frac = (status == self.I).mean(axis=1, keepdims=True)   # (B, 1)
        p_inf = params["beta"] * i_frac                           # (B, 1) force of infection
        d_inf = rng.random(status.shape, dtype=xp.float32)
        d_rec = rng.random(status.shape, dtype=xp.float32)
        new_inf = (status == self.S) & (d_inf < p_inf)
        new_rec = (status == self.I) & (d_rec < params["gamma"])
        status = xp.where(new_inf, xp.int8(self.I), status)
        status = xp.where(new_rec, xp.int8(self.R), status)
        return {"status": status, "N": state["N"]}

    def record(self, state):
        return {"I_frac": (state["status"] == self.I).mean(axis=1)}   # (B,)


def smac_batch_calibrate(
    model,
    param_bounds: Dict[str, Tuple[float, float]],
    loss_fn: Callable[[Dict], "array"],
    n_agents: int,
    steps: int,
    *,
    rounds: int = 8,
    batch_size: int = 64,
    fixed_params: Optional[Dict[str, float]] = None,
    seed: int = 0,
    quiet: bool = True,
) -> Tuple[Optional[Dict[str, float]], list]:
    """Batched Bayesian calibration: SMAC + the GPU ensemble.

    Each round, SMAC's ask/tell interface proposes ``batch_size`` distinct
    configurations; :class:`GPUEnsembleRunner` evaluates *all of them in one
    GPU pass* as a ``(batch_size, N)`` tensor; the per-run losses are told back
    so SMAC's surrogate improves for the next round. This combines SMAC's
    sample-efficient search with GPU batch throughput -- ``rounds * batch_size``
    evaluations in only ``rounds`` sequential model-update steps and ``rounds``
    GPU launches.

    Args:
        model: a batched model (see module docstring).
        param_bounds: ``name -> (low, high)`` for the calibrated parameters.
        loss_fn: ``trajectories -> (B,)`` per-run losses (e.g. SSE vs a target).
        fixed_params: parameters held constant across all runs.
        rounds, batch_size: search budget (total evals = rounds * batch_size).

    Returns ``(best_config, history)`` where history is the per-round best loss.
    """
    try:
        from ConfigSpace import ConfigurationSpace, Float
        from smac import HyperparameterOptimizationFacade, Scenario
        from smac.runhistory import TrialValue
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "smac_batch_calibrate requires `smac` and `ConfigSpace`. "
            "Install with: pip install smac"
        ) from e

    if quiet:
        import logging
        logging.getLogger("smac").setLevel(logging.ERROR)

    runner = GPUEnsembleRunner(model)
    names = list(param_bounds)
    fixed = fixed_params or {}

    cs = ConfigurationSpace(seed=seed)
    cs.add([Float(name, bounds) for name, bounds in param_bounds.items()])
    scenario = Scenario(cs, n_trials=rounds * batch_size, deterministic=True, seed=seed)
    smac = HyperparameterOptimizationFacade(
        scenario, lambda config, seed=0: 0.0, overwrite=True
    )

    history = []
    for _ in range(rounds):
        infos = [smac.ask() for _ in range(batch_size)]
        params = {nm: _np.array([info.config[nm] for info in infos], dtype=_np.float64)
                  for nm in names}
        for k, v in fixed.items():
            params[k] = _np.full(batch_size, v, dtype=_np.float64)

        trajectories = runner.run(n_agents, steps, params)
        losses = to_host(loss_fn(trajectories))
        for info, loss in zip(infos, losses):
            smac.tell(info, TrialValue(cost=float(loss)))
        history.append(float(min(losses)))

    incumbent = smac.intensifier.get_incumbent()
    best = dict(incumbent) if incumbent is not None else None
    return best, history
