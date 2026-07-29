"""Uniform adapters for every calibration method AMBER supports.

Each ``calibrate(method, ...)`` minimises the *same* problem loss and returns a
uniform record: recovered parameters, best loss, number of evaluations, wall
time, and a best-loss-so-far curve (for sample-efficiency plots). The CPU
methods (grid / random / Bayesian / SMAC) share a single ``loss(theta)``; the
GPU batched ensemble (SIR only) evaluates a whole batch of candidates per GPU
pass via ``smac_batch_calibrate``.
"""

import logging
import time

import numpy as np


def make_loss(problem, observed, eval_seed):
    """Return ``(loss_fn, names)`` where ``loss_fn(theta_vec)`` runs the model
    once at ``eval_seed`` and returns the final cumulative SSE vs ``observed``."""
    problem.OBSERVED = np.asarray(observed, dtype=np.float64)
    names = list(problem.BOUNDS)
    fixed = problem.FIXED

    def loss(theta_vec):
        theta = {n: float(v) for n, v in zip(names, theta_vec)}
        m = problem({**fixed, **theta, "seed": int(eval_seed), "show_progress": False})
        return float(m.run()["model"]["loss"].to_numpy()[-1])

    return loss, names


def _best_so_far(losses):
    best, out = np.inf, []
    for v in losses:
        best = min(best, v)
        out.append(best)
    return out


# --- CPU search loops (each faithfully *is* the named method) --------------

def _grid(loss, bounds, budget):
    d = len(bounds)
    g = max(2, int(round(budget ** (1.0 / d))))
    axes = [np.linspace(lo, hi, g) for lo, hi in bounds]
    pts = np.stack([gg.ravel() for gg in np.meshgrid(*axes, indexing="ij")], axis=1)
    return [(p, loss(p)) for p in pts]


def _random(loss, bounds, budget, seed):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    out = []
    for _ in range(budget):
        p = lo + rng.random(len(bounds)) * (hi - lo)
        out.append((p, loss(p)))
    return out


def _bayesian(loss, bounds, budget, seed):
    from skopt import gp_minimize
    from skopt.space import Real
    space = [Real(lo, hi) for lo, hi in bounds]
    evals = []

    def f(x):
        v = loss(np.asarray(x, dtype=float))
        evals.append((np.asarray(x, dtype=float), v))
        return v

    gp_minimize(f, space, n_calls=budget, n_initial_points=min(10, budget),
                random_state=seed)
    return evals


def _smac(loss, names, bounds, budget, seed):
    from ConfigSpace import ConfigurationSpace, Float
    from smac import HyperparameterOptimizationFacade, Scenario
    from smac.runhistory import TrialValue
    logging.getLogger("smac").setLevel(logging.ERROR)

    cs = ConfigurationSpace(seed=seed)
    cs.add([Float(n, bounds[i]) for i, n in enumerate(names)])
    scenario = Scenario(cs, n_trials=budget, deterministic=True, seed=seed)
    smac = HyperparameterOptimizationFacade(
        scenario, lambda config, seed=0: 0.0, overwrite=True)

    evals = []
    for _ in range(budget):
        info = smac.ask()
        x = np.array([info.config[n] for n in names], dtype=float)
        v = loss(x)
        smac.tell(info, TrialValue(cost=v))
        evals.append((x, v))
    return evals


# --- GPU batched ensemble (SIR only -- it has a batched model) -------------

def _gpu_sir(problem, observed, budget, seed):
    from ambr.gpu import get_array_module
    from ambr.gpu_ensemble import BatchedWellMixedSIR, smac_batch_calibrate

    obs = np.asarray(observed, dtype=np.float64)
    fixed = problem.FIXED

    def loss_fn(traj):
        xp = get_array_module()
        I = traj["I_frac"]                                   # (B, steps)
        o = xp.asarray(obs, dtype=I.dtype).reshape(1, -1)
        return ((I - o) ** 2).sum(axis=1)

    batch_size = 32
    rounds = max(1, budget // batch_size)
    t0 = time.perf_counter()
    best, history = smac_batch_calibrate(
        BatchedWellMixedSIR(), problem.BOUNDS, loss_fn,
        fixed["n"], fixed["steps"],
        rounds=rounds, batch_size=batch_size,
        fixed_params={"i0_frac": fixed["i0_frac"]}, seed=seed)
    wall = time.perf_counter() - t0

    curve = _best_so_far([h for h in history for _ in range(batch_size)])
    return {
        "theta_hat": best or {},
        "best_loss": float(min(history)) if history else float("inf"),
        "n_evals": rounds * batch_size,
        "wall_time": wall,
        "curve": curve,
    }


# --- dispatcher ------------------------------------------------------------

def calibrate(method, problem, observed, budget, seed, eval_seed):
    """Run one calibration method, return a uniform result record."""
    if method == "gpu_ensemble":
        return _gpu_sir(problem, observed, budget, seed)

    loss, names = make_loss(problem, observed, eval_seed)
    bounds = [problem.BOUNDS[n] for n in names]
    t0 = time.perf_counter()
    if method == "grid":
        evals = _grid(loss, bounds, budget)
    elif method == "random":
        evals = _random(loss, bounds, budget, seed)
    elif method == "bayesian":
        evals = _bayesian(loss, bounds, budget, seed)
    elif method == "smac":
        evals = _smac(loss, names, bounds, budget, seed)
    else:
        raise ValueError(f"unknown method {method!r}")
    wall = time.perf_counter() - t0

    losses = [e[1] for e in evals]
    best_i = int(np.argmin(losses))
    return {
        "theta_hat": {n: float(v) for n, v in zip(names, evals[best_i][0])},
        "best_loss": float(losses[best_i]),
        "n_evals": len(evals),
        "wall_time": wall,
        "curve": _best_so_far(losses),
    }
