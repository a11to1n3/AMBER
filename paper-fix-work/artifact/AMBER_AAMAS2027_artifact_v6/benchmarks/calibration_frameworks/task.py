"""Shared calibration task for the cross-framework benchmark.

Every framework calibrates the *same* well-mixed SIR: recover (beta, gamma) from
an observed infected-fraction curve. The dynamics are fixed so the task is
identical everywhere:

    status in {S=0, I=1, R=2}; k = i0_frac*N initially infected.
    each step (synchronous, using the step-start I fraction):
        S -> I with prob  beta * I_fraction
        I -> R with prob  gamma
    record the I fraction after each step  ->  a length-`steps` curve.

A framework supplies ``sir_curve(beta, gamma, n, steps, seed) -> curve`` using
its own engine; the common optimiser evaluates a shared, seeded set of candidate
(beta, gamma) pairs (a fair random search) and the wall-clock difference reflects
each framework's evaluation throughput.
"""

import time

import numpy as np

GROUND_TRUTH = {"beta": 0.35, "gamma": 0.08}
BOUNDS = {"beta": (0.05, 0.60), "gamma": (0.02, 0.25)}
N = 3000
STEPS = 50
I0_FRAC = 0.02


def reference_sir_curve(beta, gamma, n=N, steps=STEPS, seed=0, i0_frac=I0_FRAC):
    """Canonical well-mixed SIR (NumPy) -- generates the observed data."""
    rng = np.random.default_rng(seed)
    status = np.zeros(n, dtype=np.int8)
    status[: max(1, int(i0_frac * n))] = 1
    curve = np.empty(steps, dtype=np.float64)
    for t in range(steps):
        i_frac = float((status == 1).mean())
        r1 = rng.random(n)
        r2 = rng.random(n)
        new = status.copy()
        new[(status == 0) & (r1 < beta * i_frac)] = 1
        new[(status == 1) & (r2 < gamma)] = 2
        status = new
        curve[t] = float((status == 1).mean())
    return curve


def make_observed(seeds):
    """Mean infected-fraction curve at ground truth (the data to calibrate to)."""
    curves = [reference_sir_curve(seed=s, **GROUND_TRUTH) for s in seeds]
    return np.mean(np.stack(curves), axis=0)


def make_candidates(k, seed):
    """Shared seeded (beta, gamma) candidates -- the common optimiser."""
    rng = np.random.default_rng(seed)
    lo = np.array([BOUNDS["beta"][0], BOUNDS["gamma"][0]])
    hi = np.array([BOUNDS["beta"][1], BOUNDS["gamma"][1]])
    return [tuple(lo + rng.random(2) * (hi - lo)) for _ in range(k)]


def loss(curve, observed):
    return float(np.sum((np.asarray(curve, dtype=np.float64) - observed) ** 2))


def recovery_error(theta, gt=GROUND_TRUTH, bounds=BOUNDS):
    """Normalised RMS recovery error (fraction of search range)."""
    return float(np.sqrt(np.mean(
        [((theta[k] - gt[k]) / (bounds[k][1] - bounds[k][0])) ** 2 for k in gt])))


def run_common_optimizer(sir_curve, observed, candidates, n, steps, eval_seed):
    """Evaluate every shared candidate with one framework; return best + timing."""
    best_loss, best_theta, curve_best = np.inf, None, []
    t0 = time.perf_counter()
    for (beta, gamma) in candidates:
        l = loss(sir_curve(beta, gamma, n, steps, eval_seed), observed)
        if l < best_loss:
            best_loss, best_theta = l, {"beta": float(beta), "gamma": float(gamma)}
        curve_best.append(best_loss)
    wall = time.perf_counter() - t0
    return {
        "theta_hat": best_theta, "best_loss": best_loss,
        "n_evals": len(candidates), "wall_s": wall,
        "evals_per_s": len(candidates) / max(wall, 1e-9), "curve": curve_best,
    }


def validation_loss(sir_curve, observed, theta, val_seeds, n=N, steps=STEPS):
    """Out-of-sample loss of recovered parameters on held-out seeds."""
    return float(np.mean([
        loss(sir_curve(theta["beta"], theta["gamma"], n, steps, s), observed)
        for s in val_seeds]))
