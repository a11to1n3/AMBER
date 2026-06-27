"""Fast regression guard for the calibration benchmark (no optional deps).

Checks the loss landscape and that the dependency-free methods (grid, random)
recover parameters near ground truth. SMAC / skopt / cupy paths are exercised
by the full benchmark run, not here.

Run: python test_calibration_smoke.py   (or: pytest test_calibration_smoke.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import calib_models as cm          # noqa: E402
from methods import calibrate      # noqa: E402


def test_loss_minimised_at_ground_truth():
    for problem in cm.PROBLEMS.values():
        observed = cm.make_observed(problem, [0, 1, 2])
        problem.OBSERVED = observed

        def loss_at(theta, seed=7):
            m = problem({**problem.FIXED, **theta, "seed": seed, "show_progress": False})
            return m.run()["model"]["loss"].to_numpy()[-1]

        at_truth = np.mean([loss_at(problem.GROUND_TRUTH, s) for s in (7, 8, 9)])
        wrong = {k: (problem.BOUNDS[k][1] if v < sum(problem.BOUNDS[k]) / 2
                     else problem.BOUNDS[k][0])
                 for k, v in problem.GROUND_TRUTH.items()}
        at_wrong = np.mean([loss_at(wrong, s) for s in (7, 8, 9)])
        assert at_wrong > 5 * at_truth, f"weak loss signal for {problem.__name__}"


def test_grid_and_random_recover_sir():
    problem = cm.SIRCalib
    observed = cm.make_observed(problem, [0, 1, 2, 3])
    for method in ("grid", "random"):
        r = calibrate(method, problem, observed, budget=36, seed=0, eval_seed=100)
        assert r["theta_hat"]["beta"] == r["theta_hat"]["beta"]  # not NaN
        err = abs(r["theta_hat"]["beta"] - 0.35) + abs(r["theta_hat"]["gamma"] - 0.08)
        assert err < 0.20, f"{method} recovery too far: {r['theta_hat']}"


if __name__ == "__main__":
    test_loss_minimised_at_ground_truth()
    test_grid_and_random_recover_sir()
    print("calibration smoke tests passed")
