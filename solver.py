"""OPTIMISE@Wits baseline solver.

Zip this file on its own and upload it to confirm that your submission pipeline
works. It is deliberately simple and is not expected to win: on the hidden suite
it scores in the low 30s.

Rules recap:
  * the file must be named solver.py and sit at the top level of the zip;
  * it must define a top-level solve(problem);
  * only Python's standard library subset and NumPy may be imported.
"""
from __future__ import annotations

import numpy as np

from optimise import BudgetExceeded, GradientUnavailable


def _finite_difference_gradient(problem, x, f0):
    """Forward-difference gradient using at most n additional evaluations."""
    n = problem.n
    if problem.remaining < n:
        raise BudgetExceeded("not enough budget for a finite-difference gradient")
    h = 1.0e-6 * (1.0 + np.linalg.norm(x)) / max(1.0, np.sqrt(n))
    g = np.zeros(n)
    for j in range(n):
        xp = x.copy()
        xp[j] += h
        g[j] = (problem.value(xp) - f0) / h
    return g


def solve(problem):
    """Return a valid candidate minimiser for one hidden problem."""
    rng = np.random.default_rng(problem.seed)
    x = np.array(problem.x0, dtype=float, copy=True)
    best_x = x.copy()
    best_f = problem.value(x)
    stagnation = 0

    while problem.remaining > 3:
        f0 = problem.value(x)
        try:
            if problem.has_gradient and problem.remaining >= 2:
                g = problem.gradient(x)
            else:
                g = _finite_difference_gradient(problem, x, f0)
        except (BudgetExceeded, GradientUnavailable):
            break

        ng = np.linalg.norm(g)
        if not np.isfinite(ng) or ng < 1.0e-9:
            break
        d = -g / ng

        alpha = min(1.0, 1.0 + np.linalg.norm(x))
        accepted = False
        while problem.remaining > 0 and alpha >= 1.0e-10:
            trial = x + alpha * d
            ft = problem.value(trial)
            if ft <= f0 - 1.0e-4 * alpha * ng:
                x = trial
                accepted = True
                if ft < best_f:
                    best_f = ft
                    best_x = trial.copy()
                    stagnation = 0
                else:
                    stagnation += 1
                break
            alpha *= 0.5

        if not accepted:
            stagnation += 1
            if problem.remaining > 0 and stagnation >= 2:
                scale = 0.05 * (1.0 + np.linalg.norm(best_x))
                trial = best_x + scale * rng.normal(size=problem.n)
                ft = problem.value(trial)
                if ft < best_f:
                    x, best_x, best_f = trial, trial.copy(), ft
                else:
                    x = best_x.copy()
                stagnation = 0
            else:
                break

    return {
        "x": best_x,
        "status": "ok",
        "method": "baseline descent with backtracking",
    }