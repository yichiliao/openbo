"""Simple Bayesian optimization loop using the scratch GP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from metabo.acquisition.ei import expected_improvement_minimization
from metabo.models.gp_scratch import GPScratch

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class BORunResult:
    """Container for BO observations and best-so-far trajectory."""

    x_obs: NDArray[np.float64]
    y_obs: NDArray[np.float64]
    best_y_history: NDArray[np.float64]


def run_bo_scratch(
    objective: Objective,
    bounds: list[tuple[float, float]],
    n_init: int = 5,
    n_iter: int = 25,
    n_candidates: int = 256,
    seed: int | None = 0,
) -> BORunResult:
    """Run a minimal BO loop for a maximization objective.

    EI is defined for minimization, so we minimize `-objective(x)` internally.
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lower = np.array([b[0] for b in bounds], dtype=np.float64)
    upper = np.array([b[1] for b in bounds], dtype=np.float64)

    x_obs = rng.uniform(lower, upper, size=(n_init, d)).astype(np.float64)
    y_obs = np.asarray(objective(x_obs), dtype=np.float64)
    if y_obs.shape != (n_init,):
        raise ValueError(f"Objective must return shape ({n_init},), got {y_obs.shape}.")

    best_y_history: list[float] = [float(np.max(y_obs))]

    gp = GPScratch(lengthscale=1.5, variance=1.0, noise=1e-6)
    for _ in range(n_iter):
        gp.fit(x_obs, -y_obs)
        x_cand = rng.uniform(lower, upper, size=(n_candidates, d)).astype(np.float64)
        mean_min, var_min = gp.posterior(x_cand)
        best_min = float(np.min(-y_obs))
        ei = expected_improvement_minimization(mean_min, var_min, best_min)
        next_idx = int(np.argmax(ei))
        x_next = x_cand[next_idx : next_idx + 1]
        y_next = np.asarray(objective(x_next), dtype=np.float64)

        x_obs = np.vstack([x_obs, x_next])
        y_obs = np.concatenate([y_obs, y_next])
        best_y_history.append(float(np.max(y_obs)))

    return BORunResult(
        x_obs=x_obs.astype(np.float64),
        y_obs=y_obs.astype(np.float64),
        best_y_history=np.asarray(best_y_history, dtype=np.float64),
    )
