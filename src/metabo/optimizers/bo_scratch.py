"""Simple Bayesian optimization loop using the scratch GP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import qmc

from metabo.acquisition.ei import expected_improvement_maximization
from metabo.models.gp_scratch import GPScratch

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _ei_scalar(gp: GPScratch, x: NDArray[np.float64], best_y: float) -> float:
    mean, var = gp.posterior(x[None, :])
    return float(expected_improvement_maximization(mean, var, best_y)[0])


def _maximize_ei_lbfgsb(
    gp: GPScratch,
    x0: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    best_y: float,
) -> tuple[NDArray[np.float64], float]:
    """Bounded EI maximization from a single start."""

    def objective(x: NDArray[np.float64]) -> float:
        return -_ei_scalar(gp, x.astype(np.float64), best_y)

    bounds = [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]
    result = minimize(
        objective,
        x0.astype(np.float64),
        method="L-BFGS-B",
        bounds=bounds,
    )
    x_opt = np.asarray(result.x, dtype=np.float64)
    return x_opt, _ei_scalar(gp, x_opt, best_y)


def _sobol_in_bounds(
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample Sobol points in [lower, upper]."""
    d = lower.shape[0]
    seed = int(rng.integers(0, 2**31 - 1))
    engine = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = engine.random(n).astype(np.float64)
    return lower + (upper - lower) * u


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
    n_candidates: int = 512,
    n_starts: int = 8,
    search_strategy: str = "multistart",
    kernel_type: str = "matern52", # "rbf" or "matern52"
    optimize_hyperparameters: bool = True,
    seed: int | None = 0,
) -> BORunResult:
    """Run a minimal BO loop for a maximization objective.

    `search_strategy` controls how the next EI maximizer is found:
    - "multistart": Sobol pool + L-BFGS-B refinement from top starts
    - "grid": dense Sobol candidate scan
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

    gp = GPScratch(
        lengthscale=np.full(d, 1.5, dtype=np.float64),
        variance=1.0,
        noise=1e-6,
        kernel_type=kernel_type,
        optimize_hyperparameters=optimize_hyperparameters,
        optimize_noise=False,
        optimizer_seed=seed,
    )
    for _ in range(n_iter):
        gp.fit(x_obs, y_obs)
        best_y = float(np.max(y_obs))

        if search_strategy == "multistart":
            # Multi-start EI search with Sobol candidates + L-BFGS-B refinement.
            x_pool = _sobol_in_bounds(lower, upper, n_candidates, rng)
            mean_pool, var_pool = gp.posterior(x_pool)
            ei_pool = expected_improvement_maximization(mean_pool, var_pool, best_y)

            n_starts_eff = int(max(1, min(n_starts, n_candidates)))
            start_indices = np.argsort(ei_pool)[-n_starts_eff:]
            x_starts = x_pool[start_indices]

            best_start_idx = int(np.argmax(ei_pool[start_indices]))
            best_x = x_starts[best_start_idx].copy()
            best_ei = float(ei_pool[start_indices][best_start_idx])

            for x_start in x_starts:
                x_refined, ei_refined = _maximize_ei_lbfgsb(
                    gp=gp, x0=x_start, lower=lower, upper=upper, best_y=best_y
                )
                if ei_refined > best_ei:
                    best_ei = ei_refined
                    best_x = x_refined.copy()
            x_next = best_x[None, :]
        elif search_strategy == "grid":
            # Always use Sobol sequence for dense global EI scan.
            x_grid = _sobol_in_bounds(lower, upper, n_candidates, rng)
            mean_grid, var_grid = gp.posterior(x_grid)
            ei_grid = expected_improvement_maximization(mean_grid, var_grid, best_y)
            best_idx = int(np.argmax(ei_grid))
            x_next = x_grid[best_idx : best_idx + 1]
        else:
            raise ValueError(
                f"Unknown search_strategy '{search_strategy}'. "
                "Use 'multistart' or 'grid'."
            )
        y_next = np.asarray(objective(x_next), dtype=np.float64)

        x_obs = np.vstack([x_obs, x_next])
        y_obs = np.concatenate([y_obs, y_next])
        best_y_history.append(float(np.max(y_obs)))

    return BORunResult(
        x_obs=x_obs.astype(np.float64),
        y_obs=y_obs.astype(np.float64),
        best_y_history=np.asarray(best_y_history, dtype=np.float64),
    )
