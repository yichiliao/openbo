"""Simple Bayesian optimization loop using the scratch GP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from metabo.acquisition.ei import expected_improvement_maximization
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
    n_candidates: int = 1024,
    n_starts: int = 8,
    local_steps: int = 20,
    search_strategy: str = "multistart",
    kernel_type: str = "matern52", # "rbf" or "matern52"
    optimize_hyperparameters: bool = True,
    seed: int | None = 0,
) -> BORunResult:
    """Run a minimal BO loop for a maximization objective.

    `search_strategy` controls how the next EI maximizer is found:
    - "multistart": random pool + local hill-climb around best starts
    - "grid": dense Cartesian grid over the box bounds
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
            # Multi-start EI search:
            # 1) global random pool, 2) pick best starts, 3) local random refinement.
            x_pool = rng.uniform(lower, upper, size=(n_candidates, d)).astype(np.float64)
            mean_pool, var_pool = gp.posterior(x_pool)
            ei_pool = expected_improvement_maximization(mean_pool, var_pool, best_y)

            n_starts_eff = int(max(1, min(n_starts, n_candidates)))
            start_indices = np.argsort(ei_pool)[-n_starts_eff:]
            x_starts = x_pool[start_indices]

            best_start_idx = int(np.argmax(ei_pool[start_indices]))
            best_x = x_starts[best_start_idx].copy()
            best_ei = float(ei_pool[start_indices][best_start_idx])
            step_scale = 0.1 * (upper - lower)

            for x_start in x_starts:
                x_curr = x_start.copy()
                mean_curr, var_curr = gp.posterior(x_curr[None, :])
                curr_ei = float(
                    expected_improvement_maximization(mean_curr, var_curr, best_y)[0]
                )
                for step in range(local_steps):
                    decay = 0.95**step
                    noise = rng.normal(0.0, step_scale * decay, size=d)
                    x_try = np.clip(x_curr + noise, lower, upper)
                    mean_try, var_try = gp.posterior(x_try[None, :])
                    ei_try = float(
                        expected_improvement_maximization(mean_try, var_try, best_y)[0]
                    )
                    if ei_try > curr_ei:
                        x_curr = x_try
                        curr_ei = ei_try
                    if curr_ei > best_ei:
                        best_ei = curr_ei
                        best_x = x_curr.copy()
            x_next = best_x[None, :]
        elif search_strategy == "grid":
            points_per_dim = int(np.ceil(n_candidates ** (1.0 / d)))
            axes = [
                np.linspace(lower[j], upper[j], points_per_dim, dtype=np.float64)
                for j in range(d)
            ]
            mesh = np.meshgrid(*axes, indexing="ij")
            x_grid = np.stack([m.reshape(-1) for m in mesh], axis=1).astype(np.float64)
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
