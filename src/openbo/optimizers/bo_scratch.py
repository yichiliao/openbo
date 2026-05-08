"""Simple Bayesian optimization loop using the scratch GP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import qmc

from openbo.acquisition.ei import expected_improvement_maximization
from openbo.models.gp_scratch import GPScratch

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
    final_gp_state: dict[str, object]


@dataclass
class ScratchConfig:
    """Configuration for scratch BO sequential optimizer."""

    bounds: list[tuple[float, float]]
    n_init: int = 5
    n_candidates: int = 512
    n_starts: int = 8
    search_strategy: str = "multistart"
    kernel_type: str = "matern52"
    optimize_hyperparameters: bool = True
    seed: int | None = 0


class ScratchSequentialOptimizer:
    """Ask/tell-style scratch BO optimizer state machine."""

    def __init__(self, config: ScratchConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.d = len(config.bounds)
        self.lower = np.array([b[0] for b in config.bounds], dtype=np.float64)
        self.upper = np.array([b[1] for b in config.bounds], dtype=np.float64)

        self.x_obs = np.empty((0, self.d), dtype=np.float64)
        self.y_obs = np.empty((0,), dtype=np.float64)
        self.best_y_history: list[float] = []

        self.gp = GPScratch(
            lengthscale=np.full(self.d, 1.5, dtype=np.float64),
            variance=1.0,
            noise=1e-6,
            kernel_type=config.kernel_type,
            optimize_hyperparameters=config.optimize_hyperparameters,
            optimize_noise=False,
            optimizer_seed=config.seed,
        )

    def bootstrap(self, objective: Objective) -> None:
        """Collect random initial observations."""
        if self.config.n_init <= 0:
            return
        x_init = self.rng.uniform(
            self.lower, self.upper, size=(self.config.n_init, self.d)
        ).astype(np.float64)
        y_init = np.asarray(objective(x_init), dtype=np.float64)
        if y_init.shape != (self.config.n_init,):
            raise ValueError(
                f"Objective must return shape ({self.config.n_init},), got {y_init.shape}."
            )
        self.observe(x_init, y_init)

    def suggest(self) -> NDArray[np.float64]:
        """Suggest next point batch of shape (1, d)."""
        if self.x_obs.shape[0] == 0:
            raise ValueError("Cannot suggest without observations. Call bootstrap() first.")

        self.gp.fit(self.x_obs, self.y_obs)
        best_y = float(np.max(self.y_obs))

        if self.config.search_strategy == "multistart":
            x_pool = _sobol_in_bounds(
                self.lower, self.upper, self.config.n_candidates, self.rng
            )
            mean_pool, var_pool = self.gp.posterior(x_pool)
            ei_pool = expected_improvement_maximization(mean_pool, var_pool, best_y)

            n_starts_eff = int(
                max(1, min(self.config.n_starts, self.config.n_candidates))
            )
            start_indices = np.argsort(ei_pool)[-n_starts_eff:]
            x_starts = x_pool[start_indices]

            best_start_idx = int(np.argmax(ei_pool[start_indices]))
            best_x = x_starts[best_start_idx].copy()
            best_ei = float(ei_pool[start_indices][best_start_idx])

            for x_start in x_starts:
                x_refined, ei_refined = _maximize_ei_lbfgsb(
                    gp=self.gp,
                    x0=x_start,
                    lower=self.lower,
                    upper=self.upper,
                    best_y=best_y,
                )
                if ei_refined > best_ei:
                    best_ei = ei_refined
                    best_x = x_refined.copy()
            return best_x[None, :]

        if self.config.search_strategy == "grid":
            x_grid = _sobol_in_bounds(
                self.lower, self.upper, self.config.n_candidates, self.rng
            )
            mean_grid, var_grid = self.gp.posterior(x_grid)
            ei_grid = expected_improvement_maximization(mean_grid, var_grid, best_y)
            best_idx = int(np.argmax(ei_grid))
            return x_grid[best_idx : best_idx + 1]

        raise ValueError(
            f"Unknown search_strategy '{self.config.search_strategy}'. "
            "Use 'multistart' or 'grid'."
        )

    def observe(
        self,
        x_new: NDArray[np.float64],
        y_new: NDArray[np.float64],
    ) -> None:
        """Tell optimizer new observations."""
        x_new = np.asarray(x_new, dtype=np.float64)
        y_new = np.asarray(y_new, dtype=np.float64)
        if x_new.ndim != 2 or x_new.shape[1] != self.d:
            raise ValueError(f"x_new must have shape (n, {self.d}), got {x_new.shape}.")
        if y_new.ndim != 1 or y_new.shape[0] != x_new.shape[0]:
            raise ValueError("y_new must have shape (n,) and match x_new rows.")

        self.x_obs = np.vstack([self.x_obs, x_new])
        self.y_obs = np.concatenate([self.y_obs, y_new])
        self.best_y_history.append(float(np.max(self.y_obs)))

    def result(self) -> BORunResult:
        """Build run result from current state."""
        self.gp.fit(self.x_obs, self.y_obs)
        lengthscale_arr = np.asarray(self.gp.lengthscale, dtype=np.float64).reshape(-1)
        final_gp_state: dict[str, object] = {
            "kernel_type": self.gp.kernel_type,
            "lengthscale": [float(v) for v in lengthscale_arr],
            "variance": float(self.gp.variance),
            "noise": float(self.gp.noise),
            "standardize_targets": bool(self.gp.standardize_targets),
            "optimize_hyperparameters": bool(self.gp.optimize_hyperparameters),
            "optimize_noise": bool(self.gp.optimize_noise),
            "y_mean": float(self.gp.y_mean),
            "y_std": float(self.gp.y_std),
            "n_observations": int(self.x_obs.shape[0]),
            "dim": int(self.d),
        }
        return BORunResult(
            x_obs=self.x_obs.astype(np.float64),
            y_obs=self.y_obs.astype(np.float64),
            best_y_history=np.asarray(self.best_y_history, dtype=np.float64),
            final_gp_state=final_gp_state,
        )


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
    optimizer = ScratchSequentialOptimizer(
        ScratchConfig(
            bounds=bounds,
            n_init=n_init,
            n_candidates=n_candidates,
            n_starts=n_starts,
            search_strategy=search_strategy,
            kernel_type=kernel_type,
            optimize_hyperparameters=optimize_hyperparameters,
            seed=seed,
        )
    )
    optimizer.bootstrap(objective)
    for _ in range(n_iter):
        x_next = optimizer.suggest()
        y_next = np.asarray(objective(x_next), dtype=np.float64)
        optimizer.observe(x_next, y_next)
    return optimizer.result()
