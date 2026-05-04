"""Simple benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metabo.optimizers.bo_botorch import run_bo_botorch
from metabo.optimizers.bo_scratch import run_bo_scratch
from metabo.optimizers.random_search import RandomSearch
from metabo.test_functions.registry import get_function_spec


@dataclass
class BenchmarkResult:
    """Container for basic benchmark outputs."""

    best_value: float
    best_x: list[float]
    y_values: list[float]


def run_simple_benchmark(
    function_name: str = "branin",
    n_evals: int = 20,
    seed: int | None = 0,
    method: str = "random",
    n_init: int | None = None,
    n_iter: int | None = None,
) -> BenchmarkResult:
    """Run a minimal benchmark with random or BO methods."""
    if n_evals <= 0:
        raise ValueError("n_evals must be positive.")

    def resolve_bo_budget(
        total_evals: int,
        init_points: int | None,
        bo_steps: int | None,
    ) -> tuple[int, int]:
        """Resolve BO budget from optional n_init/n_iter values."""
        if init_points is None and bo_steps is None:
            # Teaching-friendly default: 20% random init, rest BO.
            init_points = max(3, int(round(0.2 * total_evals)))
            init_points = min(init_points, max(total_evals - 1, 1))
            bo_steps = max(total_evals - init_points, 1)
            return init_points, bo_steps
        if init_points is None:
            if bo_steps is None:
                raise ValueError("Internal error: bo_steps should be resolved.")
            init_points = max(total_evals - bo_steps, 1)
            return init_points, bo_steps
        if bo_steps is None:
            bo_steps = max(total_evals - init_points, 1)
            return init_points, bo_steps
        if init_points <= 0 or bo_steps <= 0:
            raise ValueError("n_init and n_iter must be positive when provided.")
        return init_points, bo_steps

    spec = get_function_spec(function_name)
    objective = spec.objective
    bounds = spec.bounds

    if method == "random":
        optimizer = RandomSearch(bounds=bounds, seed=seed)
        x, y = optimizer.run(objective=objective, n_evals=n_evals)
    elif method == "bo_scratch":
        resolved_n_init, resolved_n_iter = resolve_bo_budget(n_evals, n_init, n_iter)
        result = run_bo_scratch(
            objective=objective,
            bounds=bounds,
            n_init=resolved_n_init,
            n_iter=resolved_n_iter,
            seed=seed,
        )
        x, y = result.x_obs, result.y_obs
    elif method == "bo_botorch":
        resolved_n_init, resolved_n_iter = resolve_bo_budget(n_evals, n_init, n_iter)
        result = run_bo_botorch(
            objective=objective,
            bounds=bounds,
            n_init=resolved_n_init,
            n_iter=resolved_n_iter,
            seed=seed,
        )
        x, y = result.x_obs, result.y_obs
    else:
        raise ValueError(f"Unknown method: {method}")

    best_idx = int(np.argmax(y))
    return BenchmarkResult(
        best_value=float(y[best_idx]),
        best_x=[float(v) for v in x[best_idx]],
        y_values=[float(v) for v in y],
    )
