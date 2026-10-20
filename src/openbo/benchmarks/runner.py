"""Simple benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openbo.optimizers.bo_botorch import BORunResult as BoTorchRunResult, run_bo_botorch
from openbo.optimizers.bo_scratch import BORunResult as ScratchRunResult, run_bo_scratch
from openbo.optimizers.bo_taf import run_bo_taf
from openbo.optimizers.random_search import RandomSearch
from openbo.test_functions.registry import get_function_spec


@dataclass
class BenchmarkResult:
    """Container for basic benchmark outputs."""

    best_value: float
    best_x: list[float]
    x_values: list[list[float]]
    y_values: list[float]
    metadata: dict[str, object] | None = None


def run_simple_benchmark(
    function_name: str = "branin",
    n_evals: int = 20,
    seed: int | None = 0,
    method: str = "random",
    n_init: int | None = None,
    n_iter: int | None = None,
    noise_std: float = 0.0,
    cap_at_optimum: bool = False,
    taf_run_dir: str | None = None,
    taf_rho: float = 1.0,
    taf_weight_mode: str = "taf_m",
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
        # Validate any value the caller provided up front. Previously this check
        # only ran when BOTH were provided, so a single non-positive value slipped
        # through (e.g. --n-iter 0 silently degraded BO to pure random search).
        if init_points is not None and init_points <= 0:
            raise ValueError("n_init must be positive when provided.")
        if bo_steps is not None and bo_steps <= 0:
            raise ValueError("n_iter must be positive when provided.")
        if init_points is None and bo_steps is None:
            # Deriving the split from n_evals needs room for >=1 init AND >=1 BO
            # step, which is impossible within a single evaluation. (An explicit
            # init/iter pair below is honored as-is and does not read n_evals.)
            if total_evals < 2:
                raise ValueError("BO methods need n_evals >= 2 (init point + BO step).")
            # Teaching-friendly default: 20% random init, rest BO.
            init_points = max(3, int(round(0.2 * total_evals)))
            init_points = min(init_points, total_evals - 1)
            bo_steps = total_evals - init_points
            return init_points, bo_steps
        # Exactly one value was provided: derive the other from n_evals so the two
        # sum to exactly n_evals. Clamping with max(..., 1) instead would silently
        # overrun the budget (e.g. n_evals=10 with n_init=20 ran 21 evaluations),
        # making a side-by-side comparison against another method unfair.
        if init_points is None:
            assert bo_steps is not None  # both-None handled above
            if bo_steps >= total_evals:
                raise ValueError(
                    "n_iter must be < n_evals (need room for at least one init point)."
                )
            return total_evals - bo_steps, bo_steps
        if bo_steps is None:
            if init_points >= total_evals:
                raise ValueError(
                    "n_init must be < n_evals (need room for at least one BO step)."
                )
            return init_points, total_evals - init_points
        return init_points, bo_steps

    spec = get_function_spec(
        function_name,
        noise_std=noise_std,
        noise_seed=seed,
        cap_at_optimum=cap_at_optimum,
    )
    objective = spec.objective
    bounds = spec.bounds

    metadata: dict[str, object] | None = None
    result: ScratchRunResult | BoTorchRunResult
    if method == "random":
        optimizer = RandomSearch(bounds=bounds, seed=seed)
        x, y = optimizer.run(objective=objective, n_evals=n_evals)
    elif method in {"bo_scratch", "bo_scratch_multistart", "bo_scratch_grid"}:
        resolved_n_init, resolved_n_iter = resolve_bo_budget(n_evals, n_init, n_iter)
        scratch_strategy = (
            "grid" if method == "bo_scratch_grid" else "multistart"
        )
        result = run_bo_scratch(
            objective=objective,
            bounds=bounds,
            n_init=resolved_n_init,
            n_iter=resolved_n_iter,
            search_strategy=scratch_strategy,
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
    elif method in {"bo_taf", "bo_taf_m", "bo_taf_r"}:
        if taf_run_dir is None:
            raise ValueError("taf_run_dir is required for TAF methods.")
        resolved_taf_mode = (
            "taf_m" if method == "bo_taf_m" else
            "taf_r" if method == "bo_taf_r" else
            taf_weight_mode
        )
        # Honor a user-provided n_init instead of silently ignoring it. Default
        # (n_init=None) keeps pure transfer: 0 random init and n_iter=n_evals.
        if n_init is not None and n_init < 0:
            raise ValueError("n_init must be non-negative.")
        # Validate n_iter here too (the TAF path does not go through
        # resolve_bo_budget), so an explicit n_iter=0 fails clearly instead of
        # reaching run_bo_taf and erroring with "finished without any observations".
        if n_iter is not None and n_iter <= 0:
            raise ValueError("n_iter must be positive when provided.")
        taf_n_init = 0 if n_init is None else n_init
        if n_iter is not None:
            resolved_n_iter = n_iter
        elif taf_n_init > 0:
            # Derive the BO steps from the total budget so init + iter == n_evals
            # exactly (no off-by-one overrun when n_init is large).
            if taf_n_init >= n_evals:
                raise ValueError(
                    "n_init must be < n_evals for TAF methods (need room for a BO step)."
                )
            resolved_n_iter = n_evals - taf_n_init
        else:
            resolved_n_iter = n_evals
        result = run_bo_taf(
            objective=objective,
            bounds=bounds,
            taf_run_dir=taf_run_dir,
            n_init=taf_n_init,
            n_iter=resolved_n_iter,
            rho=taf_rho,
            taf_weight_mode=resolved_taf_mode,
            seed=seed,
        )
        x, y = result.x_obs, result.y_obs
        metadata = {
            "taf_acquisition_trace": result.final_gp_state.get("taf_acquisition_trace", []),
            "taf_rho": result.final_gp_state.get("taf_rho"),
            "taf_weight_mode": result.final_gp_state.get("taf_weight_mode"),
            "n_sources": result.final_gp_state.get("n_sources"),
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    best_idx = int(np.argmax(y))
    return BenchmarkResult(
        best_value=float(y[best_idx]),
        best_x=[float(v) for v in x[best_idx]],
        x_values=[[float(v) for v in row] for row in x],
        y_values=[float(v) for v in y],
        metadata=metadata,
    )
