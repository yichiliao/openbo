"""Registry utilities for test functions and metadata."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Callable, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from openbo.test_functions.synthetic import (
    KNOWN_OPTIMA,
    ackley,
    branin,
    hartmann6,
    mccormick,
    rastrigin,
    rosenbrock,
    sphere,
)
from openbo.test_functions.tasks import (
    TASK_DIMS,
    TaskVariantSpec,
    make_variant_objective,
    noise_rng,
)

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _estimate_reachable_optimum(
    clean_objective: Objective, dim: int, upper_bound: float
) -> float:
    """Estimate the maximum of a noise-free variant objective over ``[0, 1]^d``.

    The affine input transform clips into a sub-box of ``[0, 1]^d``, so the base
    optimum location can be unreachable and the true reachable maximum can be
    strictly below ``upper_bound = output_scale * base_optimum``. For ``d <= 2``
    we bracket the maximum on a dense grid and polish the best points with
    L-BFGS-B; for higher dimensions we conservatively return the analytic upper
    bound (a valid over-estimate that never claims false convergence). The result
    is clamped to ``upper_bound`` since the global maximum can never be exceeded.
    """
    if dim > 2:
        warnings.warn(
            "Reported variant optimum for dim>2 is the analytic upper bound "
            "(output_scale*base_optimum); it may be unreachable after input "
            "clipping, so log-regret can plateau above 0."
        )
        return upper_bound

    n = 129
    axes = [np.linspace(0.0, 1.0, n, dtype=np.float64)] * dim
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([m.ravel() for m in mesh], axis=1).astype(np.float64)
    vals = np.asarray(clean_objective(grid), dtype=np.float64)
    best = float(np.max(vals))

    def neg(x: NDArray[np.float64]) -> float:
        return -float(np.asarray(clean_objective(x[None, :]), dtype=np.float64)[0])

    top_starts = grid[np.argsort(vals)[-5:]]
    for x0 in top_starts:
        res = minimize(neg, x0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * dim)
        if np.isfinite(res.fun):
            best = max(best, -float(res.fun))
    best = float(min(best, upper_bound))

    # When the base optimum is still reachable, the search lands only a hair below
    # the analytic bound (empirically <=1.2e-7 across the registry, versus ~0.36
    # when it is genuinely unreachable). Snap those cases back to the exact
    # analytic value: an under-estimated optimum would let best_y exceed it and
    # make regret negative, which is exactly what this estimate exists to prevent.
    # Erring toward the upper bound is the safe direction.
    tol = 1e-6 * max(1.0, abs(upper_bound))
    if upper_bound - best <= tol:
        return float(upper_bound)
    return best


@dataclass(frozen=True)
class FunctionSpec:
    """Description of a test function used by the benchmark code."""

    name: str
    objective: Objective
    bounds: list[tuple[float, float]]
    dim: int
    optimum: float | None = None


REGISTRY: dict[str, FunctionSpec] = {
    "branin": FunctionSpec(
        name="branin",
        objective=branin,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        dim=2,
        optimum=KNOWN_OPTIMA["branin"],
    ),
    "sphere": FunctionSpec(
        name="sphere",
        objective=sphere,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        dim=2,
        optimum=KNOWN_OPTIMA["sphere"],
    ),
    "ackley": FunctionSpec(
        name="ackley",
        objective=ackley,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        dim=2,
        optimum=KNOWN_OPTIMA["ackley"],
    ),
    "rastrigin": FunctionSpec(
        name="rastrigin",
        objective=rastrigin,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        dim=2,
        optimum=KNOWN_OPTIMA["rastrigin"],
    ),
    "rosenbrock": FunctionSpec(
        name="rosenbrock",
        objective=rosenbrock,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        dim=2,
        optimum=KNOWN_OPTIMA["rosenbrock"],
    ),
    "hartmann6": FunctionSpec(
        name="hartmann6",
        objective=hartmann6,
        bounds=[(0.0, 1.0)] * 6,
        dim=6,
        optimum=KNOWN_OPTIMA["hartmann6"],
    ),
    "mccormick": FunctionSpec(
        name="mccormick",
        objective=mccormick,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        dim=2,
        optimum=KNOWN_OPTIMA["mccormick"],
    ),
}


def get_test_function(name: str) -> Objective:
    """Return the objective callable for a test function."""
    try:
        return REGISTRY[name].objective
    except KeyError as exc:
        raise KeyError(f"Unknown test function: {name}") from exc


def get_function_spec(
    name: str,
    *,
    noise_std: float = 0.0,
    noise_seed: int | None = None,
    cap_at_optimum: bool = False,
) -> FunctionSpec:
    """Return full metadata for a test function.

    By default objectives are deterministic. Set ``noise_std > 0`` to add
    Gaussian output noise; set ``cap_at_optimum=True`` to clip noisy outputs so
    they never exceed the known optimum for the task.
    """
    try:
        spec = REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown test function: {name}") from exc
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    if noise_std == 0.0:
        return spec

    rng = noise_rng(noise_seed)
    noisy_function = cast(Callable[..., NDArray[np.float64]], spec.objective)

    def noisy_objective(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return noisy_function(
            x,
            noise_std=noise_std,
            rng=rng,
            cap_at_optimum=cap_at_optimum,
        )

    return FunctionSpec(
        name=spec.name,
        objective=noisy_objective,
        bounds=spec.bounds,
        dim=spec.dim,
        optimum=spec.optimum,
    )


def make_variant_function_spec(
    base_name: str,
    variant: TaskVariantSpec,
    variant_name: str | None = None,
) -> FunctionSpec:
    """Create one task variant from a base function."""
    base = get_function_spec(base_name)
    variant_optimum: float | None = None
    if base.optimum is not None and variant.output_scale >= 0.0:
        upper_bound = variant.output_scale * base.optimum
        # The base optimum can become unreachable once the affine transform clips
        # into a sub-box of [0,1]^d, so estimate the true reachable maximum on the
        # noise-free variant rather than assuming output_scale*base_optimum.
        clean_variant = make_variant_objective(
            base.objective,
            replace(variant, noise_std=0.0, cap_at_optimum=False),
            dim=base.dim,
            base_optimum=base.optimum,
        )
        variant_optimum = _estimate_reachable_optimum(
            clean_variant, base.dim, upper_bound
        )
    # Cap noisy outputs at the SAME reachable optimum we report, so capped
    # observations never exceed FunctionSpec.optimum (which would make regret
    # negative and log-regret NaN) for shifted/scaled variants, not just identity.
    objective = make_variant_objective(
        base.objective,
        variant,
        dim=base.dim,
        base_optimum=base.optimum,
        cap_value=variant_optimum,
    )
    return FunctionSpec(
        name=variant_name or f"{base_name}_variant",
        objective=objective,
        bounds=base.bounds,
        dim=base.dim,
        optimum=variant_optimum,
    )


def make_branin_family(
    n_tasks: int,
    seed: int = 0,
    max_input_shift: float = 0.05,
    max_input_scale_delta: float = 0.1,
    max_output_scale_delta: float = 0.1,
) -> list[FunctionSpec]:
    """Create a list of Branin variants for transfer/meta-learning."""
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive.")

    dim = TASK_DIMS["branin"]
    rng = np.random.default_rng(seed)
    family: list[FunctionSpec] = []
    for idx in range(n_tasks):
        shift = tuple(rng.uniform(-max_input_shift, max_input_shift, size=dim).tolist())
        scale = tuple(
            rng.uniform(
                1.0 - max_input_scale_delta,
                1.0 + max_input_scale_delta,
                size=dim,
            ).tolist()
        )
        output_scale = float(
            rng.uniform(1.0 - max_output_scale_delta, 1.0 + max_output_scale_delta)
        )
        variant = TaskVariantSpec(
            input_shift=shift,
            input_scale=scale,
            output_scale=output_scale,
            noise_std=0.0,
            cap_at_optimum=False,
            seed=seed + idx,
        )
        family.append(
            make_variant_function_spec(
                base_name="branin",
                variant=variant,
                variant_name=f"branin_variant_{idx:03d}",
            )
        )
    return family
