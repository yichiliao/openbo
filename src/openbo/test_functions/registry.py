"""Registry utilities for test functions and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from openbo.test_functions.synthetic import (
    KNOWN_OPTIMA,
    ackley,
    branin,
    hartmann6,
    rastrigin,
    rosenbrock,
    sphere,
)
from openbo.test_functions.tasks import TASK_DIMS, TaskVariantSpec, make_variant_objective

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


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

    rng = np.random.default_rng(noise_seed)

    def noisy_objective(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return spec.objective(
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
    objective = make_variant_objective(
        base.objective,
        variant,
        dim=base.dim,
        base_optimum=base.optimum,
    )
    variant_optimum: float | None = None
    if base.optimum is not None and variant.output_scale >= 0.0:
        variant_optimum = variant.output_scale * base.optimum
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
