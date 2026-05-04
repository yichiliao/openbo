"""Registry utilities for test functions and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from metabo.test_functions.synthetic import branin, sphere

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class FunctionSpec:
    """Description of a test function used by the benchmark code."""

    name: str
    objective: Objective
    bounds: list[tuple[float, float]]
    optimum: float | None = None


REGISTRY: dict[str, FunctionSpec] = {
    "branin": FunctionSpec(
        name="branin",
        objective=branin,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        # Standard Branin minimum is ~0.397887, objective is sign-flipped.
        optimum=-0.39788735772973816,
    ),
    "sphere": FunctionSpec(
        name="sphere",
        objective=sphere,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        optimum=0.0,
    ),
}


def get_test_function(name: str) -> Objective:
    """Return the objective callable for a test function."""
    try:
        return REGISTRY[name].objective
    except KeyError as exc:
        raise KeyError(f"Unknown test function: {name}") from exc


def get_function_spec(name: str) -> FunctionSpec:
    """Return full metadata for a test function."""
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown test function: {name}") from exc
