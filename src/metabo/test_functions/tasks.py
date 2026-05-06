"""Task metadata and variant specifications for test functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

TASK_BOUNDS: dict[str, list[tuple[float, float]]] = {
    "branin": [(0.0, 1.0), (0.0, 1.0)],
    "sphere": [(0.0, 1.0), (0.0, 1.0)],
    "ackley": [(0.0, 1.0), (0.0, 1.0)],
    "rastrigin": [(0.0, 1.0), (0.0, 1.0)],
    "rosenbrock": [(0.0, 1.0), (0.0, 1.0)],
    "hartmann6": [(0.0, 1.0)] * 6,
}

TASK_DIMS: dict[str, int] = {
    name: len(bounds) for name, bounds in TASK_BOUNDS.items()
}

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class TaskVariantSpec:
    """Affine input/output perturbation for a base objective.

    All input transforms are defined in normalized coordinates.
    """

    input_shift: tuple[float, ...]
    input_scale: tuple[float, ...]
    output_scale: float = 1.0
    noise_std: float = 0.0
    seed: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize variant spec to a JSON-friendly dictionary."""
        return {
            "input_shift": list(self.input_shift),
            "input_scale": list(self.input_scale),
            "output_scale": self.output_scale,
            "noise_std": self.noise_std,
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> "TaskVariantSpec":
        """Construct variant spec from a JSON-like dictionary."""
        return TaskVariantSpec(
            input_shift=tuple(float(v) for v in data["input_shift"]),  # type: ignore[index]
            input_scale=tuple(float(v) for v in data["input_scale"]),  # type: ignore[index]
            output_scale=float(data["output_scale"]),  # type: ignore[arg-type]
            noise_std=float(data["noise_std"]),  # type: ignore[arg-type]
            seed=None if data["seed"] is None else int(data["seed"]),  # type: ignore[index]
        )

    def validate_for_dim(self, dim: int) -> None:
        """Validate that the variant is compatible with `dim`."""
        if len(self.input_shift) != dim:
            raise ValueError(
                f"input_shift has dim {len(self.input_shift)} but function dim is {dim}."
            )
        if len(self.input_scale) != dim:
            raise ValueError(
                f"input_scale has dim {len(self.input_scale)} but function dim is {dim}."
            )
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative.")


def make_variant_objective(base_objective: Objective, variant: TaskVariantSpec, dim: int) -> Objective:
    """Wrap a base objective with an affine input/output variant."""
    variant.validate_for_dim(dim)
    shift = np.array(variant.input_shift, dtype=np.float64)
    scale = np.array(variant.input_scale, dtype=np.float64)
    rng = np.random.default_rng(variant.seed)

    def objective(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != dim:
            raise ValueError(f"Expected x shape (n, {dim}) or ({dim},), got {x.shape}.")
        x_variant = np.clip(scale * x + shift, 0.0, 1.0)
        y = variant.output_scale * base_objective(x_variant)
        if variant.noise_std > 0:
            y = y + rng.normal(0.0, variant.noise_std, size=y.shape)
        return np.asarray(y, dtype=np.float64)

    return objective
