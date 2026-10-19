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
    "mccormick": [(0.0, 1.0), (0.0, 1.0)],
}

TASK_DIMS: dict[str, int] = {
    name: len(bounds) for name, bounds in TASK_BOUNDS.items()
}

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]

# Namespace id so objective-noise RNGs never share a bit stream with the
# search/optimizer RNGs (which use np.random.default_rng(seed) directly). Without
# this, a task whose noise seed equals the optimizer seed couples the noise
# realization to the search points.
_NOISE_STREAM_ID = 0x0B0


def noise_rng(seed: int | None) -> np.random.Generator:
    """Return a decoupled RNG for objective output noise."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([int(seed), _NOISE_STREAM_ID]))


@dataclass(frozen=True)
class TaskVariantSpec:
    """Affine input/output perturbation for a base objective.

    All input transforms are defined in normalized coordinates.
    """

    input_shift: tuple[float, ...]
    input_scale: tuple[float, ...]
    output_scale: float = 1.0
    noise_std: float = 0.0
    cap_at_optimum: bool = False
    seed: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize variant spec to a JSON-friendly dictionary."""
        return {
            "input_shift": list(self.input_shift),
            "input_scale": list(self.input_scale),
            "output_scale": self.output_scale,
            "noise_std": self.noise_std,
            "cap_at_optimum": self.cap_at_optimum,
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> "TaskVariantSpec":
        """Construct variant spec from a JSON-like dictionary."""
        return TaskVariantSpec(
            input_shift=tuple(float(v) for v in data["input_shift"]),  # type: ignore[index]
            input_scale=tuple(float(v) for v in data["input_scale"]),  # type: ignore[index]
            output_scale=float(data["output_scale"]),  # type: ignore[arg-type]
            noise_std=float(data.get("noise_std", 0.0)),  # type: ignore[arg-type]
            cap_at_optimum=bool(data.get("cap_at_optimum", False)),
            seed=None if data.get("seed") is None else int(data["seed"]),  # type: ignore[index]
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


def make_variant_objective(
    base_objective: Objective,
    variant: TaskVariantSpec,
    dim: int,
    base_optimum: float | None = None,
    cap_value: float | None = None,
) -> Objective:
    """Wrap a base objective with an affine input/output variant.

    When ``cap_at_optimum`` is set, noisy outputs are clipped at ``cap_value`` if
    the caller supplies one (its reachable optimum), otherwise at the analytic
    ``output_scale * base_optimum``. Passing ``cap_value`` keeps the cap
    consistent with the optimum reported by the caller.
    """
    variant.validate_for_dim(dim)
    shift = np.array(variant.input_shift, dtype=np.float64)
    scale = np.array(variant.input_scale, dtype=np.float64)
    rng = noise_rng(variant.seed)
    variant_optimum: float | None = None
    if variant.cap_at_optimum and variant.output_scale >= 0.0:
        if cap_value is not None:
            variant_optimum = float(cap_value)
        elif base_optimum is not None:
            variant_optimum = variant.output_scale * base_optimum

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
            if variant_optimum is not None:
                y = np.minimum(y, variant_optimum)
        return np.asarray(y, dtype=np.float64)

    return objective
