"""Simple random search baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class RandomSearch:
    """Uniform random search over box constraints."""

    bounds: list[tuple[float, float]]
    seed: int | None = None

    def run(self, objective: Objective, n_evals: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sample points uniformly and evaluate objective."""
        if n_evals <= 0:
            raise ValueError("n_evals must be positive.")
        rng = np.random.default_rng(self.seed)
        lower = np.array([b[0] for b in self.bounds], dtype=np.float64)
        upper = np.array([b[1] for b in self.bounds], dtype=np.float64)
        x = rng.uniform(lower, upper, size=(n_evals, len(self.bounds))).astype(np.float64)
        y = np.asarray(objective(x), dtype=np.float64)
        if y.shape != (n_evals,):
            raise ValueError(f"Objective should return shape ({n_evals},), got {y.shape}.")
        return x, y
