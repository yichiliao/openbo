"""Expected Improvement acquisition implementations."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def expected_improvement_maximization(
    mean: NDArray[np.float64],
    variance: NDArray[np.float64],
    best_y: float,
) -> NDArray[np.float64]:
    """Compute EI for maximization from predictive mean/variance."""
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    std = np.sqrt(np.maximum(variance, 1e-12))

    z = (mean - best_y) / std
    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * z**2)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    ei = (mean - best_y) * cdf + std * pdf
    return np.maximum(ei, 0.0).astype(np.float64)


def expected_improvement_minimization(
    mean: NDArray[np.float64],
    variance: NDArray[np.float64],
    best_y: float,
) -> NDArray[np.float64]:
    """Backward-compatible minimization EI wrapper via sign flip."""
    return expected_improvement_maximization(-mean, variance, -best_y)
