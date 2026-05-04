"""Kernel implementations used by the scratch GP."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def rbf_kernel(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    lengthscale: float,
    variance: float,
) -> NDArray[np.float64]:
    """Compute an RBF kernel matrix between two point sets."""
    if lengthscale <= 0:
        raise ValueError("lengthscale must be positive.")
    diff = x1[:, None, :] - x2[None, :, :]
    sq_dist = np.sum(diff**2, axis=2)
    return variance * np.exp(-0.5 * sq_dist / (lengthscale**2))
