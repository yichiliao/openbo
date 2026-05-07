"""Kernel implementations used by the scratch GP."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def _resolve_lengthscale(
    lengthscale: float | NDArray[np.float64],
    d: int,
) -> NDArray[np.float64]:
    """Return ARD lengthscale vector of shape (d,)."""
    ls = np.asarray(lengthscale, dtype=np.float64)
    if ls.ndim == 0:
        val = float(ls)
        if val <= 0:
            raise ValueError("lengthscale must be positive.")
        return np.full(d, val, dtype=np.float64)
    if ls.ndim != 1 or ls.shape[0] != d:
        raise ValueError(f"lengthscale must be scalar or shape ({d},).")
    if np.any(ls <= 0):
        raise ValueError("all ARD lengthscales must be positive.")
    return ls


def rbf_kernel(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    lengthscale: float | NDArray[np.float64],
    variance: float,
) -> NDArray[np.float64]:
    """Compute an RBF kernel matrix between two point sets."""
    if variance <= 0:
        raise ValueError("variance must be positive.")
    d = x1.shape[1]
    ls = _resolve_lengthscale(lengthscale, d)
    scaled_diff = (x1[:, None, :] - x2[None, :, :]) / ls[None, None, :]
    sq_dist = np.sum(scaled_diff**2, axis=2)
    return variance * np.exp(-0.5 * sq_dist)


def matern52_kernel(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    lengthscale: float | NDArray[np.float64],
    variance: float,
) -> NDArray[np.float64]:
    """Compute a Matern-5/2 kernel matrix between two point sets."""
    if variance <= 0:
        raise ValueError("variance must be positive.")
    d = x1.shape[1]
    ls = _resolve_lengthscale(lengthscale, d)
    scaled_diff = (x1[:, None, :] - x2[None, :, :]) / ls[None, None, :]
    sq_dist = np.sum(scaled_diff**2, axis=2)
    r = np.sqrt(np.maximum(sq_dist, 1e-24))
    sqrt5_r = np.sqrt(5.0) * r
    return variance * (1.0 + sqrt5_r + (5.0 / 3.0) * sq_dist) * np.exp(-sqrt5_r)
