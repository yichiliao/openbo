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
        if not np.isfinite(val) or val <= 0:
            raise ValueError("lengthscale must be finite and positive.")
        return np.full(d, val, dtype=np.float64)
    if ls.ndim != 1 or ls.shape[0] != d:
        raise ValueError(f"lengthscale must be scalar or shape ({d},).")
    if not np.all(np.isfinite(ls)) or np.any(ls <= 0):
        raise ValueError("all ARD lengthscales must be finite and positive.")
    return ls


def _scaled_squared_distances(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    lengthscale: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return squared ARD distances, allowing infinity for distant finite inputs.

    Subtract first to preserve equal large coordinates. If subtraction itself
    overflows, scale before subtracting instead: e.g. ``(1e308 - -1e308)/1e308``
    has distance 2, even though its unscaled difference cannot be represented.
    """
    if not np.all(np.isfinite(x1)) or not np.all(np.isfinite(x2)):
        raise ValueError("kernel inputs must be finite.")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        diff = x1[:, None, :] - x2[None, :, :]
        scaled = diff / lengthscale[None, None, :]
        overflow = np.isinf(diff)
        if np.any(overflow):
            scaled_first = (
                x1[:, None, :] / lengthscale[None, None, :]
                - x2[None, :, :] / lengthscale[None, None, :]
            )
            scaled = np.where(overflow, scaled_first, scaled)
        return np.sum(scaled**2, axis=2)


def rbf_kernel(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    lengthscale: float | NDArray[np.float64],
    variance: float,
) -> NDArray[np.float64]:
    """Compute an RBF kernel matrix between two point sets."""
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("variance must be finite and positive.")
    d = x1.shape[1]
    ls = _resolve_lengthscale(lengthscale, d)
    sq_dist = _scaled_squared_distances(x1, x2, ls)
    with np.errstate(under="ignore"):
        values = variance * np.exp(-0.5 * sq_dist)
        tail = sq_dist > 1200.0
        # Retain representable covariances when a large variance compensates
        # for an exponential that would underflow if evaluated on its own.
        values[tail] = np.exp(np.log(variance) - 0.5 * sq_dist[tail])
        return values


def matern52_kernel(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    lengthscale: float | NDArray[np.float64],
    variance: float,
) -> NDArray[np.float64]:
    """Compute a Matern-5/2 kernel matrix between two point sets."""
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("variance must be finite and positive.")
    d = x1.shape[1]
    ls = _resolve_lengthscale(lengthscale, d)
    sq_dist = _scaled_squared_distances(x1, x2, ls)
    r = np.sqrt(sq_dist)
    sqrt5_r = np.sqrt(5.0) * r
    values = np.zeros_like(sq_dist)
    regular = np.isfinite(sq_dist) & (sqrt5_r <= 600.0)
    tail = np.isfinite(sq_dist) & ~regular
    with np.errstate(under="ignore"):
        unit = (
            1.0 + sqrt5_r[regular] + (5.0 / 3.0) * sq_dist[regular]
        ) * np.exp(-sqrt5_r[regular])
        values[regular] = variance * np.minimum(unit, 1.0)
        z = sqrt5_r[tail]
        log_polynomial = 2.0 * np.log(z) - np.log(3.0) + np.log1p(3.0 / z + 3.0 / z / z)
        values[tail] = np.exp(np.log(variance) + log_polynomial - z)
    # Infinite squared distances retain their exact stationary-kernel limit 0,
    # without evaluating the indeterminate product infinity * exp(-infinity).
    return values
