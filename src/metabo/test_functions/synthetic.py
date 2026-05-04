"""Synthetic benchmark test functions."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_2d(x: ArrayLike, dim: int) -> NDArray[np.float64]:
    """Return `x` as a 2D float array with shape `(n, dim)`."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"Expected shape (n, {dim}) or ({dim},), got {arr.shape}.")
    return arr


def _from_unit_cube(
    x_unit: NDArray[np.float64],
    bounds: list[tuple[float, float]],
) -> NDArray[np.float64]:
    """Linearly map points from `[0, 1]^d` to provided bounds."""
    lower = np.array([b[0] for b in bounds], dtype=np.float64)
    upper = np.array([b[1] for b in bounds], dtype=np.float64)
    return lower + x_unit * (upper - lower)


def branin(x: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the Branin function.

    Input is expected in normalized domain `[0, 1]^2` and is linearly mapped to:
    - `x1 in [-5, 10]`
    - `x2 in [0, 15]`
    """
    arr = _as_2d(x, dim=2)
    arr_native = _from_unit_cube(arr, bounds=[(-5.0, 10.0), (0.0, 15.0)])
    x1 = arr_native[:, 0]
    x2 = arr_native[:, 1]

    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    y_min = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    # Flip sign so larger is better (maximization objective).
    return (-y_min).astype(np.float64)


def sphere(x: ArrayLike) -> NDArray[np.float64]:
    """Evaluate maximization Sphere on `[0, 1]^2`, mapped to `[-5, 5]^2`."""
    arr = _as_2d(x, dim=2)
    arr_native = _from_unit_cube(arr, bounds=[(-5.0, 5.0), (-5.0, 5.0)])
    return (-np.sum(arr_native**2, axis=1, dtype=float)).astype(np.float64)
