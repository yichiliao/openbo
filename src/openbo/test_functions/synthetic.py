"""Synthetic benchmark test functions."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

BRANIN_MAXIMUM: float = -0.39788735772973816
SPHERE_MAXIMUM: float = 0.0
ACKLEY_MAXIMUM: float = 0.0
RASTRIGIN_MAXIMUM: float = 0.0
ROSENBROCK_MAXIMUM: float = 0.0
HARTMANN6_MAXIMUM: float = 3.322368011415515

KNOWN_OPTIMA: dict[str, float] = {
    "branin": BRANIN_MAXIMUM,
    "sphere": SPHERE_MAXIMUM,
    "ackley": ACKLEY_MAXIMUM,
    "rastrigin": RASTRIGIN_MAXIMUM,
    "rosenbrock": ROSENBROCK_MAXIMUM,
    "hartmann6": HARTMANN6_MAXIMUM,
}


def _apply_optional_noise(
    y: NDArray[np.float64],
    *,
    optimum: float,
    noise_std: float,
    rng: np.random.Generator | None,
    cap_at_optimum: bool,
) -> NDArray[np.float64]:
    """Optionally add Gaussian noise and cap by known optimum."""
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    if noise_std == 0.0:
        return y.astype(np.float64)

    rng_eff = np.random.default_rng() if rng is None else rng
    noisy = y + rng_eff.normal(loc=0.0, scale=noise_std, size=y.shape)
    if cap_at_optimum:
        noisy = np.minimum(noisy, optimum)
    return noisy.astype(np.float64)


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


def branin(
    x: ArrayLike,
    *,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    cap_at_optimum: bool = False,
) -> NDArray[np.float64]:
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
    y = (-y_min).astype(np.float64)
    return _apply_optional_noise(
        y,
        optimum=BRANIN_MAXIMUM,
        noise_std=noise_std,
        rng=rng,
        cap_at_optimum=cap_at_optimum,
    )


def sphere(
    x: ArrayLike,
    *,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    cap_at_optimum: bool = False,
) -> NDArray[np.float64]:
    """Evaluate maximization Sphere on `[0, 1]^2`, mapped to `[-5, 5]^2`."""
    arr = _as_2d(x, dim=2)
    arr_native = _from_unit_cube(arr, bounds=[(-5.0, 5.0), (-5.0, 5.0)])
    y = (-np.sum(arr_native**2, axis=1, dtype=float)).astype(np.float64)
    return _apply_optional_noise(
        y,
        optimum=SPHERE_MAXIMUM,
        noise_std=noise_std,
        rng=rng,
        cap_at_optimum=cap_at_optimum,
    )


def ackley(
    x: ArrayLike,
    *,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    cap_at_optimum: bool = False,
) -> NDArray[np.float64]:
    """Evaluate maximization Ackley objective on `[0, 1]^2`."""
    arr = _as_2d(x, dim=2)
    arr_native = _from_unit_cube(arr, bounds=[(-5.0, 5.0), (-5.0, 5.0)])
    d = arr_native.shape[1]
    sq_term = np.mean(arr_native**2, axis=1)
    cos_term = np.mean(np.cos(2.0 * math.pi * arr_native), axis=1)
    y_min = (
        -20.0 * np.exp(-0.2 * np.sqrt(sq_term))
        - np.exp(cos_term)
        + 20.0
        + math.e
    )
    y = (-y_min).astype(np.float64)
    return _apply_optional_noise(
        y,
        optimum=ACKLEY_MAXIMUM,
        noise_std=noise_std,
        rng=rng,
        cap_at_optimum=cap_at_optimum,
    )


def rastrigin(
    x: ArrayLike,
    *,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    cap_at_optimum: bool = False,
) -> NDArray[np.float64]:
    """Evaluate maximization Rastrigin objective on `[0, 1]^2`."""
    arr = _as_2d(x, dim=2)
    arr_native = _from_unit_cube(arr, bounds=[(-5.12, 5.12), (-5.12, 5.12)])
    d = arr_native.shape[1]
    y_min = 10.0 * d + np.sum(
        arr_native**2 - 10.0 * np.cos(2.0 * math.pi * arr_native),
        axis=1,
        dtype=float,
    )
    y = (-y_min).astype(np.float64)
    return _apply_optional_noise(
        y,
        optimum=RASTRIGIN_MAXIMUM,
        noise_std=noise_std,
        rng=rng,
        cap_at_optimum=cap_at_optimum,
    )


def rosenbrock(
    x: ArrayLike,
    *,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    cap_at_optimum: bool = False,
) -> NDArray[np.float64]:
    """Evaluate maximization Rosenbrock objective on `[0, 1]^2`."""
    arr = _as_2d(x, dim=2)
    arr_native = _from_unit_cube(arr, bounds=[(-2.0, 2.0), (-2.0, 2.0)])
    x1 = arr_native[:, 0]
    x2 = arr_native[:, 1]
    y_min = (1.0 - x1) ** 2 + 100.0 * (x2 - x1**2) ** 2
    y = (-y_min).astype(np.float64)
    return _apply_optional_noise(
        y,
        optimum=ROSENBROCK_MAXIMUM,
        noise_std=noise_std,
        rng=rng,
        cap_at_optimum=cap_at_optimum,
    )


def hartmann6(
    x: ArrayLike,
    *,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    cap_at_optimum: bool = False,
) -> NDArray[np.float64]:
    """Evaluate maximization Hartmann6 objective on `[0, 1]^6`."""
    arr = _as_2d(x, dim=6)
    alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=np.float64)
    a = np.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ],
        dtype=np.float64,
    )
    p = 1e-4 * np.array(
        [
            [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
            [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
            [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
            [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
        ],
        dtype=np.float64,
    )
    diff = arr[:, None, :] - p[None, :, :]
    inner = np.sum(a[None, :, :] * diff**2, axis=2)
    y_min = -np.sum(alpha[None, :] * np.exp(-inner), axis=1)
    y = (-y_min).astype(np.float64)
    return _apply_optional_noise(
        y,
        optimum=HARTMANN6_MAXIMUM,
        noise_std=noise_std,
        rng=rng,
        cap_at_optimum=cap_at_optimum,
    )
