"""Simulated users for the TAF+ simulations (Liao et al., CHI 2024, App. B.1-B.2).

The base function has 4 inputs and 3 maximized objectives. Each objective is a
2D "Sphere" ``1 - 8 * ((u - c1)^2 + (v - c2)^2)`` on a pair of inputs:
``(x1, x2)``, ``(x2, x3)`` and ``(x3, x4)`` with the centres of App. Table 6.
``x2`` and ``x3`` are shared, so no input maximizes all three.

A user shifts every input, ``x' = x + s`` with ``s_i ~ U(-r_s/2, r_s/2)``, and
scales the output by ``S ~ U(1 - r_c/2, 1 + r_c/2)``. Observations get Gaussian
noise with std 0.05. The paper does not say whether the noise is added before or
after scaling; here it is added after, as observation noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

SPHERE_CENTRES = ((0.55, 0.4), (0.6, 0.45), (0.65, 0.35))
SPHERE_PAIRS = ((0, 1), (1, 2), (2, 3))
SPHERE_COEFFICIENT = 8.0
SPHERE_INPUT_DIM = 4
SPHERE_NOISE_STD = 0.05


def three_spheres(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the noise-free base function, shape ``(n, 3)``, for ``x`` of shape ``(n, 4)``."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.shape[1] != SPHERE_INPUT_DIM:
        raise ValueError(f"x must have shape (n, {SPHERE_INPUT_DIM}).")
    return np.stack(
        [
            1.0 - SPHERE_COEFFICIENT * np.sum((x[:, list(pair)] - np.asarray(c)) ** 2, axis=1)
            for pair, c in zip(SPHERE_PAIRS, SPHERE_CENTRES)
        ],
        axis=1,
    )


@dataclass(frozen=True)
class SphereUser:
    """One simulated user: an input shift and an output scale of the base function."""

    shift: tuple[float, ...]
    scale: float
    noise_std: float = SPHERE_NOISE_STD

    def noise_free(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the user's noise-free objectives, shape ``(n, 3)``."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return self.scale * three_spheres(x + np.asarray(self.shift))

    def observe(self, x: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
        """Return noisy objectives, shape ``(n, 3)``."""
        y = self.noise_free(x)
        if self.noise_std > 0.0:
            y = y + rng.normal(0.0, self.noise_std, size=y.shape)
        return y

    def weighted_optimum(self, objective_weights: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        """Return the maximizer in ``[0, 1]^4`` of the noise-free weighted sum, and its value.

        The weighted sum is a concave quadratic, so one bounded L-BFGS-B run
        finds the global maximum.
        """
        alpha = np.asarray(objective_weights, dtype=np.float64)
        result = minimize(
            lambda z: -float(self.noise_free(z[None, :])[0] @ alpha),
            np.full(SPHERE_INPUT_DIM, 0.5),
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * SPHERE_INPUT_DIM,
            options={"ftol": 1e-15, "gtol": 1e-12},
        )
        return np.asarray(result.x, dtype=np.float64), -float(result.fun)


def make_sphere_users(
    n_users: int, shift_range: float, scale_range: float, seed: int
) -> list[SphereUser]:
    """Sample `n_users` users with shifts ``U(-r_s/2, r_s/2)`` and scales ``U(1 - r_c/2, 1 + r_c/2)``."""
    if n_users < 1:
        raise ValueError("n_users must be positive.")
    if shift_range < 0.0 or not 0.0 <= scale_range < 2.0:
        raise ValueError("shift_range must be >= 0 and scale_range in [0, 2).")
    rng = np.random.default_rng(seed)
    users = []
    for _ in range(n_users):
        shift = rng.uniform(-shift_range / 2, shift_range / 2, size=SPHERE_INPUT_DIM)
        scale = rng.uniform(1.0 - scale_range / 2, 1.0 + scale_range / 2)
        users.append(SphereUser(tuple(float(s) for s in shift), float(scale)))
    return users
