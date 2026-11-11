"""User families of the NAF+ synthetic tests (Liao et al., CHI 2026, App. B).

Synthetic Test 1, the double sphere: two objectives on ``[0, 1]^2``,

    f_k(x) = 1 - gamma * sum_i (x_i - c_k,i)^2,    gamma = 8,

centred at ``c_1 = (0.4, 0.4)`` and ``c_2 = (0.6, 0.6)``; both are maximized.
A user shifts the input, ``x' = x + delta`` with ``delta ~ U(-r/2, r/2)^2``, and
scales the output by ``S ~ U(1 - s/2, 1 + s/2)``. The paper scales the weighted
sum; since the sum is linear, scaling each objective is the same thing, and it
lets per-objective consumers (TAF+, the server) see the user's own values.

Novel users (Test 1.2) move the centres to ``(0.3, 0.3)`` and ``(0.7, 0.7)``.

The paper does not state ``r`` and ``s``. The defaults are MetaBO's family
ranges (``r = 0.2``, a shift in ``[-0.1, 0.1]``; ``s = 0.2``, a scale in
``[0.9, 1.1]``) until the author confirms them (``docs/plans/naf_plus.md``,
author question 2).

The scalarized objective ``S * sum_k w_k f_k(x + delta)`` with weights on the
simplex is an isotropic quadratic centred at ``sum_k w_k c_k - delta``, so its
maximizer on the unit box is that centre clipped per coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from openbo.acquisition.naf import validate_weights

DOUBLE_SPHERE_GAMMA = 8.0
TRAINING_CENTRES = ((0.4, 0.4), (0.6, 0.6))
NOVEL_CENTRES = ((0.3, 0.3), (0.7, 0.7))
# The wide family (OpenBO, 2026-11-11; plan "Design stance"): the centres far apart, so the
# weights move the optimum across most of the square; novel users use the other diagonal.
WIDE_TRAINING_CENTRES = ((0.2, 0.2), (0.8, 0.8))
WIDE_NOVEL_CENTRES = ((0.2, 0.8), (0.8, 0.2))
# Family name -> (training centres, novel centres).
DOUBLE_SPHERE_FAMILIES = {
    "double_sphere": (TRAINING_CENTRES, NOVEL_CENTRES),
    "double_sphere_wide": (WIDE_TRAINING_CENTRES, WIDE_NOVEL_CENTRES),
}
DEFAULT_SHIFT_RANGE = 0.2
DEFAULT_SCALE_RANGE = 0.2


def double_sphere_objectives(
    x: ArrayLike,
    centres: tuple[tuple[float, ...], ...] = TRAINING_CENTRES,
    gamma: float = DOUBLE_SPHERE_GAMMA,
) -> NDArray[np.float64]:
    """Return ``1 - gamma * ||x - c_k||^2`` for each centre, shape ``(n, L)``."""
    arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
    c = np.asarray(centres, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != arr.shape[1]:
        raise ValueError(f"centres must have shape (L, {arr.shape[1]}).")
    return 1.0 - gamma * ((arr[:, None, :] - c[None, :, :]) ** 2).sum(axis=-1)


def sample_weights(n_objectives: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Draw objective weights uniformly from the simplex (Dirichlet(1)).

    For two objectives this is the paper's ``alpha ~ U[0, 1]``, ``w = (alpha, 1 - alpha)``.
    """
    if n_objectives < 1:
        raise ValueError("n_objectives must be positive.")
    return rng.dirichlet(np.ones(n_objectives))


@dataclass(frozen=True)
class DoubleSphereUser:
    """One synthetic user of the double-sphere family."""

    shift: tuple[float, float]
    scale: float
    centres: tuple[tuple[float, float], ...] = TRAINING_CENTRES
    gamma: float = DOUBLE_SPHERE_GAMMA

    @property
    def n_objectives(self) -> int:
        """Number of objectives ``L``."""
        return len(self.centres)

    def objectives(self, x: ArrayLike) -> NDArray[np.float64]:
        """The user's per-objective values at ``x`` (``(n, 2)`` or ``(2,)``), shape ``(n, L)``."""
        arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return self.scale * double_sphere_objectives(arr + np.asarray(self.shift), self.centres, self.gamma)

    def __call__(self, x: ArrayLike, weights: ArrayLike) -> NDArray[np.float64]:
        """The scalarized objective ``sum_k w_k f_k`` at ``x``, shape ``(n,)``."""
        w = validate_weights(weights, self.n_objectives)
        return self.objectives(x) @ w

    def argmax(self, weights: ArrayLike) -> NDArray[np.float64]:
        """Maximizer of the scalarized objective inside ``[0, 1]^2``."""
        w = validate_weights(weights, self.n_objectives)
        centre = w @ np.asarray(self.centres, dtype=np.float64) - np.asarray(self.shift)
        return np.clip(centre, 0.0, 1.0)

    def optimum(self, weights: ArrayLike) -> float:
        """Maximum of the scalarized objective inside ``[0, 1]^2``."""
        return float(self(self.argmax(weights)[None, :], weights)[0])


@dataclass(frozen=True)
class WeightedTask:
    """A user with fixed weights: a single-objective task with ``__call__`` and ``optimum``.

    This is the form NAF episodes and ``fit_family_hyperparameters`` take.
    """

    user: DoubleSphereUser
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        """Check the weights."""
        validate_weights(self.weights, self.user.n_objectives)

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """The scalarized objective, shape ``(n,)``."""
        return self.user(x, self.weights)

    @property
    def optimum(self) -> float:
        """Maximum inside ``[0, 1]^2``."""
        return self.user.optimum(self.weights)


def sample_double_sphere_users(
    n_users: int,
    seed: int,
    shift_range: float = DEFAULT_SHIFT_RANGE,
    scale_range: float = DEFAULT_SCALE_RANGE,
    centres: tuple[tuple[float, float], ...] = TRAINING_CENTRES,
) -> list[DoubleSphereUser]:
    """Draw ``n_users`` users: shifts ``U(-r/2, r/2)^2`` and scales ``U(1 - s/2, 1 + s/2)``."""
    if n_users < 1:
        raise ValueError("n_users must be positive.")
    if shift_range < 0.0:
        raise ValueError("shift_range must be non-negative.")
    if not 0.0 <= scale_range < 2.0:
        raise ValueError("scale_range must be in [0, 2) so that scales stay positive.")
    rng = np.random.default_rng(seed)
    u = rng.random((n_users, 3))
    shifts = shift_range * (u[:, :2] - 0.5)
    scales = 1.0 + scale_range * (u[:, 2] - 0.5)
    return [
        DoubleSphereUser(shift=(float(s[0]), float(s[1])), scale=float(c), centres=centres)
        for s, c in zip(shifts, scales)
    ]


def sample_weighted_tasks(
    n_tasks: int,
    seed: int,
    shift_range: float = DEFAULT_SHIFT_RANGE,
    scale_range: float = DEFAULT_SCALE_RANGE,
    centres: tuple[tuple[float, float], ...] = TRAINING_CENTRES,
) -> list[WeightedTask]:
    """Draw ``n_tasks`` users, each with its own weights from :func:`sample_weights`."""
    users = sample_double_sphere_users(n_tasks, seed, shift_range, scale_range, centres)
    rng = np.random.default_rng(seed + 1)
    return [WeightedTask(user, tuple(float(v) for v in sample_weights(user.n_objectives, rng))) for user in users]
