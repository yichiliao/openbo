"""ConBO's mixed acquisition: population EI handing over to user EI.

Follows Liao et al., *Continual Human-in-the-Loop Optimization* (CHI 2025):

    a_t(x) = w_t * EI_pop(x) + (1 - w_t) * EI_user(x)

with a weight that stays at 1 for the first ``alpha1`` iterations of a user,
then falls linearly by ``alpha2`` per iteration until it reaches 0. Both EIs
are in raw output units, so they can be added directly.

Which incumbent each EI uses is not stated in the paper (``docs/plans/conbo.md``,
open question 5). :func:`default_incumbents` documents the choice used here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from openbo.acquisition.ei import expected_improvement_maximization

Prediction = tuple[NDArray[np.float64], NDArray[np.float64]]


def population_weight(t: int, alpha1: float = 5.0, alpha2: float = 0.2) -> float:
    """Return the population model's weight ``w_t`` at 1-based iteration `t`.

    ``w_t = 1`` for ``t <= alpha1``, ``1 - (t - alpha1) * alpha2`` up to
    ``t = alpha1 + 1 / alpha2``, and 0 after that. The paper's user study uses
    ``alpha1 = 5``, ``alpha2 = 0.2``, so the weight reaches 0 at ``t = 10``.
    """
    if t < 1:
        raise ValueError("t must be at least 1.")
    if alpha1 < 0.0:
        raise ValueError("alpha1 must be non-negative.")
    if alpha2 <= 0.0:
        raise ValueError("alpha2 must be positive.")
    return float(np.clip(1.0 - (t - alpha1) * alpha2, 0.0, 1.0))


def default_incumbents(
    y_user: ArrayLike,
    pop_mean: ArrayLike,
) -> tuple[float, float]:
    """Return the incumbents ``(best_pop, best_user)`` for the two EIs.

    Both use the best value the current user has produced so far, so the two
    EIs measure improvement over the same reference. Before the user has any
    observation (a user with no random starts), both use the best predicted
    population mean over the candidates instead. With the default schedule the
    user EI has zero weight at that point anyway.
    """
    y = np.asarray(y_user, dtype=np.float64).ravel()
    if y.size > 0:
        best = float(np.max(y))
        return best, best
    mean = np.asarray(pop_mean, dtype=np.float64).ravel()
    if mean.size == 0:
        raise ValueError("Need user observations or population predictions.")
    best_pop = float(np.max(mean))
    return best_pop, best_pop


def mixed_expected_improvement(
    weight: float,
    pop_prediction: Prediction | None,
    user_prediction: Prediction | None,
    best_pop: float,
    best_user: float,
) -> NDArray[np.float64]:
    """Return ``weight * EI_pop + (1 - weight) * EI_user`` over candidates.

    Each prediction is ``(mean, variance)`` at the same candidates. A term with
    zero weight is skipped, so its prediction may be ``None`` (no population
    model yet, or no user data yet).
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1].")
    if weight > 0.0 and pop_prediction is None:
        raise ValueError("pop_prediction is required when weight > 0.")
    if weight < 1.0 and user_prediction is None:
        raise ValueError("user_prediction is required when weight < 1.")

    terms: list[NDArray[np.float64]] = []
    shapes: set[tuple[int, ...]] = set()
    if weight > 0.0:
        assert pop_prediction is not None
        mean, var = pop_prediction
        shapes.add(np.shape(mean))
        terms.append(weight * expected_improvement_maximization(mean, var, best_pop))
    if weight < 1.0:
        assert user_prediction is not None
        mean, var = user_prediction
        shapes.add(np.shape(mean))
        terms.append((1.0 - weight) * expected_improvement_maximization(mean, var, best_user))
    if len(shapes) > 1:
        raise ValueError("Population and user predictions must cover the same candidates.")
    return np.sum(terms, axis=0).astype(np.float64)
