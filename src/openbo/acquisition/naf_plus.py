"""NAF+'s novelty-aware mix of the neural AF and EI (Liao et al., CHI 2026, Sec. 4.5-4.6).

Plan: ``docs/plans/naf_plus.md``, Phase 3.

- eq. 3 and the two-tailed p-value: ``z = (y - mu(x)) / sigma(x)`` under the
  novelty detector's prediction, ``p = 2 (1 - Phi(|z|))``.
- eq. 5: the EI weight from the running mean ``p_bar`` of the user's
  p-values: 1 at ``p_bar = 0``, 0 from ``p_bar >= tau`` on, linear between.
- eq. 4: ``A(x) = lambda EI(x) + (1 - lambda) naf(x)``.

OpenBO's calibrated alternative (plan, "Design stance", items 2-3): the
statistic ``S_n = sum_i z_i^2`` over a user's ``n`` observations, turned into
a p-value by simulation (the share of training-like users with a statistic at
least as large, :func:`calibrated_p_value`), with ``tau = 0.05``; and EI mixed
with the neural AF as policies (a coin with probability ``lambda`` picks EI's
point), not as scores.

The paper does not say how the neural AF's values and EI are put on one scale
(author question 1). Here both are min-max normalized over a fixed reference
set, the adaptive set's global grid, each step. A fixed reference keeps the
scores of the grid and of the local clouds (scored in separate calls)
comparable. With ``lambda = 0`` the pure neural AF score is used unchanged, so
NAF+ then picks exactly what NAF picks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from openbo.acquisition.ei import expected_improvement_maximization
from openbo.acquisition.naf import ScoreFn, gp_mean_std

NOVELTY_THRESHOLD = 0.1
NAF_ONLY_ITERATIONS = 3
# The false-alarm rate of the calibrated test: tau for calibrated p-values.
CALIBRATED_TAU = 0.05


def novelty_p_values(mean: ArrayLike, std: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Two-tailed p-values of observations ``y`` under ``N(mean, std^2)`` (eq. 3)."""
    m = np.asarray(mean, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64)
    obs = np.asarray(y, dtype=np.float64)
    if m.shape != s.shape or m.shape != obs.shape:
        raise ValueError("mean, std and y must have the same shape.")
    if np.any(s <= 0.0):
        raise ValueError("std must be positive.")
    z = (obs - m) / s
    return 2.0 * ndtr(-np.abs(z))


def novelty_z_scores(mean: ArrayLike, std: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Standardized residuals ``(y - mean) / std`` of observations under the detector."""
    m = np.asarray(mean, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64)
    obs = np.asarray(y, dtype=np.float64)
    if m.shape != s.shape or m.shape != obs.shape:
        raise ValueError("mean, std and y must have the same shape.")
    if np.any(s <= 0.0):
        raise ValueError("std must be positive.")
    return (obs - m) / s


def novelty_statistic(z: ArrayLike) -> float:
    """The calibrated test's statistic ``sum z_i^2`` (large for users unlike the population)."""
    return float(np.sum(np.square(np.asarray(z, dtype=np.float64))))


def calibrated_p_value(null_statistics: ArrayLike, statistic: float) -> float:
    """``(1 + #{null >= statistic}) / (N + 1)``: the share of ``N`` simulated typical users at least as extreme.

    The ``+1`` counts the observed user itself, so the p-value is never 0 and
    a test at level ``tau`` has a false-alarm rate of at most ``tau``.
    """
    null = np.asarray(null_statistics, dtype=np.float64).reshape(-1)
    if null.size == 0:
        raise ValueError("null_statistics must not be empty.")
    return float((1 + np.count_nonzero(null >= statistic)) / (null.size + 1))


def ei_weight(p_bar: float, tau: float = NOVELTY_THRESHOLD) -> float:
    """The EI weight ``lambda_EI`` of eq. 5 for a mean p-value ``p_bar``."""
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1].")
    if not 0.0 <= p_bar <= 1.0:
        raise ValueError("p_bar must be in [0, 1].")
    if p_bar >= tau:
        return 0.0
    return (tau - p_bar) / tau


def ei_score_fn(gp: object, best_y: float) -> ScoreFn:
    """Analytic EI on the (fixed-hyperparameter) GP, as a score function."""

    def score(x: NDArray[np.float64]) -> NDArray[np.float64]:
        mean, std = gp_mean_std(gp, x)
        return expected_improvement_maximization(mean, std**2, best_y)

    return score


def _minmax(values: NDArray[np.float64]) -> tuple[float, float]:
    """Offset and scale mapping ``values`` onto ``[0, 1]`` (scale 1 for constant values)."""
    lo, hi = float(np.min(values)), float(np.max(values))
    return lo, (hi - lo) if hi > lo else 1.0


def mixed_score_fn(
    naf_score: ScoreFn,
    ei_score: ScoreFn,
    lam: float,
    reference: NDArray[np.float64],
) -> ScoreFn:
    """``A(x) = lam EI(x) + (1 - lam) naf(x)`` (eq. 4), each min-max normalized over ``reference``.

    Points outside ``reference`` may score outside ``[0, 1]``; the
    normalization is a fixed affine map per step, so their ranking is kept.
    ``lam = 0`` returns ``naf_score`` itself and ``lam = 1`` returns ``ei_score``.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1].")
    if lam == 0.0:
        return naf_score
    if lam == 1.0:
        return ei_score
    naf_lo, naf_scale = _minmax(np.asarray(naf_score(reference), dtype=np.float64))
    ei_lo, ei_scale = _minmax(np.asarray(ei_score(reference), dtype=np.float64))

    def score(x: NDArray[np.float64]) -> NDArray[np.float64]:
        naf = (np.asarray(naf_score(x), dtype=np.float64) - naf_lo) / naf_scale
        ei = (np.asarray(ei_score(x), dtype=np.float64) - ei_lo) / ei_scale
        return lam * ei + (1.0 - lam) * naf

    return score
