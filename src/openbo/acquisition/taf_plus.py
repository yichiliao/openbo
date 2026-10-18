"""TAF+: the transfer acquisition function with objective weights and decay.

Follows Liao et al., *A Meta-Bayesian Approach for Rapid Online Parametric
Optimization for Wrist-based Interactions* (CHI 2024). Everything TAF+ inherits
from TAF follows the original TAF paper instead, wherever the two disagree:
Wistuba, Schilling and Schmidt-Thieme, *Scalable Gaussian process-based
transfer surrogates for hyperparameter optimization* (Machine Learning, 2018).
See ``docs/plans/taf_plus.md`` for each such decision.

For candidates ``x``, earlier users (population models) ``j = 1..M``, the
current user (target, index ``M+1``) and objectives ``l = 1..L``:

    I_j^l(x)   = max(mu_j^l(x) - ref_j^l, 0)                (original eq. 47)
    ref_j^l    = max over the target's evaluated points of mu_j^l
    EI^l(x)    = expected improvement of the target GP over its best y^l
    PEI_j^+(x) = sum_l alpha_l * I_j^l(x)                   (CHI eq. 4)
    EI^+(x)    = sum_l alpha_l * EI^l(x)
    w_j^+(x)   = delta(t) * sum_l alpha_l * w_j^l(x)         (CHI eq. 5)
    w_{M+1}^+  = sum_l alpha_l * w_{M+1}^l(x)
    TAF+(x)    = (sum_j w_j^+ I_j^+ + w_{M+1}^+ EI^+) / (sum_j w_j^+ + w_{M+1}^+)

The weights ``w^l`` are TAF-PoE (original section 6.1, the variance-based
weights the CHI paper uses, which it calls TAF-M) or TAF-R (original section
6.2, Epanechnikov kernel on a pairwise-ranking distance). With one objective
and ``delta = 1`` this is the original TAF; with ``delta = 0`` it is the
target's EI alone.

Everything is in maximization form and in raw output units: each source term
stays on its own source's scale, as in the original.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from openbo.acquisition.conbo import population_weight
from openbo.acquisition.ei import expected_improvement_maximization
from openbo.acquisition.taf import epanechnikov_weight

TAF_PLUS_WEIGHT_MODES = ("poe", "taf_r")


def decay_factor(t: int, d1: int, d2: float) -> float:
    """Return the population-weight decay ``delta(t)`` (CHI 2024 eq. 6).

    ``delta(t) = 1`` for ``t <= d1``, ``1 - (t - d1) * d2`` up to
    ``t = d1 + 1 / d2``, and 0 after that. ``t`` counts iterations from 1.
    The paper's study uses ``d1 = 2``, ``d2 = 0.3``.
    """
    if d1 < 0 or int(d1) != d1:
        raise ValueError("d1 must be a non-negative integer.")
    if not 0.0 < d2 <= 1.0:
        raise ValueError("d2 must be in (0, 1].")
    return population_weight(t, alpha1=float(d1), alpha2=float(d2))


def source_references(source_mean_at_target: ArrayLike) -> NDArray[np.float64]:
    """Return each source's reference ``ref_j^l`` from its means at the target's points.

    The original's ``y_{D_i}^min`` is the best value on source ``i`` among the
    configurations tried so far on the target. The sources have no data at
    those points, so their posterior mean stands in.

    Input shape ``(M, t, L)``; output shape ``(M, L)``.
    """
    mean = np.asarray(source_mean_at_target, dtype=np.float64)
    if mean.ndim != 3 or mean.shape[1] == 0:
        raise ValueError(
            "source_mean_at_target must have shape (M, t, L) with t >= 1."
        )
    return np.max(mean, axis=1).astype(np.float64)


def predicted_improvement(mean: ArrayLike, reference: ArrayLike) -> NDArray[np.float64]:
    """Return ``max(mean - reference, 0)``, the original's predicted improvement."""
    return np.maximum(
        np.asarray(mean, dtype=np.float64) - np.asarray(reference, dtype=np.float64),
        0.0,
    )


def poe_weights(variance: ArrayLike) -> NDArray[np.float64]:
    """Return TAF-PoE weights ``sigma^-2 / K`` for ``K`` models (original eqs. 38-39).

    ``variance`` has the model axis first, with every model of the combination
    (sources and target) included, so ``K = variance.shape[0]``.
    """
    var = np.asarray(variance, dtype=np.float64)
    if var.ndim < 1 or var.shape[0] == 0:
        raise ValueError("variance must have a leading model axis of length >= 1.")
    return (1.0 / np.maximum(var, 1e-300) / var.shape[0]).astype(np.float64)


def ranking_distance(y_target: ArrayLike, source_mean: ArrayLike) -> float:
    """Return the fraction of ordered pairs on which a source and the target disagree.

    Each model's descriptor holds ``1[f(x_i) > f(x_j)]`` for every ordered pair
    ``i != j`` of the target's evaluated points (original eq. 45): the target
    uses its observations, a source its posterior mean. The distance is the
    fraction of pairs whose indicators differ, a Kendall-tau distance. With
    fewer than two points there are no pairs, and the distance is 0.
    """
    y = np.asarray(y_target, dtype=np.float64).ravel()
    mu = np.asarray(source_mean, dtype=np.float64).ravel()
    if y.shape != mu.shape:
        raise ValueError("y_target and source_mean must have the same length.")
    t = y.shape[0]
    if t < 2:
        return 0.0
    target_gt = y[:, None] > y[None, :]
    source_gt = mu[:, None] > mu[None, :]
    off_diagonal = ~np.eye(t, dtype=bool)
    return float(np.sum((target_gt != source_gt) & off_diagonal) / (t * (t - 1)))


def taf_r_weights(
    y_target: ArrayLike,
    source_mean_at_target: ArrayLike,
    rho: float = 1.0,
) -> tuple[NDArray[np.float64], float]:
    """Return TAF-R weights: ``(source weights (M, L), target weight)``.

    Each source's weight per objective is the Epanechnikov kernel of its
    ranking distance to the target (original eqs. 41-43, 45). The target's own
    weight is the kernel at distance 0, ``3/4``.
    """
    y = np.asarray(y_target, dtype=np.float64)
    mean = np.asarray(source_mean_at_target, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("y_target must have shape (t, L).")
    if mean.ndim != 3 or mean.shape[1:] != y.shape:
        raise ValueError("source_mean_at_target must have shape (M, t, L).")
    n_sources, _, n_obj = mean.shape
    weights = np.empty((n_sources, n_obj), dtype=np.float64)
    for j in range(n_sources):
        for obj in range(n_obj):
            distance = ranking_distance(y[:, obj], mean[j, :, obj])
            weights[j, obj] = epanechnikov_weight(distance, rho)
    return weights, epanechnikov_weight(0.0, rho)


def population_mean(
    source_mean: ArrayLike,
    source_var: ArrayLike,
    objective_weights: ArrayLike,
    weight_mode: str = "poe",
) -> NDArray[np.float64]:
    """Return the population's combined prediction of the weighted objective, shape ``(n,)``.

    Used for the first step, before the new user has data and TAF is undefined.
    It is the original's product-of-experts mean (eq. 37, SGPT-PoE) over the
    sources only, ``sum_j w_j^+ mu_j^+ / sum_j w_j^+``, with the objectives
    combined as in CHI eqs. 4-5. Under ``"taf_r"`` no ranking exists yet, so
    every source gets the same weight and this is a plain average.
    """
    mean = np.asarray(source_mean, dtype=np.float64)
    var = np.asarray(source_var, dtype=np.float64)
    if mean.ndim != 3 or mean.shape[0] == 0 or var.shape != mean.shape:
        raise ValueError("source_mean and source_var must have shape (M, n, L) with M >= 1.")
    alpha = _check_objective_weights(objective_weights, mean.shape[2])
    if weight_mode == "poe":
        w_plus = poe_weights(var) @ alpha
    elif weight_mode == "taf_r":
        w_plus = np.ones(mean.shape[:2], dtype=np.float64)
    else:
        raise ValueError(f"weight_mode must be one of {TAF_PLUS_WEIGHT_MODES}.")
    mean_plus = mean @ alpha
    return (np.sum(w_plus * mean_plus, axis=0) / np.maximum(np.sum(w_plus, axis=0), 1e-300)).astype(np.float64)


def _check_objective_weights(alpha: ArrayLike, n_obj: int) -> NDArray[np.float64]:
    a = np.asarray(alpha, dtype=np.float64).ravel()
    if a.shape != (n_obj,):
        raise ValueError(f"objective_weights must have length {n_obj}.")
    if np.any(a < 0.0):
        raise ValueError("objective_weights must be non-negative.")
    if abs(float(a.sum()) - 1.0) > 1e-9:
        raise ValueError("objective_weights must sum to 1.")
    return a


def taf_plus_acquisition(
    target_mean: ArrayLike,
    target_var: ArrayLike,
    target_best: ArrayLike,
    source_mean: ArrayLike,
    source_var: ArrayLike,
    source_reference: ArrayLike,
    objective_weights: ArrayLike,
    decay: float = 1.0,
    weight_mode: str = "poe",
    ranking_weights: tuple[ArrayLike, float] | None = None,
) -> NDArray[np.float64]:
    """Return TAF+ at ``n`` candidates (CHI 2024 eq. 3).

    Shapes: ``target_mean``/``target_var`` ``(n, L)``, ``target_best`` ``(L,)``,
    ``source_mean``/``source_var`` ``(M, n, L)``, ``source_reference``
    ``(M, L)``, ``objective_weights`` ``(L,)``. ``M`` may be 0.

    ``weight_mode="poe"`` uses per-point inverse variances. ``"taf_r"`` takes
    the output of :func:`taf_r_weights` through ``ranking_weights``.
    """
    t_mean = np.asarray(target_mean, dtype=np.float64)
    t_var = np.asarray(target_var, dtype=np.float64)
    if t_mean.ndim != 2 or t_var.shape != t_mean.shape:
        raise ValueError("target_mean and target_var must have shape (n, L).")
    n, n_obj = t_mean.shape
    best = np.asarray(target_best, dtype=np.float64).ravel()
    if best.shape != (n_obj,):
        raise ValueError(f"target_best must have length {n_obj}.")
    s_mean = np.asarray(source_mean, dtype=np.float64).reshape(-1, n, n_obj)
    s_var = np.asarray(source_var, dtype=np.float64).reshape(-1, n, n_obj)
    if s_var.shape != s_mean.shape:
        raise ValueError("source_mean and source_var must have shape (M, n, L).")
    n_sources = s_mean.shape[0]
    ref = np.asarray(source_reference, dtype=np.float64).reshape(-1, n_obj)
    if ref.shape != (n_sources, n_obj):
        raise ValueError(f"source_reference must have shape ({n_sources}, {n_obj}).")
    alpha = _check_objective_weights(objective_weights, n_obj)
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be in [0, 1].")

    ei = np.stack(
        [
            expected_improvement_maximization(t_mean[:, obj], t_var[:, obj], float(best[obj]))
            for obj in range(n_obj)
        ],
        axis=1,
    )
    improvement = predicted_improvement(s_mean, ref[:, None, :])

    if weight_mode == "poe":
        all_var = np.concatenate([s_var, t_var[None]], axis=0)
        w = poe_weights(all_var)
        w_sources, w_target = w[:-1], w[-1]
    elif weight_mode == "taf_r":
        if ranking_weights is None:
            raise ValueError("weight_mode='taf_r' needs ranking_weights.")
        r_sources = np.asarray(ranking_weights[0], dtype=np.float64).reshape(-1, n_obj)
        if r_sources.shape != (n_sources, n_obj):
            raise ValueError(f"ranking_weights[0] must have shape ({n_sources}, {n_obj}).")
        w_sources = np.broadcast_to(r_sources[:, None, :], (n_sources, n, n_obj))
        w_target = np.full((n, n_obj), float(ranking_weights[1]))
    else:
        raise ValueError(f"weight_mode must be one of {TAF_PLUS_WEIGHT_MODES}.")

    # Within-model combination over objectives (CHI eqs. 4-5).
    ei_plus = ei @ alpha
    improvement_plus = improvement @ alpha
    w_target_plus = w_target @ alpha
    w_sources_plus = float(decay) * (w_sources @ alpha)

    # Between-model combination (CHI eq. 3, original eq. 46).
    numerator = w_target_plus * ei_plus + np.sum(w_sources_plus * improvement_plus, axis=0)
    denominator = w_target_plus + np.sum(w_sources_plus, axis=0)
    return (numerator / np.maximum(denominator, 1e-300)).astype(np.float64)
