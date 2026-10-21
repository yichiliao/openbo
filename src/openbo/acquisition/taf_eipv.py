"""TAF-EIPV: the original TAF's acquisition in hypervolume units (``docs/plans/taf_eipv.md``).

The original TAF (Wistuba, Schilling and Schmidt-Thieme, Machine Learning 2018,
eqs. 46-47) blends the target's EI with each source's predicted improvement:

    a(x) = (w_t * EI_t(x) + sum_i w_i * I_i(x)) / (w_t + sum_i w_i).

Here the target term is the expected increase in Pareto volume (IEIPV or CEIPV)
and ``I_i(x)`` is the hypervolume that source ``i``'s posterior mean adds to a
reference front: the front of what source ``i`` predicts at the designs the
target has already evaluated (the original's "best value achieved so far on
D_i"). With no target data the reference is empty. All objectives are
maximized.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from openbo.acquisition.taf import epanechnikov_weight
from openbo.acquisition.taf_plus import ranking_distance
from openbo.benchmarks.hypervolume import hypervolume_gain, nondominated_cells

# Candidates scored per chunk, so (candidates x cells x objectives) stays small.
_CHUNK = 256
# log-acquisition floor, as in optimizers/mobo.py.
LOG_FLOOR = -1e4


def reference_cells(
    reference_front: ArrayLike,
    v_ref: ArrayLike,
    w_ref: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cells of ``[v_ref, w_ref]`` not dominated by ``reference_front``: ``(lower, upper)``, each ``(K, L)``.

    An empty front (shape ``(0, L)``) leaves the whole box open.
    """
    v = np.asarray(v_ref, dtype=np.float64).reshape(-1)
    w = np.asarray(w_ref, dtype=np.float64).reshape(-1)
    if v.shape != w.shape:
        raise ValueError("v_ref and w_ref must have one value per objective.")
    front = np.asarray(reference_front, dtype=np.float64).reshape(-1, v.shape[0])
    if front.shape[0] == 0:
        return v[None, :], w[None, :]
    return nondominated_cells(front, v, w)


def source_improvement(
    mean: ArrayLike,
    cells: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Hypervolume each row of ``mean`` adds to a reference front, shape ``(n,)``.

    ``cells`` comes from :func:`reference_cells`. Exact for points below
    ``w_ref``; above it the gain is capped at ``w_ref``.
    """
    mean_arr = np.atleast_2d(np.asarray(mean, dtype=np.float64))
    lower, upper = cells
    out = np.empty(mean_arr.shape[0], dtype=np.float64)
    for s in range(0, mean_arr.shape[0], _CHUNK):
        out[s : s + _CHUNK] = hypervolume_gain(mean_arr[s : s + _CHUNK], lower, upper)
    return out


def taf_r_source_weights(
    y_target: ArrayLike,
    source_mean_at_target: ArrayLike,
    rho: float = 1.0,
) -> NDArray[np.float64]:
    """TAF-R weights, one per source, shape ``(M,)``.

    Source ``i``'s distance is the fraction of ordered pairs of the target's
    observations that its posterior mean ranks differently, pooled over the
    objectives (the mean of the per-objective ``ranking_distance``); its weight
    is the Epanechnikov kernel ``gamma(d / rho)``. Not normalized, as in the
    original. With fewer than two observations every source gets ``3/4``.
    """
    y = np.asarray(y_target, dtype=np.float64)
    mean = np.asarray(source_mean_at_target, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("y_target must have shape (t, L).")
    if mean.ndim != 3 or mean.shape[1:] != y.shape:
        raise ValueError("source_mean_at_target must have shape (M, t, L).")
    weights = np.empty(mean.shape[0], dtype=np.float64)
    for i in range(mean.shape[0]):
        distance = float(np.mean([ranking_distance(y[:, l], mean[i, :, l]) for l in range(y.shape[1])]))
        weights[i] = epanechnikov_weight(distance, rho)
    return weights


def log_taf_eipv(
    log_target: ArrayLike | None,
    source_terms: ArrayLike,
    source_weights: ArrayLike,
    target_weight: float,
) -> NDArray[np.float64]:
    """``log a(x)`` of the TAF blend, floored at ``LOG_FLOOR``, shape ``(n,)``.

    ``log_target`` is the log target acquisition per candidate, or ``None``
    before the target has data (then the target leaves the blend). Source terms
    ``(M, n)`` and weights ``(M,)``; sources with weight 0 leave both the
    numerator and the denominator. Zero source terms add nothing.
    """
    terms = np.asarray(source_terms, dtype=np.float64)
    weights = np.asarray(source_weights, dtype=np.float64).reshape(-1)
    if terms.ndim != 2 or terms.shape[0] != weights.shape[0]:
        raise ValueError("source_terms must have shape (M, n) matching source_weights (M,).")
    if np.any(weights < 0.0) or target_weight < 0.0:
        raise ValueError("weights must be non-negative.")
    active = weights > 0.0
    pieces = []
    denominator = float(np.sum(weights[active]))
    if log_target is not None and target_weight > 0.0:
        pieces.append(np.log(target_weight) + np.asarray(log_target, dtype=np.float64))
        denominator += target_weight
    if np.any(active):
        with np.errstate(divide="ignore"):
            pieces.extend(np.log(weights[active])[:, None] + np.log(terms[active]))
    if not pieces or denominator <= 0.0:
        raise ValueError("TAF-EIPV needs a target term or a source with positive weight.")
    stacked = np.vstack([np.atleast_2d(p) for p in pieces])
    return np.maximum(logsumexp(stacked, axis=0) - np.log(denominator), LOG_FLOOR)
