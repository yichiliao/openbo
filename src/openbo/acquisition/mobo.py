"""Expected increase in Pareto volume (EIPV) acquisitions for multi-objective BO.

All objectives are maximized. The objective space between ``v_ref`` and
``w_ref`` is split into front-aligned cells (see
:mod:`openbo.benchmarks.hypervolume`), and a new observation ``y`` gains

    gain(y) = sum_C prod_l g_{C,l}(y_l),   g(y) = (min(y, w_C) - v_C)_+

over the non-dominated cells ``C``. EIPV is the expectation of this gain under
the predictive distribution ``N(mean, cov)`` of the objectives at ``x``.

Shah and Ghahramani (2016) write the integrand as ``(y - v_C)`` restricted to
the cell that contains ``y`` (eq. 3). Taken literally that credits only the
part of the gain inside one cell and badly undercounts it, so the default
``form="exact"`` uses the full gain above. ``form="paper"`` keeps the literal
version for comparison.

* :func:`eipv_mc` estimates EIPV by Monte Carlo (ground truth for tests).
* :func:`ieipv` is exact for independent objectives (it ignores the
  off-diagonal of ``cov``).
* :func:`ceipv` is the paper's approximation for correlated objectives. Each
  factor of the gain is replaced by a scaled Gaussian with the same mass,
  mean and variance, so every cell term becomes a Gaussian density
  ``N(lambda; mean, cov + diag(tau^2))`` in closed form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp, ndtr

from openbo.benchmarks.hypervolume import cell_grid, hypervolume_gain

EIPVForm = Literal["exact", "paper"]

# Largest number of (candidate, term) or (sample, cell) pairs held at once.
_CHUNK_BUDGET = 1 << 21
# Above this fraction of kept (candidate, term) pairs, CEIPV evaluates densely.
_DENSE_FRACTION = 0.25


def _as_predictive(
    mean: ArrayLike, cov: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return mean ``(m, L)`` and covariance ``(m, L, L)`` arrays."""
    mean_arr = np.asarray(mean, dtype=np.float64)
    cov_arr = np.asarray(cov, dtype=np.float64)
    if mean_arr.ndim == 1:
        mean_arr = mean_arr[None, :]
    if cov_arr.ndim == 2:
        cov_arr = cov_arr[None, :, :]
    m, n_objectives = mean_arr.shape
    if cov_arr.shape != (m, n_objectives, n_objectives):
        raise ValueError(
            f"cov must have shape {(m, n_objectives, n_objectives)}, got {cov_arr.shape}."
        )
    return mean_arr, cov_arr


def _as_cells(
    lower: ArrayLike, upper: ArrayLike, n_objectives: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return cell corner arrays of shape ``(K, L)``."""
    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)
    if lower_arr.ndim != 2 or lower_arr.shape != upper_arr.shape:
        raise ValueError("lower and upper must both have shape (K, L).")
    if lower_arr.shape[1] != n_objectives:
        raise ValueError(
            f"Cells have {lower_arr.shape[1]} objectives, predictions have {n_objectives}."
        )
    if not np.all(np.isfinite(lower_arr)) or not np.all(np.isfinite(upper_arr)):
        raise ValueError("Cell corners must be finite.")
    return lower_arr, upper_arr


def _check_form(form: str) -> None:
    """Raise if ``form`` is not a known EIPV form."""
    if form not in ("exact", "paper"):
        raise ValueError(f"Unknown EIPV form {form!r}; expected 'exact' or 'paper'.")


def _paper_gain(
    y: NDArray[np.float64], lower: NDArray[np.float64], upper: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the paper's literal integrand ``prod_l (y_l - v_C,l)`` in the cell holding ``y``."""
    inside = np.all((y[:, None, :] > lower[None]) & (y[:, None, :] <= upper[None]), axis=2)
    return np.sum(np.prod(y[:, None, :] - lower[None], axis=2) * inside, axis=1)


def _covariance_sqrt(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``S`` with ``S S^T = cov`` for a batch of PSD matrices, even if singular."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs * np.sqrt(np.maximum(eigvals, 0.0))[:, None, :]


def eipv_mc(
    mean: ArrayLike,
    cov: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    n_samples: int = 4096,
    rng: np.random.Generator | None = None,
    form: EIPVForm = "exact",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Monte Carlo EIPV estimates and their standard errors, both ``(m,)``.

    Every candidate uses the same standard normal draws (common random
    numbers), which keeps comparisons between candidates low-variance.
    """
    _check_form(form)
    mean_arr, cov_arr = _as_predictive(mean, cov)
    m, n_objectives = mean_arr.shape
    lower_arr, upper_arr = _as_cells(lower, upper, n_objectives)
    generator = rng if rng is not None else np.random.default_rng()
    eps = generator.standard_normal((n_samples, n_objectives))
    sqrt_cov = _covariance_sqrt(cov_arr)

    gain_fn = hypervolume_gain if form == "exact" else _paper_gain
    chunk = max(1, _CHUNK_BUDGET // max(1, lower_arr.shape[0]))
    estimate = np.zeros(m, dtype=np.float64)
    stderr = np.zeros(m, dtype=np.float64)
    for i in range(m):
        y = mean_arr[i] + eps @ sqrt_cov[i].T
        gains = np.concatenate(
            [gain_fn(y[s : s + chunk], lower_arr, upper_arr) for s in range(0, n_samples, chunk)]
        )
        estimate[i] = gains.mean()
        stderr[i] = gains.std(ddof=1) / math.sqrt(n_samples) if n_samples > 1 else math.inf
    return estimate, stderr


def _partial_expectation(
    mean: NDArray[np.float64], std: NDArray[np.float64], a: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``E[(y - a)_+]`` for ``y ~ N(mean, std^2)``."""
    z = (mean - a) / std
    return std * np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi) + (mean - a) * ndtr(z)


def ieipv(
    mean: ArrayLike,
    cov: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    form: EIPVForm = "exact",
) -> NDArray[np.float64]:
    """Return EIPV in closed form, treating the objectives as independent.

    Only the diagonal of ``cov`` is used. For a diagonal ``cov`` the result is
    exact, so this is both the independent-objective acquisition and the
    correlation ablation for :func:`ceipv`.
    """
    _check_form(form)
    mean_arr, cov_arr = _as_predictive(mean, cov)
    n_objectives = mean_arr.shape[1]
    lower_arr, upper_arr = _as_cells(lower, upper, n_objectives)
    std = np.sqrt(np.maximum(np.diagonal(cov_arr, axis1=1, axis2=2), 1e-300))

    chunk = max(1, _CHUNK_BUDGET // max(1, lower_arr.shape[0]))
    out = np.empty(mean_arr.shape[0], dtype=np.float64)
    for s in range(0, mean_arr.shape[0], chunk):
        mu = mean_arr[s : s + chunk, None, :]
        sd = std[s : s + chunk, None, :]
        # E[(min(y, w) - v)_+] = E[(y - v)_+] - E[(y - w)_+]
        factors = _partial_expectation(mu, sd, lower_arr[None]) - _partial_expectation(
            mu, sd, upper_arr[None]
        )
        if form == "paper":
            # Drop the mass above w: E[(y - v) 1[v < y <= w]].
            factors = factors - (upper_arr - lower_arr)[None] * ndtr((mu - upper_arr[None]) / sd)
        out[s : s + chunk] = np.sum(np.prod(np.maximum(factors, 0.0), axis=2), axis=1)
    return out


@dataclass(frozen=True)
class CEIPVTerms:
    """Gaussian terms of the CEIPV approximation; they depend only on the cells.

    The approximate EIPV is ``sum_t exp(log_weight[t]) * N(center[t]; mean, cov + diag(tau2[t]))``.
    """

    log_weight: NDArray[np.float64]
    center: NDArray[np.float64]
    tau2: NDArray[np.float64]


def ceipv_terms(
    front: ArrayLike,
    v_ref: ArrayLike,
    w_ref: ArrayLike,
    *,
    form: EIPVForm = "exact",
    overflow: float | tuple[float, ...] = (0.1, 0.25, 0.5),
) -> CEIPVTerms:
    """Moment-match the gain inside every grid cell with scaled Gaussians.

    For ``y`` in grid cell ``j``, a non-dominated cell ``i <= j`` contributes
    ``prod_l`` of ``(y_l - v_j,l)`` where ``i_l = j_l`` and of the width of
    ``i`` where ``i_l < j_l``. Grouping by the set ``S`` of objectives with
    ``i_l = j_l`` gives

        gain(y) 1[y in j] = sum_S c_S(j) prod_{l in S} ramp_j,l(y_l) prod_{l not in S} box_j,l(y_l),

    where ``ramp = (y - v) 1[v < y <= w]`` and ``box = 1[v < y <= w]`` on cell
    ``j``, and ``c_S(j)`` comes from exclusive prefix sums of the
    non-dominated mask. Every factor is local to one cell and is replaced by a
    Gaussian with the same mass, mean and variance:

    * ramp of width ``d``: ``(d^2 / 2) N(y; (2w + v) / 3, d^2 / 18)``. The paper
      prints the mean as ``(2w + v) / 2``, which is not the ramp's mean (and
      not shift-equivariant).
    * box of width ``d``: ``d N(y; (v + w) / 2, d^2 / 12)``.

    ``form="paper"`` keeps only ``S = all objectives`` (the paper's eq. 3).
    ``overflow`` adds a layer of width ``overflow * (w_ref - v_ref)`` above
    ``w_ref`` in each objective, which credits predictions beyond ``w_ref``
    with the gain capped at ``w_ref``; ``0`` disables it.
    """
    _check_form(form)
    layers = np.atleast_1d(np.asarray(overflow, dtype=np.float64))
    if np.any(layers < 0.0) or np.any(np.diff(layers) <= 0.0):
        raise ValueError("overflow must be non-negative and strictly increasing.")
    layers = layers[layers > 0.0]
    coords, nondominated = cell_grid(front, v_ref, w_ref)
    n_objectives = len(coords)
    if nondominated.size == 0 or not np.any(nondominated):
        empty = np.empty((0, n_objectives), dtype=np.float64)
        return CEIPVTerms(np.empty(0, dtype=np.float64), empty, empty.copy())

    if layers.size and form == "exact":
        coords = [np.append(c, c[-1] + layers * (c[-1] - c[0])) for c in coords]
        nondominated = np.pad(nondominated, [(0, layers.size)] * n_objectives)
    widths = [np.diff(c) for c in coords]
    shape = nondominated.shape

    def along(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
        """Broadcast a per-interval vector along ``axis`` of the grid."""
        view = [1] * n_objectives
        view[axis] = -1
        return values.reshape(view)

    patterns = [(True,) * n_objectives]
    if form == "exact":
        patterns = list(product((True, False), repeat=n_objectives))

    weights, centers, taus = [], [], []
    for pattern in patterns:
        coef = nondominated.astype(np.float64)
        for axis, is_ramp in enumerate(pattern):
            if not is_ramp:
                # Sum over cells strictly below j along this axis, each with its full width.
                coef = coef * along(widths[axis], axis)
                coef = np.concatenate(
                    [np.zeros_like(np.take(coef, [0], axis=axis)), np.cumsum(coef, axis=axis)],
                    axis=axis,
                )
                coef = np.take(coef, np.arange(shape[axis]), axis=axis)
        idx = np.nonzero(coef > 0.0)
        if idx[0].size == 0:
            continue
        log_weight = np.log(coef[idx])
        center = np.empty((idx[0].size, n_objectives))
        tau2 = np.empty((idx[0].size, n_objectives))
        for axis, is_ramp in enumerate(pattern):
            lo = coords[axis][idx[axis]]
            width = widths[axis][idx[axis]]
            if is_ramp:
                log_weight += np.log(0.5 * width**2)
                center[:, axis] = lo + 2.0 * width / 3.0
                tau2[:, axis] = width**2 / 18.0
            else:
                log_weight += np.log(width)
                center[:, axis] = lo + 0.5 * width
                tau2[:, axis] = width**2 / 12.0
        weights.append(log_weight)
        centers.append(center)
        taus.append(tau2)
    return CEIPVTerms(np.concatenate(weights), np.concatenate(centers), np.concatenate(taus))




def _log_gaussian_density(
    diff: NDArray[np.float64], cov: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``log N(diff; 0, cov)`` over leading batch dimensions.

    Uses explicit formulas for one to three objectives, which are much faster
    than batched LAPACK calls on tiny matrices.
    """
    n_objectives = diff.shape[-1]
    if n_objectives == 1:
        det = cov[..., 0, 0]
        quad = diff[..., 0] ** 2 / det
    elif n_objectives == 2:
        a, b, d = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
        det = a * d - b * b
        x0, x1 = diff[..., 0], diff[..., 1]
        quad = (d * x0 * x0 - 2.0 * b * x0 * x1 + a * x1 * x1) / det
    elif n_objectives == 3:
        a, b, c = cov[..., 0, 0], cov[..., 0, 1], cov[..., 0, 2]
        d, e, f = cov[..., 1, 1], cov[..., 1, 2], cov[..., 2, 2]
        adj00, adj01, adj02 = d * f - e * e, c * e - b * f, b * e - c * d
        adj11, adj12, adj22 = a * f - c * c, b * c - a * e, a * d - b * b
        det = a * adj00 + b * adj01 + c * adj02
        x0, x1, x2 = diff[..., 0], diff[..., 1], diff[..., 2]
        quad = (
            adj00 * x0 * x0
            + adj11 * x1 * x1
            + adj22 * x2 * x2
            + 2.0 * (adj01 * x0 * x1 + adj02 * x0 * x2 + adj12 * x1 * x2)
        ) / det
    else:
        chol = np.linalg.cholesky(cov)
        sol = np.linalg.solve(chol, diff[..., None])[..., 0]
        quad = np.sum(sol**2, axis=-1)
        det = np.prod(np.diagonal(chol, axis1=-2, axis2=-1), axis=-1) ** 2
    return -0.5 * (quad + np.log(det) + n_objectives * math.log(2.0 * math.pi))


def log_ceipv(
    mean: ArrayLike,
    cov: ArrayLike,
    terms: CEIPVTerms,
    *,
    prune_z: float | None = 7.0,
) -> NDArray[np.float64]:
    """Return ``log CEIPV`` for each candidate, shape ``(m,)``.

    Terms are local to one cell, so most are negligible for a given candidate.
    With ``prune_z`` set, a term is skipped when its center lies more than
    ``prune_z`` marginal standard deviations from the candidate's mean in some
    objective. Adding ``diag(tau^2)`` keeps every conditional variance at
    least ``tau^2``, so a skipped term is below ``exp(-prune_z^2 / 2)`` times
    its peak. The nearest term is always kept, so far from the front the log
    value stays finite and informative (it is then a lower bound). Pass
    ``prune_z=None`` to sum every term. The result is ``-inf`` only when there
    are no terms.
    """
    mean_arr, cov_arr = _as_predictive(mean, cov)
    m, n_objectives = mean_arr.shape
    if terms.center.ndim != 2 or terms.center.shape[1] != n_objectives:
        raise ValueError("CEIPV terms and predictions disagree on the number of objectives.")
    n_terms = terms.center.shape[0]
    out = np.full(m, -np.inf, dtype=np.float64)
    if n_terms == 0:
        return out

    term_chunk = min(n_terms, _CHUNK_BUDGET)
    cand_chunk = max(1, _CHUNK_BUDGET // term_chunk)
    marginal_var = np.diagonal(cov_arr, axis1=1, axis2=2)
    for t in range(0, n_terms, term_chunk):
        center = terms.center[t : t + term_chunk]
        tau2 = terms.tau2[t : t + term_chunk]
        log_weight = terms.log_weight[t : t + term_chunk]
        for s in range(0, m, cand_chunk):
            mu = mean_arr[s : s + cand_chunk]
            diff = center[None] - mu[:, None, :]
            keep = None
            if prune_z is not None:
                z2 = np.max(diff**2 / (marginal_var[s : s + cand_chunk, None, :] + tau2[None]), axis=2)
                keep = z2 <= prune_z**2
                keep[np.arange(z2.shape[0]), np.argmin(z2, axis=1)] = True
            rows = np.arange(s, s + mu.shape[0])
            if keep is None or keep.mean() > _DENSE_FRACTION:
                # Most terms matter: broadcasting beats gathering the kept pairs.
                total_cov = cov_arr[rows, None] + tau2[None, :, :, None] * np.eye(n_objectives)
                log_terms = log_weight[None] + _log_gaussian_density(diff, total_cov)
                if keep is not None:
                    log_terms = np.where(keep, log_terms, -np.inf)
                chunk_log = logsumexp(log_terms, axis=1)
            else:
                cand_idx, term_idx = np.nonzero(keep)
                total_cov = cov_arr[s + cand_idx].copy()
                diag = np.arange(n_objectives)
                total_cov[:, diag, diag] += tau2[term_idx]
                flat = log_weight[term_idx] + _log_gaussian_density(
                    diff[cand_idx, term_idx], total_cov
                )
                # cand_idx is sorted and every candidate keeps at least one term,
                # so each candidate's terms form one contiguous run.
                starts = np.flatnonzero(np.r_[True, cand_idx[1:] != cand_idx[:-1]])
                peak = np.maximum.reduceat(flat, starts)
                run_peak = np.repeat(peak, np.diff(np.r_[starts, cand_idx.size]))
                chunk_log = peak + np.log(np.add.reduceat(np.exp(flat - run_peak), starts))
            out[rows] = np.logaddexp(out[rows], chunk_log)
    return out


def ceipv(
    mean: ArrayLike,
    cov: ArrayLike,
    terms: CEIPVTerms,
    *,
    prune_z: float | None = 7.0,
) -> NDArray[np.float64]:
    """Return the paper's correlated EIPV approximation, shape ``(m,)``."""
    return np.exp(log_ceipv(mean, cov, terms, prune_z=prune_z))
