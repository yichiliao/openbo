"""Generative replay of stored user GPs for ConBO's population model.

Follows Liao et al., *Continual Human-in-the-Loop Optimization* (CHI 2025),
Section 4: after each user, every stored user GP is queried on a set of replay
inputs, predictions whose variance is not below a threshold ``lambda`` are
dropped, and the BNN is trained to match the per-input average of the remaining
means and variances (eq. 7, :meth:`PopulationBNN.fit_replay`).

``lambda`` is in raw output units, as in the paper (5 in the user study, 50
for Branin, 10 for McCormick). The paper tuned it per problem, and no single
scale-free value reproduces both simulated settings (``docs/plans/conbo.md``,
Phase 3). :func:`standardized_variance_threshold` converts a threshold given
relative to the spread of the replayed means, for problems without a tuned value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc

from openbo.models.gp_scratch import GPScratch


@dataclass(frozen=True)
class ReplayTargets:
    """Filtered replay data for :meth:`PopulationBNN.fit_replay`.

    Attributes:
        x: Replay inputs where at least one user passed the filter, shape `(m, d)`.
        mean: Average predicted mean over the qualifying users, shape `(m,)`.
        var: Average predicted variance over the qualifying users, shape `(m,)`.
        n_users: Number of qualifying users per input, shape `(m,)`.
        kept_fraction: Fraction of all replay inputs kept for each user, shape `(U,)`.
        excluded_users: Indices of users that passed the filter nowhere.
    """

    x: NDArray[np.float64]
    mean: NDArray[np.float64]
    var: NDArray[np.float64]
    n_users: NDArray[np.int64]
    kept_fraction: NDArray[np.float64]
    excluded_users: tuple[int, ...]


def replay_inputs(
    dim: int,
    grid_points_per_dim: int = 30,
    n_random: int = 50,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Return replay inputs in `[0, 1]^dim`: a grid plus uniform random points.

    The paper uses a 30 x 30 grid plus 50 random points in its simulations
    (20 x 20 + 100 in the user study). Above two dimensions a full grid is too
    large, so a scrambled Sobol set of ``grid_points_per_dim ** 2`` points
    replaces it.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if grid_points_per_dim < 2:
        raise ValueError("grid_points_per_dim must be at least 2.")
    if n_random < 0:
        raise ValueError("n_random must be non-negative.")
    if dim <= 2:
        axis = np.linspace(0.0, 1.0, grid_points_per_dim)
        mesh = np.meshgrid(*([axis] * dim), indexing="ij")
        base = np.stack([m.ravel() for m in mesh], axis=1)
    else:
        base = qmc.Sobol(d=dim, scramble=True, seed=seed).random(grid_points_per_dim**2)
    random = np.random.default_rng(seed).random((n_random, dim))
    return np.vstack([base, random]).astype(np.float64)


def _predict_all(
    user_gps: Sequence[GPScratch], x: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Stack every user's posterior mean and variance at `x`, shape `(U, n)`."""
    if len(user_gps) == 0:
        raise ValueError("Replay needs at least one stored user GP.")
    preds = [gp.posterior(x) for gp in user_gps]
    return np.stack([m for m, _ in preds]), np.stack([v for _, v in preds])


def _as_inputs(x: ArrayLike) -> NDArray[np.float64]:
    """Return replay inputs as a float array of shape `(n, d)`."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"x must have shape (n, d) with n > 0, got {arr.shape}.")
    return arr


def standardized_variance_threshold(
    user_gps: Sequence[GPScratch],
    x: ArrayLike,
    variance_fraction: float,
) -> float:
    """Convert a standardized threshold into raw units.

    Returns ``variance_fraction * Var(all users' predicted means at x)``, i.e. a
    threshold expressed in the same standardized units the BNN trains in.
    """
    if variance_fraction <= 0.0:
        raise ValueError("variance_fraction must be positive.")
    means, _ = _predict_all(user_gps, _as_inputs(x))
    spread = float(np.var(means))
    return variance_fraction * (spread if spread > 1e-12 else 1.0)


def replay_targets(
    user_gps: Sequence[GPScratch],
    x: ArrayLike,
    max_variance: float | None,
) -> ReplayTargets:
    """Query stored user GPs at `x`, filter by variance and average (paper eq. 7).

    A pair (user ``i``, input ``s``) qualifies when ``var_i(s) < max_variance``.
    Each kept input's targets are the averages over its qualifying users;
    inputs with no qualifying user are dropped. ``max_variance=None`` disables
    the filter (the paper's "without variance filter" ablation).
    """
    if max_variance is not None and max_variance <= 0.0:
        raise ValueError("max_variance must be positive or None.")
    x_arr = _as_inputs(x)
    means, variances = _predict_all(user_gps, x_arr)
    if max_variance is None:
        qualifies = np.ones_like(variances, dtype=bool)
    else:
        qualifies = variances < max_variance

    n_users = qualifies.sum(axis=0)
    keep = n_users > 0
    counts = n_users[keep]
    mean = np.where(qualifies, means, 0.0).sum(axis=0)[keep] / counts
    var = np.where(qualifies, variances, 0.0).sum(axis=0)[keep] / counts
    kept_fraction = qualifies.mean(axis=1)
    excluded = tuple(int(i) for i in np.flatnonzero(kept_fraction == 0.0))
    return ReplayTargets(
        x=x_arr[keep],
        mean=mean.astype(np.float64),
        var=var.astype(np.float64),
        n_users=counts.astype(np.int64),
        kept_fraction=kept_fraction.astype(np.float64),
        excluded_users=excluded,
    )
