"""TAF+ workflow steps 2 and 3 (Liao et al., CHI 2024, sections 4.4-4.5).

- Step 2, :func:`optimize_objective_weights`: Algorithm 1 (App. A.2). Each
  population user has rated their own Pareto-optimal designs. For every
  candidate weight vector, take each user's design with the highest weighted
  sum and add up its (rescaled) rating; return the best-scoring vector.
- Step 3, :func:`search_decay`: leave-one-out simulation. Each population user
  in turn plays the new user, with the posterior mean of their own GPs as the
  objective, and the others as the population. Every ``(d1, d2)`` pair on a
  grid is scored by the mean weighted value of the evaluated designs over users
  and iterations: what a user experiences at each iteration. (Best-so-far
  saturates after a few steps, before the decay can act, so pairs tie.)

Neither step exists in the original TAF, so both follow the CHI paper.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from openbo.optimizers.bo_taf_plus import (
    PopulationUser,
    TAFPlusConfig,
    TAFPlusPopulation,
    run_bo_taf_plus,
)

# The paper's grid (section 6.3).
PAPER_D1_VALUES = tuple(range(1, 10))
PAPER_D2_VALUES = (0.1, 0.2, 0.3)


def objective_weight_grid(n_objectives: int, step: float = 0.1) -> NDArray[np.float64]:
    """Return every weight vector with entries in multiples of `step`, all non-zero, summing to 1.

    Algorithm 1 uses "all possible non-zero weight combinations to the first
    decimal point", i.e. ``step = 0.1``. Shape ``(K, n_objectives)``.
    """
    if n_objectives < 1:
        raise ValueError("n_objectives must be at least 1.")
    n_steps = round(1.0 / step)
    if n_steps < n_objectives or abs(n_steps * step - 1.0) > 1e-9:
        raise ValueError("step must divide 1 into at least n_objectives parts.")
    rows = [
        combo
        for combo in itertools.product(range(1, n_steps + 1), repeat=n_objectives)
        if sum(combo) == n_steps
    ]
    return np.asarray(rows, dtype=np.float64) / n_steps


def normalize_ratings(ratings: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    """Rescale one user's ratings so the lowest is 1 and the highest 100 (Algorithm 1, line 8).

    If every rating is the same, all become 100. The algorithm divides by zero
    there; any constant leaves the argmax over weight vectors unchanged.
    """
    r = np.asarray(ratings, dtype=np.float64).ravel()
    if r.size == 0:
        raise ValueError("ratings must not be empty.")
    low, high = float(r.min()), float(r.max())
    if high - low <= 0.0:
        return np.full_like(r, 100.0)
    return 1.0 + 99.0 * (r - low) / (high - low)


@dataclass
class ObjectiveWeightResult:
    """Outcome of :func:`optimize_objective_weights`."""

    best: tuple[float, ...]
    candidates: NDArray[np.float64]
    scores: NDArray[np.float64]


def optimize_objective_weights(
    pareto_y: Sequence[NDArray[np.float64]],
    ratings: Sequence[Sequence[float] | NDArray[np.float64]],
    candidate_weights: NDArray[np.float64] | None = None,
    step: float = 0.1,
) -> ObjectiveWeightResult:
    """Return the objective weights whose weighted-sum choices earn the highest total rating.

    ``pareto_y[u]`` has shape ``(n_u, L)``: user ``u``'s Pareto-optimal
    objective values (maximized). ``ratings[u]`` has length ``n_u``. Ties
    between weight vectors, and between a user's designs, go to the first.
    """
    if len(pareto_y) != len(ratings) or not pareto_y:
        raise ValueError("pareto_y and ratings must be non-empty and the same length.")
    ys = [np.asarray(y, dtype=np.float64) for y in pareto_y]
    n_obj = ys[0].shape[1] if ys[0].ndim == 2 else 0
    for y, r in zip(ys, ratings):
        if y.ndim != 2 or y.shape[1] != n_obj or y.shape[0] == 0:
            raise ValueError("Each pareto_y entry must have shape (n_u, L) with n_u >= 1.")
        if len(np.ravel(r)) != y.shape[0]:
            raise ValueError("Each ratings entry must have one rating per design.")
    scaled = [normalize_ratings(r) for r in ratings]
    weights = (
        objective_weight_grid(n_obj, step)
        if candidate_weights is None
        else np.asarray(candidate_weights, dtype=np.float64)
    )
    if weights.ndim != 2 or weights.shape[1] != n_obj:
        raise ValueError(f"candidate_weights must have shape (K, {n_obj}).")
    scores = np.zeros(weights.shape[0], dtype=np.float64)
    for y, r in zip(ys, scaled):
        chosen = np.argmax(y @ weights.T, axis=0)
        scores += r[chosen]
    best = weights[int(np.argmax(scores))]
    return ObjectiveWeightResult(tuple(float(v) for v in best), weights, scores)


def population_user_objective(
    user: PopulationUser,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Return a simulated user: the posterior means of a population user's GPs, shape ``(n, L)``."""

    def objective(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return np.stack([gp.posterior(x)[0] for gp in user.gps], axis=1)

    return objective


def leave_one_out_curves(
    population: TAFPlusPopulation,
    config: TAFPlusConfig,
    n_iter: int = 10,
) -> NDArray[np.float64]:
    """Run TAF+ once per population user, with the others as the population.

    Returns the weighted value ``f(x_t) @ alpha`` of each evaluated design,
    shape ``(n_users, n_iter)``. The objective is a GP mean, so it is noise-free;
    ``np.maximum.accumulate`` along the last axis gives the best-so-far.
    """
    if population.n_users < 2:
        raise ValueError("Leave-one-out needs at least two population users.")
    curves = []
    for user in population.users:
        result = run_bo_taf_plus(
            population_user_objective(user), config, population.subset(user.name), n_iter
        )
        curves.append(result.weighted_y)
    return np.asarray(curves, dtype=np.float64)


def _curves_for_pair(
    args: tuple[TAFPlusPopulation, TAFPlusConfig, int],
) -> NDArray[np.float64]:
    population, config, n_iter = args
    return leave_one_out_curves(population, config, n_iter)


@dataclass
class DecaySearchResult:
    """Outcome of :func:`search_decay`. ``scores[i, k]`` belongs to ``(d1_values[i], d2_values[k])``."""

    best: tuple[int, float]
    d1_values: tuple[int, ...]
    d2_values: tuple[float, ...]
    scores: NDArray[np.float64]
    curves: NDArray[np.float64]  # (len(d1), len(d2), n_users, n_iter), per-iteration values
    no_decay_curves: NDArray[np.float64]  # (n_users, n_iter), TAF without decay


def search_decay(
    population: TAFPlusPopulation,
    config: TAFPlusConfig,
    d1_values: Sequence[int] = PAPER_D1_VALUES,
    d2_values: Sequence[float] = PAPER_D2_VALUES,
    n_iter: int = 10,
    n_workers: int = 1,
) -> DecaySearchResult:
    """Grid-search the decay ``(d1, d2)`` by leave-one-out simulation (CHI section 4.5).

    A pair's score is the mean weighted value of the evaluated designs over all
    users and iterations (the area under per-iteration curves like the paper's
    Fig. 7; the paper does not say whether it plots per-iteration or
    best-so-far values). Ties go to
    the first pair in ``d1``-major order. ``config`` supplies everything but
    ``decay``, ``d1`` and ``d2``. Also runs the no-decay baseline for reference.
    """
    d1s = tuple(int(v) for v in d1_values)
    d2s = tuple(float(v) for v in d2_values)
    if not d1s or not d2s:
        raise ValueError("d1_values and d2_values must not be empty.")
    jobs = [(population, replace(config, decay=True, d1=d1, d2=d2), n_iter) for d1 in d1s for d2 in d2s]
    jobs.append((population, replace(config, decay=False), n_iter))
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            all_curves = list(pool.map(_curves_for_pair, jobs))
    else:
        all_curves = [_curves_for_pair(job) for job in jobs]
    no_decay = all_curves.pop()
    curves = np.asarray(all_curves, dtype=np.float64).reshape(len(d1s), len(d2s), *no_decay.shape)
    scores = curves.mean(axis=(2, 3))
    i, k = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return DecaySearchResult((d1s[i], d2s[k]), d1s, d2s, scores, curves, no_decay)
