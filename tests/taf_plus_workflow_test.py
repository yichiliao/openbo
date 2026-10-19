"""Tests for TAF+ workflow steps 2 (objective weights) and 3 (decay search)."""

from __future__ import annotations

import numpy as np
import pytest

from openbo.optimizers.bo_taf_plus import TAFPlusConfig, TAFPlusPopulation
from openbo.optimizers.taf_plus_workflow import (
    normalize_ratings,
    objective_weight_grid,
    optimize_objective_weights,
    population_user_objective,
    search_decay,
)


def test_weight_grid_is_all_nonzero_tenths_summing_to_one() -> None:
    grid = objective_weight_grid(3)
    assert grid.shape == (36, 3)  # compositions of 10 into 3 positive parts: C(9, 2)
    np.testing.assert_allclose(grid.sum(axis=1), 1.0)
    assert np.all(grid >= 0.1 - 1e-12)
    assert len({tuple(np.round(r, 6)) for r in grid}) == 36
    np.testing.assert_allclose(objective_weight_grid(1), [[1.0]])
    with pytest.raises(ValueError, match="step"):
        objective_weight_grid(3, step=0.3)


def test_ratings_are_rescaled_to_1_100() -> None:
    np.testing.assert_allclose(normalize_ratings([20, 60, 40]), [1.0, 100.0, 50.5])
    np.testing.assert_allclose(normalize_ratings([7, 7]), [100.0, 100.0])


def test_paper_table_1_picks_the_second_weight_vector() -> None:
    # CHI 2024 Table 1: designs A, B, C with ratings 75, 100, 1.
    pareto = [np.array([[7, 2, 5], [5, 4, 8], [3, 9, 2]], dtype=float)]
    candidates = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5], [0.2, 0.6, 0.2]])
    result = optimize_objective_weights(pareto, [[75, 100, 1]], candidate_weights=candidates)
    assert result.best == (0.2, 0.3, 0.5)
    # Rescaled ratings: A = 1 + 99 * 74/99 = 75, B = 100, C = 1.
    np.testing.assert_allclose(result.scores, [75.0, 100.0, 1.0])


def test_population_scores_add_up_across_users() -> None:
    # User 1 prefers their objective-1 design, user 2 their objective-2 design,
    # and user 2 cares more (the other design is rated lowest for both).
    pareto = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]
    ratings = [[100, 1], [1, 100], [1, 100]]
    result = optimize_objective_weights(pareto, ratings)
    assert result.best[1] > result.best[0]
    assert result.scores.max() == pytest.approx(201.0)


def test_objective_weight_validation() -> None:
    with pytest.raises(ValueError, match="same length"):
        optimize_objective_weights([np.ones((2, 2))], [])
    with pytest.raises(ValueError, match="one rating per design"):
        optimize_objective_weights([np.ones((2, 2))], [[1.0]])
    with pytest.raises(ValueError, match="candidate_weights"):
        optimize_objective_weights([np.ones((2, 2))], [[1.0, 2.0]], candidate_weights=np.ones((3, 3)) / 3)


@pytest.fixture(scope="module")
def population() -> TAFPlusPopulation:
    population = TAFPlusPopulation(input_dim=2, n_objectives=2)
    rng = np.random.default_rng(0)
    for i, shift in enumerate((0.0, 0.05, -0.05)):
        x = rng.random((12, 2))
        y = np.stack([-np.sum((x - (0.3 + shift)) ** 2, axis=1), -np.sum((x - (0.7 + shift)) ** 2, axis=1)], axis=1)
        population.add_user(f"u{i}", x, y, seed=i)
    return population


def test_population_user_objective_is_the_gp_mean(population: TAFPlusPopulation) -> None:
    user = population.users[0]
    x = np.array([[0.2, 0.4], [0.9, 0.1]])
    expected = np.stack([gp.posterior(x)[0] for gp in user.gps], axis=1)
    np.testing.assert_allclose(population_user_objective(user)(x), expected)


def test_search_decay_shapes_and_parallel_matches_serial(population: TAFPlusPopulation) -> None:
    config = TAFPlusConfig(input_dim=2, objective_weights=(0.5, 0.5), n_candidates=64, refine_suggestion=False)
    serial = search_decay(population, config, d1_values=(1, 2), d2_values=(0.5, 1.0), n_iter=3)
    assert serial.scores.shape == (2, 2)
    assert serial.curves.shape == (2, 2, 3, 3)
    assert serial.no_decay_curves.shape == (3, 3)
    assert serial.best in [(d1, d2) for d1 in (1, 2) for d2 in (0.5, 1.0)]
    i = serial.d1_values.index(serial.best[0])
    k = serial.d2_values.index(serial.best[1])
    assert serial.scores[i, k] == serial.scores.max()
    parallel = search_decay(population, config, d1_values=(1, 2), d2_values=(0.5, 1.0), n_iter=3, n_workers=2)
    np.testing.assert_allclose(parallel.curves, serial.curves)
