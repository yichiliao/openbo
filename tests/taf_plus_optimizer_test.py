"""Tests for the TAF+ population store and ask/tell optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from openbo.optimizers.bo_taf_plus import (
    TAFPlusConfig,
    TAFPlusPopulation,
    TAFPlusSequentialOptimizer,
    run_bo_taf_plus,
)

CENTRES = (np.array([0.3, 0.7]), np.array([0.7, 0.6]))


def make_objective(shift: float):
    """Two maximized bowls whose centres move by `shift` along both axes."""

    def objective(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return np.stack(
            [-np.sum((x - (c + shift)) ** 2, axis=1) for c in CENTRES], axis=1
        )

    return objective


def make_population(shifts=(0.0, 0.02, -0.02), n_points: int = 15) -> TAFPlusPopulation:
    population = TAFPlusPopulation(input_dim=2, n_objectives=2)
    rng = np.random.default_rng(1)
    for i, shift in enumerate(shifts):
        x = rng.random((n_points, 2))
        population.add_user(f"user_{i:03d}", x, make_objective(shift)(x), seed=i)
    return population


CONFIG = TAFPlusConfig(input_dim=2, objective_weights=(0.5, 0.5), n_candidates=256)


@pytest.fixture(scope="module")
def population() -> TAFPlusPopulation:
    return make_population()


def test_first_point_can_be_the_centre(population: TAFPlusPopulation) -> None:
    config = TAFPlusConfig(input_dim=2, objective_weights=(0.5, 0.5), n_candidates=256, first_point="centre")
    optimizer = TAFPlusSequentialOptimizer(config, population)
    np.testing.assert_array_equal(optimizer.suggest(), [[0.5, 0.5]])
    assert optimizer.decay_history == [None]


def test_first_point_maximizes_the_population_mean(population: TAFPlusPopulation) -> None:
    # The weighted sum of the two bowls peaks at the mean of their centres, and
    # the population users are only slightly shifted from the new one.
    x0 = TAFPlusSequentialOptimizer(CONFIG, population).suggest()
    optimum = 0.5 * (CENTRES[0] + CENTRES[1])
    assert np.linalg.norm(x0[0] - optimum) < 0.1


def test_empty_population_starts_at_the_centre() -> None:
    optimizer = TAFPlusSequentialOptimizer(CONFIG, TAFPlusPopulation(2, 2))
    np.testing.assert_array_equal(optimizer.suggest(), [[0.5, 0.5]])


def test_decay_history_follows_eq6(population: TAFPlusPopulation) -> None:
    result = run_bo_taf_plus(make_objective(0.0), CONFIG, population, n_iter=7)
    assert result.decay_history[0] is None
    np.testing.assert_allclose(result.decay_history[1:], [1.0, 0.7, 0.4, 0.1, 0.0, 0.0], atol=1e-12)
    assert result.x_obs.shape == (7, 2) and result.y_obs.shape == (7, 2)
    np.testing.assert_allclose(result.weighted_y, result.y_obs @ np.array([0.5, 0.5]))
    assert np.all(np.diff(result.best_so_far) >= 0.0)


def test_population_guides_the_second_point(population: TAFPlusPopulation) -> None:
    config = TAFPlusConfig(input_dim=2, objective_weights=(0.5, 0.5), n_candidates=256, first_point="centre")
    optimizer = TAFPlusSequentialOptimizer(config, population)
    objective = make_objective(0.0)
    x0 = optimizer.suggest()
    optimizer.observe(x0, objective(x0))
    x1 = optimizer.suggest()
    # CHI eq. 4 sums per-objective clipped improvements, so TAF+ heads for the
    # point that improves one objective most (bowl 1's centre: improvement 0.08
    # on objective 1), not for the weighted-sum optimum between the bowls
    # (about 0.04 + 0.01). Bowl 1 beats bowl 2 because the centre point is
    # further from it.
    assert np.linalg.norm(x1[0] - CENTRES[0]) < 0.1


def test_full_decay_matches_an_empty_population(population: TAFPlusPopulation) -> None:
    objective = make_objective(0.0)
    config = TAFPlusConfig(input_dim=2, objective_weights=(0.5, 0.5), d1=0, d2=1.0, n_candidates=256, first_point="centre")
    with_sources = TAFPlusSequentialOptimizer(config, population)
    without = TAFPlusSequentialOptimizer(config, TAFPlusPopulation(2, 2))
    for opt in (with_sources, without):
        x0 = opt.suggest()
        opt.observe(x0, objective(x0))
    # t = 2 > d1 + 1/d2, so delta = 0 and only the user's EI is left.
    np.testing.assert_allclose(with_sources.suggest(), without.suggest())
    assert with_sources.decay_history[-1] == 0.0


def test_save_load_round_trip(population: TAFPlusPopulation, tmp_path) -> None:
    population.save(tmp_path)
    loaded = TAFPlusPopulation.load(tmp_path)
    assert [u.name for u in loaded.users] == [u.name for u in population.users]
    x = np.random.default_rng(3).random((5, 2))
    for a, b in zip(population.users, loaded.users):
        np.testing.assert_array_equal(a.y, b.y)
        for gp_a, gp_b in zip(a.gps, b.gps):
            np.testing.assert_allclose(gp_a.posterior(x)[0], gp_b.posterior(x)[0], rtol=1e-10)


def test_a_run_leaves_the_population_untouched(population: TAFPlusPopulation) -> None:
    before = [(u.x.copy(), u.y.copy(), [dict(s) for s in u.gp_states]) for u in population.users]
    run_bo_taf_plus(make_objective(0.05), CONFIG, population, n_iter=4)
    for (x, y, states), user in zip(before, population.users):
        np.testing.assert_array_equal(user.x, x)
        np.testing.assert_array_equal(user.y, y)
        assert user.gp_states == states


def test_taf_r_mode_runs(population: TAFPlusPopulation) -> None:
    config = TAFPlusConfig(input_dim=2, objective_weights=(0.3, 0.7), weight_mode="taf_r", n_candidates=128)
    result = run_bo_taf_plus(make_objective(0.0), config, population, n_iter=4)
    assert result.y_obs.shape == (4, 2)


def test_single_objective_accepts_flat_y() -> None:
    population = TAFPlusPopulation(input_dim=2, n_objectives=1)
    x = np.random.default_rng(0).random((8, 2))
    population.add_user("u", x, -np.sum((x - 0.4) ** 2, axis=1))
    config = TAFPlusConfig(input_dim=2, objective_weights=(1.0,), decay=False, n_candidates=128)
    optimizer = TAFPlusSequentialOptimizer(config, population)
    x0 = optimizer.suggest()
    optimizer.observe(x0, np.array([-0.02]))
    assert optimizer.suggest().shape == (1, 2)
    assert optimizer.current_decay() == 1.0


def test_subset_leaves_one_user_out(population: TAFPlusPopulation) -> None:
    rest = population.subset("user_001")
    assert [u.name for u in rest.users] == ["user_000", "user_002"]
    with pytest.raises(ValueError, match="not in the population"):
        rest.subset("user_001")


def test_validation(population: TAFPlusPopulation) -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        TAFPlusConfig(input_dim=2, objective_weights=(0.5, 0.6))
    with pytest.raises(ValueError, match="d2"):
        TAFPlusConfig(input_dim=2, objective_weights=(1.0,), d2=0.0)
    with pytest.raises(ValueError, match="weight_mode"):
        TAFPlusConfig(input_dim=2, objective_weights=(1.0,), weight_mode="taf_m")
    with pytest.raises(ValueError, match="n_objectives"):
        TAFPlusSequentialOptimizer(TAFPlusConfig(input_dim=2, objective_weights=(1.0,)), population)
    with pytest.raises(ValueError, match="already"):
        population.add_user("user_000", np.zeros((2, 2)), np.zeros((2, 2)))
    optimizer = TAFPlusSequentialOptimizer(CONFIG, population)
    with pytest.raises(ValueError, match="y must have shape"):
        optimizer.observe(np.zeros((1, 2)), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="finite"):
        optimizer.observe(np.zeros((1, 2)), np.array([[np.nan, 0.0]]))


def test_one_observation_keeps_a_usable_target_variance(population: TAFPlusPopulation) -> None:
    # GPScratch standardizes a single point with std 1e-12; TAF+ resets the
    # scale so the target's TAF-PoE weight does not swamp the population.
    optimizer = TAFPlusSequentialOptimizer(CONFIG, population)
    x0 = optimizer.suggest()
    optimizer.observe(x0, make_objective(0.0)(x0))
    optimizer.suggest()
    _, var = optimizer.gps[0].posterior(np.array([[0.1, 0.9]]))
    assert var[0] > 1e-3


@pytest.mark.parametrize("resample", [True, False])
def test_candidate_pool_is_redrawn_each_step_only_when_resampling(population: TAFPlusPopulation, resample: bool) -> None:
    config = TAFPlusConfig(input_dim=2, objective_weights=(0.5, 0.5), n_candidates=64, resample_candidates=resample)
    optimizer = TAFPlusSequentialOptimizer(config, population)
    objective = make_objective(0.0)
    pools = []
    for _ in range(3):
        x = optimizer.suggest()
        pools.append(optimizer.candidates.copy())
        optimizer.observe(x, objective(x))
    assert np.array_equal(pools[0], pools[1]) is not resample
    assert np.array_equal(pools[1], pools[2]) is not resample
