"""Tests for the NAF+ double-sphere family and its fixed GP (NAF+, Phase 1)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from openbo.optimizers.naf_plus import (
    DOUBLE_SPHERE_GP_HYPERPARAMETERS,
    DOUBLE_SPHERE_WIDE_GP_HYPERPARAMETERS,
    select_family_hyperparameters,
)
from openbo.test_functions.naf_plus_families import (
    DEFAULT_SCALE_RANGE,
    DEFAULT_SHIFT_RANGE,
    NOVEL_CENTRES,
    TRAINING_CENTRES,
    WIDE_NOVEL_CENTRES,
    WIDE_TRAINING_CENTRES,
    DoubleSphereUser,
    WeightedTask,
    double_sphere_objectives,
    sample_double_sphere_users,
    sample_weighted_tasks,
    sample_weights,
    validate_weights,
)


def test_objectives_follow_the_paper_definition() -> None:
    # f_k(x) = 1 - 8 ||x - c_k||^2: each sphere peaks at 1 on its centre.
    values = double_sphere_objectives(np.asarray(TRAINING_CENTRES))
    assert values.shape == (2, 2)
    np.testing.assert_allclose(np.diag(values), 1.0)
    np.testing.assert_allclose(values[0, 1], 1.0 - 8.0 * 2 * 0.2**2)
    assert double_sphere_objectives(np.array([0.5, 0.5])).shape == (1, 2)
    with pytest.raises(ValueError, match="centres"):
        double_sphere_objectives(np.zeros((1, 3)))


def test_user_shifts_the_input_and_scales_every_objective() -> None:
    user = DoubleSphereUser(shift=(0.05, -0.02), scale=1.07)
    x = np.random.default_rng(0).random((10, 2))
    expected = 1.07 * double_sphere_objectives(x + np.array([0.05, -0.02]))
    np.testing.assert_allclose(user.objectives(x), expected)
    np.testing.assert_allclose(user(x, (0.3, 0.7)), expected @ np.array([0.3, 0.7]))
    task = WeightedTask(user, (0.3, 0.7))
    np.testing.assert_allclose(task(x), user(x, (0.3, 0.7)))
    assert task.optimum == user.optimum((0.3, 0.7))


@pytest.mark.parametrize("centres", [TRAINING_CENTRES, NOVEL_CENTRES, WIDE_TRAINING_CENTRES, WIDE_NOVEL_CENTRES])
def test_analytic_optimum_is_never_beaten(centres: tuple[tuple[float, float], ...]) -> None:
    users = sample_double_sphere_users(250, seed=3, centres=centres)
    rng = np.random.default_rng(4)
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101)), axis=-1).reshape(-1, 2)
    for user in users:
        w = sample_weights(2, rng)
        values = user(grid, w)
        start = grid[np.argmax(values)]
        polished = minimize(lambda z: -user(z[None, :], w)[0], start, bounds=[(0.0, 1.0)] * 2, method="L-BFGS-B")
        best = max(values.max(), -polished.fun)
        assert user.optimum(w) >= best - 1e-12
        assert np.all((user.argmax(w) >= 0.0) & (user.argmax(w) <= 1.0))


def test_argmax_is_clipped_to_the_unit_box() -> None:
    user = DoubleSphereUser(shift=(0.5, -0.7), scale=1.0)
    w = (0.5, 0.5)
    # Unclipped centre (0.5, 0.5) - shift = (0.0, 1.2): the second coordinate is clipped.
    np.testing.assert_allclose(user.argmax(w), [0.0, 1.0])
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201)), axis=-1).reshape(-1, 2)
    assert user.optimum(w) >= user(grid, w).max() - 1e-12


def test_sampled_users_stay_in_range_and_are_seeded() -> None:
    users = sample_double_sphere_users(500, seed=0)
    shifts = np.array([u.shift for u in users])
    scales = np.array([u.scale for u in users])
    assert np.all(np.abs(shifts) <= DEFAULT_SHIFT_RANGE / 2)
    assert np.all(np.abs(scales - 1.0) <= DEFAULT_SCALE_RANGE / 2)
    assert np.ptp(shifts) > 0.9 * DEFAULT_SHIFT_RANGE  # the range is actually used
    assert users == sample_double_sphere_users(500, seed=0)
    assert users != sample_double_sphere_users(500, seed=1)
    novel = sample_double_sphere_users(3, seed=0, centres=NOVEL_CENTRES)
    assert all(u.centres == NOVEL_CENTRES for u in novel)
    with pytest.raises(ValueError, match="scale_range"):
        sample_double_sphere_users(3, seed=0, scale_range=2.0)


def test_weights_are_validated_and_sampled_on_the_simplex() -> None:
    rng = np.random.default_rng(0)
    w = np.array([sample_weights(2, rng) for _ in range(4000)])
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert np.all(w >= 0.0)
    # Two objectives: w_1 ~ U[0, 1], the paper's alpha.
    assert abs(w[:, 0].mean() - 0.5) < 0.02
    assert abs(w[:, 0].var() - 1.0 / 12.0) < 0.01
    with pytest.raises(ValueError, match="sum to 1"):
        validate_weights((0.5, 0.6), 2)
    with pytest.raises(ValueError, match="2 entries"):
        validate_weights((1.0,), 2)
    with pytest.raises(ValueError, match="sum to 1"):
        WeightedTask(DoubleSphereUser(shift=(0.0, 0.0), scale=1.0), (1.2, -0.2))


def test_weighted_tasks_are_seeded() -> None:
    tasks = sample_weighted_tasks(20, seed=5)
    assert tasks == sample_weighted_tasks(20, seed=5)
    assert len({t.weights for t in tasks}) == 20


def test_double_sphere_gp_hyperparameters_match_their_selection() -> None:
    selected = select_family_hyperparameters(sample_weighted_tasks(60, seed=11), dim=2, seed=0)
    assert selected == DOUBLE_SPHERE_GP_HYPERPARAMETERS
    # Prior std close to Branin's (sqrt 2) and to the family's spread.
    assert 1.0 < np.sqrt(selected.variance) < 1.5


def test_wide_gp_hyperparameters_match_their_selection() -> None:
    tasks = sample_weighted_tasks(60, seed=11, centres=WIDE_TRAINING_CENTRES)
    assert select_family_hyperparameters(tasks, dim=2, seed=0) == DOUBLE_SPHERE_WIDE_GP_HYPERPARAMETERS


def test_wide_optima_stay_inside_the_square_and_move_with_the_weights() -> None:
    users = sample_double_sphere_users(200, seed=5, centres=WIDE_TRAINING_CENTRES)
    for user in users:
        a, b = user.argmax((1.0, 0.0)), user.argmax((0.0, 1.0))
        assert np.all((a > 0.0) & (a < 1.0)) and np.all((b > 0.0) & (b < 1.0))  # never clipped
        assert np.linalg.norm(b - a) == pytest.approx(0.6 * np.sqrt(2))


def test_select_family_hyperparameters_validates_its_inputs() -> None:
    tasks = sample_weighted_tasks(2, seed=0)
    with pytest.raises(ValueError, match="tasks"):
        select_family_hyperparameters([], dim=2)
    with pytest.raises(ValueError, match="lengthscales"):
        select_family_hyperparameters(tasks, dim=2, lengthscales=(0.0,))
    with pytest.raises(ValueError, match="n_observations"):
        select_family_hyperparameters(tasks, dim=2, n_observations=())
