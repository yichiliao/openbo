"""Tests for MetaBO's Branin family and the fixed family GPs (NAF, Phase 1)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from openbo.optimizers.naf import (
    BRANIN_GP_HYPERPARAMETERS,
    FamilyGPHyperparameters,
    fit_family_hyperparameters,
)
from openbo.test_functions.metabo_families import (
    NORMALIZED_BRANIN_MAXIMUM,
    BraninTask,
    branin_grid_tasks,
    normalized_branin,
    sample_branin_tasks,
)
from openbo.test_functions.synthetic import branin


def test_normalized_branin_matches_the_reference_normalization() -> None:
    x = np.random.default_rng(0).random((20, 2))
    # Reference: -(f - 54.44) / 51.44 with f the textbook (minimized) Branin.
    np.testing.assert_allclose(normalized_branin(x), -(-branin(x) - 54.44) / 51.44)
    assert NORMALIZED_BRANIN_MAXIMUM == pytest.approx(1.0505854, abs=1e-6)


def test_task_translates_inputs_and_scales_outputs() -> None:
    task = BraninTask(translation=(0.05, -0.08), scaling=1.07)
    x = np.array([[0.2, 0.7], [0.9, 0.1]])
    np.testing.assert_allclose(task(x), 1.07 * normalized_branin(x - np.array([0.05, -0.08])))
    assert task(task.argmax)[0] == pytest.approx(task.optimum)


@pytest.mark.parametrize("task", sample_branin_tasks(5, seed=3) + branin_grid_tasks()[:5])
def test_tracked_optimum_is_the_maximum_in_the_domain(task: BraninTask) -> None:
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101)), -1).reshape(-1, 2)
    start = grid[np.argmax(task(grid))]
    result = minimize(lambda z: -task(z)[0], start, bounds=[(0.0, 1.0)] * 2)
    assert -result.fun <= task.optimum + 1e-9
    assert np.all((task.argmax >= 0.0) & (task.argmax <= 1.0))


def test_sampled_and_grid_tasks_stay_in_range() -> None:
    tasks = sample_branin_tasks(100, seed=0) + branin_grid_tasks(50)
    t = np.array([task.translation for task in tasks])
    s = np.array([task.scaling for task in tasks])
    assert np.all(np.abs(t) <= 0.1) and np.all((s >= 0.9) & (s <= 1.1))
    assert sample_branin_tasks(3, seed=0) == tasks[:3]
    grid = branin_grid_tasks(50)
    assert len(set(grid)) == 50 and grid == branin_grid_tasks(50)


def test_family_gp_has_fixed_hyperparameters_and_no_standardization() -> None:
    gp = BRANIN_GP_HYPERPARAMETERS.new_gp()
    x = np.random.default_rng(1).random((10, 2))
    gp.fit(x, normalized_branin(x))
    np.testing.assert_allclose(gp.lengthscale, [0.235, 0.578])
    assert gp.variance == 2.0 and gp.y_mean == 0.0 and gp.y_std == 1.0
    # Far from the data, the zero-mean GP falls back to its prior.
    mean, var = gp.posterior(np.array([[5.0, 5.0]]))
    assert abs(mean[0]) < 1e-8 and var[0] == pytest.approx(2.0)
    assert FamilyGPHyperparameters.from_dict(BRANIN_GP_HYPERPARAMETERS.to_dict()) == BRANIN_GP_HYPERPARAMETERS


def test_fit_family_hyperparameters_returns_a_usable_gp() -> None:
    """A fit on one task should predict about as well as the reference's hyperparameters."""
    hp = fit_family_hyperparameters(normalized_branin, dim=2, n_points=40, seed=0)
    assert len(hp.lengthscale) == 2 and hp.variance > 0.0
    x = np.random.default_rng(2).random((30, 2))
    x_test = np.random.default_rng(3).random((200, 2))
    errors = []
    for family_hp in (hp, BRANIN_GP_HYPERPARAMETERS):
        gp = family_hp.new_gp()
        gp.fit(x, normalized_branin(x))
        mean, _ = gp.posterior(x_test)
        errors.append(np.sqrt(np.mean((mean - normalized_branin(x_test)) ** 2)))
    assert errors[0] < 2.0 * errors[1]
