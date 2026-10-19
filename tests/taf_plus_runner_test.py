"""Tests for the TAF+ sphere users and simulation runner."""

from __future__ import annotations

import numpy as np
import pytest

from openbo.benchmarks.taf_plus_runner import (
    TAF_PLUS_METHODS,
    SimulationSettings,
    build_population,
    run_taf_plus_method,
    simulation_users,
)
from openbo.test_functions.taf_plus_users import (
    SPHERE_CENTRES,
    SphereUser,
    make_sphere_users,
    three_spheres,
)


def test_each_sphere_peaks_at_its_centre() -> None:
    x = np.array([[0.55, 0.4, 0.5, 0.5], [0.5, 0.6, 0.45, 0.5], [0.5, 0.5, 0.65, 0.35]])
    y = three_spheres(x)
    for k in range(3):
        assert y[k, k] == pytest.approx(1.0)
    assert SPHERE_CENTRES[0] == (0.55, 0.4)


def test_users_shift_inputs_and_scale_outputs() -> None:
    user = SphereUser(shift=(0.1, 0.0, 0.0, -0.05), scale=1.2, noise_std=0.0)
    x = np.array([[0.45, 0.4, 0.5, 0.5]])
    np.testing.assert_allclose(user.noise_free(x), 1.2 * three_spheres(x + np.array([0.1, 0.0, 0.0, -0.05])))
    noisy = SphereUser(shift=(0.0,) * 4, scale=1.0).observe(np.full((2000, 4), 0.5), np.random.default_rng(0))
    assert np.std(noisy[:, 0]) == pytest.approx(0.05, rel=0.1)


def test_sampled_users_stay_in_their_ranges() -> None:
    users = make_sphere_users(50, shift_range=0.2, scale_range=0.1, seed=3)
    shifts = np.array([u.shift for u in users])
    scales = np.array([u.scale for u in users])
    assert np.all(np.abs(shifts) <= 0.1) and np.all(np.abs(scales - 1.0) <= 0.05)
    assert make_sphere_users(3, 0.2, 0.1, seed=3) == users[:3]


def test_weighted_optimum_beats_a_dense_sample() -> None:
    user = make_sphere_users(1, 0.2, 0.2, seed=1)[0]
    alpha = np.array([0.3, 0.5, 0.2])
    x_best, value = user.weighted_optimum(alpha)
    sample = np.random.default_rng(0).random((20000, 4))
    assert value >= float(np.max(user.noise_free(sample) @ alpha)) - 1e-9
    assert value == pytest.approx(float(user.noise_free(x_best[None, :])[0] @ alpha))


TINY = SimulationSettings(n_population=2, n_new=1, population_init=6, population_iter=1, n_evals=4, bo_init=2, n_candidates=64)


@pytest.fixture(scope="module")
def population(tmp_path_factory):
    return build_population("sim2_r01", 0, TINY, tmp_path_factory.mktemp("pop"))


def test_population_is_cached(population, tmp_path) -> None:
    assert population.n_users == 2
    assert population.users[0].x.shape == (7, 4)
    built = build_population("sim2_r01", 0, TINY, tmp_path)
    again = build_population("sim2_r01", 0, TINY, tmp_path)
    np.testing.assert_array_equal(built.users[1].y, again.users[1].y)


@pytest.mark.parametrize("method", TAF_PLUS_METHODS)
def test_every_method_runs_on_the_new_users(method, population) -> None:
    result = run_taf_plus_method(method, "sim2_r01", (0.3, 0.5, 0.2), 0, population, TINY, d1=1, d2=0.5)
    assert len(result["users"]) == 1
    record = result["users"][0]
    assert np.asarray(record["x"]).shape == (4, 4)
    assert np.asarray(record["y"]).shape == (4, 3)
    regret = np.asarray(record["regret"])
    instant = np.asarray(record["instant_regret"])
    assert np.all(regret >= -1e-9) and np.all(np.diff(regret) <= 1e-12)
    np.testing.assert_allclose(regret, np.minimum.accumulate(instant))
    _, new_users = simulation_users("sim2_r01", 0, TINY)
    assert record["shift"] == list(new_users[0].shift)
