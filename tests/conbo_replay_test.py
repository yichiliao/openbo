"""Tests for ConBO generative replay."""

from __future__ import annotations

import numpy as np
import pytest

from openbo.models.bnn import BNNConfig, PopulationBNN
from openbo.models.conbo_replay import (
    replay_inputs,
    replay_targets,
    standardized_variance_threshold,
)
from openbo.models.gp_scratch import GPScratch


class _StubGP:
    """Fixed per-input predictions, standing in for a fitted GP."""

    def __init__(self, mean: list[float], var: list[float]) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.var = np.asarray(var, dtype=np.float64)

    def posterior(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert x.shape[0] == self.mean.shape[0]
        return self.mean.copy(), self.var.copy()


X3 = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])


def test_replay_inputs_grid_plus_random() -> None:
    x = replay_inputs(2, grid_points_per_dim=30, n_random=50, seed=0)
    assert x.shape == (950, 2)
    assert np.all((x >= 0.0) & (x <= 1.0))
    grid = {tuple(p) for p in np.round(x[:900], 12)}
    assert (0.0, 0.0) in grid and (1.0, 1.0) in grid
    np.testing.assert_array_equal(x, replay_inputs(2, 30, 50, seed=0))
    assert not np.array_equal(x[900:], replay_inputs(2, 30, 50, seed=1)[900:])


def test_replay_inputs_use_sobol_above_two_dims() -> None:
    x = replay_inputs(4, grid_points_per_dim=8, n_random=5)
    assert x.shape == (69, 4)
    assert np.all((x >= 0.0) & (x <= 1.0))


def test_replay_inputs_validation() -> None:
    with pytest.raises(ValueError, match="dim"):
        replay_inputs(0)
    with pytest.raises(ValueError, match="grid_points_per_dim"):
        replay_inputs(2, grid_points_per_dim=1)
    with pytest.raises(ValueError, match="n_random"):
        replay_inputs(2, n_random=-1)


def test_filter_and_averages_match_hand_computation() -> None:
    users = [
        _StubGP([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        _StubGP([3.0, 3.0, 3.0], [0.5, 10.0, 10.0]),
        _StubGP([9.0, 9.0, 9.0], [100.0, 100.0, 100.0]),
    ]
    targets = replay_targets(users, X3, max_variance=2.0)  # type: ignore[arg-type]
    np.testing.assert_array_equal(targets.x, X3)
    np.testing.assert_allclose(targets.mean, [2.0, 1.0, 1.0])
    np.testing.assert_allclose(targets.var, [0.75, 1.0, 1.0])
    np.testing.assert_array_equal(targets.n_users, [2, 1, 1])
    np.testing.assert_allclose(targets.kept_fraction, [1.0, 1.0 / 3.0, 0.0])
    assert targets.excluded_users == (2,)


def test_inputs_without_qualifying_users_are_dropped() -> None:
    users = [
        _StubGP([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        _StubGP([3.0, 3.0, 3.0], [0.5, 10.0, 10.0]),
    ]
    targets = replay_targets(users, X3, max_variance=0.6)  # type: ignore[arg-type]
    np.testing.assert_array_equal(targets.x, X3[:1])
    np.testing.assert_allclose(targets.mean, [3.0])
    assert targets.excluded_users == (0,)


def test_no_filter_averages_every_user() -> None:
    users = [_StubGP([1.0, 2.0, 3.0], [1.0, 5.0, 9.0]), _StubGP([3.0, 4.0, 5.0], [3.0, 7.0, 11.0])]
    targets = replay_targets(users, X3, max_variance=None)  # type: ignore[arg-type]
    np.testing.assert_allclose(targets.mean, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(targets.var, [2.0, 6.0, 10.0])
    assert targets.excluded_users == ()


def test_standardized_threshold_scales_with_the_spread_of_means() -> None:
    users = [_StubGP([0.0, 2.0, 4.0], [1.0, 1.0, 1.0])]
    expected = 0.5 * np.var([0.0, 2.0, 4.0])
    assert standardized_variance_threshold(users, X3, 0.5) == pytest.approx(expected)  # type: ignore[arg-type]
    scaled = [_StubGP([0.0, 20.0, 40.0], [1.0, 1.0, 1.0])]
    assert standardized_variance_threshold(scaled, X3, 0.5) == pytest.approx(100.0 * expected)  # type: ignore[arg-type]
    flat = [_StubGP([3.0, 3.0, 3.0], [1.0, 1.0, 1.0])]
    assert standardized_variance_threshold(flat, X3, 0.5) == pytest.approx(0.5)  # type: ignore[arg-type]


def test_validation() -> None:
    users = [_StubGP([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])]
    with pytest.raises(ValueError, match="at least one"):
        replay_targets([], X3, max_variance=1.0)
    with pytest.raises(ValueError, match="max_variance"):
        replay_targets(users, X3, max_variance=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="x must have shape"):
        replay_targets(users, X3[0], max_variance=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="variance_fraction"):
        standardized_variance_threshold(users, X3, 0.0)  # type: ignore[arg-type]


def _fitted_gp(x: np.ndarray, y: np.ndarray) -> GPScratch:
    gp = GPScratch(lengthscale=0.2, variance=1.0, noise=1e-4)
    gp.fit(x, y)
    return gp


def test_filter_keeps_inputs_near_a_users_data() -> None:
    rng = np.random.default_rng(0)
    x_obs = rng.random((15, 2)) * 0.3
    gp = _fitted_gp(x_obs, np.sin(5.0 * x_obs.sum(axis=1)))
    x = replay_inputs(2, grid_points_per_dim=15, n_random=0)
    targets = replay_targets([gp], x, max_variance=0.05 * gp.y_std**2)
    assert 0.0 < targets.kept_fraction[0] < 0.5
    assert targets.x.max() < 0.7
    assert not any(np.allclose(p, [1.0, 1.0]) for p in targets.x)


def test_bnn_learns_the_replayed_population_mean() -> None:
    rng = np.random.default_rng(1)
    users = []
    for shift in (0.0, 0.1):
        x_obs = rng.random((30, 2))
        users.append(_fitted_gp(x_obs, np.sin(4.0 * (x_obs[:, 0] + shift)) + x_obs[:, 1]))
    x = replay_inputs(2, grid_points_per_dim=15, n_random=20)
    targets = replay_targets(users, x, max_variance=0.1)
    assert targets.x.shape[0] > 0.8 * x.shape[0]
    model = PopulationBNN(BNNConfig(input_dim=2))
    model.fit_replay(targets.x, targets.mean, targets.var, epochs=400)
    mean, _ = model.predict(targets.x)
    assert np.sqrt(np.mean((mean - targets.mean) ** 2)) < 0.1
