"""Tests for NAF+'s novelty detector and EI mixing (NAF+, Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from openbo.acquisition.ei import expected_improvement_maximization
from openbo.acquisition.naf_plus import (
    calibrated_p_value,
    novelty_statistic,
    novelty_z_scores,
    NOVELTY_THRESHOLD,
    ei_score_fn,
    ei_weight,
    mixed_score_fn,
    novelty_p_values,
)
from openbo.models.bnn import BNNConfig
from openbo.optimizers.naf_plus import (
    DOUBLE_SPHERE_GP_HYPERPARAMETERS,
    NoveltyDetector,
    build_detector_dataset,
    detector_config,
    fit_detector,
    weight_grid,
)
from openbo.test_functions.naf_plus_families import sample_double_sphere_users, sample_weights


def test_p_values_are_two_tailed_normal_tails() -> None:
    rng = np.random.default_rng(0)
    mean, std, y = rng.normal(size=50), rng.uniform(0.1, 2.0, 50), rng.normal(size=50)
    expected = 2.0 * norm.sf(np.abs((y - mean) / std))
    np.testing.assert_allclose(novelty_p_values(mean, std, y), expected, rtol=1e-12)
    assert novelty_p_values([0.0], [1.0], [0.0])[0] == 1.0
    with pytest.raises(ValueError, match="positive"):
        novelty_p_values([0.0], [0.0], [1.0])
    with pytest.raises(ValueError, match="same shape"):
        novelty_p_values([0.0, 1.0], [1.0], [1.0])


def test_ei_weight_follows_eq_5() -> None:
    tau = NOVELTY_THRESHOLD
    assert ei_weight(0.0) == 1.0  # an entirely novel user: EI only
    assert ei_weight(tau) == 0.0 and ei_weight(0.7) == 0.0  # a typical user: the neural AF only
    assert ei_weight(0.05) == pytest.approx(0.5)
    assert ei_weight(tau - 1e-9) == pytest.approx(0.0, abs=1e-7)  # continuous at tau
    assert ei_weight(0.2, tau=0.4) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="tau"):
        ei_weight(0.1, tau=0.0)
    with pytest.raises(ValueError, match="p_bar"):
        ei_weight(1.5)


def test_ei_score_fn_is_ei_on_the_gp_posterior() -> None:
    gp = DOUBLE_SPHERE_GP_HYPERPARAMETERS.new_gp()
    rng = np.random.default_rng(1)
    x_obs = rng.random((5, 2))
    gp.fit(x_obs, np.sin(x_obs.sum(axis=1)))
    x = rng.random((30, 2))
    mean, var = gp.posterior(x)
    expected = expected_improvement_maximization(mean, var, 0.8)
    np.testing.assert_allclose(ei_score_fn(gp, 0.8)(x), expected, rtol=1e-10)


def test_mixed_score_is_eq_4_on_a_fixed_reference() -> None:
    def naf(x: np.ndarray) -> np.ndarray:
        return 10.0 * x[:, 0]  # range [0, 10] on the reference

    def ei(x: np.ndarray) -> np.ndarray:
        return 0.5 - 0.5 * x[:, 0]  # range [0, 0.5] on the reference

    reference = np.linspace(0.0, 1.0, 11)[:, None]
    assert mixed_score_fn(naf, ei, 0.0, reference) is naf  # NAF+ picks exactly what NAF picks
    assert mixed_score_fn(naf, ei, 1.0, reference) is ei
    mixed = mixed_score_fn(naf, ei, 0.25, reference)
    x = np.array([[0.0], [0.4], [1.0], [1.2]])  # 1.2 lies outside the reference
    expected = 0.25 * (1.0 - x[:, 0]) + 0.75 * x[:, 0]
    np.testing.assert_allclose(mixed(x), expected)
    with pytest.raises(ValueError, match="lam"):
        mixed_score_fn(naf, ei, 1.5, reference)


def test_weight_grid_includes_the_corners() -> None:
    grid = weight_grid(2)
    assert grid.shape == (11, 2)
    np.testing.assert_allclose(grid[:, 0], np.linspace(0.0, 1.0, 11))
    np.testing.assert_allclose(grid.sum(axis=1), 1.0)
    assert weight_grid(3, step=0.5).shape == (6, 3)
    with pytest.raises(ValueError, match="divide"):
        weight_grid(2, step=0.3)


def test_detector_dataset_evaluates_fresh_users_per_weight() -> None:
    weights = weight_grid(2, step=0.5)
    inputs, y = build_detector_dataset(sample_double_sphere_users, weights, dim=2, n_users=3, points_per_dim=4)
    assert inputs.shape == (3 * 3 * 16, 4) and y.shape == (3 * 3 * 16,)
    # Block (weight k, user u) is user u of sampler seed 1009 k on the 4 x 4 grid.
    user = sample_double_sphere_users(3, 1009 * 1)[2]
    block = slice((1 * 3 + 2) * 16, (1 * 3 + 3) * 16)
    np.testing.assert_allclose(inputs[block, 2:], np.tile(weights[1], (16, 1)))
    np.testing.assert_allclose(y[block], user(inputs[block, :2], weights[1]))


def test_detector_saves_loads_and_checks_inputs(tmp_path) -> None:
    inputs, y = build_detector_dataset(sample_double_sphere_users, weight_grid(2, 0.5), 2, n_users=2, points_per_dim=4)
    config = detector_config(4, hidden_units=16, n_predict_samples=10)
    detector = fit_detector(inputs, y, 2, 2, epochs=5, config=config)
    detector.save(tmp_path)
    loaded = NoveltyDetector.load(tmp_path)
    x = np.random.default_rng(0).random((7, 2))
    for a, b in zip(detector.predict(x, (0.3, 0.7)), loaded.predict(x, (0.3, 0.7))):
        np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError, match="weights 2 entries"):
        detector.predict(x, (1.0,))
    with pytest.raises(ValueError, match="shape"):
        fit_detector(inputs[:, :3], y, 2, 2, epochs=1)
    with pytest.raises(ValueError, match="input_dim"):
        fit_detector(inputs, y, 2, 2, epochs=1, config=BNNConfig(input_dim=3))


def test_detector_flags_users_far_from_the_population() -> None:
    """A small detector: typical users stay above tau on average, strongly novel ones fall below."""
    weights = weight_grid(2, step=0.25)
    inputs, y = build_detector_dataset(sample_double_sphere_users, weights, 2, n_users=10, points_per_dim=8)
    detector = fit_detector(inputs, y, 2, 2, epochs=400, config=detector_config(4, hidden_units=64))
    rng = np.random.default_rng(5)

    def p_bars(users: list) -> np.ndarray:
        out = []
        for user in users:
            w = sample_weights(2, rng)
            x = rng.random((6, 2))
            out.append(detector.p_values(x, w, user(x, w)).mean())
        return np.asarray(out)

    typical = p_bars(sample_double_sphere_users(20, seed=50))
    far = p_bars(sample_double_sphere_users(20, seed=51, centres=((0.05, 0.05), (0.95, 0.95))))
    assert np.mean(typical >= NOVELTY_THRESHOLD) > 0.8
    assert np.mean(far < NOVELTY_THRESHOLD) > 0.8


def test_calibrated_test_helpers() -> None:
    z = novelty_z_scores([0.0, 1.0], [1.0, 2.0], [1.5, 0.0])
    np.testing.assert_allclose(z, [1.5, -0.5])
    np.testing.assert_allclose(novelty_p_values([0.0, 1.0], [1.0, 2.0], [1.5, 0.0]), 2 * norm.sf(np.abs(z)))
    assert novelty_statistic(z) == pytest.approx(2.5)
    # (1 + #{null >= s}) / (N + 1)
    assert calibrated_p_value([1.0, 2.0, 3.0], 2.0) == pytest.approx(3 / 4)
    assert calibrated_p_value([1.0, 2.0, 3.0], 10.0) == pytest.approx(1 / 4)
    assert calibrated_p_value([1.0, 2.0, 3.0], 0.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="empty"):
        calibrated_p_value([], 1.0)
    with pytest.raises(ValueError, match="positive"):
        novelty_z_scores([0.0], [0.0], [1.0])
