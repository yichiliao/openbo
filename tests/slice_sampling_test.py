"""Tests for the coordinate-wise slice sampler and GP hyperparameter sampling."""

from __future__ import annotations

import numpy as np
import pytest

from openbo.models.multi_output_gp import MultiOutputGP
from openbo.models.slice_sampling import slice_sample


def test_recovers_correlated_gaussian_moments() -> None:
    """Samples from a correlated 2D Gaussian should match its mean and covariance."""
    mean = np.array([1.0, -2.0])
    cov = np.array([[1.0, 0.8], [0.8, 2.0]])
    precision = np.linalg.inv(cov)

    def log_density(x: np.ndarray) -> float:
        diff = x - mean
        return float(-0.5 * diff @ precision @ diff)

    samples = slice_sample(log_density, [0.0, 0.0], 4000, np.random.default_rng(0), burn_in=100)
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.12)
    np.testing.assert_allclose(np.cov(samples.T), cov, atol=0.2)


def test_respects_bounds_and_truncation() -> None:
    """A standard normal truncated to [0, 1] should never leave the box."""
    samples = slice_sample(
        lambda x: float(-0.5 * x @ x),
        [0.5],
        3000,
        np.random.default_rng(1),
        lower=[0.0],
        upper=[1.0],
        widths=5.0,
    )
    assert np.all((samples >= 0.0) & (samples <= 1.0))
    # Mean of N(0, 1) truncated to [0, 1] is about 0.4598.
    assert samples.mean() == pytest.approx(0.4598, abs=0.03)


def test_is_deterministic_and_validates_inputs() -> None:
    """The same rng seed should give the same chain; bad inputs should raise."""
    log_density = lambda x: float(-0.5 * x @ x)  # noqa: E731
    first = slice_sample(log_density, [0.0, 0.0], 50, np.random.default_rng(2), thin=3)
    second = slice_sample(log_density, [0.0, 0.0], 50, np.random.default_rng(2), thin=3)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (50, 2)
    with pytest.raises(ValueError, match="within"):
        slice_sample(log_density, [2.0], 5, np.random.default_rng(0), lower=[0.0], upper=[1.0])
    with pytest.raises(ValueError, match="thin"):
        slice_sample(log_density, [0.0], 5, np.random.default_rng(0), thin=0)
    with pytest.raises(ValueError, match="finite"):
        slice_sample(lambda x: -np.inf, [0.0], 5, np.random.default_rng(0))


def test_gp_hyperparameter_samples_are_valid_and_usable() -> None:
    """GP samples should stay in bounds and give PSD posteriors via with_hyperparameters."""
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(15, 2))
    y = np.stack([np.sin(3 * x[:, 0]), np.cos(3 * x[:, 1]) + x[:, 0]], axis=1)
    gp = MultiOutputGP(mode="slf", optimizer_seed=0)
    with pytest.raises(ValueError, match="fit"):
        gp.sample_hyperparameters(2, rng)
    gp.fit(x, y)
    assert gp.log_posterior(gp._theta) == pytest.approx(-gp._negative_log_posterior(gp._theta)[0])

    samples = gp.sample_hyperparameters(5, rng, burn_in=5, thin=1)
    bounds = np.array(gp._bounds())
    assert samples.shape == (5, gp._theta.shape[0])
    assert np.all((samples >= bounds[:, 0]) & (samples <= bounds[:, 1]))
    assert len({tuple(row) for row in samples}) > 1

    x_test = rng.uniform(size=(4, 2))
    map_mean, _ = gp.posterior(x_test)
    for theta in samples:
        twin = gp.with_hyperparameters(theta)
        mean, cov = twin.posterior(x_test)
        assert mean.shape == (4, 2)
        assert np.all(np.linalg.eigvalsh(cov) > -1e-9)
    # The copy must not change the original model.
    np.testing.assert_allclose(gp.posterior(x_test)[0], map_mean)
    with pytest.raises(ValueError, match="shape"):
        gp.with_hyperparameters(samples[0][:-1])
