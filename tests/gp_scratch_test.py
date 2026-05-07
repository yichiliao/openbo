"""Tests for scratch GP posterior."""

from __future__ import annotations

import numpy as np

from metabo.models.gp_scratch import GPScratch


def test_gp_scratch_fit_stores_training_data() -> None:
    """Fit should store training arrays."""
    model = GPScratch()
    x = np.array([[0.0], [1.0]], dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)
    model.fit(x, y)
    assert model.x_train.shape == (2, 1)
    assert model.y_train.shape == (2,)


def test_gp_scratch_posterior_shape() -> None:
    """Posterior should return vectors of shape (n_test,)."""
    model = GPScratch(lengthscale=1.0, variance=1.0, noise=1e-6)
    x_train = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y_train = np.array([0.0, -1.0, -4.0], dtype=np.float64)
    model.fit(x_train, y_train)

    x_test = np.array([[0.5], [1.5]], dtype=np.float64)
    mean, var = model.posterior(x_test)
    assert mean.shape == (2,)
    assert var.shape == (2,)


def test_gp_scratch_matern52_ard_kernel() -> None:
    """GP should support Matern-5/2 with ARD lengthscales."""
    model = GPScratch(
        kernel_type="matern52",
        lengthscale=np.array([0.8, 1.2], dtype=np.float64),
        variance=1.0,
        noise=1e-6,
    )
    x_train = np.array([[0.0, 0.0], [1.0, 0.5], [0.2, 1.2]], dtype=np.float64)
    y_train = np.array([0.0, 1.0, -0.5], dtype=np.float64)
    model.fit(x_train, y_train)

    x_test = np.array([[0.1, 0.2], [0.8, 0.7]], dtype=np.float64)
    mean, var = model.posterior(x_test)
    assert mean.shape == (2,)
    assert var.shape == (2,)
    assert np.all(var >= 0.0)


def test_gp_hyperparameter_optimization_improves_mll() -> None:
    """Hyperparameter fitting should not worsen negative log marginal likelihood."""
    rng = np.random.default_rng(0)
    x_train = rng.uniform(0.0, 1.0, size=(10, 2)).astype(np.float64)
    y_train = (np.sin(4.0 * x_train[:, 0]) + 0.3 * np.cos(3.0 * x_train[:, 1])).astype(
        np.float64
    )

    model = GPScratch(
        kernel_type="matern52",
        lengthscale=np.array([1.5, 1.5], dtype=np.float64),
        variance=1.0,
        noise=1e-6,
        optimize_hyperparameters=False,
    )
    model.x_train = x_train
    model.y_train = y_train
    model.lengthscale = model._resolve_lengthscale(2)
    theta0 = model._pack_hyperparameters()
    nll_before = model._negative_log_marginal_likelihood(theta0)

    model.optimize_hyperparameters = True
    model.fit(x_train, y_train)
    theta_opt = model._pack_hyperparameters()
    nll_after = model._negative_log_marginal_likelihood(theta_opt)

    assert np.isfinite(nll_before)
    assert np.isfinite(nll_after)
    assert nll_after <= nll_before + 1e-6
