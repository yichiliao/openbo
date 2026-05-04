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
