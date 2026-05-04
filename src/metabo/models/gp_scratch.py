"""Simple Gaussian Process regression implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from metabo.models.kernels import rbf_kernel


@dataclass
class GPScratch:
    """Teaching-oriented GP regression with an RBF kernel."""

    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-6

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit GP by caching train data and solving kernel system."""
        self.x_train = np.asarray(x, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64)
        if self.x_train.ndim != 2:
            raise ValueError("x must have shape (n, d).")
        if self.y_train.ndim != 1 or self.y_train.shape[0] != self.x_train.shape[0]:
            raise ValueError("y must have shape (n,) and match x rows.")

        k_xx = rbf_kernel(
            self.x_train,
            self.x_train,
            lengthscale=self.lengthscale,
            variance=self.variance,
        )
        n = self.x_train.shape[0]
        self.k_train = k_xx + (self.noise + 1e-10) * np.eye(n, dtype=np.float64)
        self.k_inv = np.linalg.inv(self.k_train)
        self.alpha = self.k_inv @ self.y_train

    def posterior(self, x_test: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return posterior mean and variance at test points."""
        x_test = np.asarray(x_test, dtype=np.float64)
        if x_test.ndim == 1:
            x_test = x_test[None, :]
        if x_test.ndim != 2:
            raise ValueError("x_test must have shape (m, d) or (d,).")

        k_xs = rbf_kernel(
            self.x_train,
            x_test,
            lengthscale=self.lengthscale,
            variance=self.variance,
        )
        k_ss = rbf_kernel(
            x_test,
            x_test,
            lengthscale=self.lengthscale,
            variance=self.variance,
        )
        mean = k_xs.T @ self.alpha
        cov = k_ss - k_xs.T @ self.k_inv @ k_xs
        var = np.maximum(np.diag(cov), 1e-12)
        return mean.astype(np.float64), var.astype(np.float64)
