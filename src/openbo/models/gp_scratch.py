"""Simple Gaussian Process regression implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from openbo.models.kernels import matern52_kernel, rbf_kernel


@dataclass
class GPScratch:
    """Teaching-oriented GP regression with selectable kernels."""

    lengthscale: float | NDArray[np.float64] = 1.0
    variance: float = 1.0
    noise: float = 1e-6
    kernel_type: Literal["rbf", "matern52"] = "matern52"
    optimize_hyperparameters: bool = False
    optimizer_maxiter: int = 75
    optimizer_restarts: int = 8
    optimizer_seed: int | None = None
    standardize_targets: bool = True
    optimize_noise: bool = False
    y_mean: float = field(default=0.0, init=False)
    y_std: float = field(default=1.0, init=False)
    y_train_norm: NDArray[np.float64] | None = field(default=None, init=False)

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit GP by caching train data and solving kernel system."""
        self.x_train = np.asarray(x, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64)
        if self.x_train.ndim != 2:
            raise ValueError("x must have shape (n, d).")
        if self.y_train.ndim != 1 or self.y_train.shape[0] != self.x_train.shape[0]:
            raise ValueError("y must have shape (n,) and match x rows.")

        d = self.x_train.shape[1]
        self.lengthscale = self._resolve_lengthscale(d)
        self._set_target_scaling()

        if self.optimize_hyperparameters:
            self._optimize_hyperparameters()

        k_xx = self._kernel(self.x_train, self.x_train)
        n = self.x_train.shape[0]
        self.k_train = k_xx + (self.noise + 1e-10) * np.eye(n, dtype=np.float64)
        self.l_chol = self._robust_cholesky(self.k_train)
        assert self.y_train_norm is not None
        self.alpha = self._solve_cholesky(self.l_chol, self.y_train_norm)

    def posterior(
        self, x_test: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return posterior mean and variance at test points (original y scale)."""
        x_test = np.asarray(x_test, dtype=np.float64)
        if x_test.ndim == 1:
            x_test = x_test[None, :]
        if x_test.ndim != 2:
            raise ValueError("x_test must have shape (m, d) or (d,).")

        k_xs = self._kernel(self.x_train, x_test)
        k_ss = self._kernel(x_test, x_test)
        mean_norm = k_xs.T @ self.alpha
        v = np.linalg.solve(self.l_chol, k_xs)
        cov_norm = k_ss - v.T @ v
        var_norm = np.maximum(np.diag(cov_norm), 1e-12)
        mean = mean_norm * self.y_std + self.y_mean
        var = var_norm * (self.y_std**2)
        return mean.astype(np.float64), var.astype(np.float64)

    def _set_target_scaling(self) -> None:
        if self.standardize_targets:
            self.y_mean = float(np.mean(self.y_train))
            self.y_std = float(np.std(self.y_train) + 1e-12)
            self.y_train_norm = ((self.y_train - self.y_mean) / self.y_std).astype(
                np.float64
            )
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
            self.y_train_norm = self.y_train.astype(np.float64)

    def _kernel(
        self,
        x1: NDArray[np.float64],
        x2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.kernel_type == "rbf":
            return rbf_kernel(x1, x2, self.lengthscale, self.variance)
        if self.kernel_type == "matern52":
            return matern52_kernel(x1, x2, self.lengthscale, self.variance)
        raise ValueError(f"Unknown kernel_type '{self.kernel_type}'.")

    def _resolve_lengthscale(self, d: int) -> NDArray[np.float64]:
        ls = np.asarray(self.lengthscale, dtype=np.float64)
        if ls.ndim == 0:
            val = float(ls)
            if val <= 0:
                raise ValueError("lengthscale must be positive.")
            return np.full(d, val, dtype=np.float64)
        if ls.ndim != 1 or ls.shape[0] != d:
            raise ValueError(f"lengthscale must be scalar or shape ({d},).")
        if np.any(ls <= 0):
            raise ValueError("all ARD lengthscales must be positive.")
        return ls.astype(np.float64)

    def _robust_cholesky(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Cholesky with escalating jitter for numerical robustness."""
        n = matrix.shape[0]
        eye = np.eye(n, dtype=np.float64)
        jitter = 1e-10
        for _ in range(8):
            try:
                return np.linalg.cholesky(matrix + jitter * eye)
            except np.linalg.LinAlgError:
                jitter *= 10.0
        raise np.linalg.LinAlgError("Cholesky failed even after jitter escalation.")

    def _solve_cholesky(
        self, chol_l: NDArray[np.float64], rhs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        tmp = np.linalg.solve(chol_l, rhs)
        return np.linalg.solve(chol_l.T, tmp)

    def _pack_hyperparameters(self) -> NDArray[np.float64]:
        log_ls = np.log(np.asarray(self.lengthscale, dtype=np.float64))
        parts = [log_ls, np.array([np.log(self.variance)], dtype=np.float64)]
        if self.optimize_noise:
            parts.append(np.array([np.log(self.noise)], dtype=np.float64))
        return np.concatenate(parts)

    def _unpack_hyperparameters(self, theta: NDArray[np.float64]) -> None:
        d = self.x_train.shape[1]
        self.lengthscale = np.exp(theta[:d])
        self.variance = float(np.exp(theta[d]))
        if self.optimize_noise:
            self.noise = float(np.exp(theta[d + 1]))

    def _negative_log_marginal_likelihood(self, theta: NDArray[np.float64]) -> float:
        if not np.isfinite(theta).all():
            return float("inf")

        if (
            self.y_train_norm is None
            or self.y_train_norm.shape != self.y_train.shape
        ):
            self._set_target_scaling()

        d = self.x_train.shape[1]
        lengthscale = np.exp(theta[:d])
        variance = float(np.exp(theta[d]))
        if self.optimize_noise:
            noise = float(np.exp(theta[d + 1]))
        else:
            noise = float(self.noise)

        if variance <= 0 or noise <= 0 or np.any(lengthscale <= 0):
            return float("inf")

        if self.kernel_type == "rbf":
            k_xx = rbf_kernel(self.x_train, self.x_train, lengthscale, variance)
        elif self.kernel_type == "matern52":
            k_xx = matern52_kernel(self.x_train, self.x_train, lengthscale, variance)
        else:
            return float("inf")

        n = self.x_train.shape[0]
        k_train = k_xx + (noise + 1e-10) * np.eye(n, dtype=np.float64)
        try:
            l_chol = self._robust_cholesky(k_train)
        except np.linalg.LinAlgError:
            return float("inf")

        y_norm = self.y_train_norm
        assert y_norm is not None
        alpha = self._solve_cholesky(l_chol, y_norm)
        log_det_k = 2.0 * np.sum(np.log(np.diag(l_chol)))
        quad_term = float(y_norm.T @ alpha)
        return 0.5 * (quad_term + log_det_k + n * np.log(2.0 * np.pi))

    def _optimize_hyperparameters(self) -> None:
        theta0 = self._pack_hyperparameters()
        d = self.x_train.shape[1]
        bounds = [(-6.0, 4.0)] * d + [(-8.0, 6.0)]
        if self.optimize_noise:
            bounds.append((-12.0, 0.0))
        lower = np.array([b[0] for b in bounds], dtype=np.float64)
        upper = np.array([b[1] for b in bounds], dtype=np.float64)
        theta0 = np.clip(theta0, lower, upper)

        rng = np.random.default_rng(self.optimizer_seed)
        starts: list[NDArray[np.float64]] = [theta0]
        n_random = max(0, int(self.optimizer_restarts) - 1)
        for _ in range(n_random):
            starts.append(rng.uniform(lower, upper))

        best_fun: float | None = None
        best_x: NDArray[np.float64] | None = None
        best_success = False

        for start in starts:
            start_clipped = np.clip(np.asarray(start, dtype=np.float64), lower, upper)
            result = minimize(
                self._negative_log_marginal_likelihood,
                start_clipped,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.optimizer_maxiter},
            )
            if result.success and np.isfinite(result.fun):
                if best_fun is None or result.fun < best_fun:
                    best_fun = float(result.fun)
                    best_x = np.asarray(result.x, dtype=np.float64)
                    best_success = True

        if best_success and best_x is not None:
            self._unpack_hyperparameters(best_x)
        else:
            self._unpack_hyperparameters(theta0)
