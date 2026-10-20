"""Multi-output GP for correlated objectives (MOBO).

All objectives are observed at the same inputs. With ``L`` objectives and
``S`` latent Matern-5/2 kernels ``k_s`` (unit variance, ARD lengthscales),
the covariance between objectives is

    Cov(f_l(x), f_l'(x')) = sum_s B_s[l, l'] k_s(x, x'),

where the ``L x L`` coregionalization matrices ``B_s`` depend on ``mode``:

- ``"independent"``: ``S = L`` and ``B_s = v_s e_s e_s^T``, i.e. one GP per
  objective with its own lengthscales and variance.
- ``"mt"`` (multi-task, Bonilla et al. 2008): ``S = 1`` and ``B = A A^T`` with
  ``A`` lower-triangular; every objective shares one input kernel.
- ``"slf"`` (semiparametric latent factor, Teh et al. 2004): ``S = L`` and
  ``B_s = a_s a_s^T``, where ``a_s`` is column ``s`` of a lower-triangular
  ``A``; each latent function has its own lengthscales.

Each objective has its own Gaussian noise. Targets are standardized per
objective. Hyperparameters are fitted by MAP with L-BFGS-B and analytic
gradients: log lengthscales and log noise variances have Gaussian priors, and
the entries of ``A`` (or the log variances in ``"independent"`` mode) have
standard normal priors, as in Shah and Ghahramani (2016, Section 4).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import LinAlgError, cho_solve, cholesky, solve_triangular
from scipy.linalg.lapack import dpotri
from scipy.optimize import minimize

from openbo.models.slice_sampling import slice_sample

Mode = Literal["independent", "mt", "slf"]

_SQRT5 = math.sqrt(5.0)
_LOG_LENGTHSCALE_BOUNDS = (math.log(1e-2), math.log(10.0))
_LOG_NOISE_BOUNDS = (math.log(1e-8), 0.0)
_COREG_BOUND = 5.0


def _matern52_and_grad_factor(
    sq_dist: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return unit-variance Matern-5/2 values and the lengthscale-gradient factor.

    ``sq_dist`` holds scaled squared distances ``r^2``. The derivative of the
    kernel with respect to ``log l_j`` is ``factor * (x_j - x'_j)^2 / l_j^2``.
    """
    r = np.sqrt(np.maximum(sq_dist, 0.0))
    exp_term = np.exp(-_SQRT5 * r)
    k = (1.0 + _SQRT5 * r + (5.0 / 3.0) * sq_dist) * exp_term
    factor = (5.0 / 3.0) * (1.0 + _SQRT5 * r) * exp_term
    return k, factor


@dataclass
class MultiOutputGP:
    """GP over ``L`` correlated objectives with MAP-fitted hyperparameters."""

    mode: Mode = "slf"
    optimize_hyperparameters: bool = True
    optimize_noise: bool = True
    noise: float = 1e-4
    optimizer_restarts: int = 4
    refit_restarts: int = 1
    optimizer_maxiter: int = 200
    optimizer_seed: int | None = None
    lengthscale_prior: tuple[float, float] = (math.log(0.5), 1.0)
    noise_prior: tuple[float, float] = (math.log(1e-3), 2.0)
    lengthscales: NDArray[np.float64] = field(init=False, repr=False)
    coregionalization: NDArray[np.float64] = field(init=False, repr=False)
    noise_variances: NDArray[np.float64] = field(init=False, repr=False)
    y_mean: NDArray[np.float64] = field(init=False, repr=False)
    y_std: NDArray[np.float64] = field(init=False, repr=False)

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        """Fit to inputs ``(n, d)`` and objective values ``(n, L)``.

        The first fit runs ``optimizer_restarts`` starts (the default
        hyperparameters plus draws from the prior). Later fits with matching
        shapes warm-start from the previous hyperparameters and run
        ``refit_restarts`` starts, which is much cheaper inside a BO loop.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if x_arr.ndim != 2 or x_arr.shape[0] == 0 or x_arr.shape[1] == 0:
            raise ValueError("x must have shape (n, d).")
        if y_arr.ndim != 2 or y_arr.shape[0] != x_arr.shape[0] or y_arr.shape[1] == 0:
            raise ValueError("y must have shape (n, L) and match x rows.")
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("x and y must be finite.")
        if self.mode not in ("independent", "mt", "slf"):
            raise ValueError(f"Unknown mode '{self.mode}'.")
        self.x_train = x_arr

        self.n_objectives = y_arr.shape[1]
        self.dim = self.x_train.shape[1]
        self.n_latent = 1 if self.mode == "mt" else self.n_objectives
        self.y_mean = y_arr.mean(axis=0)
        raw_std = y_arr.std(axis=0)
        self.y_std = np.where(raw_std > 1e-8, raw_std, 1.0)
        y_norm = (y_arr - self.y_mean) / self.y_std
        # Objective-major stacking: block l holds objective l at all inputs.
        self._y_vec = y_norm.T.reshape(-1)
        diff = self.x_train[:, None, :] - self.x_train[None, :, :]
        self._sq_diff = np.moveaxis(diff**2, 2, 0)

        previous = getattr(self, "_theta", None)
        restarts = self.refit_restarts
        if previous is None or previous.shape != (self._n_params(),):
            previous = self._default_theta()
            restarts = self.optimizer_restarts
        theta = (
            self._optimize(previous, restarts)
            if self.optimize_hyperparameters
            else previous
        )
        self._set_state(theta)

    def posterior(
        self, x_test: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the latent mean ``(m, L)`` and joint covariance ``(m, L, L)``."""
        x_star = np.asarray(x_test, dtype=np.float64)
        if x_star.ndim == 1:
            x_star = x_star[None, :]
        if x_star.ndim != 2 or x_star.shape[1] != self.dim:
            raise ValueError(f"x_test must have shape (m, {self.dim}) or ({self.dim},).")

        n, n_obj = self.x_train.shape[0], self.n_objectives
        m = x_star.shape[0]
        # cross[j, l, l' * n + i] = Cov(f_l(x*_j), y_l'(x_i)).
        cross = np.zeros((m, n_obj, n_obj, n), dtype=np.float64)
        for s in range(self.n_latent):
            scaled = (x_star[:, None, :] - self.x_train[None, :, :]) / self.lengthscales[s]
            k_star, _ = _matern52_and_grad_factor(np.sum(scaled**2, axis=2))
            cross += np.einsum("ab,ji->jabi", self.coregionalization[s], k_star)
        cross_flat = cross.reshape(m, n_obj, n_obj * n)

        mean_norm = cross_flat @ self._alpha
        v = solve_triangular(
            self._chol, cross_flat.reshape(m * n_obj, -1).T, lower=True
        ).T.reshape(m, n_obj, -1)
        prior_cov = self.coregionalization.sum(axis=0)
        cov_norm = prior_cov[None, :, :] - np.einsum("jak,jbk->jab", v, v)
        cov_norm = 0.5 * (cov_norm + np.transpose(cov_norm, (0, 2, 1)))

        scale = np.outer(self.y_std, self.y_std)
        mean = mean_norm * self.y_std + self.y_mean
        return mean.astype(np.float64), (cov_norm * scale).astype(np.float64)

    def output_correlation(self) -> NDArray[np.float64]:
        """Return the prior correlation between objectives at the same input."""
        cov = self.coregionalization.sum(axis=0)
        std = np.sqrt(np.diag(cov))
        return cov / np.outer(std, std)

    # Parameter vector: [log lengthscales (S * d), coregionalization params,
    # log noise variances (L, only when optimize_noise)].

    def _n_coreg(self) -> int:
        n_obj = self.n_objectives
        return n_obj if self.mode == "independent" else n_obj * (n_obj + 1) // 2

    def _n_params(self) -> int:
        n = self.n_latent * self.dim + self._n_coreg()
        return n + (self.n_objectives if self.optimize_noise else 0)

    def _split(
        self, theta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return log lengthscales ``(S, d)``, coregionalization params, log noise."""
        n_ls = self.n_latent * self.dim
        log_ls = theta[:n_ls].reshape(self.n_latent, self.dim)
        coreg = theta[n_ls : n_ls + self._n_coreg()]
        if self.optimize_noise:
            log_noise = theta[n_ls + self._n_coreg() :]
        else:
            log_noise = np.full(self.n_objectives, math.log(self.noise))
        return log_ls, coreg, log_noise

    def _lower_triangular(self, coreg: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.zeros((self.n_objectives, self.n_objectives), dtype=np.float64)
        a[np.tril_indices(self.n_objectives)] = coreg
        return a

    def _coregionalization(self, coreg: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``B`` with shape ``(S, L, L)``."""
        n_obj = self.n_objectives
        if self.mode == "independent":
            b = np.zeros((n_obj, n_obj, n_obj), dtype=np.float64)
            b[np.arange(n_obj), np.arange(n_obj), np.arange(n_obj)] = np.exp(coreg)
            return b
        a = self._lower_triangular(coreg)
        if self.mode == "mt":
            return (a @ a.T)[None, :, :]
        return np.einsum("ls,ks->slk", a, a)

    def _coreg_grad(
        self, coreg: NDArray[np.float64], m: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Chain ``dNLL/dB_s = m[s] / 2`` back to the coregionalization params."""
        n_obj = self.n_objectives
        if self.mode == "independent":
            return 0.5 * m[np.arange(n_obj), np.arange(n_obj), np.arange(n_obj)] * np.exp(coreg)
        a = self._lower_triangular(coreg)
        if self.mode == "mt":
            grad_a = m[0] @ a
        else:
            grad_a = np.stack([m[s] @ a[:, s] for s in range(n_obj)], axis=1)
        return grad_a[np.tril_indices(n_obj)]

    def _default_theta(self) -> NDArray[np.float64]:
        log_ls = np.full(self.n_latent * self.dim, math.log(0.5))
        if self.mode == "independent":
            coreg = np.zeros(self.n_objectives)
        else:
            coreg = np.eye(self.n_objectives)[np.tril_indices(self.n_objectives)]
            if self.mode == "slf":
                # Small off-diagonal mixing so every latent starts with gradient.
                coreg = np.where(coreg == 0.0, 0.1, coreg)
        parts = [log_ls, coreg]
        if self.optimize_noise:
            parts.append(np.full(self.n_objectives, math.log(self.noise)))
        return np.concatenate(parts)

    def _bounds(self) -> list[tuple[float, float]]:
        bounds = [_LOG_LENGTHSCALE_BOUNDS] * (self.n_latent * self.dim)
        bounds += [(-_COREG_BOUND, _COREG_BOUND)] * self._n_coreg()
        if self.optimize_noise:
            bounds += [_LOG_NOISE_BOUNDS] * self.n_objectives
        return bounds

    def _sample_theta(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw a restart point from the priors, clipped to the bounds."""
        mu_ls, sd_ls = self.lengthscale_prior
        parts = [
            rng.normal(mu_ls, sd_ls, size=self.n_latent * self.dim),
            rng.normal(0.0, 1.0, size=self._n_coreg()),
        ]
        if self.optimize_noise:
            mu_n, sd_n = self.noise_prior
            parts.append(rng.normal(mu_n, sd_n, size=self.n_objectives))
        bounds = np.array(self._bounds())
        return np.clip(np.concatenate(parts), bounds[:, 0], bounds[:, 1])

    def _full_covariance(
        self, theta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        """Return the stacked training covariance, kernels ``K_s`` and grad factors."""
        log_ls, coreg, log_noise = self._split(theta)
        b = self._coregionalization(coreg)
        n = self.x_train.shape[0]
        kernels, factors = [], []
        full = np.zeros((self.n_objectives * n, self.n_objectives * n), dtype=np.float64)
        for s in range(self.n_latent):
            inv_sq_ls = np.exp(-2.0 * log_ls[s])
            k_s, factor = _matern52_and_grad_factor(
                np.einsum("jik,j->ik", self._sq_diff, inv_sq_ls)
            )
            kernels.append(k_s)
            factors.append(factor)
            full += np.kron(b[s], k_s)
        full += np.diag(np.repeat(np.exp(log_noise), n) + 1e-8)
        return full, kernels, factors

    def _negative_log_posterior(
        self, theta: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the MAP objective (negative log marginal likelihood + priors) and gradient."""
        full, kernels, factors = self._full_covariance(theta)
        try:
            chol = cholesky(full, lower=True)
        except LinAlgError:
            return 1e25, np.zeros_like(theta)

        n_obj, n = self.n_objectives, self.x_train.shape[0]
        alpha = cho_solve((chol, True), self._y_vec)
        nll = (
            0.5 * float(self._y_vec @ alpha)
            + float(np.sum(np.log(np.diag(chol))))
            + 0.5 * n_obj * n * math.log(2.0 * math.pi)
        )

        # dNLL/dtheta = tr(W dK/dtheta) / 2 with W = K^{-1} - alpha alpha^T.
        k_inv, info = dpotri(chol, lower=1)
        if info != 0:
            return 1e25, np.zeros_like(theta)
        k_inv = np.tril(k_inv) + np.tril(k_inv, -1).T
        w = k_inv - np.outer(alpha, alpha)
        w_blocks = w.reshape(n_obj, n, n_obj, n)
        log_ls, coreg, log_noise = self._split(theta)
        b = self._coregionalization(coreg)

        grad_ls = np.empty((self.n_latent, self.dim), dtype=np.float64)
        m = np.empty((self.n_latent, n_obj, n_obj), dtype=np.float64)
        for s in range(self.n_latent):
            m[s] = np.tensordot(w_blocks, kernels[s], axes=([1, 3], [0, 1]))
            w_tilde = np.tensordot(b[s], w_blocks, axes=([0, 1], [0, 2]))
            inv_sq_ls = np.exp(-2.0 * log_ls[s])
            grad_ls[s] = 0.5 * np.einsum("ik,jik->j", w_tilde * factors[s], self._sq_diff) * inv_sq_ls
        grad_parts = [grad_ls.reshape(-1), self._coreg_grad(coreg, m)]
        if self.optimize_noise:
            traces = np.einsum("aiai->a", w_blocks)
            grad_parts.append(0.5 * np.exp(log_noise) * traces)
        grad = np.concatenate(grad_parts)

        prior, prior_grad = self._negative_log_prior(theta)
        return nll + prior, grad + prior_grad

    def _negative_log_prior(
        self, theta: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the negative log prior (up to constants) and its gradient."""
        grad = np.zeros_like(theta)
        mu_ls, sd_ls = self.lengthscale_prior
        n_ls = self.n_latent * self.dim
        prior = 0.5 * float(np.sum(((theta[:n_ls] - mu_ls) / sd_ls) ** 2))
        grad[:n_ls] = (theta[:n_ls] - mu_ls) / sd_ls**2
        coreg_slice = slice(n_ls, n_ls + self._n_coreg())
        prior += 0.5 * float(np.sum(theta[coreg_slice] ** 2))
        grad[coreg_slice] = theta[coreg_slice]
        if self.optimize_noise:
            mu_n, sd_n = self.noise_prior
            noise_slice = slice(n_ls + self._n_coreg(), None)
            prior += 0.5 * float(np.sum(((theta[noise_slice] - mu_n) / sd_n) ** 2))
            grad[noise_slice] = (theta[noise_slice] - mu_n) / sd_n**2
        return prior, grad

    def log_posterior(self, theta: ArrayLike) -> float:
        """Return the unnormalized log posterior of hyperparameters ``theta``.

        This is the negated MAP objective without its gradient, which is much
        cheaper and is all a slice sampler needs. It is ``-inf`` when the
        covariance is not positive definite.
        """
        theta_arr = np.asarray(theta, dtype=np.float64)
        full, _, _ = self._full_covariance(theta_arr)
        try:
            chol = cholesky(full, lower=True)
        except LinAlgError:
            return -math.inf
        alpha = cho_solve((chol, True), self._y_vec)
        nll = (
            0.5 * float(self._y_vec @ alpha)
            + float(np.sum(np.log(np.diag(chol))))
            + 0.5 * self._y_vec.shape[0] * math.log(2.0 * math.pi)
        )
        return -(nll + self._negative_log_prior(theta_arr)[0])

    def sample_hyperparameters(
        self,
        n_samples: int,
        rng: np.random.Generator,
        *,
        start: ArrayLike | None = None,
        burn_in: int = 20,
        thin: int = 2,
    ) -> NDArray[np.float64]:
        """Draw hyperparameter vectors from the posterior by slice sampling.

        Requires a prior call to :meth:`fit`. The chain starts at ``start``
        (default: the fitted hyperparameters) and stays within the MAP bounds.
        Returns an array of shape ``(n_samples, n_params)``.
        """
        if not hasattr(self, "_theta"):
            raise ValueError("Call fit() before sampling hyperparameters.")
        bounds = np.array(self._bounds())
        theta0 = np.clip(
            self._theta if start is None else np.asarray(start, dtype=np.float64),
            bounds[:, 0],
            bounds[:, 1],
        )
        return slice_sample(
            self.log_posterior,
            theta0,
            n_samples,
            rng,
            widths=1.0,
            burn_in=burn_in,
            thin=thin,
            lower=bounds[:, 0],
            upper=bounds[:, 1],
        )

    def with_hyperparameters(self, theta: ArrayLike) -> MultiOutputGP:
        """Return a copy fitted to the same data with hyperparameters ``theta``."""
        if not hasattr(self, "_theta"):
            raise ValueError("Call fit() before copying with new hyperparameters.")
        theta_arr = np.asarray(theta, dtype=np.float64)
        if theta_arr.shape != self._theta.shape:
            raise ValueError(f"theta must have shape {self._theta.shape}, got {theta_arr.shape}.")
        twin = copy.copy(self)
        twin._set_state(theta_arr)
        return twin

    def _optimize(self, theta0: NDArray[np.float64], restarts: int) -> NDArray[np.float64]:
        bounds = self._bounds()
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(self.optimizer_seed)
        starts = [np.clip(theta0, lower, upper)]
        starts += [self._sample_theta(rng) for _ in range(max(0, restarts - 1))]

        best_theta, best_value = starts[0], math.inf
        for start in starts:
            result = minimize(
                self._negative_log_posterior,
                start,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.optimizer_maxiter},
            )
            if np.isfinite(result.fun) and result.fun < best_value:
                best_value = float(result.fun)
                best_theta = np.asarray(result.x, dtype=np.float64)
        return best_theta

    def _set_state(self, theta: NDArray[np.float64]) -> None:
        """Store hyperparameters and the factorization used by ``posterior``."""
        self._theta = theta
        log_ls, coreg, log_noise = self._split(theta)
        self.lengthscales = np.exp(log_ls)
        self.coregionalization = self._coregionalization(coreg)
        self.noise_variances = np.exp(log_noise)
        full, _, _ = self._full_covariance(theta)
        jitter = 0.0
        for _ in range(8):
            try:
                self._chol = cholesky(full + jitter * np.eye(full.shape[0]), lower=True)
                break
            except LinAlgError:
                jitter = 1e-8 if jitter == 0.0 else jitter * 10.0
        else:
            raise LinAlgError("Cholesky failed even after jitter escalation.")
        self._alpha = cho_solve((self._chol, True), self._y_vec)
