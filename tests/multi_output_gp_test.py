"""Tests for the multi-output GP used by MOBO."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from openbo.models.gp_scratch import GPScratch
from openbo.models.multi_output_gp import MultiOutputGP
from openbo.test_functions.multi_objective import make_slf_gp_problem

MODES = ["independent", "mt", "slf"]


def _toy_data(n: int = 15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return inputs (n, 3) and three loosely related objectives (n, 3)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 3))
    y = np.stack(
        [
            np.sin(4.0 * x[:, 0]) + x[:, 1],
            -np.sin(4.0 * x[:, 0]) + 0.3 * x[:, 2],
            x.sum(axis=1),
        ],
        axis=1,
    )
    return x, y


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("optimize_noise", [True, False])
def test_map_gradient_matches_finite_differences(mode: str, optimize_noise: bool) -> None:
    """The analytic MAP gradient should match finite differences."""
    x, y = _toy_data()
    gp = MultiOutputGP(mode=mode, optimize_noise=optimize_noise, optimize_hyperparameters=False)
    gp.fit(x, y)
    rng = np.random.default_rng(1)
    for _ in range(3):
        theta = gp._sample_theta(rng)
        _, grad = gp._negative_log_posterior(theta)
        grad_fd = approx_fprime(theta, lambda t: gp._negative_log_posterior(t)[0], 1e-6)
        scale = max(1.0, float(np.max(np.abs(grad_fd))))
        assert np.max(np.abs(grad - grad_fd)) / scale < 1e-4


def test_independent_mode_matches_gp_scratch() -> None:
    """With the same hyperparameters, independent mode equals one GPScratch per objective."""
    x, y = _toy_data(n=20)
    gp = MultiOutputGP(mode="independent", optimizer_seed=0)
    gp.fit(x, y)
    x_test = np.random.default_rng(2).uniform(size=(7, 3))
    mean, cov = gp.posterior(x_test)

    for k in range(y.shape[1]):
        single = GPScratch(
            lengthscale=gp.lengthscales[k],
            variance=float(gp.coregionalization[k, k, k]),
            noise=float(gp.noise_variances[k]),
            kernel_type="matern52",
        )
        single.fit(x, y[:, k])
        mean_k, var_k = single.posterior(x_test)
        np.testing.assert_allclose(mean[:, k], mean_k, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(cov[:, k, k], var_k, rtol=1e-4, atol=1e-8)
    off_diagonal = cov[:, ~np.eye(y.shape[1], dtype=bool)]
    np.testing.assert_allclose(off_diagonal, 0.0, atol=1e-10)


@pytest.mark.parametrize("mode", MODES)
def test_posterior_shapes_and_psd_covariance(mode: str) -> None:
    """Posterior covariances should be symmetric positive semi-definite (L x L)."""
    x, y = _toy_data()
    gp = MultiOutputGP(mode=mode, optimizer_seed=0)
    gp.fit(x, y)
    mean, cov = gp.posterior(np.random.default_rng(3).uniform(size=(25, 3)))
    assert mean.shape == (25, 3)
    assert cov.shape == (25, 3, 3)
    np.testing.assert_allclose(cov, np.transpose(cov, (0, 2, 1)))
    assert np.all(np.linalg.eigvalsh(cov) > -1e-9)

    mean_one, cov_one = gp.posterior(x[0])
    assert mean_one.shape == (1, 3) and cov_one.shape == (1, 3, 3)


@pytest.mark.parametrize("mode", ["independent", "slf"])
def test_interpolates_noise_free_data(mode: str) -> None:
    """At training inputs the posterior should reproduce noise-free targets."""
    x, y = _toy_data(n=25)
    gp = MultiOutputGP(mode=mode, optimizer_seed=0)
    gp.fit(x, y)
    mean, cov = gp.posterior(x)
    scale = y.std(axis=0)
    assert np.max(np.abs(mean - y) / scale) < 0.05
    assert np.max(np.diagonal(cov, axis1=1, axis2=2) / scale**2) < 0.01


@pytest.mark.parametrize("mode", MODES)
def test_target_scaling_is_equivariant(mode: str) -> None:
    """Affinely rescaled targets should give rescaled posteriors.

    Standardization makes the rescaled problem identical up to round-off, which
    can move the optimizer slightly, so ``gp_b`` reuses ``gp_a``'s
    hyperparameters to test the scaling code exactly.
    """
    x, y = _toy_data()
    shift, scale = np.array([3.0, -1.0, 0.5]), np.array([10.0, 0.1, 2.0])
    gp_a = MultiOutputGP(mode=mode, optimizer_seed=0)
    gp_a.fit(x, y)
    gp_b = MultiOutputGP(mode=mode, optimize_hyperparameters=False)
    gp_b._theta = gp_a._theta.copy()
    gp_b.fit(x, y * scale + shift)
    x_test = np.random.default_rng(4).uniform(size=(5, 3))
    mean_a, cov_a = gp_a.posterior(x_test)
    mean_b, cov_b = gp_b.posterior(x_test)
    np.testing.assert_allclose(mean_b, mean_a * scale + shift, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(cov_b, cov_a * np.outer(scale, scale), rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize("mode", ["mt", "slf"])
@pytest.mark.parametrize("rho", [-0.9, 0.9])
def test_correlated_modes_recover_correlation_sign(mode: str, rho: float) -> None:
    """On SLF-GP draws, the fitted output correlation should have the sign of rho."""
    spec = make_slf_gp_problem(rho=rho, seed=0, n_reference=1024)
    x = np.random.default_rng(5).uniform(size=(40, 3))
    gp = MultiOutputGP(mode=mode, optimizer_seed=0)
    gp.fit(x, spec.objective(x))
    corr = gp.output_correlation()[0, 1]
    assert np.sign(corr) == np.sign(rho)
    assert abs(corr) > 0.5


def test_mt_captures_perfect_negative_correlation() -> None:
    """If f2 = -f1 exactly, the multi-task model should learn correlation near -1."""
    x, y = _toy_data(n=20)
    gp = MultiOutputGP(mode="mt", optimizer_seed=0)
    gp.fit(x, np.stack([y[:, 0], -y[:, 0]], axis=1))
    assert gp.output_correlation()[0, 1] < -0.99


def test_fit_is_deterministic_and_warm_refit_is_consistent() -> None:
    """The same seed should give the same fit; refitting the same data keeps it."""
    x, y = _toy_data()
    gp_a = MultiOutputGP(mode="slf", optimizer_seed=7)
    gp_b = MultiOutputGP(mode="slf", optimizer_seed=7)
    gp_a.fit(x, y)
    gp_b.fit(x, y)
    np.testing.assert_allclose(gp_a._theta, gp_b._theta)

    value_before = gp_a._negative_log_posterior(gp_a._theta)[0]
    gp_a.fit(x, y)
    assert gp_a._negative_log_posterior(gp_a._theta)[0] <= value_before + 1e-6


def test_input_validation() -> None:
    """Bad shapes and unknown modes should raise clear errors."""
    x, y = _toy_data()
    with pytest.raises(ValueError, match=r"\(n, L\)"):
        MultiOutputGP().fit(x, y[:, 0])
    with pytest.raises(ValueError, match=r"\(n, d\)"):
        MultiOutputGP().fit(x[:, 0], y)
    with pytest.raises(ValueError, match="Unknown mode"):
        MultiOutputGP(mode="lmc").fit(x, y)  # type: ignore[arg-type]
    gp = MultiOutputGP(optimize_hyperparameters=False)
    gp.fit(x, y)
    with pytest.raises(ValueError, match="x_test"):
        gp.posterior(np.zeros((2, 4)))
