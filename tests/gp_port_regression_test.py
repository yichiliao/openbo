"""Regressions for GP defects discovered while validating the native Rust port."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from openbo.models.gp_scratch import GPScratch
from openbo.models.kernels import matern52_kernel, rbf_kernel


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
def test_noiseless_likelihood_is_finite_and_optimization_improves_it(kernel: str) -> None:
    rng = np.random.default_rng(0)
    x = rng.random((10, 2))
    y = np.sin(4.0 * x[:, 0]) + 0.3 * np.cos(3.0 * x[:, 1])
    gp = GPScratch(
        lengthscale=1.5,
        noise=0.0,
        kernel_type=kernel,
        optimizer_restarts=1,
        optimizer_maxiter=40,
        optimizer_seed=0,
    )
    gp.fit(x, y)
    before = gp._negative_log_marginal_likelihood(gp._pack_hyperparameters())
    assert np.isfinite(before)

    gp.optimize_hyperparameters = True
    gp.fit(x, y)
    after = gp._negative_log_marginal_likelihood(gp._pack_hyperparameters())
    assert np.isfinite(after)
    assert after < before - 0.1
    assert gp.noise == 0.0


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
@pytest.mark.parametrize("parameter", ["lengthscale", "variance"])
def test_bounded_optimization_cannot_worsen_out_of_box_initial_model(
    kernel: str, parameter: str
) -> None:
    if parameter == "lengthscale":
        x = np.array([[0.0], [1e-4], [1.0]])
        y = np.array([0.0, 1.0, 0.0])
        lengthscale, variance, noise = 1e-5, 1.0, 1e-6
    else:
        x = np.array([[0.0], [0.4], [1.0]])
        y = np.array([0.0, 1e-4, 0.0])
        lengthscale, variance, noise = 0.3, 1e-8, 1e-12
    gp = GPScratch(
        lengthscale=lengthscale,
        variance=variance,
        noise=noise,
        kernel_type=kernel,
        standardize_targets=False,
        optimizer_maxiter=10,
        optimizer_restarts=1,
        optimizer_seed=0,
    )
    gp.fit(x, y)
    before = gp._negative_log_marginal_likelihood(gp._pack_hyperparameters())
    gp.optimize_hyperparameters = True
    gp.fit(x, y)
    after = gp._negative_log_marginal_likelihood(gp._pack_hyperparameters())
    assert np.isfinite(before)
    assert after <= before + 1e-10


def test_zero_initial_noise_can_be_optimized_without_log_zero_warning() -> None:
    x = np.array([[0.0], [0.25], [0.7], [1.0]])
    y = np.array([0.0, 0.8, 0.3, -0.7])
    gp = GPScratch(
        noise=0.0,
        optimize_noise=True,
        optimize_hyperparameters=True,
        optimizer_restarts=1,
        optimizer_maxiter=10,
        optimizer_seed=0,
    )
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        gp.fit(x, y)
        theta = gp._pack_hyperparameters()
    assert np.isfinite(theta).all()
    assert np.isfinite(gp._negative_log_marginal_likelihood(theta))
    assert gp.noise >= 0.0


def test_iteration_limit_can_keep_a_verified_likelihood_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.linspace(0.0, 1.0, 8)[:, None]
    y = np.sin(8.0 * x[:, 0])
    gp = GPScratch(lengthscale=2.0, optimizer_restarts=1)
    gp.fit(x, y)
    before = gp._negative_log_marginal_likelihood(gp._pack_hyperparameters())
    candidate = np.log(np.array([0.2, 1.0]))
    assert gp._negative_log_marginal_likelihood(candidate) < before
    monkeypatch.setattr(
        "openbo.models.gp_scratch.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False, x=candidate, fun=-1e9),
    )
    gp.optimize_hyperparameters = True
    gp.fit(x, y)
    np.testing.assert_allclose(gp.lengthscale, [0.2], rtol=1e-14)
    assert gp._negative_log_marginal_likelihood(gp._pack_hyperparameters()) < before


def test_optimizer_status_and_reported_fun_cannot_replace_a_better_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.array([[0.0], [1e-4], [1.0]])
    y = np.array([0.0, 1.0, 0.0])
    gp = GPScratch(lengthscale=1e-5, standardize_targets=False, optimizer_restarts=1)
    gp.fit(x, y)
    before = gp.posterior(x)
    original_lengthscale = np.asarray(gp.lengthscale).copy()
    original_variance, original_noise = gp.variance, gp.noise
    monkeypatch.setattr(
        "openbo.models.gp_scratch.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=True, x=np.array([-6.0, 0.0]), fun=-1e9
        ),
    )
    gp.optimize_hyperparameters = True
    gp.fit(x, y)
    np.testing.assert_array_equal(gp.lengthscale, original_lengthscale)
    assert gp.variance == original_variance
    assert gp.noise == original_noise
    for actual, expected in zip(gp.posterior(x), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "stage",
    [
        "_resolve_lengthscale",
        "_set_target_scaling",
        "_optimize_hyperparameters",
        "_kernel",
        "_robust_cholesky",
        "_solve_cholesky",
    ],
)
def test_every_failed_refit_preserves_full_previous_state(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    gp = GPScratch(
        lengthscale=0.7,
        variance=1.3,
        noise=1e-3,
        mean_constant=0.27,
        fixed_target_mean=2.3,
        fixed_target_std=0.8,
    )
    x = np.array([[0.0, 0.0], [0.4, 0.9], [1.0, 1.0]])
    gp.fit(x, np.array([1.0, -2.0, 0.7]))
    query = np.array([[0.1, 0.3], [0.7, 0.6]])
    expected = gp.posterior(query)
    gp.optimize_hyperparameters = stage == "_optimize_hyperparameters"
    snapshot = vars(gp).copy()

    def fail(candidate: GPScratch, *args: object, **kwargs: object) -> None:
        # Exercise rollback even after several derived attributes have changed.
        candidate.variance = 13.0
        candidate.noise = 0.5
        candidate.mean_constant = -1.2
        candidate.y_mean = 7.0
        candidate.y_std = 2.0
        candidate.temporary_fit_attribute = True  # type: ignore[attr-defined]
        raise RuntimeError("injected fit failure")

    with monkeypatch.context() as context:
        context.setattr(GPScratch, stage, fail)
        with pytest.raises(RuntimeError, match="injected fit failure"):
            gp.fit(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), np.array([7.0, 9.0]))
    assert vars(gp).keys() == snapshot.keys()
    assert all(value is snapshot[key] for key, value in vars(gp).items())
    for actual, previous in zip(gp.posterior(query), expected):
        np.testing.assert_array_equal(actual, previous)


def test_failed_initial_fit_leaves_no_partial_training_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gp = GPScratch(lengthscale=0.7)
    before = vars(gp).copy()

    def fail(*args: object, **kwargs: object) -> None:
        raise np.linalg.LinAlgError("injected Cholesky failure")

    with monkeypatch.context() as context:
        context.setattr(GPScratch, "_robust_cholesky", fail)
        with pytest.raises(np.linalg.LinAlgError):
            gp.fit(np.array([[0.0], [1.0]]), np.array([1.0, 2.0]))
    assert vars(gp).keys() == before.keys()
    assert all(value is before[key] for key, value in vars(gp).items())
    gp.fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([1.0, 2.0]))
    assert gp.x_train.shape == (2, 2)
    np.testing.assert_array_equal(gp.lengthscale, [0.7, 0.7])


def test_validation_failure_keeps_previous_fitted_predictions() -> None:
    gp = GPScratch()
    gp.fit(np.array([[0.0], [1.0]]), np.array([1.0, 2.0]))
    query = np.array([[0.3], [0.8]])
    expected = gp.posterior(query)
    with pytest.raises(ValueError, match="finite"):
        gp.fit(np.array([[0.0], [np.nan]]), np.array([3.0, 4.0]))
    for actual, previous in zip(gp.posterior(query), expected):
        np.testing.assert_array_equal(actual, previous)


@pytest.mark.parametrize("kernel", [rbf_kernel, matern52_kernel])
def test_extreme_finite_coordinates_have_finite_stationary_covariances(kernel: object) -> None:
    x = np.array([[0.0], [1e200], [-1e200], [np.finfo(np.float64).max]])
    with np.errstate(all="raise"):
        covariance = kernel(x, x, 1.0, 1.7)  # type: ignore[operator]
    np.testing.assert_array_equal(covariance, 1.7 * np.eye(len(x)))
    with np.errstate(all="raise"):
        covariance = kernel(x, x, 1e-308, 1.7)  # type: ignore[operator]
    np.testing.assert_array_equal(covariance, 1.7 * np.eye(len(x)))


@pytest.mark.parametrize("kernel", [rbf_kernel, matern52_kernel])
def test_overflowing_difference_is_scaled_before_subtraction_when_needed(kernel: object) -> None:
    x = np.array([[1e308], [-1e308]])
    with np.errstate(all="raise"):
        actual = kernel(x, x, 1e308, 1.3)  # type: ignore[operator]
        expected = kernel(np.array([[1.0], [-1.0]]), np.array([[1.0], [-1.0]]), 1.0, 1.3)  # type: ignore[operator]
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()
    assert actual[0, 1] > 0.0


@pytest.mark.parametrize("kernel", [rbf_kernel, matern52_kernel])
@pytest.mark.parametrize("parameter", ["variance", "scalar_lengthscale", "ard_lengthscale"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_kernel_hyperparameters_must_be_finite(
    kernel: object, parameter: str, nonfinite: float
) -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    lengthscale = (
        nonfinite if parameter == "scalar_lengthscale"
        else np.array([1.0, nonfinite]) if parameter == "ard_lengthscale"
        else 1.0
    )
    variance = nonfinite if parameter == "variance" else 1.0
    with pytest.raises(ValueError, match="finite"):
        kernel(x, x, lengthscale, variance)  # type: ignore[operator]


@pytest.mark.parametrize("kernel", [rbf_kernel, matern52_kernel])
def test_large_variance_preserves_representable_exponential_tails(kernel: object) -> None:
    x = np.array([[0.0]])
    q = np.array([[40.0]]) if kernel is rbf_kernel else np.array([[750.0 / np.sqrt(5.0)]])
    with np.errstate(all="raise"):
        value = kernel(x, q, 1.0, 1e308)[0, 0]  # type: ignore[operator]
    if kernel is rbf_kernel:
        expected = np.exp(np.log(1e308) - 800.0)
    else:
        expected = np.exp(np.log(1e308) + np.log(1.0 + 750.0 + 750.0**2 / 3.0) - 750.0)
    assert value > 0.0
    np.testing.assert_allclose(value, expected, rtol=2e-12)


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
def test_extreme_finite_gp_training_inputs_keep_covariance_and_predictions_finite(kernel: str) -> None:
    gp = GPScratch(kernel_type=kernel, lengthscale=1e-308)
    x = np.array([[0.0], [1e200], [-1e200]])
    with np.errstate(all="raise"):
        gp.fit(x, np.array([1.0, 2.0, -1.0]))
        mean, variance = gp.posterior(np.array([[0.0], [1e200], [1e300]]))
    assert np.isfinite(gp.k_train).all()
    assert np.isfinite(gp.l_chol).all()
    assert np.isfinite(gp.alpha).all()
    assert np.isfinite(mean).all()
    assert np.isfinite(variance).all()
    assert np.all(variance > 0.0)


@pytest.mark.parametrize("cache", ["y_train_norm", "k_train", "l_chol", "alpha", "y_mean", "y_std"])
def test_nonfinite_fitted_cache_is_rejected_before_commit(
    cache: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    gp = GPScratch()
    gp.fit(np.array([[0.0], [1.0]]), np.array([1.0, 2.0]))
    query = np.array([[0.3], [0.8]])
    expected = gp.posterior(query)
    before = vars(gp).copy()
    build = GPScratch._fit_in_place

    def poison(candidate: GPScratch, x: np.ndarray, y: np.ndarray) -> None:
        build(candidate, x, y)
        setattr(candidate, cache, np.nan if cache in {"y_mean", "y_std"} else np.array([np.nan]))

    with monkeypatch.context() as context:
        context.setattr(GPScratch, "_fit_in_place", poison)
        with pytest.raises(ValueError, match="non-finite state"):
            gp.fit(np.array([[0.1], [0.7]]), np.array([3.0, -1.0]))
    assert vars(gp).keys() == before.keys()
    assert all(value is before[key] for key, value in vars(gp).items())
    for actual, previous in zip(gp.posterior(query), expected):
        np.testing.assert_array_equal(actual, previous)


def test_target_scaling_overflow_does_not_publish_an_invalid_gp() -> None:
    gp = GPScratch()
    gp.fit(np.array([[0.0], [1.0]]), np.array([1.0, 2.0]))
    query = np.array([[0.3]])
    expected = gp.posterior(query)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="non-finite state"),
    ):
        gp.fit(np.array([[0.1], [0.7]]), np.array([-1e200, 1e200]))
    for actual, previous in zip(gp.posterior(query), expected):
        np.testing.assert_array_equal(actual, previous)
