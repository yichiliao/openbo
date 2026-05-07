"""Tests for scratch acquisition and BO loops."""

from __future__ import annotations

import numpy as np

from metabo.acquisition.ei import expected_improvement_maximization
from metabo.optimizers.bo_botorch import run_bo_botorch
from metabo.optimizers.bo_scratch import run_bo_scratch
from metabo.test_functions.registry import get_function_spec


def test_scratch_ei_shape() -> None:
    """EI output shape should match mean/variance shape."""
    mean = np.array([0.1, -0.2, 0.0], dtype=np.float64)
    variance = np.array([0.5, 0.2, 1.0], dtype=np.float64)
    ei = expected_improvement_maximization(mean, variance, best_y=0.3)
    assert ei.shape == (3,)


def test_scratch_bo_runs_small_loop() -> None:
    """Scratch BO should run and produce expected trajectory length."""
    spec = get_function_spec("branin")
    result = run_bo_scratch(spec.objective, spec.bounds, n_init=3, n_iter=2, seed=0)
    assert result.x_obs.shape == (5, 2)
    assert result.y_obs.shape == (5,)
    assert result.best_y_history.shape == (3,)


def test_scratch_bo_runs_small_loop_with_matern52() -> None:
    """Scratch BO should run with Matern-5/2 ARD kernel and hyperparameter learning."""
    spec = get_function_spec("branin")
    result = run_bo_scratch(
        spec.objective,
        spec.bounds,
        n_init=3,
        n_iter=2,
        kernel_type="matern52",
        optimize_hyperparameters=True,
        seed=0,
    )
    assert result.x_obs.shape == (5, 2)
    assert result.y_obs.shape == (5,)
    assert result.best_y_history.shape == (3,)


def test_botorch_bo_runs_small_loop() -> None:
    """BoTorch BO should run for a tiny setup."""
    spec = get_function_spec("branin")
    result = run_bo_botorch(spec.objective, spec.bounds, n_init=3, n_iter=2, seed=0)
    assert result.x_obs.shape == (5, 2)
    assert result.y_obs.shape == (5,)
    assert result.best_y_history.shape == (3,)
