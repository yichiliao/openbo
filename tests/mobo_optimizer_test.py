"""Tests for the MOBO optimizers (NumPy CEIPV/IEIPV and the BoTorch baseline)."""

from __future__ import annotations

import numpy as np
import pytest

from openbo.benchmarks.hypervolume import hypervolume, pareto_mask
from openbo.optimizers.mobo import (
    MOBOConfig,
    MOBOSequentialOptimizer,
    _value_and_grad,
    run_mo_random,
    run_mobo,
)
from openbo.optimizers.mobo_botorch import MOBoTorchConfig, run_mobo_botorch
from openbo.test_functions.multi_objective import get_mo_function_spec

SPEC = get_mo_function_spec("oka2")
BOUNDS = [(0.0, 1.0)] * SPEC.dim


def _config(**overrides: object) -> MOBOConfig:
    """Return a small, fast MOBO config on OKA2."""
    settings: dict[str, object] = dict(
        bounds=BOUNDS,
        v_ref=SPEC.v_ref,
        w_ref=SPEC.w_ref,
        n_candidates=64,
        n_starts=2,
        optimizer_maxiter=10,
        seed=0,
    )
    settings.update(overrides)
    return MOBOConfig(**settings)  # type: ignore[arg-type]


def _check_result(result, n_init: int, n_iter: int) -> None:
    """Shared checks on a MOBORunResult."""
    n = n_init + n_iter
    assert result.x_obs.shape == (n, SPEC.dim)
    assert result.y_obs.shape == (n, SPEC.n_objectives)
    assert np.all((result.x_obs >= 0.0) & (result.x_obs <= 1.0))
    np.testing.assert_allclose(result.y_obs, SPEC.objective(result.x_obs))
    assert result.hv_history.shape == (n_iter + 1,)
    assert np.all(np.diff(result.hv_history) >= -1e-12)
    assert result.hv_history[-1] == pytest.approx(hypervolume(result.y_obs, SPEC.v_ref))
    mask = pareto_mask(result.y_obs)
    np.testing.assert_allclose(result.pareto_y, result.y_obs[mask])
    np.testing.assert_allclose(result.pareto_x, result.x_obs[mask])


def test_value_and_grad_matches_analytic_gradient_and_respects_bounds() -> None:
    """Batched forward differences should match the true gradient, also at the bounds."""

    def fn(x: np.ndarray) -> np.ndarray:
        assert np.all((x >= 0.0) & (x <= 1.0))
        return -np.sum((x - 0.3) ** 2, axis=1)

    lower, upper = np.zeros(3), np.ones(3)
    for x in [np.array([0.1, 0.5, 0.9]), np.array([1.0, 0.0, 1.0])]:
        value, grad = _value_and_grad(fn, x, lower, upper)
        assert value == pytest.approx(fn(x[None])[0])
        np.testing.assert_allclose(grad, -2.0 * (x - 0.3), atol=1e-5)


@pytest.mark.parametrize(
    ("model", "acquisition", "n_rescore"),
    [("slf", "ceipv", None), ("mt", "ceipv", 16), ("independent", "ieipv", None)],
)
def test_run_mobo_produces_consistent_results(
    model: str, acquisition: str, n_rescore: int | None
) -> None:
    """Every model/acquisition combination should run and record valid results."""
    result = run_mobo(
        SPEC.objective,
        _config(model=model, acquisition=acquisition, n_rescore=n_rescore),
        n_iter=3,
    )
    _check_result(result, n_init=5, n_iter=3)
    state = result.final_model_state
    assert state["model"] == model and state["acquisition"] == acquisition
    assert np.asarray(state["output_correlation"]).shape == (2, 2)


def test_mobo_is_deterministic_for_a_seed() -> None:
    """The same seed should give the same suggestions."""
    first = run_mobo(SPEC.objective, _config(), n_iter=2)
    second = run_mobo(SPEC.objective, _config(), n_iter=2)
    np.testing.assert_allclose(first.x_obs, second.x_obs)


def test_ask_tell_interface_and_validation() -> None:
    """Suggestions should be (1, d) in bounds; bad inputs should raise."""
    optimizer = MOBOSequentialOptimizer(_config())
    with pytest.raises(ValueError, match="bootstrap"):
        optimizer.suggest()
    optimizer.bootstrap(SPEC.objective)
    x_next = optimizer.suggest()
    assert x_next.shape == (1, SPEC.dim)
    assert np.all((x_next >= 0.0) & (x_next <= 1.0))
    with pytest.raises(ValueError, match="y_new"):
        optimizer.observe(x_next, SPEC.objective(x_next)[:, 0])
    with pytest.raises(ValueError, match="x_new"):
        optimizer.observe(x_next[:, :2], SPEC.objective(x_next))
    with pytest.raises(ValueError, match="acquisition"):
        MOBOSequentialOptimizer(_config(acquisition="ehvi"))
    with pytest.raises(ValueError, match="same length"):
        MOBOSequentialOptimizer(_config(w_ref=[0.0, 0.0, 0.0]))


def test_random_baseline_layout_matches_mobo() -> None:
    """The random floor should use the same history layout as MOBO."""
    result = run_mo_random(SPEC.objective, BOUNDS, SPEC.v_ref, n_init=5, n_iter=4, seed=0)
    _check_result(result, n_init=5, n_iter=4)
    assert result.hv_history[0] == pytest.approx(hypervolume(result.y_obs[:5], SPEC.v_ref))


def test_botorch_baseline_runs() -> None:
    """The qLogNEHVI baseline should run and record valid results."""
    config = MOBoTorchConfig(
        bounds=BOUNDS, v_ref=SPEC.v_ref, num_restarts=2, raw_samples=32, mc_samples=32, seed=0
    )
    result = run_mobo_botorch(SPEC.objective, config, n_iter=2)
    _check_result(result, n_init=5, n_iter=2)
    assert result.final_model_state["acquisition"] == "qLogNEHVI"


def test_benchmark_dispatch_runs_every_cheap_method() -> None:
    """run_mobo_method should dispatch names and reject unknown ones."""
    from openbo.benchmarks.mobo_runner import get_mo_problem, run_mobo_method

    spec = get_mo_problem("slf_gp", rho=-0.5)
    assert spec.n_objectives == 2 and spec.dim == 3
    for method in ["random", "mobo_ceipv_mt", "mobo_ieipv"]:
        result = run_mobo_method(
            method, spec, n_init=4, n_iter=1, seed=0, **({} if method == "random" else {"n_candidates": 32})
        )
        assert result.y_obs.shape == (5, 2)
        assert result.hv_history.shape == (2,)
    with pytest.raises(ValueError, match="Unknown method"):
        run_mobo_method("parego", spec)


def test_slice_sampled_hyperparameters_run_and_continue_the_chain() -> None:
    """hyperparameters='slice' should run, record the mode and keep a chain state."""
    optimizer = MOBOSequentialOptimizer(
        _config(hyperparameters="slice", n_hyper_samples=3, slice_burn_in=2, slice_thin=1)
    )
    optimizer.bootstrap(SPEC.objective)
    for _ in range(2):
        x_next = optimizer.suggest()
        assert optimizer._chain_state is not None
        optimizer.observe(x_next, SPEC.objective(x_next))
    result = optimizer.result()
    _check_result(result, n_init=5, n_iter=2)
    assert result.final_model_state["hyperparameters"] == "slice"
    with pytest.raises(ValueError, match="hyperparameters"):
        MOBOSequentialOptimizer(_config(hyperparameters="nuts"))
