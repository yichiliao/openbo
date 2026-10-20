"""Regression coverage for upstream integration and model replay."""

import json

import numpy as np
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize

from openbo.benchmarks.mobo_runner import run_mobo_method
from openbo.models.gp_scratch import GPScratch
from openbo.models.multi_output_gp import MultiOutputGP
from openbo.optimizers.bo_botorch import (
    BoTorchConfig,
    BoTorchSequentialOptimizer,
    _gp_state_from_single_task_gp,
)
from openbo.optimizers.bo_taf import _load_source_surrogates
from openbo.optimizers.mobo_botorch import MOBoTorchConfig, MOBoTorchSequentialOptimizer, run_mobo_botorch
from openbo.test_functions.multi_objective import get_mo_function_spec


def test_upstream_mobo_dispatcher_accepts_config_and_returns_its_result_type():
    spec = get_mo_function_spec("oka2")
    result = run_mobo_method("mobo_botorch", spec, n_init=3, n_iter=0, seed=0)
    assert result.x_obs.shape == (3, spec.dim)
    assert result.hv_history.shape == (1,)
    assert result.pareto_x.shape[0] == result.pareto_y.shape[0]
    assert result.final_model_state["acquisition"] == "qLogNEHVI"


def test_reference_point_aliases_must_agree():
    config = MOBoTorchConfig(bounds=[(0.0, 1.0)], v_ref=[-1.0, -1.0])
    assert config.ref_point == config.v_ref
    with pytest.raises(ValueError, match="must agree"):
        MOBoTorchConfig(bounds=config.bounds, ref_point=[0.0, 0.0], v_ref=[-1.0, -1.0])


@pytest.mark.parametrize("calling_style", ["config_keyword", "iterations_positional", "original"])
def test_botorch_helper_keeps_both_public_calling_conventions(calling_style):
    config = MOBoTorchConfig(bounds=[(0.0, 1.0)], v_ref=[-1.0, -1.0], n_init=3)

    def objective(x):
        return np.column_stack([-x[:, 0]**2, -(1 - x[:, 0])**2])

    if calling_style == "config_keyword":
        result = run_mobo_botorch(objective, config=config, n_iter=0)
    elif calling_style == "iterations_positional":
        result = run_mobo_botorch(objective, config, 0)
    else:
        legacy = run_mobo_botorch(objective, config.bounds, [-1.0, -1.0], n_init=3, n_iter=0)
        assert legacy.hypervolume_history.shape == (3,)
        return
    assert result.hv_history.shape == (1,)
    assert result.x_obs.shape == (3, 1)


@pytest.mark.parametrize("mode", ["independent", "mt", "slf"])
@pytest.mark.parametrize("n", [1, 3])
def test_constant_multi_output_targets_retain_uncertainty(mode, n):
    gp = MultiOutputGP(mode=mode, optimize_hyperparameters=False)
    gp.fit(np.linspace(0, 0.2, n)[:, None], np.full((n, 2), 2.5))
    mean, covariance = gp.posterior([[0.9]])
    assert np.all(np.isfinite(mean))
    assert np.all(np.diagonal(covariance, axis1=1, axis2=2) > 0.01)


@pytest.mark.parametrize("kind", ["scratch", "multi_output"])
def test_nonfinite_refit_does_not_poison_previous_gp(kind):
    gp = GPScratch() if kind == "scratch" else MultiOutputGP(optimize_hyperparameters=False)
    x = np.array([[0.1], [0.8]])
    y = np.array([1.0, 2.0]) if kind == "scratch" else np.array([[1.0, 2.0], [2.0, 3.0]])
    gp.fit(x, y)
    before = gp.posterior([[0.5]])
    with pytest.raises(ValueError, match="finite"):
        gp.fit(x, np.full_like(y, np.nan))
    after = gp.posterior([[0.5]])
    for expected, actual in zip(before, after):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("multi", [False, True])
def test_botorch_rejects_nonfinite_observations_without_changing_state(multi):
    optimizer = (
        MOBoTorchSequentialOptimizer(MOBoTorchConfig(bounds=[(0.0, 1.0)], v_ref=[-1.0, -1.0]))
        if multi else BoTorchSequentialOptimizer(BoTorchConfig(bounds=[(0.0, 1.0)]))
    )
    y = np.array([[np.nan, 1.0]]) if multi else np.array([np.nan])
    with pytest.raises(ValueError, match="finite"):
        optimizer.observe(np.array([[0.5]]), y)
    assert optimizer.x_obs.shape[0] == optimizer.y_obs.shape[0] == 0


def test_saved_botorch_source_replays_mean_variance_and_scaling(tmp_path):
    x = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.double)
    y = torch.tensor([[8.0], [10.0], [9.0], [15.0]], dtype=torch.double)
    model = SingleTaskGP(x, y, outcome_transform=Standardize(m=1))
    model.mean_module.constant.data.fill_(1.7)
    model.eval()
    state = _gp_state_from_single_task_gp(model, dim=1, n_observations=4)
    (tmp_path / "gp_states").mkdir()
    (tmp_path / "trajectories").mkdir()
    (tmp_path / "gp_states" / "source.json").write_text(json.dumps({"gp_state": state}))
    (tmp_path / "trajectories" / "source.json").write_text(
        json.dumps({"x_values": x.tolist(), "y_values": y.squeeze(-1).tolist()})
    )
    restored = _load_source_surrogates(tmp_path)[0].gp
    query = torch.tensor([[0.15], [0.5], [0.95]], dtype=torch.double)
    with torch.no_grad():
        expected = model.posterior(query)
    mean, variance = restored.posterior(query.numpy())
    np.testing.assert_allclose(mean, expected.mean.squeeze(-1).numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(variance, expected.variance.squeeze(-1).numpy(), rtol=1e-6, atol=1e-8)
