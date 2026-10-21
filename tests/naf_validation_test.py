"""Regression checks for NAF checkpoint selection and atomic observations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openbo.models.gp_scratch import GPScratch
from openbo.optimizers.naf import (
    BOEpisode,
    FamilyGPHyperparameters,
    NAFConfig,
    NAFSequentialOptimizer,
    save_naf,
    train_naf,
)
from rust.tools.export_python_naf import export_naf


def _config() -> NAFConfig:
    return NAFConfig(
        dim=1, budget=2, hidden=(8,), points_per_dim=5, n_local_maxima=2,
        n_local=8, n_iterations=1, steps_per_iteration=4, epochs=1,
        minibatch_size=2, learning_rate=1e-2,
    )


HP = FamilyGPHyperparameters((0.25,), 1.0)


class Peak:
    optimum = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return -(x[:, 0] - 0.37) ** 2


def test_best_policy_is_the_policy_whose_returns_were_evaluated() -> None:
    config = _config()
    initial = config.new_af()
    result = train_naf(config, HP, task_pool=[Peak()])
    assert result.best_iteration == 0
    for before, best in zip(initial.parameters(), result.best_af.parameters()):
        torch.testing.assert_close(before, best, rtol=0, atol=0)
    assert any(not torch.equal(a, b) for a, b in zip(initial.parameters(), result.af.parameters()))


@pytest.mark.parametrize(
    ("x", "y"),
    [([[np.nan]], [0.0]), ([[0.5]], [np.inf]), ([[1.1]], [0.0]),
     ([[-0.1]], [0.0]), (np.empty((0, 1)), []), ([[0.1], [0.2], [0.3]], [1.0, 2.0, 3.0])],
)
def test_invalid_observations_leave_optimizer_unchanged(x, y) -> None:
    config = _config()
    optimizer = NAFSequentialOptimizer(config.new_af(), HP, config.budget, config.adaptive_set())
    before = optimizer.suggest()
    gp = optimizer.gp
    with pytest.raises(ValueError):
        optimizer.observe(np.asarray(x), np.asarray(y))
    assert optimizer.t == 0 and optimizer.gp is gp
    np.testing.assert_array_equal(optimizer.suggest(), before)


def test_failed_gp_fit_does_not_commit_observations(monkeypatch) -> None:
    config = _config()
    optimizer = NAFSequentialOptimizer(config.new_af(), HP, config.budget, config.adaptive_set())
    optimizer.observe(np.array([[0.2]]), np.array([0.1]))
    before_x, before_y, gp = optimizer.x_obs.copy(), optimizer.y_obs.copy(), optimizer.gp

    def fail_fit(self, x, y):
        self.x_train = np.zeros((10, 1))
        raise np.linalg.LinAlgError("simulated factorization failure")

    monkeypatch.setattr(GPScratch, "fit", fail_fit)
    with pytest.raises(np.linalg.LinAlgError):
        optimizer.observe(np.array([[0.3]]), np.array([0.2]))
    np.testing.assert_array_equal(optimizer.x_obs, before_x)
    np.testing.assert_array_equal(optimizer.y_obs, before_y)
    assert optimizer.gp is gp and gp.x_train.shape == (1, 1)


def test_nonfinite_task_does_not_poison_episode() -> None:
    config = _config()
    episode = BOEpisode(lambda x: np.full(x.shape[0], np.nan), HP, config.adaptive_set(), config.budget)
    episode.observation(config.new_af())
    with pytest.raises(ValueError, match="finite"):
        episode.step(0)
    assert episode.t == 0 and episode.candidates is not None
    with pytest.raises(ValueError, match="bounds"):
        episode.step(-1)


def test_portable_export_contains_weights_configuration_and_parity_cases(tmp_path) -> None:
    config = _config()
    af = config.new_af()
    save_naf(tmp_path / "python", af, config, HP, name="best_af")
    output = tmp_path / "portable.json"
    exported = export_naf(tmp_path / "python", output, samples=3)
    assert output.is_file()
    assert exported["format"] == "openbo-python-naf-export"
    assert exported["config"]["dim"] == 1
    assert exported["gp_config"]["standardize_targets"] is False
    assert len(exported["state_dict"]) == len(af.state_dict())
    assert len(exported["parity_cases"]) == 3
    for case in exported["parity_cases"]:
        np.testing.assert_array_equal(af.logits(np.array(case["features"])), case["logits"])
    with pytest.raises(ValueError, match="filename"):
        export_naf(tmp_path / "python", output, name="../escape")
