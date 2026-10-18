"""Tests for the optional weight-conditioned neural AF (``n_weights > 0``)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from openbo.acquisition.naf import NeuralAF, af_weights, n_naf_features, naf_features, validate_weights
from openbo.optimizers.naf import (
    FamilyGPHyperparameters,
    NAFConfig,
    NAFSequentialOptimizer,
    load_naf,
    run_episodes,
    save_naf,
    train_naf,
    value_features,
)


@dataclass(frozen=True)
class WeightedPeaks:
    """A 1D two-objective task: peaks at 0.2 and 0.8, scalarized with ``(a, 1 - a)``."""

    a: float
    scaling: float = 1.0

    @property
    def weights(self) -> tuple[float, float]:
        return (self.a, 1.0 - self.a)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)[:, 0]
        f1 = 1.0 - 20.0 * (x - 0.2) ** 2
        f2 = 1.0 - 20.0 * (x - 0.8) ** 2
        return self.scaling * (self.a * f1 + (1.0 - self.a) * f2)

    @property
    def optimum(self) -> float:
        centre = 0.2 * self.a + 0.8 * (1.0 - self.a)
        return float(self(np.array([[centre]]))[0])


TOY_HP = FamilyGPHyperparameters(lengthscale=(0.2,), variance=1.0)
TOY_POOL = [WeightedPeaks(a, s) for a in (0.0, 1.0) for s in np.linspace(0.9, 1.1, 5)]


def _toy_config(**overrides: object) -> NAFConfig:
    settings: dict[str, object] = dict(
        dim=1,
        budget=3,
        hidden=(32, 32),
        n_local=50,
        n_iterations=2,
        steps_per_iteration=60,
        minibatch_size=20,
        learning_rate=3e-3,
        seed=0,
        n_weights=2,
    )
    settings.update(overrides)
    return NAFConfig(**settings)  # type: ignore[arg-type]


def test_weights_are_appended_to_every_feature_row() -> None:
    x = np.random.default_rng(0).random((4, 2))
    plain = naf_features(np.zeros(4), np.ones(4), x, 1, 20)
    weighted = naf_features(np.zeros(4), np.ones(4), x, 1, 20, weights=np.array([0.3, 0.7]))
    np.testing.assert_array_equal(weighted[:, : plain.shape[1]], plain)
    np.testing.assert_array_equal(weighted[:, -2:], np.tile([0.3, 0.7], (4, 1)))
    assert weighted.shape[1] == n_naf_features(2, n_weights=2) == n_naf_features(2) + 2


def test_af_weights_checks_what_the_af_reads() -> None:
    plain, weighted = NeuralAF(2), NeuralAF(2, n_weights=2)
    assert af_weights(plain, (0.3, 0.7)) is None  # a plain NAF ignores weights
    np.testing.assert_array_equal(af_weights(weighted, (0.3, 0.7)), [0.3, 0.7])
    with pytest.raises(ValueError, match="pass weights"):
        af_weights(weighted, None)
    with pytest.raises(ValueError, match="got 3"):
        af_weights(weighted, (0.2, 0.3, 0.5))
    with pytest.raises(ValueError, match="n_weights"):
        NeuralAF(2, n_weights=-1)
    with pytest.raises(ValueError, match="n_weights"):
        NAFConfig(dim=2, n_weights=-1)


def test_validate_weights_requires_the_simplex() -> None:
    np.testing.assert_array_equal(validate_weights((0.25, 0.75), 2), [0.25, 0.75])
    with pytest.raises(ValueError, match="3 entries"):
        validate_weights((0.5, 0.5), 3)
    with pytest.raises(ValueError, match="sum to 1"):
        validate_weights((1.2, -0.2), 2)


def test_value_features_append_the_step_weights() -> None:
    plain = value_features(np.array([0.0, 1.0]), 20)
    weighted = value_features(np.array([0.0, 1.0]), 20, np.array([[0.3, 0.7], [0.3, 0.7]]))
    assert plain.shape == (2, 2) and weighted.shape == (2, 4)
    torch.testing.assert_close(weighted[:, :2], plain)


def test_weights_change_the_suggestion_and_are_validated(tmp_path) -> None:
    # A hand-set AF on features [mu, sigma, x1, x2, t, T, w1, w2]: the logit is
    # relu(x1 + w1 - 1) - relu(x1 + w2 - 1), so +x1 for w = (1, 0) and -x1 for w = (0, 1).
    config = NAFConfig(dim=2, budget=20, hidden=(2,), n_local=50, n_weights=2)
    af = config.new_af()
    with torch.no_grad():
        first, last = af.net[0], af.net[2]
        first.weight.zero_()
        first.weight[0, 2] = first.weight[0, 6] = 1.0
        first.weight[1, 2] = first.weight[1, 7] = 1.0
        first.bias.fill_(-1.0)
        last.weight.copy_(torch.tensor([[1.0, -1.0]]))
        last.bias.zero_()
    hp = FamilyGPHyperparameters(lengthscale=(0.35, 0.35), variance=1.41)
    save_naf(tmp_path, af, config, hp)
    pick_1 = NAFSequentialOptimizer.from_directory(tmp_path, name="af", weights=(1.0, 0.0)).suggest()
    pick_2 = NAFSequentialOptimizer.from_directory(tmp_path, name="af", weights=(0.0, 1.0)).suggest()
    assert pick_1[0, 0] == 1.0 and pick_2[0, 0] == 0.0
    loaded, loaded_config, _ = load_naf(tmp_path)
    assert loaded_config.n_weights == 2 and loaded.n_weights == 2
    with pytest.raises(ValueError, match="pass weights"):
        NAFSequentialOptimizer.from_directory(tmp_path, name="af")
    with pytest.raises(ValueError, match="sum to 1"):
        NAFSequentialOptimizer.from_directory(tmp_path, name="af", weights=(0.5, 0.6))


def test_weighted_training_is_reproducible_and_runs_in_parallel() -> None:
    config = _toy_config()
    first = train_naf(config, TOY_HP, task_pool=TOY_POOL)
    second = train_naf(config, TOY_HP, task_pool=TOY_POOL)
    assert first.mean_return == second.mean_return
    parallel = train_naf(_toy_config(n_workers=2), TOY_HP, task_pool=TOY_POOL)
    assert len(parallel.mean_return) == 2
    # Episodes on a weighted task need a weight-conditioned AF to carry the weights.
    rollout = run_episodes(first.af, TOY_POOL[:2], TOY_HP, config.adaptive_set(), 3, np.random.default_rng(0))
    assert [tuple(w) for w in rollout.weights] == [TOY_POOL[0].weights] * 3 + [TOY_POOL[1].weights] * 3
