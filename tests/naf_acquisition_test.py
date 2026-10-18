"""Tests for the neural AF, its features and the adaptive candidate set (NAF, Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openbo.acquisition.naf import (
    AdaptiveSet,
    NeuralAF,
    gp_mean_std,
    n_naf_features,
    naf_features,
    naf_score_fn,
    regular_grid,
)
from openbo.optimizers.naf import BRANIN_GP_HYPERPARAMETERS
from openbo.test_functions.metabo_families import normalized_branin


def test_features_are_raw_mu_sigma_x_t_budget() -> None:
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    features = naf_features(np.array([1.0, -2.0]), np.array([0.5, 0.25]), x, t=3, budget=30)
    np.testing.assert_allclose(features, [[1.0, 0.5, 0.1, 0.2, 3, 30], [-2.0, 0.25, 0.3, 0.4, 3, 30]])
    without_x = naf_features(np.array([1.0, -2.0]), np.array([0.5, 0.25]), x, t=3, budget=30, use_x=False)
    assert without_x.shape == (2, n_naf_features(2, use_x=False)) == (2, 4)
    with pytest.raises(ValueError, match="t must be"):
        naf_features(np.zeros(2), np.ones(2), x, t=31, budget=30)


def test_neural_af_is_seeded_and_starts_near_uniform() -> None:
    af = NeuralAF(dim=2, seed=0)
    assert [m.out_features for m in af.net if isinstance(m, torch.nn.Linear)] == [200, 200, 200, 200, 1]
    features = naf_features(np.random.default_rng(0).normal(size=50), np.ones(50), np.random.default_rng(1).random((50, 2)), 0, 30)
    logits = af.logits(features)
    assert logits.shape == (50,)
    # N(0, 0.01^2) weights through four layers give logits near zero: a near-uniform policy.
    assert np.max(np.abs(logits)) < 1e-3
    np.testing.assert_array_equal(logits, NeuralAF(dim=2, seed=0).logits(features))
    assert not np.array_equal(logits, NeuralAF(dim=2, seed=1).logits(features))


def test_neural_af_forward_is_differentiable_and_batched() -> None:
    af = NeuralAF(dim=2, seed=0)
    features = torch.randn(3, 7, 6, dtype=torch.float32)
    out = af(features)
    assert out.shape == (3, 7)
    out.sum().backward()
    assert all(p.grad is not None for p in af.parameters())


def test_regular_grid_matches_the_reference_size() -> None:
    grid = regular_grid(31, 2)
    assert grid.shape == (961, 2)
    assert grid.min() == 0.0 and grid.max() == 1.0
    assert len(np.unique(grid[:, 0])) == 31


def test_adaptive_set_adds_local_maxima_near_the_best_grid_points() -> None:
    adaptive = AdaptiveSet.build(dim=2, seed=0)
    assert adaptive.size == 966 and adaptive.local_offsets.shape == (1000, 2)
    assert np.all(np.abs(adaptive.local_offsets) <= 1.0 / 31 + 1e-12)
    target = np.array([0.517, 0.283])

    def score(x: np.ndarray) -> np.ndarray:
        return -np.sum((x - target) ** 2, axis=1)

    points, scores = adaptive.candidates(score)
    assert points.shape == (966, 2) and scores.shape == (966,)
    np.testing.assert_allclose(scores, score(points))
    # The local points refine the best grid point, so the argmax beats every grid point.
    best = adaptive.maximize(score)
    assert score(best)[0] >= np.max(score(adaptive.global_grid))
    assert np.linalg.norm(best[0] - target) < 0.01
    assert np.all((points >= 0.0) & (points <= 1.0))


def test_local_clouds_are_clipped_at_the_boundary() -> None:
    adaptive = AdaptiveSet.build(dim=2, seed=0)
    best = adaptive.maximize(lambda x: x.sum(axis=1))
    np.testing.assert_allclose(best, [[1.0, 1.0]])


def test_gp_scores_use_the_prior_before_any_data() -> None:
    gp = BRANIN_GP_HYPERPARAMETERS.new_gp()
    x = np.random.default_rng(0).random((5, 2))
    mean, std = gp_mean_std(gp, x)
    np.testing.assert_allclose(mean, 0.0)
    np.testing.assert_allclose(std, np.sqrt(2.0))
    x_train = np.random.default_rng(1).random((8, 2))
    gp.fit(x_train, normalized_branin(x_train))
    mean, std = gp_mean_std(gp, x_train)
    np.testing.assert_allclose(mean, normalized_branin(x_train), atol=1e-4)
    assert np.all(std < 1e-3)
    af = NeuralAF(dim=2, seed=0)
    score = naf_score_fn(af, gp, t=8, budget=30)
    expected = af.logits(naf_features(*gp_mean_std(gp, x), x, 8, 30))
    np.testing.assert_allclose(score(x), expected)
    assert score(x).dtype == np.float64
