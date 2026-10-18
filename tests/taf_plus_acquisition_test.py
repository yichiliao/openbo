"""Tests for the TAF+ acquisition (CHI 2024 on top of the original TAF)."""

from __future__ import annotations

import numpy as np
import pytest

from openbo.acquisition.ei import expected_improvement_maximization
from openbo.acquisition.taf_plus import (
    decay_factor,
    poe_weights,
    population_mean,
    predicted_improvement,
    ranking_distance,
    source_references,
    taf_plus_acquisition,
    taf_r_weights,
)

RNG = np.random.default_rng(0)
N, M, L = 7, 3, 2
T_MEAN = RNG.normal(size=(N, L))
T_VAR = RNG.uniform(0.1, 1.0, size=(N, L))
T_BEST = np.array([0.5, -0.2])
S_MEAN = RNG.normal(size=(M, N, L))
S_VAR = RNG.uniform(0.05, 0.5, size=(M, N, L))
S_REF = RNG.normal(size=(M, L))


def _original_taf(t_mean, t_var, t_best, s_mean, s_var, s_ref):
    """Original TAF-PoE (Wistuba et al. 2018, eqs. 38-39, 46-47), one objective, written out."""
    k = s_mean.shape[0] + 1
    w_t = 1.0 / t_var / k
    num = w_t * expected_improvement_maximization(t_mean, t_var, t_best)
    den = w_t.copy()
    for j in range(s_mean.shape[0]):
        w_j = 1.0 / s_var[j] / k
        num += w_j * np.maximum(s_mean[j] - s_ref[j], 0.0)
        den += w_j
    return num / den


def test_decay_matches_eq6_breakpoints() -> None:
    values = [decay_factor(t, d1=2, d2=0.3) for t in range(1, 8)]
    np.testing.assert_allclose(values, [1, 1, 0.7, 0.4, 0.1, 0, 0], atol=1e-12)


def test_decay_validation() -> None:
    with pytest.raises(ValueError, match="d1"):
        decay_factor(1, d1=-1, d2=0.1)
    with pytest.raises(ValueError, match="d1"):
        decay_factor(1, d1=1.5, d2=0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="d2"):
        decay_factor(1, d1=1, d2=0.0)
    with pytest.raises(ValueError, match="d2"):
        decay_factor(1, d1=1, d2=1.5)


def test_reference_is_the_best_source_mean_over_target_points() -> None:
    mean_at_target = np.array([[[1.0], [3.0], [2.0]], [[-1.0], [-4.0], [-2.0]]])
    np.testing.assert_allclose(source_references(mean_at_target), [[3.0], [-1.0]])
    with pytest.raises(ValueError, match="t >= 1"):
        source_references(np.zeros((2, 0, 1)))


def test_predicted_improvement_is_relu_of_the_mean() -> None:
    np.testing.assert_allclose(predicted_improvement([0.0, 1.0, 3.0], 1.0), [0.0, 0.0, 2.0])


def test_poe_weights_are_inverse_variance_over_model_count() -> None:
    np.testing.assert_allclose(poe_weights([[0.5], [2.0]]), [[1.0], [0.25]])


def test_single_objective_without_decay_is_the_original_taf() -> None:
    value = taf_plus_acquisition(
        T_MEAN[:, :1], T_VAR[:, :1], T_BEST[:1], S_MEAN[..., :1], S_VAR[..., :1], S_REF[:, :1], [1.0]
    )
    expected = _original_taf(T_MEAN[:, 0], T_VAR[:, 0], T_BEST[0], S_MEAN[..., 0], S_VAR[..., 0], S_REF[:, 0])
    np.testing.assert_allclose(value, expected, rtol=1e-12)


def test_full_decay_leaves_only_the_target_ei() -> None:
    alpha = np.array([0.3, 0.7])
    value = taf_plus_acquisition(T_MEAN, T_VAR, T_BEST, S_MEAN, S_VAR, S_REF, alpha, decay=0.0)
    ei = np.stack([expected_improvement_maximization(T_MEAN[:, k], T_VAR[:, k], T_BEST[k]) for k in range(L)], axis=1)
    np.testing.assert_allclose(value, ei @ alpha, rtol=1e-12)


def test_no_sources_gives_the_target_ei() -> None:
    alpha = np.array([0.5, 0.5])
    value = taf_plus_acquisition(T_MEAN, T_VAR, T_BEST, np.zeros((0, N, L)), np.zeros((0, N, L)), np.zeros((0, L)), alpha)
    ei = np.stack([expected_improvement_maximization(T_MEAN[:, k], T_VAR[:, k], T_BEST[k]) for k in range(L)], axis=1)
    np.testing.assert_allclose(value, ei @ alpha, rtol=1e-12)


def test_within_and_between_model_combination_match_eqs_3_to_5() -> None:
    alpha = np.array([0.4, 0.6])
    decay = 0.35
    value = taf_plus_acquisition(T_MEAN, T_VAR, T_BEST, S_MEAN, S_VAR, S_REF, alpha, decay=decay)

    n_models = M + 1
    ei = np.stack([expected_improvement_maximization(T_MEAN[:, k], T_VAR[:, k], T_BEST[k]) for k in range(L)], axis=1)
    ei_plus = sum(alpha[k] * ei[:, k] for k in range(L))
    w_t_plus = sum(alpha[k] / T_VAR[:, k] / n_models for k in range(L))
    num = w_t_plus * ei_plus
    den = w_t_plus.copy()
    for j in range(M):
        pei_plus = sum(alpha[k] * max_(S_MEAN[j, :, k] - S_REF[j, k]) for k in range(L))
        w_j_plus = decay * sum(alpha[k] / S_VAR[j, :, k] / n_models for k in range(L))
        num += w_j_plus * pei_plus
        den += w_j_plus
    np.testing.assert_allclose(value, num / den, rtol=1e-12)


def max_(a: np.ndarray) -> np.ndarray:
    return np.maximum(a, 0.0)


def test_one_hot_objective_weights_select_one_objective() -> None:
    both = taf_plus_acquisition(T_MEAN, T_VAR, T_BEST, S_MEAN, S_VAR, S_REF, [0.0, 1.0], decay=0.5)
    only_second = taf_plus_acquisition(
        T_MEAN[:, 1:], T_VAR[:, 1:], T_BEST[1:], S_MEAN[..., 1:], S_VAR[..., 1:], S_REF[:, 1:], [1.0], decay=0.5
    )
    np.testing.assert_allclose(both, only_second, rtol=1e-12)


def test_ranking_distance_counts_discordant_ordered_pairs() -> None:
    y = np.array([0.0, 1.0, 2.0])
    assert ranking_distance(y, [10.0, 20.0, 30.0]) == 0.0
    assert ranking_distance(y, [30.0, 20.0, 10.0]) == 1.0
    # Swapping the top two disagrees on 1 of 3 unordered pairs.
    assert ranking_distance(y, [0.0, 2.0, 1.0]) == pytest.approx(1.0 / 3.0)
    assert ranking_distance([1.0], [5.0]) == 0.0


def test_taf_r_weights_use_the_epanechnikov_kernel() -> None:
    y = np.array([[0.0], [1.0], [2.0]])
    mean = np.array([[[0.0], [1.0], [2.0]], [[2.0], [1.0], [0.0]], [[0.0], [2.0], [1.0]]])
    sources, target = taf_r_weights(y, mean, rho=1.0)
    assert target == pytest.approx(0.75)
    np.testing.assert_allclose(sources[:, 0], [0.75, 0.0, 0.75 * (1 - 1 / 9)])


def test_taf_r_mode_uses_constant_weights() -> None:
    w_sources = np.array([[0.75, 0.0], [0.5, 0.25], [0.0, 0.0]])
    alpha = np.array([0.5, 0.5])
    value = taf_plus_acquisition(
        T_MEAN, T_VAR, T_BEST, S_MEAN, S_VAR, S_REF, alpha, weight_mode="taf_r", ranking_weights=(w_sources, 0.75)
    )
    ei = np.stack([expected_improvement_maximization(T_MEAN[:, k], T_VAR[:, k], T_BEST[k]) for k in range(L)], axis=1)
    pei = max_(S_MEAN - S_REF[:, None, :]) @ alpha
    ws = w_sources @ alpha
    expected = (0.75 * (ei @ alpha) + np.sum(ws[:, None] * pei, axis=0)) / (0.75 + ws.sum())
    np.testing.assert_allclose(value, expected, rtol=1e-12)


def test_validation() -> None:
    args = (T_MEAN, T_VAR, T_BEST, S_MEAN, S_VAR, S_REF)
    with pytest.raises(ValueError, match="sum to 1"):
        taf_plus_acquisition(*args, [0.5, 0.6])
    with pytest.raises(ValueError, match="non-negative"):
        taf_plus_acquisition(*args, [1.5, -0.5])
    with pytest.raises(ValueError, match="length 2"):
        taf_plus_acquisition(*args, [1.0])
    with pytest.raises(ValueError, match="decay"):
        taf_plus_acquisition(*args, [0.5, 0.5], decay=1.5)
    with pytest.raises(ValueError, match="weight_mode"):
        taf_plus_acquisition(*args, [0.5, 0.5], weight_mode="taf_m")
    with pytest.raises(ValueError, match="ranking_weights"):
        taf_plus_acquisition(*args, [0.5, 0.5], weight_mode="taf_r")
    with pytest.raises(ValueError, match="source_reference"):
        taf_plus_acquisition(T_MEAN, T_VAR, T_BEST, S_MEAN, S_VAR, S_REF[:2], [0.5, 0.5])


def test_population_mean_is_the_poe_mean_of_the_sources() -> None:
    alpha = np.array([0.4, 0.6])
    value = population_mean(S_MEAN, S_VAR, alpha)
    w = (1.0 / S_VAR / M) @ alpha
    expected = np.sum(w * (S_MEAN @ alpha), axis=0) / np.sum(w, axis=0)
    np.testing.assert_allclose(value, expected, rtol=1e-12)
    np.testing.assert_allclose(population_mean(S_MEAN, S_VAR, alpha, "taf_r"), (S_MEAN @ alpha).mean(axis=0))
    with pytest.raises(ValueError, match="M >= 1"):
        population_mean(np.zeros((0, N, L)), np.zeros((0, N, L)), alpha)
