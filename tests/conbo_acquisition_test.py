"""Tests for ConBO's weight schedule and mixed EI."""

from __future__ import annotations

import numpy as np
import pytest

from openbo.acquisition.conbo import (
    default_incumbents,
    mixed_expected_improvement,
    population_weight,
)
from openbo.acquisition.ei import expected_improvement_maximization

POP = (np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.5, 0.1]))
USER = (np.array([2.0, 0.5, -1.0]), np.array([0.2, 0.2, 0.2]))


def test_weight_schedule_matches_the_paper_breakpoints() -> None:
    weights = [population_weight(t, alpha1=5, alpha2=0.2) for t in range(1, 13)]
    np.testing.assert_allclose(weights, [1, 1, 1, 1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0], atol=1e-12)


def test_weight_schedule_with_zero_hold() -> None:
    assert population_weight(1, alpha1=0, alpha2=0.5) == pytest.approx(0.5)
    assert population_weight(2, alpha1=0, alpha2=0.5) == 0.0


def test_weight_schedule_validation() -> None:
    with pytest.raises(ValueError, match="t must"):
        population_weight(0)
    with pytest.raises(ValueError, match="alpha1"):
        population_weight(1, alpha1=-1)
    with pytest.raises(ValueError, match="alpha2"):
        population_weight(1, alpha2=0)


def test_mixed_ei_interpolates_the_two_eis() -> None:
    ei_pop = expected_improvement_maximization(*POP, 1.0)
    ei_user = expected_improvement_maximization(*USER, 0.5)
    for w in (0.0, 0.3, 1.0):
        mixed = mixed_expected_improvement(w, POP, USER, best_pop=1.0, best_user=0.5)
        np.testing.assert_allclose(mixed, w * ei_pop + (1 - w) * ei_user)


def test_endpoints_need_only_one_model() -> None:
    np.testing.assert_allclose(
        mixed_expected_improvement(1.0, POP, None, 1.0, 1.0),
        expected_improvement_maximization(*POP, 1.0),
    )
    np.testing.assert_allclose(
        mixed_expected_improvement(0.0, None, USER, 1.0, 1.0),
        expected_improvement_maximization(*USER, 1.0),
    )


def test_handover_moves_the_argmax_from_population_to_user() -> None:
    picks = [
        int(np.argmax(mixed_expected_improvement(population_weight(t), POP, USER, 1.0, 1.0)))
        for t in range(1, 11)
    ]
    assert picks[0] == 2  # the population model prefers the last candidate
    assert picks[-1] == 0  # the user model prefers the first


def test_mixed_ei_validation() -> None:
    with pytest.raises(ValueError, match="weight"):
        mixed_expected_improvement(1.5, POP, USER, 0.0, 0.0)
    with pytest.raises(ValueError, match="pop_prediction"):
        mixed_expected_improvement(0.5, None, USER, 0.0, 0.0)
    with pytest.raises(ValueError, match="user_prediction"):
        mixed_expected_improvement(0.5, POP, None, 0.0, 0.0)
    with pytest.raises(ValueError, match="same candidates"):
        mixed_expected_improvement(0.5, POP, (np.zeros(2), np.ones(2)), 0.0, 0.0)


def test_incumbents_use_user_best_or_population_mean() -> None:
    assert default_incumbents([0.1, 0.7, 0.3], POP[0]) == (0.7, 0.7)
    assert default_incumbents([], POP[0]) == (2.0, 2.0)
    with pytest.raises(ValueError, match="Need user"):
        default_incumbents([], [])
