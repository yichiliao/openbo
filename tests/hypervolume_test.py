"""Tests for Pareto front and hypervolume utilities."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from openbo.benchmarks.hypervolume import (
    hypervolume,
    hypervolume_gain,
    nondominated_cells,
    pareto_front,
    pareto_mask,
)


def _hypervolume_inclusion_exclusion(y: np.ndarray, v_ref: np.ndarray) -> float:
    """Exact hypervolume by inclusion-exclusion over boxes (small n only)."""
    total = 0.0
    for size in range(1, y.shape[0] + 1):
        sign = 1.0 if size % 2 == 1 else -1.0
        for subset in combinations(range(y.shape[0]), size):
            corner = np.min(y[list(subset)], axis=0)
            total += sign * float(np.prod(np.maximum(corner - v_ref, 0.0)))
    return total


def _hypervolume_sweep_2d(y: np.ndarray, v_ref: np.ndarray) -> float:
    """Exact 2D hypervolume by sweeping points in decreasing first objective."""
    total = 0.0
    best_second = v_ref[1]
    for y1, y2 in sorted(y.tolist(), key=lambda row: -row[0]):
        if y1 <= v_ref[0] or y2 <= best_second:
            continue
        total += (y1 - v_ref[0]) * (y2 - best_second)
        best_second = y2
    return total


def test_pareto_mask_simple_example() -> None:
    """Dominated rows should be removed and trade-off rows kept."""
    y = np.array([[1.0, 3.0], [2.0, 2.0], [1.5, 1.5], [3.0, 1.0], [0.5, 0.5]])
    assert pareto_mask(y).tolist() == [True, True, False, True, False]


def test_pareto_mask_keeps_first_duplicate_only() -> None:
    """Identical non-dominated rows should appear once in the front."""
    y = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 2.0]])
    assert pareto_mask(y).tolist() == [True, True, False]
    assert pareto_front(y).shape == (2, 2)


def test_pareto_mask_weak_domination() -> None:
    """A row equal in one objective and worse in another is dominated."""
    y = np.array([[1.0, 2.0], [1.0, 1.0]])
    assert pareto_mask(y).tolist() == [True, False]


def test_pareto_mask_empty() -> None:
    """Empty input should give an empty mask."""
    assert pareto_mask(np.empty((0, 3))).shape == (0,)


@pytest.mark.parametrize("n_objectives", [2, 3])
def test_hypervolume_matches_inclusion_exclusion(n_objectives: int) -> None:
    """Grid-cell hypervolume should match inclusion-exclusion exactly."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.uniform(-1.0, 1.0, size=(7, n_objectives))
        v_ref = np.full(n_objectives, -1.2)
        expected = _hypervolume_inclusion_exclusion(y, v_ref)
        assert hypervolume(y, v_ref) == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_hypervolume_and_cells_with_ties() -> None:
    """Shared coordinates, duplicates and points on v_ref should stay exact."""
    rng = np.random.default_rng(7)
    for _ in range(100):
        n_objectives = int(rng.integers(1, 4))
        y = rng.integers(0, 4, size=(int(rng.integers(1, 8)), n_objectives)).astype(float)
        v_ref = np.zeros(n_objectives)
        w_ref = np.full(n_objectives, 4.0)
        hv = hypervolume(y, v_ref)
        assert hv == pytest.approx(_hypervolume_inclusion_exclusion(y, v_ref))

        lower, upper = nondominated_cells(y, v_ref, w_ref)
        free_volume = float(np.sum(np.prod(upper - lower, axis=1)))
        assert free_volume + hv == pytest.approx(4.0**n_objectives)

        y_new = rng.integers(0, 5, size=(4, n_objectives)).astype(float)
        expected = [hypervolume(np.vstack([y, row[None, :]]), v_ref) - hv for row in y_new]
        np.testing.assert_allclose(hypervolume_gain(y_new, lower, upper), expected, atol=1e-12)


def test_hypervolume_matches_2d_sweep_on_larger_sets() -> None:
    """2D hypervolume should match the sweep algorithm on larger sets."""
    rng = np.random.default_rng(1)
    for _ in range(10):
        y = rng.normal(size=(60, 2))
        v_ref = np.array([-3.0, -3.0])
        assert hypervolume(y, v_ref) == pytest.approx(
            _hypervolume_sweep_2d(y, v_ref), rel=1e-10
        )


def test_hypervolume_matches_monte_carlo_3d() -> None:
    """3D hypervolume should agree with a Monte-Carlo estimate."""
    rng = np.random.default_rng(2)
    y = rng.uniform(0.0, 1.0, size=(30, 3))
    v_ref = np.zeros(3)
    samples = rng.uniform(0.0, 1.0, size=(200_000, 3))
    dominated = np.any(np.all(y[None, :, :] >= samples[:, None, :], axis=2), axis=1)
    assert hypervolume(y, v_ref) == pytest.approx(float(np.mean(dominated)), abs=5e-3)


def test_hypervolume_ignores_points_below_reference() -> None:
    """Points not dominating v_ref should add no volume."""
    y = np.array([[2.0, 2.0]])
    v_ref = np.zeros(2)
    with_outside = np.vstack([y, [[3.0, -1.0]], [[-1.0, 5.0]]])
    assert hypervolume(y, v_ref) == pytest.approx(4.0)
    assert hypervolume(with_outside, v_ref) == pytest.approx(4.0)


def test_hypervolume_edge_cases() -> None:
    """Empty sets and a single point should give simple volumes."""
    v_ref = np.zeros(3)
    assert hypervolume(np.empty((0, 3)), v_ref) == 0.0
    assert hypervolume(np.array([[1.0, 2.0, 3.0]]), v_ref) == pytest.approx(6.0)


def test_hypervolume_is_monotone_and_ignores_dominated_points() -> None:
    """Adding a point never lowers hypervolume; dominated points add nothing."""
    rng = np.random.default_rng(3)
    y = rng.uniform(0.0, 1.0, size=(10, 3))
    v_ref = np.zeros(3)
    base = hypervolume(y, v_ref)

    dominated_point = pareto_front(y)[:1] - 0.01
    assert hypervolume(np.vstack([y, dominated_point]), v_ref) == pytest.approx(base)
    for _ in range(10):
        extra = rng.uniform(0.0, 1.0, size=(1, 3))
        assert hypervolume(np.vstack([y, extra]), v_ref) >= base - 1e-12


@pytest.mark.parametrize("n_objectives", [2, 3])
def test_cells_partition_the_box(n_objectives: int) -> None:
    """Non-dominated cells plus the hypervolume should fill the reference box."""
    rng = np.random.default_rng(4)
    y = rng.uniform(0.0, 1.0, size=(12, n_objectives))
    v_ref = np.zeros(n_objectives)
    w_ref = np.full(n_objectives, 1.5)
    lower, upper = nondominated_cells(y, v_ref, w_ref)

    assert np.all(upper > lower)
    free_volume = float(np.sum(np.prod(upper - lower, axis=1)))
    assert free_volume + hypervolume(y, v_ref) == pytest.approx(1.5**n_objectives)


def test_cells_are_not_dominated_by_front() -> None:
    """No front point should dominate any interior point of a returned cell."""
    rng = np.random.default_rng(5)
    y = rng.uniform(0.0, 1.0, size=(15, 3))
    lower, upper = nondominated_cells(y, np.zeros(3), np.ones(3))
    centers = 0.5 * (lower + upper)
    front = pareto_front(y)
    covered = np.any(np.all(front[None, :, :] >= centers[:, None, :], axis=2), axis=1)
    assert not np.any(covered)


def test_cells_with_empty_front_is_whole_box() -> None:
    """With no observations the only cell is the whole reference box."""
    lower, upper = nondominated_cells(np.empty((0, 2)), [0.0, -1.0], [2.0, 3.0])
    np.testing.assert_allclose(lower, [[0.0, -1.0]])
    np.testing.assert_allclose(upper, [[2.0, 3.0]])


@pytest.mark.parametrize("n_objectives", [2, 3])
def test_hypervolume_gain_matches_difference(n_objectives: int) -> None:
    """Cell-based gain should equal HV(Y + y) - HV(Y) for points below w_ref."""
    rng = np.random.default_rng(6)
    y = rng.uniform(0.0, 1.0, size=(10, n_objectives))
    v_ref = np.zeros(n_objectives)
    w_ref = np.ones(n_objectives)
    lower, upper = nondominated_cells(y, v_ref, w_ref)

    y_new = rng.uniform(0.0, 1.0, size=(25, n_objectives))
    gains = hypervolume_gain(y_new, lower, upper)
    base = hypervolume(y, v_ref)
    expected = [hypervolume(np.vstack([y, row[None, :]]), v_ref) - base for row in y_new]
    np.testing.assert_allclose(gains, expected, atol=1e-12)


def test_reference_validation() -> None:
    """Bad reference points should raise clear errors."""
    y = np.array([[1.0, 1.0]])
    with pytest.raises(ValueError, match="v_ref"):
        nondominated_cells(y, [0.0], [2.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        nondominated_cells(y, [0.0, 0.0], [np.inf, 2.0])
    with pytest.raises(ValueError, match="at least v_ref"):
        nondominated_cells(y, [0.0, 0.0], [-1.0, 2.0])
    with pytest.raises(ValueError, match=r"\(n, L\)"):
        hypervolume(np.array([1.0, 2.0]), [0.0, 0.0])
