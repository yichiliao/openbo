"""Tests for the EIPV acquisitions (Monte Carlo, IEIPV and CEIPV)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from openbo.acquisition.mobo import (
    ceipv,
    ceipv_terms,
    eipv_mc,
    ieipv,
    log_ceipv,
)
from openbo.benchmarks.hypervolume import hypervolume_gain, nondominated_cells

FRONT = np.stack([np.linspace(0.0, 1.0, 6), 1.0 - np.linspace(0.0, 1.0, 6) ** 2], axis=1)
V_REF = np.array([-0.5, -0.5])
W_REF = np.array([1.5, 1.5])


def _covariances(sd: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Return 2x2 covariances from standard deviations (m, 2) and correlations (m,)."""
    cov = np.empty((sd.shape[0], 2, 2))
    cov[:, 0, 0] = sd[:, 0] ** 2
    cov[:, 1, 1] = sd[:, 1] ** 2
    cov[:, 0, 1] = cov[:, 1, 0] = rho * sd[:, 0] * sd[:, 1]
    return cov


def _random_predictions(m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return correlated predictive means and covariances near FRONT."""
    rng = np.random.default_rng(seed)
    mean = rng.uniform(0.0, 1.1, size=(m, 2))
    sd = rng.uniform(0.03, 0.4, size=(m, 2))
    rho = rng.choice([-0.9, -0.5, 0.0, 0.5, 0.9], size=m)
    return mean, _covariances(sd, rho)


def test_ramp_moments_match_the_corrected_lambda() -> None:
    """The ramp (y - v) on (v, w] has mass d^2/2, mean (2w+v)/3 and variance d^2/18."""
    v, w = -0.3, 1.7
    mass = quad(lambda y: y - v, v, w)[0]
    mean = quad(lambda y: y * (y - v), v, w)[0] / mass
    var = quad(lambda y: (y - mean) ** 2 * (y - v), v, w)[0] / mass
    assert mass == pytest.approx(0.5 * (w - v) ** 2)
    assert mean == pytest.approx((2.0 * w + v) / 3.0)
    assert mean != pytest.approx((2.0 * w + v) / 2.0)  # the value printed in the paper
    assert var == pytest.approx((w - v) ** 2 / 18.0)

    # Paper-form terms use exactly these moments, cell by cell.
    terms = ceipv_terms(FRONT, V_REF, W_REF, form="paper")
    lower, upper = nondominated_cells(FRONT, V_REF, W_REF)
    order_terms = np.lexsort(terms.center.T)
    order_cells = np.lexsort(((2.0 * upper + lower) / 3.0).T)
    np.testing.assert_allclose(terms.center[order_terms], ((2.0 * upper + lower) / 3.0)[order_cells])
    np.testing.assert_allclose(terms.tau2[order_terms], ((upper - lower) ** 2 / 18.0)[order_cells])


def test_terms_are_shift_equivariant() -> None:
    """Shifting the front and references should shift every term's center only."""
    shift = np.array([3.0, -2.0])
    base = ceipv_terms(FRONT, V_REF, W_REF)
    moved = ceipv_terms(FRONT + shift, V_REF + shift, W_REF + shift)
    np.testing.assert_allclose(moved.center, base.center + shift)
    np.testing.assert_allclose(moved.tau2, base.tau2, atol=1e-12)
    np.testing.assert_allclose(moved.log_weight, base.log_weight, atol=1e-9)


@pytest.mark.parametrize("n_objectives", [2, 3])
def test_exact_decomposition_preserves_gain_moments(n_objectives: int) -> None:
    """Total term mass and mean should equal the integrals of the exact gain.

    Moment matching keeps each factor's mass and mean, so with no overflow
    layer ``sum_t weight_t = int gain(y) dy`` and ``sum_t weight_t center_t =
    int y gain(y) dy`` over the box, exactly when the decomposition is right.
    """
    rng = np.random.default_rng(0)
    raw = rng.uniform(size=(30 if n_objectives == 2 else 10, n_objectives))
    front = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    v_ref, w_ref = np.zeros(n_objectives), np.full(n_objectives, 1.2)
    terms = ceipv_terms(front, v_ref, w_ref, overflow=0.0)
    weights = np.exp(terms.log_weight)

    # Midpoint quadrature of the exact gain over the box.
    n_grid = 400 if n_objectives == 2 else 60
    axis = (np.arange(n_grid) + 0.5) / n_grid * 1.2
    points = np.stack(np.meshgrid(*[axis] * n_objectives, indexing="ij"), axis=-1).reshape(-1, n_objectives)
    lower, upper = nondominated_cells(front, v_ref, w_ref)
    chunk = max(1, 2_000_000 // lower.shape[0])
    gains = np.concatenate(
        [hypervolume_gain(points[s : s + chunk], lower, upper) for s in range(0, points.shape[0], chunk)]
    )
    cell_volume = (1.2 / n_grid) ** n_objectives
    np.testing.assert_allclose(weights.sum(), gains.sum() * cell_volume, rtol=2e-3)
    np.testing.assert_allclose(
        weights @ terms.center, (gains @ points) * cell_volume, rtol=2e-3
    )


@pytest.mark.parametrize("form", ["exact", "paper"])
def test_ieipv_is_exact_for_independent_objectives(form: str) -> None:
    """With diagonal covariances IEIPV should match Monte Carlo within its error."""
    mean, cov = _random_predictions(30, seed=1)
    cov = cov * np.eye(2)
    lower, upper = nondominated_cells(FRONT, V_REF, W_REF)
    estimate, stderr = eipv_mc(
        mean, cov, lower, upper, n_samples=50_000, rng=np.random.default_rng(2), form=form
    )
    closed = ieipv(mean, cov, lower, upper, form=form)
    assert np.all(np.abs(closed - estimate) <= 5.0 * stderr + 1e-9)


def test_paper_form_undercounts_the_gain() -> None:
    """The literal eq. 3 integrand should be far below the exact gain."""
    mean, cov = _random_predictions(20, seed=3)
    lower, upper = nondominated_cells(FRONT, V_REF, W_REF)
    exact, _ = eipv_mc(mean, cov, lower, upper, n_samples=4000, rng=np.random.default_rng(4))
    paper, _ = eipv_mc(
        mean, cov, lower, upper, n_samples=4000, rng=np.random.default_rng(4), form="paper"
    )
    assert np.all(paper <= exact + 1e-12)
    assert paper.sum() < 0.5 * exact.sum()


@pytest.mark.parametrize("form", ["exact", "paper"])
def test_ceipv_tracks_monte_carlo_and_beats_ieipv(form: str) -> None:
    """CEIPV should approximate correlated EIPV better than ignoring correlation."""
    mean, cov = _random_predictions(80, seed=0)
    lower, upper = nondominated_cells(FRONT, V_REF, W_REF)
    estimate, _ = eipv_mc(
        mean, cov, lower, upper, n_samples=40_000, rng=np.random.default_rng(1), form=form
    )
    approx = ceipv(mean, cov, ceipv_terms(FRONT, V_REF, W_REF, form=form))
    independent = ieipv(mean, cov, lower, upper, form=form)
    ok = estimate > 1e-3
    rel_c = np.abs(approx - estimate)[ok] / estimate[ok]
    rel_i = np.abs(independent - estimate)[ok] / estimate[ok]
    assert np.median(rel_c) < 0.06
    assert np.median(rel_c) < 0.5 * np.median(rel_i)


def test_ceipv_ranks_by_correlation_like_monte_carlo() -> None:
    """Same marginals, opposite correlations: CEIPV should prefer what MC prefers.

    Near a concave front, negative correlation spreads samples along the front
    and positive correlation spreads them across it, so the two candidates
    differ in true EIPV while IEIPV cannot tell them apart.
    """
    mean = np.array([[0.7, 0.7], [0.7, 0.7]])
    cov = _covariances(np.full((2, 2), 0.25), np.array([0.9, -0.9]))
    lower, upper = nondominated_cells(FRONT, V_REF, W_REF)
    estimate, stderr = eipv_mc(mean, cov, lower, upper, n_samples=50_000, rng=np.random.default_rng(5))
    assert abs(estimate[0] - estimate[1]) > 10.0 * stderr.max()
    approx = ceipv(mean, cov, ceipv_terms(FRONT, V_REF, W_REF))
    independent = ieipv(mean, cov, lower, upper)
    assert np.sign(approx[0] - approx[1]) == np.sign(estimate[0] - estimate[1])
    assert independent[0] == pytest.approx(independent[1])


def test_three_objectives_track_monte_carlo() -> None:
    """CEIPV should also approximate EIPV with three correlated objectives."""
    rng = np.random.default_rng(6)
    raw = rng.uniform(size=(200, 3))
    front = (raw / np.linalg.norm(raw, axis=1, keepdims=True))[:12]
    v_ref, w_ref = np.zeros(3), np.full(3, 1.2)
    mean = rng.uniform(0.3, 1.0, size=(15, 3))
    cov = np.empty((15, 3, 3))
    for i in range(15):
        a = rng.normal(size=(3, 3)) * 0.15
        cov[i] = a @ a.T + 1e-4 * np.eye(3)
    lower, upper = nondominated_cells(front, v_ref, w_ref)
    estimate, _ = eipv_mc(mean, cov, lower, upper, n_samples=8000, rng=np.random.default_rng(7))
    approx = ceipv(mean, cov, ceipv_terms(front, v_ref, w_ref))
    ok = estimate > 1e-3
    assert np.median(np.abs(approx - estimate)[ok] / estimate[ok]) < 0.1


def test_pruning_matches_full_sum_and_stays_finite_far_away() -> None:
    """Pruned CEIPV should match the full sum; far candidates keep a finite log.

    Each skipped term is below ``exp(-7^2 / 2)`` of its own peak, which can
    still be ~1e-5 of a small total, hence the tolerance.
    """
    mean, cov = _random_predictions(40, seed=8)
    terms = ceipv_terms(FRONT, V_REF, W_REF)
    np.testing.assert_allclose(
        ceipv(mean, cov, terms), ceipv(mean, cov, terms, prune_z=None), rtol=1e-4, atol=1e-12
    )
    # Deep in the dominated region with tiny uncertainty: EIPV underflows, its log does not.
    far = log_ceipv(np.array([[-0.4, -0.4]]), 1e-6 * np.eye(2)[None], terms)
    assert np.isfinite(far[0])
    near = log_ceipv(np.array([[0.9, 0.9]]), 1e-2 * np.eye(2)[None], terms)
    assert near[0] > far[0]


def test_edge_cases_and_validation() -> None:
    """Empty gains and malformed inputs should be handled clearly."""
    # A front at the upper corner leaves no non-dominated cells.
    terms = ceipv_terms(W_REF[None], V_REF, W_REF, overflow=0.0)
    assert terms.center.shape == (0, 2)
    assert np.all(log_ceipv(np.zeros((3, 2)), np.tile(np.eye(2), (3, 1, 1)), terms) == -np.inf)

    with pytest.raises(ValueError, match="form"):
        ceipv_terms(FRONT, V_REF, W_REF, form="literal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="overflow"):
        ceipv_terms(FRONT, V_REF, W_REF, overflow=(0.5, 0.25))
    with pytest.raises(ValueError, match="cov"):
        ceipv(np.zeros((2, 2)), np.zeros((3, 2, 2)), ceipv_terms(FRONT, V_REF, W_REF))
    lower, upper = nondominated_cells(FRONT, V_REF, W_REF)
    with pytest.raises(ValueError, match="objectives"):
        ieipv(np.zeros((1, 3)), np.eye(3)[None], lower, upper)
