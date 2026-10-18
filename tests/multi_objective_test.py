"""Tests for multi-objective test problems."""

from __future__ import annotations

import math

import numpy as np
import pytest

from openbo.benchmarks.hypervolume import hypervolume, pareto_mask
from openbo.test_functions.multi_objective import (
    MO_REGISTRY,
    dtlz1a,
    get_mo_function_spec,
    make_slf_gp_problem,
    oka2,
    vlmop3,
)


def _oka2_front_inputs(n: int) -> np.ndarray:
    """Return unit-cube inputs on the OKA2 Pareto set."""
    t = np.linspace(-math.pi, math.pi, n)
    native = np.stack([t, 5.0 * np.cos(t), 5.0 * np.sin(t)], axis=1)
    lower = np.array([-math.pi, -5.0, -5.0])
    upper = np.array([math.pi, 5.0, 5.0])
    return (native - lower) / (upper - lower)


def _dtlz1a_front_inputs(n: int) -> np.ndarray:
    """Return inputs on the DTLZ1a Pareto set."""
    x = np.full((n, 6), 0.5)
    x[:, 0] = np.linspace(0.0, 1.0, n)
    return x


@pytest.mark.parametrize("name", sorted(MO_REGISTRY))
def test_registry_shapes(name: str) -> None:
    """Objectives should map (n, d) unit-cube inputs to (n, L) values."""
    spec = get_mo_function_spec(name)
    x = np.random.default_rng(0).uniform(size=(5, spec.dim))
    y = spec.objective(x)
    assert y.shape == (5, spec.n_objectives)
    assert y.dtype == np.float64
    assert len(spec.bounds) == spec.dim
    assert len(spec.v_ref) == len(spec.w_ref) == spec.n_objectives
    assert spec.objective(x[0]).shape == (1, spec.n_objectives)


@pytest.mark.parametrize("name", sorted(MO_REGISTRY))
def test_reference_points_bound_every_objective(name: str) -> None:
    """Objective values should lie strictly above v_ref and at most w_ref."""
    spec = get_mo_function_spec(name)
    rng = np.random.default_rng(1)
    corners = np.array(np.meshgrid(*[[0.0, 1.0]] * spec.dim)).reshape(spec.dim, -1).T
    x = np.vstack([rng.uniform(size=(100_000, spec.dim)), corners])
    y = spec.objective(x)
    assert np.all(y > np.asarray(spec.v_ref))
    assert np.all(y <= np.asarray(spec.w_ref))


@pytest.mark.parametrize("name", sorted(MO_REGISTRY))
def test_random_designs_have_positive_hypervolume(name: str) -> None:
    """With the loose v_ref, a few random points should already add volume."""
    spec = get_mo_function_spec(name)
    y = spec.objective(np.random.default_rng(8).uniform(size=(5, spec.dim)))
    assert hypervolume(y, spec.v_ref) > 0.0


def test_oka2_is_maximization_with_known_front() -> None:
    """OKA2 front points should satisfy the analytic curve and not be dominated."""
    y_front = oka2(_oka2_front_inputs(200))
    f1, f2 = y_front[:, 0], y_front[:, 1]
    # Cube roots amplify input round-off (1e-15 -> 1e-5), hence the tolerance.
    np.testing.assert_allclose(f2, (f1 - math.pi) ** 2 / (4.0 * math.pi**2) - 1.0, atol=1e-4)
    assert np.all(y_front >= np.asarray(MO_REGISTRY["oka2"].v_ref))

    # Random points should never dominate the front (maximization).
    y_rand = oka2(np.random.default_rng(2).uniform(size=(20_000, 3)))
    combined = np.vstack([y_front, y_rand])
    assert np.all(pareto_mask(combined)[: y_front.shape[0]])


def test_dtlz1a_is_maximization_with_known_front() -> None:
    """DTLZ1a front points should satisfy f1 + f2 = -0.5."""
    y_front = dtlz1a(_dtlz1a_front_inputs(101))
    np.testing.assert_allclose(y_front.sum(axis=1), -0.5, atol=1e-12)
    assert np.all(y_front <= 0.0)
    y_rand = dtlz1a(np.random.default_rng(3).uniform(size=(1000, 6)))
    assert np.all(y_rand.sum(axis=1) < -0.5)


def test_vlmop3_values_and_bounds() -> None:
    """VLMOP3 should match hand-computed values in maximization form."""
    # Native (0, 0) is unit (0.5, 0.5): f = (0, 4^2/8 + 1/27 + 15, 1 - 1.1).
    y = vlmop3(np.array([[0.5, 0.5]]))
    np.testing.assert_allclose(y, [[0.0, -(2.0 + 1.0 / 27.0 + 15.0), 0.1]])
    # Native (-2, -1) attains the second objective's bound w_ref[1] = -15.
    x = (np.array([[-2.0, -1.0]]) + 3.0) / 6.0
    assert vlmop3(x)[0, 1] == pytest.approx(MO_REGISTRY["vlmop3"].w_ref[1])


@pytest.mark.parametrize(
    ("name", "front_inputs"),
    [("oka2", _oka2_front_inputs), ("dtlz1a", _dtlz1a_front_inputs)],
)
def test_true_hypervolume_matches_dense_front(name: str, front_inputs) -> None:
    """A dense sample of the true front should approach true_hypervolume from below."""
    spec = get_mo_function_spec(name)
    assert spec.true_hypervolume is not None
    hv = hypervolume(spec.objective(front_inputs(4001)), spec.v_ref)
    assert hv <= spec.true_hypervolume * (1.0 + 1e-12)
    assert spec.true_hypervolume - hv < 2e-3


def test_get_mo_function_spec_overrides_and_errors() -> None:
    """Overrides should replace reference points; v_ref drops the true HV."""
    spec = get_mo_function_spec("dtlz1a", w_ref=[1.0, 1.0])
    assert spec.w_ref == (1.0, 1.0)
    assert spec.true_hypervolume == MO_REGISTRY["dtlz1a"].true_hypervolume

    spec = get_mo_function_spec("dtlz1a", v_ref=[-1.0, -1.0])
    assert spec.v_ref == (-1.0, -1.0)
    assert spec.true_hypervolume is None

    with pytest.raises(KeyError, match="Unknown"):
        get_mo_function_spec("zdt1")
    with pytest.raises(ValueError, match="length 2"):
        get_mo_function_spec("oka2", v_ref=[0.0, 0.0, 0.0])


def test_slf_gp_problem_is_deterministic_and_consistent() -> None:
    """The same seed should give the same objective and valid reference points."""
    spec_a = make_slf_gp_problem(rho=0.5, seed=3)
    spec_b = make_slf_gp_problem(rho=0.5, seed=3)
    x = np.random.default_rng(4).uniform(size=(500, 3))
    y = spec_a.objective(x)
    np.testing.assert_allclose(y, spec_b.objective(x))
    assert y.shape == (500, 2)
    assert spec_a.dim == 3 and spec_a.n_objectives == 2
    assert spec_a.v_ref == spec_b.v_ref and spec_a.w_ref == spec_b.w_ref

    assert np.all(y <= np.asarray(spec_a.w_ref))
    assert np.all(y > np.asarray(spec_a.v_ref))
    assert spec_a.true_hypervolume is not None and spec_a.true_hypervolume > 0.0
    assert hypervolume(y, spec_a.v_ref) <= spec_a.true_hypervolume * (1.0 + 1e-3)


@pytest.mark.parametrize("rho", [-0.9, 0.9])
def test_slf_gp_problem_correlation_sign(rho: float) -> None:
    """Objectives should be correlated across inputs with the sign of rho."""
    x = np.random.default_rng(5).uniform(size=(4000, 3))
    correlations = []
    for seed in range(5):
        y = make_slf_gp_problem(rho=rho, seed=seed, n_reference=1024).objective(x)
        correlations.append(np.corrcoef(y[:, 0], y[:, 1])[0, 1])
    assert np.sign(np.mean(correlations)) == np.sign(rho)
    assert abs(np.mean(correlations)) > 0.5


def test_local_matern52_matches_shared_kernel() -> None:
    """The cdist-based kernel should match openbo.models.kernels.matern52_kernel."""
    from openbo.models.kernels import matern52_kernel
    from openbo.test_functions.multi_objective import _matern52

    rng = np.random.default_rng(6)
    x1, x2 = rng.uniform(size=(30, 3)), rng.uniform(size=(20, 3))
    ls = np.array([0.7, 0.4, 1.0])
    np.testing.assert_allclose(_matern52(x1, x2, ls), matern52_kernel(x1, x2, ls, 1.0), atol=1e-12)


def test_slf_gp_problem_validates_rho() -> None:
    """Correlations outside [-1, 1] should be rejected."""
    with pytest.raises(ValueError, match="rho"):
        make_slf_gp_problem(rho=1.5)
