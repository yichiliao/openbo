"""Multi-objective test problems for MOBO benchmarks.

Every objective is returned in maximization form (the classic minimization
definitions are negated), inputs live in the unit cube ``[0, 1]^d`` and are
mapped to each problem's native domain internally, and objectives return
arrays of shape ``(n, L)``.

Each problem comes with two reference points for hypervolume:

- ``v_ref`` (lower): the worst value of each objective over the whole
  domain, rounded outward. Every evaluated point therefore adds hypervolume
  from the first iteration, as in the curves of Shah and Ghahramani (2016).
  A reference near the front's nadir would leave random designs at zero
  hypervolume on OKA2 and DTLZ1a; pass ``v_ref`` to override.
- ``w_ref`` (upper): an upper bound on every objective over the domain, as
  required by the cell decomposition in ``openbo.benchmarks.hypervolume``.

The problem definitions follow the usual literature versions (OKA2: Okabe et
al. 2004; VLMOP3: Van Veldhuizen and Lamont 1999; DTLZ1a: Knowles 2006).
Shah and Ghahramani (2016) define them in their supplementary material, which
may differ in details such as reference points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_solve, cholesky
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from openbo.benchmarks.hypervolume import hypervolume, pareto_front
from openbo.test_functions.synthetic import _as_2d, _from_unit_cube

MultiObjective = Callable[[ArrayLike], NDArray[np.float64]]


@dataclass(frozen=True)
class MOFunctionSpec:
    """Description of a multi-objective test problem (maximization)."""

    name: str
    objective: MultiObjective
    bounds: list[tuple[float, float]]
    dim: int
    n_objectives: int
    v_ref: tuple[float, ...]
    w_ref: tuple[float, ...]
    true_hypervolume: float | None = None


def oka2(x: ArrayLike) -> NDArray[np.float64]:
    """Evaluate maximization OKA2 on `[0, 1]^3` (3 inputs, 2 objectives).

    Native domain: `x1 in [-pi, pi]`, `x2, x3 in [-5, 5]`. The Pareto front is
    `f2 = (f1 - pi)^2 / (4 pi^2) - 1` for `f1 in [-pi, pi]` (maximization form).
    """
    arr = _as_2d(x, dim=3)
    native = _from_unit_cube(
        arr, bounds=[(-math.pi, math.pi), (-5.0, 5.0), (-5.0, 5.0)]
    )
    x1, x2, x3 = native[:, 0], native[:, 1], native[:, 2]
    f1 = x1
    f2 = (
        1.0
        - (x1 + math.pi) ** 2 / (4.0 * math.pi**2)
        + np.abs(x2 - 5.0 * np.cos(x1)) ** (1.0 / 3.0)
        + np.abs(x3 - 5.0 * np.sin(x1)) ** (1.0 / 3.0)
    )
    return -np.stack([f1, f2], axis=1).astype(np.float64)


def vlmop3(x: ArrayLike) -> NDArray[np.float64]:
    """Evaluate maximization VLMOP3 on `[0, 1]^2` (2 inputs, 3 objectives).

    Native domain: `x, y in [-3, 3]`.
    """
    arr = _as_2d(x, dim=2)
    native = _from_unit_cube(arr, bounds=[(-3.0, 3.0), (-3.0, 3.0)])
    x1, x2 = native[:, 0], native[:, 1]
    r2 = x1**2 + x2**2
    f1 = 0.5 * r2 + np.sin(r2)
    f2 = (3.0 * x1 - 2.0 * x2 + 4.0) ** 2 / 8.0 + (x1 - x2 + 1.0) ** 2 / 27.0 + 15.0
    f3 = 1.0 / (r2 + 1.0) - 1.1 * np.exp(-r2)
    return -np.stack([f1, f2, f3], axis=1).astype(np.float64)


def dtlz1a(x: ArrayLike) -> NDArray[np.float64]:
    """Evaluate maximization DTLZ1a on `[0, 1]^6` (6 inputs, 2 objectives).

    The Pareto front is `f1 + f2 = -0.5` with `f1, f2 in [-0.5, 0]`
    (maximization form), reached when `x2..x6 = 0.5`.
    """
    arr = _as_2d(x, dim=6)
    shifted = arr[:, 1:] - 0.5
    g = 100.0 * (
        5.0 + np.sum(shifted**2 - np.cos(2.0 * math.pi * shifted), axis=1)
    )
    f1 = 0.5 * arr[:, 0] * (1.0 + g)
    f2 = 0.5 * (1.0 - arr[:, 0]) * (1.0 + g)
    return -np.stack([f1, f2], axis=1).astype(np.float64)


# Worst values over each domain (multi-start L-BFGS-B, matching the analytic
# bounds): OKA2 (-pi, -5.0727), VLMOP3 (-8.2490, -61.9398, -0.1964),
# DTLZ1a (-563, -563). The reference points round these outward.
_OKA2_V_REF = (-3.2, -5.1)
_VLMOP3_V_REF = (-8.3, -62.0, -0.2)
_DTLZ1A_V_REF = (-570.0, -570.0)

# OKA2 front: f2 = (f1 - pi)^2 / (4 pi^2) - 1 for f1 in [-pi, pi]. Integrating
# over f1 in [-pi, pi] gives 2 pi (5.1 - 2/3); the strip f1 in [-3.2, -pi]
# under the front's end point (-pi, 0) adds 5.1 (3.2 - pi).
_OKA2_TRUE_HV = 2.0 * math.pi * (5.1 - 2.0 / 3.0) + 5.1 * (3.2 - math.pi)
# DTLZ1a front: f1 + f2 = -0.5; the square [-570, 0]^2 minus the triangle
# above the front.
_DTLZ1A_TRUE_HV = 570.0**2 - 0.125

MO_REGISTRY: dict[str, MOFunctionSpec] = {
    "oka2": MOFunctionSpec(
        name="oka2",
        objective=oka2,
        bounds=[(0.0, 1.0)] * 3,
        dim=3,
        n_objectives=2,
        v_ref=_OKA2_V_REF,
        w_ref=(math.pi, 0.0),
        true_hypervolume=_OKA2_TRUE_HV,
    ),
    "vlmop3": MOFunctionSpec(
        name="vlmop3",
        objective=vlmop3,
        bounds=[(0.0, 1.0)] * 2,
        dim=2,
        n_objectives=3,
        v_ref=_VLMOP3_V_REF,
        w_ref=(0.0, -15.0, 0.1),
        true_hypervolume=None,
    ),
    "dtlz1a": MOFunctionSpec(
        name="dtlz1a",
        objective=dtlz1a,
        bounds=[(0.0, 1.0)] * 6,
        dim=6,
        n_objectives=2,
        v_ref=_DTLZ1A_V_REF,
        w_ref=(0.0, 0.0),
        true_hypervolume=_DTLZ1A_TRUE_HV,
    ),
}


def get_mo_function_spec(
    name: str,
    *,
    v_ref: ArrayLike | None = None,
    w_ref: ArrayLike | None = None,
) -> MOFunctionSpec:
    """Return a registered multi-objective problem.

    ``v_ref`` and ``w_ref`` override the default reference points. Overriding
    ``v_ref`` drops ``true_hypervolume``, which depends on it.
    """
    try:
        spec = MO_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown multi-objective test function: {name}") from exc
    if v_ref is None and w_ref is None:
        return spec

    v = spec.v_ref if v_ref is None else _as_reference(v_ref, spec.n_objectives)
    w = spec.w_ref if w_ref is None else _as_reference(w_ref, spec.n_objectives)
    return MOFunctionSpec(
        name=spec.name,
        objective=spec.objective,
        bounds=spec.bounds,
        dim=spec.dim,
        n_objectives=spec.n_objectives,
        v_ref=v,
        w_ref=w,
        true_hypervolume=spec.true_hypervolume if v_ref is None else None,
    )


def _as_reference(ref: ArrayLike, n_objectives: int) -> tuple[float, ...]:
    """Return a reference point as a tuple of length ``n_objectives``."""
    arr = np.asarray(ref, dtype=np.float64).reshape(-1)
    if arr.shape != (n_objectives,):
        raise ValueError(f"Reference point must have length {n_objectives}.")
    return tuple(float(v) for v in arr)


def _matern52(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    lengthscale: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Unit-variance ARD Matern-5/2 kernel via ``cdist`` (same values as
    ``openbo.models.kernels.matern52_kernel``, without the ``(n, m, d)``
    intermediate array)."""
    r = cdist(x1 / lengthscale, x2 / lengthscale)
    sqrt5_r = math.sqrt(5.0) * r
    return (1.0 + sqrt5_r + (5.0 / 3.0) * r**2) * np.exp(-sqrt5_r)


def make_slf_gp_problem(
    rho: float,
    seed: int = 0,
    *,
    means: tuple[float, float] = (2.0, -2.0),
    lengthscales: tuple[tuple[float, ...], tuple[float, ...]] = (
        (0.7, 0.4, 1.0),
        (0.34, 0.9, 0.5),
    ),
    n_support: int = 1024,
    n_reference: int = 8192,
) -> MOFunctionSpec:
    """Draw a pair of correlated objectives from a semiparametric latent factor GP.

    This is the synthetic setup of Shah and Ghahramani (2016, Section 4):
    on `[0, 1]^3`, `f_l(x) = means[l] + sum_s A[l, s] g_s(x)`, where `g_s` are
    independent unit-variance Matern-5/2 GPs with the given ARD lengthscales
    and `A = [[1, 0], [rho, sqrt(1 - rho^2)]]`, so `f_1` and `f_2` have
    correlation `rho` at every input.

    Each `g_s` is sampled jointly on `n_support` Sobol points and evaluated
    elsewhere through the kernel interpolant, so the objective is a smooth,
    deterministic function of `x`. Reference points and ``true_hypervolume``
    are estimated from `n_reference` further Sobol points: ``v_ref`` and
    ``w_ref`` are the sampled extremes widened by 10% of each objective's
    range, and the true hypervolume is a dense-sample estimate. It is a lower bound that
    optimizers can exceed (in a 4 x 50 draw sweep, 518 of 800 runs ended above it, by up
    to 14.5%), so regret against it is clamped at ``1e-12`` and is not meaningful for
    this problem; compare hypervolume instead.
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1].")
    dim = len(lengthscales[0])
    if any(len(ls) != dim for ls in lengthscales):
        raise ValueError("All latent lengthscales must have the same length.")

    rng = np.random.default_rng(seed)
    support = qmc.Sobol(d=dim, scramble=True, seed=rng).random(n_support)
    ls_arrays = [np.asarray(ls, dtype=np.float64) for ls in lengthscales]
    weights = []
    for ls in ls_arrays:
        gram = _matern52(support, support, ls) + 1e-8 * np.eye(n_support)
        chol = cholesky(gram, lower=True)
        draw = chol @ rng.standard_normal(n_support)
        # Interpolant weights K^{-1} g, so the interpolant passes through the draw.
        weights.append(cho_solve((chol, True), draw))
    mixing = np.array([[1.0, 0.0], [rho, math.sqrt(1.0 - rho**2)]], dtype=np.float64)
    offsets = np.asarray(means, dtype=np.float64)

    def objective(x: ArrayLike) -> NDArray[np.float64]:
        arr = _as_2d(x, dim=dim)
        out = np.empty((arr.shape[0], 2), dtype=np.float64)
        for start in range(0, arr.shape[0], 2048):
            chunk = arr[start : start + 2048]
            latent = np.stack(
                [_matern52(chunk, support, ls) @ w for ls, w in zip(ls_arrays, weights)],
                axis=1,
            )
            out[start : start + 2048] = offsets + latent @ mixing.T
        return out

    reference = qmc.Sobol(d=dim, scramble=True, seed=rng).random(n_reference)
    values = objective(reference)
    front = pareto_front(values)
    value_range = values.max(axis=0) - values.min(axis=0)
    v_ref = values.min(axis=0) - 0.1 * value_range
    w_ref = values.max(axis=0) + 0.1 * value_range
    return MOFunctionSpec(
        name=f"slf_gp_rho{rho:g}_seed{seed}",
        objective=objective,
        bounds=[(0.0, 1.0)] * dim,
        dim=dim,
        n_objectives=2,
        v_ref=tuple(float(v) for v in v_ref),
        w_ref=tuple(float(v) for v in w_ref),
        true_hypervolume=hypervolume(front, v_ref),
    )
