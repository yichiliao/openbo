"""Function families of MetaBO's global benchmarks (Volpp et al., ICLR 2020).

Follows the authors' reference code where it settles details the paper leaves
open (see ``docs/plans/naf.md``, "Reference code findings"):

- The base Branin is OpenBO's maximization-form ``branin`` on ``[0, 1]^2``,
  normalized to about unit scale as in the reference: ``(y + 54.44) / 51.44``
  with ``y = -Branin(x)``, which is the reference's ``-(f - 54.44) / 51.44``.
- A task is ``s * g(x - t)``: a translation ``t ~ U(-0.1, 0.1)^2`` in the
  normalized input coordinates and a scaling ``s ~ U(0.9, 1.1)`` of the output.
  Translated points may fall outside ``[0, 1]^2``; Branin is defined there too.
- The optimum is tracked from Branin's upper-left minimizer, which stays inside
  the domain for every translation in range.
- "MetaBO-50" trains on 50 fixed tasks from a 3-D Sobol grid over ``(t_1, t_2, s)``.

Goldstein-Price (2D) and Hartmann-3 (3D) follow the same recipe. As in the
reference they are normalized to about unit scale before the task transform:
Goldstein-Price as ``-(log f - 8.693) / 2.427``, Hartmann-3 as
``-(f + 0.93) / 0.95`` with ``f`` the (minimized) textbook function. A task is
again ``s * g(x - t)`` with ``t ~ U(-0.1, 0.1)^D`` and ``s ~ U(0.9, 1.1)``, and
MetaBO-50 is a ``(D + 1)``-dimensional Sobol grid. The reference clips each
translation to a range that keeps the optimum inside the domain; the
translations here (at most 0.1) never reach those ranges, so there is no clipping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from openbo.test_functions.synthetic import BRANIN_MAXIMUM, branin

BRANIN_OFFSET = 54.44
BRANIN_SCALE = 51.44
# Branin's upper-left minimizer (-pi, 12.275), in the normalized [0, 1]^2 coordinates.
BRANIN_UPPER_LEFT = ((-math.pi + 5.0) / 15.0, 12.275 / 15.0)
NORMALIZED_BRANIN_MAXIMUM = (BRANIN_MAXIMUM + BRANIN_OFFSET) / BRANIN_SCALE
TRANSLATION_RANGE = 0.1
SCALING_RANGE = (0.9, 1.1)


def normalized_branin(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Branin in maximization form, normalized to about unit scale (maximum about 1.0506)."""
    return (branin(x) + BRANIN_OFFSET) / BRANIN_SCALE


# Goldstein-Price on [-2, 2]^2, mapped from [0, 1]^2; textbook minimum 3 at (0, -1).
GOLDSTEIN_PRICE_OFFSET = 8.693
GOLDSTEIN_PRICE_SCALE = 2.427
GOLDSTEIN_PRICE_ARGMAX = (0.5, 0.25)

# Hartmann-3 on [0, 1]^3; textbook minimum -3.86278 at the position below
# (polished to machine precision; the reference rounds it to 0.114614 ...).
HARTMANN3_OFFSET = -0.93
HARTMANN3_SCALE = 0.95
HARTMANN3_ARGMAX = (0.11458887075439354, 0.5556488906008059, 0.8525469799283214)
_HARTMANN3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_HARTMANN3_A = np.array([[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]])
_HARTMANN3_P = 1e-4 * np.array(
    [[3689.0, 1170.0, 2673.0], [4699.0, 4387.0, 7470.0], [1091.0, 8732.0, 5547.0], [381.0, 5743.0, 8828.0]]
)


def normalized_goldstein_price(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Goldstein-Price on ``[0, 1]^2`` in maximization form, log-normalized to about unit scale."""
    arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
    x1 = 4.0 * arr[:, 0] - 2.0
    x2 = 4.0 * arr[:, 1] - 2.0
    first = 1.0 + (x1 + x2 + 1.0) ** 2 * (19.0 - 14.0 * x1 + 3.0 * x1**2 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2**2)
    second = 30.0 + (2.0 * x1 - 3.0 * x2) ** 2 * (
        18.0 - 32.0 * x1 + 12.0 * x1**2 + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2**2
    )
    return -(np.log(first * second) - GOLDSTEIN_PRICE_OFFSET) / GOLDSTEIN_PRICE_SCALE


def normalized_hartmann3(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hartmann-3 on ``[0, 1]^3`` in maximization form, normalized to about unit scale."""
    arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
    exponent = np.einsum("ijk->ij", _HARTMANN3_A * (arr[:, None, :] - _HARTMANN3_P) ** 2)
    value = np.exp(-exponent) @ _HARTMANN3_ALPHA  # the textbook function is the negative of this
    return (value + HARTMANN3_OFFSET) / HARTMANN3_SCALE


NORMALIZED_GOLDSTEIN_PRICE_MAXIMUM = float(normalized_goldstein_price(np.array([GOLDSTEIN_PRICE_ARGMAX]))[0])
NORMALIZED_HARTMANN3_MAXIMUM = float(normalized_hartmann3(np.array([HARTMANN3_ARGMAX]))[0])


@dataclass(frozen=True)
class _MetaBOTask:
    """A task ``s * g(x - t)`` of a MetaBO function family; subclasses set the base function."""

    translation: tuple[float, ...]
    scaling: float

    base: ClassVar  # normalized base function on [0, 1]^dim, in maximization form
    base_argmax: ClassVar[tuple[float, ...]]
    base_maximum: ClassVar[float]

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the task at ``x`` of shape ``(n, d)`` (or ``(d,)``), returning ``(n,)``."""
        arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return self.scaling * type(self).base(arr - np.asarray(self.translation))

    @property
    def argmax(self) -> NDArray[np.float64]:
        """Location of the maximum inside ``[0, 1]^d``."""
        return np.asarray(self.base_argmax) + np.asarray(self.translation)

    @property
    def optimum(self) -> float:
        """Maximum value inside ``[0, 1]^d``."""
        return self.scaling * self.base_maximum


@dataclass(frozen=True)
class BraninTask(_MetaBOTask):
    """One MetaBO Branin task: ``s * normalized_branin(x - t)``."""

    base: ClassVar = staticmethod(normalized_branin)
    base_argmax: ClassVar[tuple[float, ...]] = BRANIN_UPPER_LEFT
    base_maximum: ClassVar[float] = NORMALIZED_BRANIN_MAXIMUM


@dataclass(frozen=True)
class GoldsteinPriceTask(_MetaBOTask):
    """One MetaBO Goldstein-Price task: ``s * normalized_goldstein_price(x - t)``."""

    base: ClassVar = staticmethod(normalized_goldstein_price)
    base_argmax: ClassVar[tuple[float, ...]] = GOLDSTEIN_PRICE_ARGMAX
    base_maximum: ClassVar[float] = NORMALIZED_GOLDSTEIN_PRICE_MAXIMUM


@dataclass(frozen=True)
class Hartmann3Task(_MetaBOTask):
    """One MetaBO Hartmann-3 task: ``s * normalized_hartmann3(x - t)``."""

    base: ClassVar = staticmethod(normalized_hartmann3)
    base_argmax: ClassVar[tuple[float, ...]] = HARTMANN3_ARGMAX
    base_maximum: ClassVar[float] = NORMALIZED_HARTMANN3_MAXIMUM


def _task_from_unit(u: NDArray[np.float64], task_class: type[_MetaBOTask] = BraninTask) -> _MetaBOTask:
    """Map a point of ``[0, 1]^(D+1)`` to a task's ``D`` translations and its scaling."""
    t = -TRANSLATION_RANGE + 2.0 * TRANSLATION_RANGE * u[:-1]
    lo, hi = SCALING_RANGE
    return task_class(translation=tuple(float(v) for v in t), scaling=float(lo + (hi - lo) * u[-1]))


def _sample_tasks(task_class: type[_MetaBOTask], dim: int, n_tasks: int, seed: int) -> list:
    if n_tasks < 1:
        raise ValueError("n_tasks must be positive.")
    rng = np.random.default_rng(seed)
    return [_task_from_unit(u, task_class) for u in rng.random((n_tasks, dim + 1))]


def _grid_tasks(task_class: type[_MetaBOTask], dim: int, n_tasks: int) -> list:
    if n_tasks < 1:
        raise ValueError("n_tasks must be positive.")
    sampler = qmc.Sobol(d=dim + 1, scramble=False)
    # Unscrambled Sobol starts at the corner (0, ..., 0); skip it so no task sits on the range boundary.
    points = sampler.random(n_tasks + 1)[1:]
    return [_task_from_unit(u, task_class) for u in points]


def sample_branin_tasks(n_tasks: int, seed: int) -> list[BraninTask]:
    """Draw ``n_tasks`` random tasks (used for test functions and full MetaBO)."""
    return _sample_tasks(BraninTask, 2, n_tasks, seed)  # type: ignore[return-value]


def branin_grid_tasks(n_tasks: int = 50) -> list[BraninTask]:
    """The fixed MetaBO-50 training (and TAF-50 source) tasks: a 3-D Sobol grid over ``(t_1, t_2, s)``."""
    return _grid_tasks(BraninTask, 2, n_tasks)  # type: ignore[return-value]


def sample_goldstein_price_tasks(n_tasks: int, seed: int) -> list[GoldsteinPriceTask]:
    """Draw ``n_tasks`` random Goldstein-Price tasks."""
    return _sample_tasks(GoldsteinPriceTask, 2, n_tasks, seed)


def goldstein_price_grid_tasks(n_tasks: int = 50) -> list[GoldsteinPriceTask]:
    """The fixed MetaBO-50 Goldstein-Price tasks: a 3-D Sobol grid over ``(t_1, t_2, s)``."""
    return _grid_tasks(GoldsteinPriceTask, 2, n_tasks)


def sample_hartmann3_tasks(n_tasks: int, seed: int) -> list[Hartmann3Task]:
    """Draw ``n_tasks`` random Hartmann-3 tasks."""
    return _sample_tasks(Hartmann3Task, 3, n_tasks, seed)


def hartmann3_grid_tasks(n_tasks: int = 50) -> list[Hartmann3Task]:
    """The fixed MetaBO-50 Hartmann-3 tasks: a 4-D Sobol grid over ``(t_1, t_2, t_3, s)``."""
    return _grid_tasks(Hartmann3Task, 3, n_tasks)
