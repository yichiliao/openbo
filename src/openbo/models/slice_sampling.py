"""Coordinate-wise slice sampling (Neal, 2003) with stepping out and shrinkage.

Used to draw GP hyperparameters from their posterior, as in Shah and
Ghahramani (2016), instead of fixing them at the MAP estimate.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

LogDensity = Callable[[NDArray[np.float64]], float]


def slice_sample(
    log_density: LogDensity,
    x0: ArrayLike,
    n_samples: int,
    rng: np.random.Generator,
    *,
    widths: ArrayLike | float = 1.0,
    burn_in: int = 0,
    thin: int = 1,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
    max_steps_out: int = 10,
) -> NDArray[np.float64]:
    """Return ``n_samples`` draws (``(n_samples, p)``) from ``exp(log_density)``.

    Each sweep updates every coordinate in turn with a univariate slice
    sampler: draw a level under the current density, step an interval of
    width ``widths[i]`` out until both ends fall below the level (at most
    ``max_steps_out`` steps per side), then sample uniformly in the interval
    and shrink it towards the current point on rejection. Points outside
    ``[lower, upper]`` have zero density. ``burn_in`` sweeps are discarded and
    one sample is kept every ``thin`` sweeps.
    """
    x = np.array(x0, dtype=np.float64).reshape(-1)
    p = x.shape[0]
    w = np.broadcast_to(np.asarray(widths, dtype=np.float64), (p,))
    lo = np.full(p, -np.inf) if lower is None else np.asarray(lower, dtype=np.float64)
    hi = np.full(p, np.inf) if upper is None else np.asarray(upper, dtype=np.float64)
    if np.any(x < lo) or np.any(x > hi):
        raise ValueError("x0 must lie within [lower, upper].")
    if n_samples < 0 or burn_in < 0 or thin < 1:
        raise ValueError("Need n_samples >= 0, burn_in >= 0 and thin >= 1.")

    def bounded(z: NDArray[np.float64]) -> float:
        """Return the log density, or -inf outside the bounds."""
        if np.any(z < lo) or np.any(z > hi):
            return -math.inf
        value = float(log_density(z))
        return value if math.isfinite(value) else -math.inf

    current = bounded(x)
    if not math.isfinite(current):
        raise ValueError("x0 must have finite log density.")

    samples = np.empty((n_samples, p), dtype=np.float64)
    n_sweeps = burn_in + n_samples * thin
    for sweep in range(n_sweeps):
        for i in rng.permutation(p):
            level = current + math.log(rng.uniform())
            left = x[i] - w[i] * rng.uniform()
            right = left + w[i]
            z = x.copy()
            for _ in range(max_steps_out):
                z[i] = left
                if bounded(z) <= level:
                    break
                left -= w[i]
            for _ in range(max_steps_out):
                z[i] = right
                if bounded(z) <= level:
                    break
                right += w[i]
            left, right = max(left, lo[i]), min(right, hi[i])
            while True:
                z[i] = rng.uniform(left, right)
                value = bounded(z)
                if value > level:
                    x, current = z, value
                    break
                if z[i] < x[i]:
                    left = z[i]
                else:
                    right = z[i]
        kept = sweep - burn_in
        if kept >= 0 and (kept + 1) % thin == 0:
            samples[kept // thin] = x
    return samples
