"""Input transform helpers for test functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def identity_transform(x: ArrayLike) -> NDArray[np.float64]:
    """Return the input as a float64 NumPy array."""
    return np.asarray(x, dtype=np.float64)
