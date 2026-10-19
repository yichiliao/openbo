"""Pareto front and hypervolume utilities for multi-objective maximization.

All objectives are maximized. The objective space between a lower reference
point ``v_ref`` and an upper corner ``w_ref`` is split into a grid of cells
whose edges are the coordinates of the Pareto points (Shah and Ghahramani,
2016, Section 2.2). Because the grid is aligned with the front, every cell is
either fully dominated by the front or fully non-dominated.

The grid has ``O((P + 1)^L)`` cells for ``P`` Pareto points and ``L``
objectives, so these helpers are meant for small ``L`` (up to about 3).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_objectives(y: ArrayLike) -> NDArray[np.float64]:
    """Return objective values as a float array of shape ``(n, L)``."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected objective values of shape (n, L), got {arr.shape}.")
    return arr


def _as_reference(ref: ArrayLike, n_objectives: int, name: str) -> NDArray[np.float64]:
    """Return a reference point as a float vector of length ``n_objectives``."""
    arr = np.asarray(ref, dtype=np.float64).reshape(-1)
    if arr.shape != (n_objectives,):
        raise ValueError(f"{name} must have length {n_objectives}, got shape {arr.shape}.")
    return arr


def pareto_mask(y: ArrayLike) -> NDArray[np.bool_]:
    """Return a mask of the non-dominated rows of ``y`` (maximization).

    A row is dominated if another row is at least as good in every objective
    and strictly better in one. Of several identical non-dominated rows only
    the first is kept, so the masked rows are distinct.
    """
    y_arr = _as_objectives(y)
    n, n_objectives = y_arr.shape
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    # Visit rows in decreasing lexicographic order (ties by original index).
    # Anything that weakly dominates a row is visited before it, and a dominated
    # dominator is itself dominated by an earlier kept row, so checking each row
    # against the rows kept so far is enough. Costs O(n * P).
    keys = (np.arange(n),) + tuple(-y_arr[:, k] for k in reversed(range(n_objectives)))
    kept = np.empty((n, n_objectives), dtype=np.float64)
    n_kept = 0
    for i in np.lexsort(keys):
        row = y_arr[i]
        if n_kept and np.any(np.all(kept[:n_kept] >= row, axis=1)):
            continue
        kept[n_kept] = row
        n_kept += 1
        mask[i] = True
    return mask


def pareto_front(y: ArrayLike) -> NDArray[np.float64]:
    """Return the distinct non-dominated rows of ``y`` with shape ``(P, L)``."""
    y_arr = _as_objectives(y)
    return y_arr[pareto_mask(y_arr)]


def _grid_coordinates(
    front: NDArray[np.float64],
    v_ref: NDArray[np.float64],
    w_ref: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.bool_]]:
    """Return per-objective grid coordinates and the grid-shaped dominated mask.

    Front points are clipped into ``[v_ref, w_ref]`` first; a point below
    ``v_ref`` in some objective then dominates no volume. The mask has one
    entry per cell, shape ``(len(coords[0]) - 1, ..., len(coords[L-1]) - 1)``.
    """
    n_objectives = v_ref.shape[0]
    clipped = np.clip(front, v_ref, w_ref)
    coords = [
        np.unique(np.concatenate(([v_ref[k]], clipped[:, k], [w_ref[k]])))
        for k in range(n_objectives)
    ]
    if any(c.shape[0] < 2 for c in coords):
        return coords, np.zeros((0,) * n_objectives, dtype=bool)

    # occupied[i_1, ..., i_L] marks a front point at those grid coordinates.
    occupied = np.zeros(tuple(c.shape[0] for c in coords), dtype=bool)
    if clipped.shape[0] > 0:
        idx = tuple(
            np.searchsorted(coords[k], clipped[:, k]) for k in range(n_objectives)
        )
        occupied[idx] = True

    # reach[i]: some front point has grid index >= i in every objective.
    reach = occupied
    for axis in range(n_objectives):
        reach = np.flip(
            np.logical_or.accumulate(np.flip(reach, axis=axis), axis=axis), axis=axis
        )
    # A cell is dominated iff a front point dominates its upper corner.
    return coords, reach[(slice(1, None),) * n_objectives]


def cell_grid(
    front: ArrayLike,
    v_ref: ArrayLike,
    w_ref: ArrayLike,
) -> tuple[list[NDArray[np.float64]], NDArray[np.bool_]]:
    """Return the front-aligned grid of ``[v_ref, w_ref]`` in grid form.

    Returns the sorted coordinates along each objective (cell ``i`` spans
    ``[coords[l][i_l], coords[l][i_l + 1]]`` in objective ``l``) and a boolean
    array, indexed by cell, that is True for cells *not* dominated by
    ``front``. This is the same partition as :func:`nondominated_cells`.
    """
    front_arr = _as_objectives(front)
    n_objectives = front_arr.shape[1]
    v, w = _as_box(v_ref, w_ref, n_objectives)
    coords, dominated = _grid_coordinates(pareto_front(front_arr), v, w)
    return coords, ~dominated


def _as_box(
    v_ref: ArrayLike, w_ref: ArrayLike, n_objectives: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return validated finite reference corners with ``w_ref >= v_ref``."""
    v = _as_reference(v_ref, n_objectives, "v_ref")
    w = _as_reference(w_ref, n_objectives, "w_ref")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(w)):
        raise ValueError("v_ref and w_ref must be finite.")
    if np.any(w < v):
        raise ValueError("w_ref must be at least v_ref in every objective.")
    return v, w


def _grid_cells(
    front: NDArray[np.float64],
    v_ref: NDArray[np.float64],
    w_ref: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Split ``[v_ref, w_ref]`` into front-aligned cells.

    Returns lower corners, upper corners (both ``(K, L)``) and a mask of the
    cells dominated by ``front``.
    """
    n_objectives = v_ref.shape[0]
    coords, dominated_grid = _grid_coordinates(front, v_ref, w_ref)
    if dominated_grid.size == 0:
        empty = np.empty((0, n_objectives), dtype=np.float64)
        return empty, empty.copy(), np.zeros(0, dtype=bool)
    dominated = dominated_grid.reshape(-1)

    cell_idx = np.meshgrid(
        *[np.arange(c.shape[0] - 1) for c in coords], indexing="ij"
    )
    lower = np.stack(
        [coords[k][cell_idx[k].reshape(-1)] for k in range(n_objectives)], axis=1
    )
    upper = np.stack(
        [coords[k][cell_idx[k].reshape(-1) + 1] for k in range(n_objectives)], axis=1
    )
    return lower, upper, dominated


def nondominated_cells(
    front: ArrayLike,
    v_ref: ArrayLike,
    w_ref: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the cells of ``[v_ref, w_ref]`` not dominated by ``front``.

    ``front`` may contain dominated rows; they do not change the result.
    ``w_ref`` must be finite and at least ``v_ref`` in every objective.
    Returns ``(lower, upper)`` corner arrays of shape ``(K, L)``; these are the
    cells ``C_nd`` that a new observation can gain volume in.
    """
    front_arr = _as_objectives(front)
    n_objectives = front_arr.shape[1]
    v, w = _as_box(v_ref, w_ref, n_objectives)

    lower, upper, dominated = _grid_cells(pareto_front(front_arr), v, w)
    return lower[~dominated], upper[~dominated]


def hypervolume(y: ArrayLike, v_ref: ArrayLike) -> float:
    """Return the volume dominated by ``y`` and bounded below by ``v_ref``."""
    y_arr = _as_objectives(y)
    n_objectives = y_arr.shape[1]
    v = _as_reference(v_ref, n_objectives, "v_ref")
    if y_arr.shape[0] == 0:
        return 0.0

    front = pareto_front(y_arr)
    w = np.maximum(front.max(axis=0), v)
    lower, upper, dominated = _grid_cells(front, v, w)
    volumes = np.prod(upper[dominated] - lower[dominated], axis=1)
    return float(np.sum(volumes))


def hypervolume_gain(
    y_new: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> NDArray[np.float64]:
    """Return the hypervolume gained by each row of ``y_new`` on its own.

    ``lower`` and ``upper`` are non-dominated cells from
    :func:`nondominated_cells`. The gain of a point ``y`` is the part of every
    such cell that ``y`` dominates, ``sum_C prod_l (min(y_l, w_C,l) - v_C,l)_+``.
    This is exact when ``y`` lies below ``w_ref``.
    """
    y_arr = _as_objectives(y_new)
    lower_arr = _as_objectives(lower)
    upper_arr = _as_objectives(upper)
    if lower_arr.shape != upper_arr.shape or lower_arr.shape[1] != y_arr.shape[1]:
        raise ValueError("lower, upper and y_new must agree on the number of objectives.")

    extent = np.minimum(y_arr[:, None, :], upper_arr[None, :, :]) - lower_arr[None, :, :]
    return np.sum(np.prod(np.maximum(extent, 0.0), axis=2), axis=1)
