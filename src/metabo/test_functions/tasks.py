"""Task metadata for test functions."""

from __future__ import annotations

TASK_BOUNDS: dict[str, list[tuple[float, float]]] = {
    "branin": [(0.0, 1.0), (0.0, 1.0)],
    "sphere": [(0.0, 1.0), (0.0, 1.0)],
}
