"""Benchmark metric placeholders."""

from __future__ import annotations

from typing import Iterable


def best_observed(values: Iterable[float]) -> float:
    """Return the minimum value from an iterable."""
    return min(values)
