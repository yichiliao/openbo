"""Tests for synthetic functions and function registry."""

from __future__ import annotations

import math

import numpy as np

from metabo.test_functions.registry import get_function_spec
from metabo.test_functions.synthetic import branin, sphere


def test_sphere_zero_is_zero() -> None:
    """Maximization sphere objective should be zero at origin."""
    y = sphere(np.array([0.5, 0.5]))
    assert np.isclose(y[0], 0.0)


def test_branin_output_shape_for_single_and_batch() -> None:
    """Branin should support both single and batched inputs."""
    y_single = branin(np.array([0.1, 0.7]))
    y_batch = branin(np.array([[0.1, 0.7], [0.0, 0.0]]))
    assert y_single.shape == (1,)
    assert y_batch.shape == (2,)


def test_branin_known_point_has_finite_value() -> None:
    """Branin should return finite values for valid inputs."""
    # Native optimum point (-pi, 12.275) mapped to unit cube.
    x1_u = (-math.pi + 5.0) / 15.0
    x2_u = 12.275 / 15.0
    y = branin(np.array([[x1_u, x2_u]]))
    assert y.shape == (1,)
    assert np.isfinite(y[0])


def test_registry_lookup_returns_branin_spec() -> None:
    """Registry should expose function metadata."""
    spec = get_function_spec("branin")
    assert spec.name == "branin"
    assert len(spec.bounds) == 2
    assert spec.optimum is not None
