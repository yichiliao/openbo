"""Tests for the McCormick function and ConBO simulated-user sequences."""

from __future__ import annotations

import math

import numpy as np
import pytest

from openbo.test_functions.families import (
    estimate_optimum,
    generate_user_variants,
    make_user_sequence,
)
from openbo.test_functions.registry import get_function_spec
from openbo.test_functions.synthetic import KNOWN_OPTIMA, MCCORMICK_MAXIMUM, mccormick
from openbo.test_functions.tasks import TASK_DIMS, TaskVariantSpec


def _mccormick_minimizer_unit() -> np.ndarray:
    """Return the McCormick minimizer mapped to `[0, 1]^2`."""
    s = -2.0 * math.pi / 3.0
    x1, x2 = (s + 1.0) / 2.0, (s - 1.0) / 2.0
    return np.array([(x1 + 1.5) / 5.5, (x2 + 3.0) / 7.0])


def test_mccormick_maximum_at_known_minimizer() -> None:
    y = mccormick(_mccormick_minimizer_unit())
    assert y.shape == (1,)
    assert y[0] == pytest.approx(MCCORMICK_MAXIMUM, abs=1e-12)
    assert MCCORMICK_MAXIMUM == pytest.approx(1.9132229549810358, abs=1e-12)


def test_mccormick_is_bounded_by_its_maximum() -> None:
    x = np.random.default_rng(0).random((2000, 2))
    assert np.all(mccormick(x) <= MCCORMICK_MAXIMUM + 1e-12)


def test_mccormick_is_registered() -> None:
    spec = get_function_spec("mccormick")
    assert spec.dim == 2
    assert spec.optimum == KNOWN_OPTIMA["mccormick"]
    assert TASK_DIMS["mccormick"] == 2


@pytest.mark.parametrize("name", ["branin", "mccormick", "sphere"])
def test_estimate_optimum_recovers_known_2d_optima(name: str) -> None:
    spec = get_function_spec(name)
    assert estimate_optimum(spec.objective, spec.dim) == pytest.approx(spec.optimum, abs=1e-7)


def test_estimate_optimum_uses_sobol_scan_above_three_dims() -> None:
    spec = get_function_spec("hartmann6")
    assert estimate_optimum(spec.objective, spec.dim) == pytest.approx(spec.optimum, abs=1e-5)


def test_estimate_optimum_validates_arguments() -> None:
    with pytest.raises(ValueError, match="dim"):
        estimate_optimum(mccormick, 0)
    with pytest.raises(ValueError, match="grid_points_per_dim"):
        estimate_optimum(mccormick, 2, grid_points_per_dim=1)


def test_user_variants_follow_the_paper_ranges() -> None:
    variants = generate_user_variants("branin", 50, shift_range=0.3, scale_range=0.2, seed=3)
    shifts = np.array([v.input_shift for v in variants])
    scales = np.array([v.output_scale for v in variants])
    assert len(variants) == 50
    assert np.all(np.abs(shifts) <= 0.15)
    assert np.all(np.abs(scales - 1.0) <= 0.1)
    assert all(v.input_scale == (1.0, 1.0) for v in variants)
    assert all(v.noise_std == 0.0 for v in variants)


def test_user_variants_are_reproducible() -> None:
    a = generate_user_variants("mccormick", 5, 0.5, 0.2, seed=1)
    b = generate_user_variants("mccormick", 5, 0.5, 0.2, seed=1)
    c = generate_user_variants("mccormick", 5, 0.5, 0.2, seed=2)
    assert a == b
    assert a != c


def test_user_variants_validate_ranges() -> None:
    with pytest.raises(ValueError, match="n_users"):
        generate_user_variants("branin", 0, 0.3, 0.2)
    with pytest.raises(ValueError, match="shift_range"):
        generate_user_variants("branin", 3, -0.1, 0.2)
    with pytest.raises(ValueError, match="scale_range"):
        generate_user_variants("branin", 3, 0.3, 2.0)


def test_user_sequence_names_and_order() -> None:
    variants = generate_user_variants("branin", 3, 0.3, 0.2, seed=0)
    users = make_user_sequence("branin", variants)
    assert [u.name for u in users] == ["user_000", "user_001", "user_002"]
    x = np.random.default_rng(0).random((4, 2))
    for user, variant in zip(users, variants):
        expected = variant.output_scale * get_function_spec("branin").objective(
            np.clip(x + np.array(variant.input_shift), 0.0, 1.0)
        )
        np.testing.assert_allclose(user.objective(x), expected)


def test_unshifted_user_keeps_the_scaled_base_optimum() -> None:
    variant = TaskVariantSpec(input_shift=(0.0, 0.0), input_scale=(1.0, 1.0), output_scale=1.1)
    (user,) = make_user_sequence("mccormick", [variant])
    assert user.optimum == pytest.approx(1.1 * MCCORMICK_MAXIMUM, abs=1e-7)


def test_shift_past_the_minimizer_lowers_the_user_optimum() -> None:
    # The McCormick minimizer sits near (0.17, 0.21) in unit coordinates, so a
    # +0.25 shift moves it outside the clipped domain.
    variant = TaskVariantSpec(input_shift=(0.25, 0.25), input_scale=(1.0, 1.0), output_scale=1.0)
    (user,) = make_user_sequence("mccormick", [variant])
    assert user.optimum < MCCORMICK_MAXIMUM - 0.05
    x = np.random.default_rng(1).random((5000, 2))
    assert np.max(user.objective(x)) <= user.optimum + 1e-9
