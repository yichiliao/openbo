"""Tests for synthetic functions and function registry."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from metabo.test_functions.families import (
    build_specs,
    generate_branin_variants,
    generate_variants,
    load_family_split,
    save_family_split,
    split_variants,
)
from metabo.test_functions.registry import (
    get_function_spec,
    make_branin_family,
    make_variant_function_spec,
)
from metabo.test_functions.synthetic import (
    ackley,
    branin,
    hartmann6,
    rastrigin,
    rosenbrock,
    sphere,
)
from metabo.test_functions.tasks import TaskVariantSpec


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
    assert spec.dim == 2
    assert spec.optimum is not None


def test_variant_function_spec_changes_output() -> None:
    """Variant wrapper should produce a valid transformed objective."""
    variant = TaskVariantSpec(
        input_shift=(0.01, -0.02),
        input_scale=(1.0, 1.0),
        output_scale=1.1,
    )
    variant_spec = make_variant_function_spec("branin", variant, "branin_shifted")
    x = np.array([[0.2, 0.3]], dtype=np.float64)
    y = variant_spec.objective(x)
    assert y.shape == (1,)
    assert np.isfinite(y[0])
    assert variant_spec.optimum is not None


def test_variant_dimension_mismatch_raises() -> None:
    """Variant creation should fail when dimensions do not match base function."""
    bad_variant = TaskVariantSpec(
        input_shift=(0.1,),
        input_scale=(1.0,),
    )
    with np.testing.assert_raises(ValueError):
        make_variant_function_spec("branin", bad_variant)


def test_branin_family_generation_is_reproducible() -> None:
    """Family generation with fixed seed should be deterministic."""
    family_a = make_branin_family(n_tasks=3, seed=7)
    family_b = make_branin_family(n_tasks=3, seed=7)
    x = np.array([[0.4, 0.6]], dtype=np.float64)
    y_a = [f.objective(x)[0] for f in family_a]
    y_b = [f.objective(x)[0] for f in family_b]
    assert len(family_a) == 3
    assert np.allclose(y_a, y_b)


def test_variant_optimum_scales_from_base() -> None:
    """Known optimum should propagate through output scaling."""
    base = get_function_spec("branin")
    assert base.optimum is not None
    variant = TaskVariantSpec(
        input_shift=(0.0, 0.0),
        input_scale=(1.0, 1.0),
        output_scale=1.1,
    )
    spec = make_variant_function_spec("branin", variant)
    expected = 1.1 * base.optimum
    assert spec.optimum is not None
    assert np.isclose(spec.optimum, expected)


def test_family_split_can_be_saved_loaded_and_reused(tmp_path: Path) -> None:
    """Saved family split should be reusable for train/test spec construction."""
    variants = generate_branin_variants(n_tasks=6, seed=3)
    split = split_variants(base_name="branin", variants=variants, train_ratio=0.5, seed=9)
    split_path = tmp_path / "branin_split.json"
    save_family_split(split, split_path)
    loaded = load_family_split(split_path)

    train_specs = build_specs(loaded.base_name, loaded.train_variants, prefix="train_task")
    test_specs = build_specs(loaded.base_name, loaded.test_variants, prefix="test_task")
    assert len(train_specs) + len(test_specs) == 6
    assert len(train_specs) > 0
    assert len(test_specs) > 0


def test_generate_variants_supports_sphere() -> None:
    """Generic family generator should support base function switching."""
    variants = generate_variants(base_name="sphere", n_tasks=4, seed=1)
    specs = build_specs(base_name="sphere", variants=variants, prefix="sphere_variant")
    assert len(specs) == 4
    assert all(spec.dim == 2 for spec in specs)


def test_new_synthetic_functions_output_shapes() -> None:
    """New functions should support expected batched shape outputs."""
    x2 = np.array([[0.2, 0.3], [0.7, 0.1]], dtype=np.float64)
    x6 = np.full((3, 6), 0.5, dtype=np.float64)
    assert ackley(x2).shape == (2,)
    assert rastrigin(x2).shape == (2,)
    assert rosenbrock(x2).shape == (2,)
    assert hartmann6(x6).shape == (3,)


def test_registry_contains_new_function_specs() -> None:
    """Registry should expose metadata for newly added functions."""
    for name, dim in [("ackley", 2), ("rastrigin", 2), ("rosenbrock", 2), ("hartmann6", 6)]:
        spec = get_function_spec(name)
        assert spec.dim == dim
        assert spec.optimum is not None
