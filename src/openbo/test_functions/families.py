"""Family generation, train/test splitting, and persistence utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import qmc

from openbo.test_functions.registry import FunctionSpec, make_variant_function_spec
from openbo.test_functions.tasks import TASK_DIMS, TaskVariantSpec


@dataclass(frozen=True)
class FamilySplit:
    """A reusable train/test split of task variants for one base function."""

    base_name: str
    train_variants: list[TaskVariantSpec]
    test_variants: list[TaskVariantSpec]


def generate_variants(
    base_name: str,
    n_tasks: int,
    seed: int = 0,
    max_input_shift: float = 0.05,
    max_input_scale_delta: float = 0.1,
    max_output_scale_delta: float = 0.1,
    noise_std: float = 0.0,
    cap_at_optimum: bool = False,
) -> list[TaskVariantSpec]:
    """Generate a list of variants for a given base function."""
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive.")
    try:
        dim = TASK_DIMS[base_name]
    except KeyError as exc:
        raise KeyError(f"Unknown base function for variants: {base_name}") from exc
    rng = np.random.default_rng(seed)
    variants: list[TaskVariantSpec] = []
    for idx in range(n_tasks):
        shift = tuple(rng.uniform(-max_input_shift, max_input_shift, size=dim).tolist())
        scale = tuple(
            rng.uniform(1.0 - max_input_scale_delta, 1.0 + max_input_scale_delta, size=dim).tolist()
        )
        output_scale = float(
            rng.uniform(1.0 - max_output_scale_delta, 1.0 + max_output_scale_delta)
        )
        variants.append(
            TaskVariantSpec(
                input_shift=shift,
                input_scale=scale,
                output_scale=output_scale,
                noise_std=noise_std,
                cap_at_optimum=cap_at_optimum,
                seed=seed + idx,
            )
        )
    return variants


def generate_branin_variants(
    n_tasks: int,
    seed: int = 0,
    max_input_shift: float = 0.05,
    max_input_scale_delta: float = 0.1,
    max_output_scale_delta: float = 0.1,
    noise_std: float = 0.0,
    cap_at_optimum: bool = False,
) -> list[TaskVariantSpec]:
    """Backward-compatible Branin-specific family generator."""
    return generate_variants(
        base_name="branin",
        n_tasks=n_tasks,
        seed=seed,
        max_input_shift=max_input_shift,
        max_input_scale_delta=max_input_scale_delta,
        max_output_scale_delta=max_output_scale_delta,
        noise_std=noise_std,
        cap_at_optimum=cap_at_optimum,
    )


def split_variants(
    base_name: str,
    variants: list[TaskVariantSpec],
    train_ratio: float = 0.8,
    seed: int = 0,
) -> FamilySplit:
    """Split variants into train/test sets with a seeded shuffle."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if len(variants) < 2:
        raise ValueError("Need at least 2 variants to create train/test split.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(variants))
    rng.shuffle(indices)

    n_train = max(1, min(len(variants) - 1, int(round(train_ratio * len(variants)))))
    train_idx = set(indices[:n_train].tolist())
    train_variants = [v for i, v in enumerate(variants) if i in train_idx]
    test_variants = [v for i, v in enumerate(variants) if i not in train_idx]
    return FamilySplit(
        base_name=base_name,
        train_variants=train_variants,
        test_variants=test_variants,
    )


def build_specs(base_name: str, variants: list[TaskVariantSpec], prefix: str) -> list[FunctionSpec]:
    """Convert variants into executable function specs."""
    return [
        make_variant_function_spec(
            base_name=base_name,
            variant=variant,
            variant_name=f"{prefix}_{idx:03d}",
        )
        for idx, variant in enumerate(variants)
    ]


def save_family_split(split: FamilySplit, path: str | Path) -> None:
    """Save split to disk in JSON format."""
    payload = {
        "base_name": split.base_name,
        "train_variants": [v.to_dict() for v in split.train_variants],
        "test_variants": [v.to_dict() for v in split.test_variants],
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_family_split(path: str | Path) -> FamilySplit:
    """Load split from disk."""
    in_path = Path(path)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    return FamilySplit(
        base_name=str(payload["base_name"]),
        train_variants=[
            TaskVariantSpec.from_dict(item) for item in payload["train_variants"]
        ],
        test_variants=[
            TaskVariantSpec.from_dict(item) for item in payload["test_variants"]
        ],
    )


def estimate_optimum(
    objective: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    dim: int,
    *,
    grid_points_per_dim: int = 201,
    n_sobol: int = 1 << 14,
    n_polish: int = 5,
    seed: int = 0,
) -> float:
    """Estimate the maximum of a deterministic objective on `[0, 1]^dim`.

    Scans a full grid for `dim <= 3` (a Sobol set otherwise), then polishes the
    best `n_polish` points with bounded L-BFGS-B.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if grid_points_per_dim < 2:
        raise ValueError("grid_points_per_dim must be at least 2.")
    if n_polish < 0:
        raise ValueError("n_polish must be non-negative.")
    if dim <= 3:
        axis = np.linspace(0.0, 1.0, grid_points_per_dim)
        mesh = np.meshgrid(*([axis] * dim), indexing="ij")
        x_scan = np.stack([m.ravel() for m in mesh], axis=1)
    else:
        x_scan = qmc.Sobol(d=dim, scramble=True, seed=seed).random(n_sobol)
    y_scan = np.asarray(objective(x_scan), dtype=np.float64)
    best = float(np.max(y_scan))

    for idx in np.argsort(y_scan)[::-1][:n_polish]:
        result = minimize(
            lambda z: -float(objective(z[None, :])[0]),
            x_scan[idx],
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * dim,
        )
        best = max(best, -float(result.fun))
    return best


def generate_user_variants(
    base_name: str,
    n_users: int,
    shift_range: float,
    scale_range: float,
    seed: int = 0,
) -> list[TaskVariantSpec]:
    """Sample simulated-user variants as in ConBO (Liao et al., CHI 2025, Appendix A).

    Each user shifts the inputs by `U(-shift_range/2, shift_range/2)` per
    dimension and scales the output by `U(1 - scale_range/2, 1 + scale_range/2)`,
    without input scaling or noise.
    """
    if n_users <= 0:
        raise ValueError("n_users must be positive.")
    if shift_range < 0.0:
        raise ValueError("shift_range must be non-negative.")
    if not 0.0 <= scale_range < 2.0:
        raise ValueError("scale_range must be in [0, 2) so output scales stay positive.")
    return generate_variants(
        base_name=base_name,
        n_tasks=n_users,
        seed=seed,
        max_input_shift=shift_range / 2.0,
        max_input_scale_delta=0.0,
        max_output_scale_delta=scale_range / 2.0,
    )


def make_user_sequence(
    base_name: str,
    variants: list[TaskVariantSpec],
    prefix: str = "user",
) -> list[FunctionSpec]:
    """Build user specs from `generate_user_variants` output, in order.

    Shifted inputs are clipped to `[0, 1]`, so a user's optimum can differ from
    the scaled base optimum; each spec's `optimum` is estimated numerically.
    """
    specs = build_specs(base_name, variants, prefix)
    return [replace(spec, optimum=estimate_optimum(spec.objective, spec.dim)) for spec in specs]
