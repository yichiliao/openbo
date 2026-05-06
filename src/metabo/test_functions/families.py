"""Family generation, train/test splitting, and persistence utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metabo.test_functions.registry import FunctionSpec, make_variant_function_spec
from metabo.test_functions.tasks import TASK_DIMS, TaskVariantSpec


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
                noise_std=0.0,
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
) -> list[TaskVariantSpec]:
    """Backward-compatible Branin-specific family generator."""
    return generate_variants(
        base_name="branin",
        n_tasks=n_tasks,
        seed=seed,
        max_input_shift=max_input_shift,
        max_input_scale_delta=max_input_scale_delta,
        max_output_scale_delta=max_output_scale_delta,
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
