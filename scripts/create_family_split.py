"""Create and save a reusable train/test split for task families."""

from __future__ import annotations

import argparse

from metabo.test_functions.families import (
    generate_variants,
    save_family_split,
    split_variants,
)
from metabo.test_functions.tasks import TASK_DIMS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a train/test split JSON for task variants."
    )
    parser.add_argument(
        "--base-function",
        choices=sorted(TASK_DIMS.keys()),
        default="branin",
        help="Base function used to generate the family.",
    )
    parser.add_argument("--n-tasks", type=int, default=50, help="Total number of tasks.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of tasks assigned to train set.",
    )
    parser.add_argument(
        "--family-seed",
        type=int,
        default=0,
        help="Seed used for variant generation and splitting.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path for saved split. "
            "Default: configs/family_splits/{base_function}_split.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Generate variants, split into train/test, and save."""
    args = parse_args()
    output_path = (
        args.output
        if args.output is not None
        else f"configs/family_splits/{args.base_function}_split.json"
    )
    variants = generate_variants(
        base_name=args.base_function,
        n_tasks=args.n_tasks,
        seed=args.family_seed,
    )
    split = split_variants(
        base_name=args.base_function,
        variants=variants,
        train_ratio=args.train_ratio,
        seed=args.family_seed,
    )
    save_family_split(split, output_path)
    print(
        f"saved_split={output_path} n_tasks={args.n_tasks} "
        f"n_train={len(split.train_variants)} n_test={len(split.test_variants)}"
    )


if __name__ == "__main__":
    main()
