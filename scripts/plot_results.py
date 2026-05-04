"""Run multiple benchmark methods and plot observed y-values."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from metabo.benchmarks.runner import run_simple_benchmark


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run random/BO methods and plot y per iteration."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["random", "bo_scratch", "bo_botorch"],
        choices=["random", "bo_scratch", "bo_botorch"],
        help="Methods to compare.",
    )
    parser.add_argument("--function", default="branin", help="Test function name.")
    parser.add_argument("--n-evals", type=int, default=30, help="Total evaluations.")
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Number of random initial points (BO methods only).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help="Number of BO iterations after initialization (BO methods only).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output",
        default="benchmark_y_values.png",
        help="Output image path for the line chart.",
    )
    return parser.parse_args()


def main() -> None:
    """Run selected methods and create a line chart of y-values."""
    args = parse_args()

    plt.figure(figsize=(9, 5))
    for method in args.methods:
        result = run_simple_benchmark(
            function_name=args.function,
            n_evals=args.n_evals,
            seed=args.seed,
            method=method,
            n_init=args.n_init,
            n_iter=args.n_iter,
        )
        iterations = list(range(1, len(result.y_values) + 1))
        plt.plot(iterations, result.y_values, marker="o", linewidth=1.5, label=method)
        print(
            f"method={method} final_best={result.best_value:.6f} "
            f"num_points={len(result.y_values)}"
        )

    plt.xlabel("iteration")
    plt.ylabel("y_at_this_iteration")
    plt.title(f"Observed y-values per iteration ({args.function})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    print(f"saved_plot={output_path}")

if __name__ == "__main__":
    main()
