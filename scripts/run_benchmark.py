"""Run a simple benchmark from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metabo.benchmarks.runner import run_simple_benchmark


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run OpenMetaBO benchmark.")
    parser.add_argument(
        "--method",
        choices=["random", "bo_botorch", "bo_scratch"],
        default="random",
        help="Optimization method.",
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
        "--results-dir",
        default="test_results",
        help="Root folder for test artifacts.",
    )
    parser.add_argument(
        "--test-id",
        default="default",
        help="Experiment id used in output filenames.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute benchmark runner and print final best value."""
    args = parse_args()
    result = run_simple_benchmark(
        function_name=args.function,
        n_evals=args.n_evals,
        seed=args.seed,
        method=args.method,
        n_init=args.n_init,
        n_iter=args.n_iter,
    )
    print(
        f"method={args.method} function={args.function} "
        f"best_value={result.best_value:.6f} best_x={result.best_x}"
    )

    trajectories_dir = Path(args.results_dir) / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    output_path = trajectories_dir / f"{args.test_id}_{args.method}_{args.function}.json"
    payload = {
        "test_id": args.test_id,
        "method": args.method,
        "function": args.function,
        "best_value": result.best_value,
        "best_x": result.best_x,
        "x_values": result.x_values,
        "y_values": result.y_values,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved_trajectory={output_path}")


if __name__ == "__main__":
    main()
