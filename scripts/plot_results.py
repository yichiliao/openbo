"""Run multiple benchmark methods and plot observed y-values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metabo.benchmarks.runner import run_simple_benchmark
from metabo.test_functions.registry import get_function_spec


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
    parser.add_argument(
        "--trajectory-dir",
        default=None,
        help=(
            "Optional directory containing saved single-run trajectories. "
            "When provided, y-values are loaded from disk instead of rerunning."
        ),
    )
    parser.add_argument(
        "--test-id",
        default="default",
        help=(
            "Experiment id used in run_benchmark trajectory filenames, "
            "e.g. {test_id}_{method}_{function}.json."
        ),
    )
    return parser.parse_args()


def _load_stored_y_values(
    trajectory_dir: Path,
    test_id: str,
    method: str,
    function_name: str,
) -> np.ndarray:
    """Load y-values from a stored single-benchmark trajectory file."""
    filename = f"{test_id}_{method}_{function_name}.json"
    path = trajectory_dir / filename
    if not path.exists():
        raise ValueError(f"Missing trajectory file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["y_values"], dtype=np.float64)


def main() -> None:
    """Run selected methods and create a line chart of log-regret values."""
    args = parse_args()
    spec = get_function_spec(args.function)
    if spec.optimum is None:
        raise ValueError(
            f"Function '{args.function}' has unknown optimum; cannot plot log-regret."
        )
    optimum = float(spec.optimum)

    plt.figure(figsize=(9, 5))
    trajectory_dir = Path(args.trajectory_dir) if args.trajectory_dir is not None else None
    for method in args.methods:
        if trajectory_dir is None:
            result = run_simple_benchmark(
                function_name=args.function,
                n_evals=args.n_evals,
                seed=args.seed,
                method=method,
                n_init=args.n_init,
                n_iter=args.n_iter,
            )
            y_values = np.asarray(result.y_values, dtype=np.float64)
            final_best = float(result.best_value)
        else:
            y_values = _load_stored_y_values(
                trajectory_dir=trajectory_dir,
                test_id=args.test_id,
                method=method,
                function_name=args.function,
            )
            final_best = float(np.max(y_values))
        diff = np.maximum(optimum - y_values, 1e-12)
        y_log_regret = np.log10(diff)
        iterations = list(range(1, len(y_values) + 1))
        plt.plot(iterations, y_log_regret, marker="o", linewidth=1.5, label=method)
        print(
            f"method={method} final_best={final_best:.6f} "
            f"final_log10_regret={y_log_regret[-1]:.6f} num_points={len(y_values)}"
        )

    plt.xlabel("iteration")
    plt.ylabel("log10(optimal_value - y_at_this_iteration)")
    plt.title(f"Log-regret per iteration ({args.function})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    print(f"saved_plot={output_path}")

if __name__ == "__main__":
    main()
