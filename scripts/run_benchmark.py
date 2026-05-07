"""Run a simple benchmark from the command line."""

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
    parser = argparse.ArgumentParser(description="Run OpenMetaBO benchmark.")
    parser.add_argument(
        "--method",
        choices=[
            "random",
            "bo_botorch",
            "bo_scratch",
            "bo_scratch_multistart",
            "bo_scratch_grid",
            "bo_taf",
        ],
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
    parser.add_argument(
        "--plot-x-locations",
        action="store_true",
        help="Plot sampled 2D x locations colored by iteration.",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional output path for x-location plot.",
    )
    parser.add_argument(
        "--noisy",
        action="store_true",
        help="Use noisy objective outputs (noise_std=0.05, capped at optimum).",
    )
    parser.add_argument(
        "--taf-run-dir",
        default=None,
        help="Path to TAF training run directory (required for method=bo_taf).",
    )
    parser.add_argument(
        "--taf-rho",
        type=float,
        default=1.0,
        help="Epanechnikov bandwidth rho for TAF-M weighting (bo_taf only).",
    )
    return parser.parse_args()


def _plot_x_locations(
    x_values: list[list[float]],
    objective: callable,
    output_path: Path,
    method: str,
    function_name: str,
) -> None:
    """Plot 2D sampled points over a function-value heatmap."""
    x_arr = np.asarray(x_values, dtype=np.float64)
    n_points = x_arr.shape[0]
    # Early iterations: light gray; late iterations: black.
    tones = np.linspace(0.8, 0.0, n_points)
    colors = np.stack([tones, tones, tones], axis=1)

    # Evaluate objective on a dense unit-square grid for a blue-red background.
    grid_n = 160
    axis = np.linspace(0.0, 1.0, grid_n, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    x_grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    z = np.asarray(objective(x_grid), dtype=np.float64).reshape(grid_n, grid_n)

    plt.figure(figsize=(6, 6))
    heat = plt.imshow(
        z,
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        cmap="coolwarm",
        alpha=0.75,
        aspect="equal",
    )
    plt.scatter(
        x_arr[:, 0],
        x_arr[:, 1],
        c=colors,
        s=45,
        edgecolors="white",
        linewidths=0.35,
    )
    plt.plot(x_arr[:, 0], x_arr[:, 1], color="0.7", linewidth=0.8, alpha=0.6)
    plt.colorbar(heat, fraction=0.046, pad=0.04, label="objective value")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"2D search trajectory ({method}, {function_name})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


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
        noise_std=0.05 if args.noisy else 0.0,
        cap_at_optimum=args.noisy,
        taf_run_dir=args.taf_run_dir,
        taf_rho=args.taf_rho,
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
        "noisy": bool(args.noisy),
        "noise_std": 0.05 if args.noisy else 0.0,
        "cap_at_optimum": bool(args.noisy),
        "best_value": result.best_value,
        "best_x": result.best_x,
        "x_values": result.x_values,
        "y_values": result.y_values,
    }
    if result.metadata is not None:
        payload["metadata"] = result.metadata
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved_trajectory={output_path}")

    if args.plot_x_locations:
        spec = get_function_spec(args.function)
        if spec.dim != 2:
            print(
                f"skip_x_location_plot function={args.function} dim={spec.dim} "
                "only_2d_supported=true"
            )
            return
        plot_path = (
            Path(args.plot_output)
            if args.plot_output is not None
            else Path(args.results_dir)
            / "plots"
            / f"{args.test_id}_{args.method}_{args.function}_x_locations.png"
        )
        _plot_x_locations(
            x_values=result.x_values,
            objective=spec.objective,
            output_path=plot_path,
            method=args.method,
            function_name=args.function,
        )
        print(f"saved_x_location_plot={plot_path}")


if __name__ == "__main__":
    main()
