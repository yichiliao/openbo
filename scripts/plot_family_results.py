"""Run family benchmarks and plot mean/std objective per iteration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metabo.optimizers.bo_botorch import run_bo_botorch
from metabo.optimizers.bo_scratch import run_bo_scratch
from metabo.optimizers.random_search import RandomSearch
from metabo.test_functions.families import build_specs, generate_variants, load_family_split
from metabo.test_functions.registry import FunctionSpec
from metabo.test_functions.tasks import TASK_DIMS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot mean/std objective trajectories across a task family."
    )
    parser.add_argument(
        "--base-function",
        choices=sorted(TASK_DIMS.keys()),
        default="branin",
        help="Base function used for on-the-fly family generation.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["random", "bo_scratch", "bo_botorch"],
        default=["random", "bo_scratch", "bo_botorch"],
        help="Methods to compare.",
    )
    parser.add_argument("--n-tasks", type=int, default=10, help="Number of family tasks.")
    parser.add_argument("--n-evals", type=int, default=30, help="Evaluations per task.")
    parser.add_argument("--family-seed", type=int, default=0, help="Family generation seed.")
    parser.add_argument(
        "--optimizer-seed",
        type=int,
        default=0,
        help="Base optimizer seed (task index is added to this).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path for raw objective mean/std plot.",
    )
    parser.add_argument(
        "--output-best-so-far",
        default=None,
        help="Optional explicit output path for best-so-far mean/std plot.",
    )
    parser.add_argument(
        "--split-path",
        default=None,
        help="Optional path to load a saved train/test family split.",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "train", "test"],
        default="all",
        help="Which split subset to use when --split-path is provided.",
    )
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
        "--trajectory-run-dir",
        default=None,
        help=(
            "Optional path to a saved run folder under test_results/trajectories. "
            "When provided, plots are generated from stored trajectories instead of rerunning."
        ),
    )
    return parser.parse_args()


def _bo_budget(n_evals: int) -> tuple[int, int]:
    """Return teaching-friendly BO budget split."""
    n_init = max(3, int(round(0.2 * n_evals)))
    n_init = min(n_init, max(n_evals - 1, 1))
    n_iter = max(n_evals - n_init, 1)
    return n_init, n_iter


def _run_one_trajectory(spec: FunctionSpec, method: str, n_evals: int, seed: int) -> np.ndarray:
    """Run one task and return y-values at each iteration."""
    if method == "random":
        optimizer = RandomSearch(bounds=spec.bounds, seed=seed)
        _, y = optimizer.run(spec.objective, n_evals=n_evals)
        return y.astype(np.float64)

    n_init, n_iter = _bo_budget(n_evals)
    if method == "bo_scratch":
        result = run_bo_scratch(
            objective=spec.objective,
            bounds=spec.bounds,
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
        )
        return result.y_obs.astype(np.float64)
    if method == "bo_botorch":
        result = run_bo_botorch(
            objective=spec.objective,
            bounds=spec.bounds,
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
        )
        return result.y_obs.astype(np.float64)
    raise ValueError(f"Unknown method: {method}")


def _to_log_regret(y: np.ndarray, optimum: float) -> np.ndarray:
    """Convert objective values to log10 regret for maximization."""
    diff = np.maximum(optimum - y, 1e-12)
    return np.log10(diff)


def _plot_mean_std(
    method_to_matrix: dict[str, np.ndarray],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot mean and standard deviation bands from trajectory matrices."""
    plt.figure(figsize=(9, 5))
    for method, y_mat in method_to_matrix.items():
        y_mean = np.mean(y_mat, axis=0)
        y_std = np.std(y_mat, axis=0)
        iterations = np.arange(1, y_mean.shape[0] + 1)
        plt.plot(iterations, y_mean, linewidth=2.0, label=f"{method} mean")
        plt.fill_between(
            iterations,
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.2,
            label=f"{method} ±1 std",
        )

    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _load_from_stored_trajectories(
    run_dir: Path,
) -> tuple[str, str, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load one method's trajectories from disk and return plot matrices."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"Missing summary.json in trajectory run dir: {run_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    method = str(summary["method"])
    base_function = str(summary.get("base_function", "unknown"))

    task_files = sorted(
        p for p in run_dir.glob("*.json") if p.name != "summary.json"
    )
    if not task_files:
        raise ValueError(f"No task trajectory files found in: {run_dir}")

    raw_rows: list[np.ndarray] = []
    best_rows: list[np.ndarray] = []
    for path in task_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        y = np.asarray(payload["y_values"], dtype=np.float64)
        y_best = np.asarray(payload["best_so_far"], dtype=np.float64)
        optimum = payload.get("optimal_value")
        if optimum is None:
            raise ValueError(
                f"Trajectory '{path.name}' missing 'optimal_value'. "
                "Re-run run_family_benchmark.py to regenerate trajectories."
            )
        raw_rows.append(_to_log_regret(y, float(optimum)))
        best_rows.append(_to_log_regret(y_best, float(optimum)))

    method_to_y_mat = {method: np.stack(raw_rows, axis=0)}
    method_to_best_mat = {method: np.stack(best_rows, axis=0)}
    return method, base_function, method_to_y_mat, method_to_best_mat


def main() -> None:
    """Generate family runs and plot mean/std at each iteration."""
    args = parse_args()
    effective_base_function = args.base_function
    if args.trajectory_run_dir is None:
        if args.split_path is None:
            variants = generate_variants(
                base_name=args.base_function,
                n_tasks=args.n_tasks,
                seed=args.family_seed,
            )
            family = build_specs(
                base_name=args.base_function,
                variants=variants,
                prefix=f"{args.base_function}_variant",
            )
        else:
            split = load_family_split(args.split_path)
            if split.base_name != args.base_function:
                raise ValueError(
                    f"Split base function is '{split.base_name}', but --base-function="
                    f"'{args.base_function}' was requested."
                )
            if args.subset == "train":
                family = build_specs(split.base_name, split.train_variants, prefix="train_task")
            elif args.subset == "test":
                family = build_specs(split.base_name, split.test_variants, prefix="test_task")
            else:
                family = build_specs(split.base_name, split.train_variants, prefix="train_task")
                family += build_specs(split.base_name, split.test_variants, prefix="test_task")

        method_to_y_mat: dict[str, np.ndarray] = {}
        method_to_best_mat: dict[str, np.ndarray] = {}
        for method in args.methods:
            raw_log_regret_trajectories: list[np.ndarray] = []
            best_log_regret_trajectories: list[np.ndarray] = []
            for idx, spec in enumerate(family):
                if spec.optimum is None:
                    raise ValueError(
                        f"Task '{spec.name}' has unknown optimum; cannot compute log-regret."
                    )
                seed = args.optimizer_seed + idx
                y = _run_one_trajectory(spec, method, args.n_evals, seed)
                y_best = np.maximum.accumulate(y)
                raw_log_regret_trajectories.append(_to_log_regret(y, spec.optimum))
                best_log_regret_trajectories.append(_to_log_regret(y_best, spec.optimum))

            y_mat = np.stack(raw_log_regret_trajectories, axis=0)
            best_mat = np.stack(best_log_regret_trajectories, axis=0)
            method_to_y_mat[method] = y_mat
            method_to_best_mat[method] = best_mat

            y_mean = np.mean(y_mat, axis=0)
            y_std = np.std(y_mat, axis=0)
            best_mean = np.mean(best_mat, axis=0)
            best_std = np.std(best_mat, axis=0)
            print(
                f"method={method} tasks={len(family)} "
                f"raw_final_mean={y_mean[-1]:.6f} raw_final_std={y_std[-1]:.6f} "
                f"best_final_mean={best_mean[-1]:.6f} best_final_std={best_std[-1]:.6f}"
            )
    else:
        run_dir = Path(args.trajectory_run_dir)
        (
            loaded_method,
            effective_base_function,
            method_to_y_mat,
            method_to_best_mat,
        ) = _load_from_stored_trajectories(run_dir)
        y_mat = method_to_y_mat[loaded_method]
        best_mat = method_to_best_mat[loaded_method]
        print(
            f"method={loaded_method} tasks={y_mat.shape[0]} "
            f"raw_final_mean={np.mean(y_mat, axis=0)[-1]:.6f} "
            f"raw_final_std={np.std(y_mat, axis=0)[-1]:.6f} "
            f"best_final_mean={np.mean(best_mat, axis=0)[-1]:.6f} "
            f"best_final_std={np.std(best_mat, axis=0)[-1]:.6f}"
        )

    plots_dir = Path(args.results_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    raw_output_path = (
        Path(args.output)
        if args.output is not None
        else plots_dir / f"{args.test_id}_{effective_base_function}_family_mean_std_plot.png"
    )
    _plot_mean_std(
        method_to_matrix=method_to_y_mat,
        title="Family benchmark: mean log10 regret with standard deviation",
        ylabel="log10(optimal_value - y)",
        output_path=raw_output_path,
    )
    print(f"saved_plot={raw_output_path}")

    best_output_path = (
        Path(args.output_best_so_far)
        if args.output_best_so_far is not None
        else plots_dir
        / f"{args.test_id}_{effective_base_function}_family_best_so_far_mean_std_plot.png"
    )
    _plot_mean_std(
        method_to_matrix=method_to_best_mat,
        title="Family benchmark: mean best-so-far log10 regret with standard deviation",
        ylabel="log10(optimal_value - best_so_far_y)",
        output_path=best_output_path,
    )
    print(f"saved_best_so_far_plot={best_output_path}")


if __name__ == "__main__":
    main()
