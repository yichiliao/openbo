"""Run benchmark methods across a family of Branin variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from metabo.optimizers.bo_botorch import run_bo_botorch
from metabo.optimizers.bo_scratch import run_bo_scratch
from metabo.optimizers.bo_taf import run_bo_taf
from metabo.optimizers.random_search import RandomSearch
from metabo.test_functions.families import (
    build_specs,
    generate_variants,
    load_family_split,
)
from metabo.test_functions.registry import FunctionSpec
from metabo.test_functions.tasks import TASK_DIMS, TaskVariantSpec


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run random/scratch/BoTorch BO across a task family."
    )
    parser.add_argument(
        "--base-function",
        choices=sorted(TASK_DIMS.keys()),
        default="branin",
        help="Base function used for on-the-fly family generation.",
    )
    parser.add_argument(
        "--method",
        choices=[
            "random",
            "bo_scratch",
            "bo_scratch_multistart",
            "bo_scratch_grid",
            "bo_botorch",
            "bo_taf",
            "bo_taf_m",
            "bo_taf_r",
        ],
        default="bo_scratch",
        help="Optimization method to evaluate.",
    )
    parser.add_argument("--n-tasks", type=int, default=5, help="Number of family tasks.")
    parser.add_argument("--n-evals", type=int, default=30, help="Evaluation budget per task.")
    parser.add_argument(
        "--family-seed",
        type=int,
        default=0,
        help="Seed used for generating family variants.",
    )
    parser.add_argument(
        "--optimizer-seed",
        type=int,
        default=0,
        help="Base seed used for each optimizer run.",
    )
    parser.add_argument(
        "--split-path",
        default=None,
        help="Optional path to load a reusable train/test family split JSON.",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "train", "test"],
        default="all",
        help="Which split subset to benchmark when split is available.",
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
        "--noisy",
        action="store_true",
        help="Enable noisy family tasks (noise_std=0.05, capped at optimum).",
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
    parser.add_argument(
        "--taf-weight-mode",
        choices=["taf_m", "taf_r"],
        default="taf_m",
        help="TAF source-weight mode: meta-feature (taf_m) or ranking-based (taf_r).",
    )
    parser.add_argument(
        "--taf-weight-modes",
        nargs="+",
        choices=["taf_m", "taf_r"],
        default=None,
        help=(
            "Optional list of TAF modes to run in one command when method=bo_taf, "
            "e.g. --taf-weight-modes taf_m taf_r."
        ),
    )
    return parser.parse_args()


def _variant_meta_features(variant: TaskVariantSpec) -> np.ndarray:
    """Explicit variant meta-features for TAF-M weighting."""
    return np.array(
        [
            *[float(v) for v in variant.input_shift],
            *[float(v) for v in variant.input_scale],
            float(variant.output_scale),
            float(variant.noise_std),
            float(1.0 if variant.cap_at_optimum else 0.0),
        ],
        dtype=np.float64,
    )


def _run_one_task(
    spec: FunctionSpec,
    method: str,
    n_evals: int,
    seed: int,
    taf_run_dir: str | None = None,
    taf_rho: float = 1.0,
    taf_weight_mode: str = "taf_m",
    taf_source_meta: dict[str, np.ndarray] | None = None,
    taf_target_meta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one method on one task and return observations `(x, y)`."""
    if method == "random":
        optimizer = RandomSearch(bounds=spec.bounds, seed=seed)
        x, y = optimizer.run(spec.objective, n_evals=n_evals)
    else:
        n_init = max(3, int(round(0.2 * n_evals)))
        n_init = min(n_init, max(n_evals - 1, 1))
        n_iter = max(n_evals - n_init, 1)
        if method in {"bo_scratch", "bo_scratch_multistart", "bo_scratch_grid"}:
            scratch_strategy = (
                "grid" if method == "bo_scratch_grid" else "multistart"
            )
            result = run_bo_scratch(
                objective=spec.objective,
                bounds=spec.bounds,
                n_init=n_init,
                n_iter=n_iter,
                search_strategy=scratch_strategy,
                seed=seed,
            )
        elif method == "bo_botorch":
            result = run_bo_botorch(
                objective=spec.objective,
                bounds=spec.bounds,
                n_init=n_init,
                n_iter=n_iter,
                seed=seed,
            )
        elif method in {"bo_taf", "bo_taf_m", "bo_taf_r"}:
            if taf_run_dir is None:
                raise ValueError("taf_run_dir is required for TAF methods.")
            resolved_taf_mode = (
                "taf_m" if method == "bo_taf_m" else
                "taf_r" if method == "bo_taf_r" else
                taf_weight_mode
            )
            result = run_bo_taf(
                objective=spec.objective,
                bounds=spec.bounds,
                taf_run_dir=taf_run_dir,
                n_init=0,
                n_iter=n_evals,
                rho=taf_rho,
                taf_weight_mode=resolved_taf_mode,
                source_meta_features=taf_source_meta,
                target_meta_features=taf_target_meta,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        x = result.x_obs
        y = result.y_obs

    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _save_trajectory_file(
    output_path: Path,
    task_name: str,
    method: str,
    optimal_value: float | None,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> None:
    """Persist per-task trajectory information as JSON."""
    payload = {
        "task_name": task_name,
        "method": method,
        "optimal_value": None if optimal_value is None else float(optimal_value),
        "n_points": int(len(y_values)),
        "x_values": [[float(v) for v in row] for row in x_values],
        "y_values": [float(v) for v in y_values],
        "best_so_far": [float(v) for v in np.maximum.accumulate(y_values)],
        "final_best": float(np.max(y_values)),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Generate family tasks, run method, and summarize performance."""
    args = parse_args()
    noise_std = 0.05 if args.noisy else 0.0
    cap_at_optimum = bool(args.noisy)
    source_meta_map: dict[str, np.ndarray] | None = None
    eval_variants: list[TaskVariantSpec] | None = None
    if args.split_path is None:
        variants = generate_variants(
            base_name=args.base_function,
            n_tasks=args.n_tasks,
            seed=args.family_seed,
            noise_std=noise_std,
            cap_at_optimum=cap_at_optimum,
        )
        family = build_specs(
            base_name=args.base_function,
            variants=variants,
            prefix=f"{args.base_function}_variant",
        )
        eval_variants = variants
    else:
        split = load_family_split(args.split_path)
        if split.base_name != args.base_function:
            raise ValueError(
                f"Split base function is '{split.base_name}', but --base-function="
                f"'{args.base_function}' was requested."
            )

        if args.subset == "train":
            chosen_variants = split.train_variants
        elif args.subset == "test":
            chosen_variants = split.test_variants
        else:
            chosen_variants = split.train_variants + split.test_variants
        if args.noisy:
            chosen_variants = [
                replace(v, noise_std=0.05, cap_at_optimum=True) for v in chosen_variants
            ]
        if args.subset == "train":
            family = build_specs(split.base_name, chosen_variants, prefix="train_task")
            eval_variants = chosen_variants
        elif args.subset == "test":
            family = build_specs(split.base_name, chosen_variants, prefix="test_task")
            eval_variants = chosen_variants
        else:
            n_train = len(split.train_variants)
            family = build_specs(split.base_name, chosen_variants[:n_train], prefix="train_task")
            family += build_specs(split.base_name, chosen_variants[n_train:], prefix="test_task")
            eval_variants = chosen_variants
        source_meta_map = {
            f"train_task_{idx:03d}": _variant_meta_features(variant)
            for idx, variant in enumerate(split.train_variants)
        }
        source_meta_map.update(
            {
                f"test_task_{idx:03d}": _variant_meta_features(variant)
                for idx, variant in enumerate(split.test_variants)
            }
        )
        print(
            f"loaded_split subset={args.subset} "
            f"n_tasks={len(family)} split_path={args.split_path}"
        )

    if args.taf_weight_modes is not None and args.method not in {"bo_taf", "bo_taf_m", "bo_taf_r"}:
        raise ValueError("--taf-weight-modes is supported only when --method=bo_taf.")
    taf_modes = (
        list(dict.fromkeys(args.taf_weight_modes))
        if args.taf_weight_modes is not None
        else [args.taf_weight_mode]
    )
    if args.method == "bo_taf_m":
        taf_modes = ["taf_m"]
    elif args.method == "bo_taf_r":
        taf_modes = ["taf_r"]
    elif args.method != "bo_taf":
        taf_modes = [args.taf_weight_mode]

    trajectories_root = Path(args.results_dir) / "trajectories"
    trajectories_root.mkdir(parents=True, exist_ok=True)
    subset_tag = args.subset if args.split_path is not None else "all"

    for taf_mode in taf_modes:
        final_bests: list[float] = []
        run_dir_name = f"{args.test_id}_{args.method}_{args.base_function}_{subset_tag}"
        if args.method in {"bo_taf", "bo_taf_m", "bo_taf_r"}:
            run_dir_name = f"{run_dir_name}_{taf_mode}"
        run_trajectories_dir = trajectories_root / run_dir_name
        run_trajectories_dir.mkdir(parents=True, exist_ok=True)

        for idx, spec in enumerate(family):
            task_seed = args.optimizer_seed + idx
            target_meta = (
                _variant_meta_features(eval_variants[idx])
                if eval_variants is not None
                else None
            )
            x_values, y_values = _run_one_task(
                spec=spec,
                method=args.method,
                n_evals=args.n_evals,
                seed=task_seed,
                taf_run_dir=args.taf_run_dir,
                taf_rho=args.taf_rho,
                taf_weight_mode=taf_mode,
                taf_source_meta=source_meta_map,
                taf_target_meta=target_meta,
            )
            final_best = float(np.max(y_values))
            final_bests.append(final_best)
            out_name = f"{spec.name}.json"
            _save_trajectory_file(
                output_path=run_trajectories_dir / out_name,
                task_name=spec.name,
                method=args.method,
                optimal_value=spec.optimum,
                x_values=x_values,
                y_values=y_values,
            )
            print(
                f"task={idx:02d} name={spec.name} method={args.method} "
                f"taf_weight_mode={taf_mode} final_best={final_best:.6f}"
            )

        avg = mean(final_bests)
        std = pstdev(final_bests) if len(final_bests) > 1 else 0.0
        print("-" * 72)
        print(
            f"summary method={args.method} taf_weight_mode={taf_mode} "
            f"n_tasks={len(final_bests)} mean_final_best={avg:.6f} std_final_best={std:.6f}"
        )
        summary_path = run_trajectories_dir / "summary.json"
        summary_payload = {
            "test_id": args.test_id,
            "method": args.method,
            "taf_weight_mode": taf_mode if args.method in {"bo_taf", "bo_taf_m", "bo_taf_r"} else None,
            "base_function": args.base_function,
            "subset": subset_tag,
            "n_tasks": len(final_bests),
            "mean_final_best": avg,
            "std_final_best": std,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"saved_trajectories_dir={run_trajectories_dir}")


if __name__ == "__main__":
    main()
