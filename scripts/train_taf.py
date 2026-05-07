"""Train TAF by running scratch BO on train tasks and storing final GPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openbo.optimizers.bo_scratch import run_bo_scratch
from openbo.test_functions.families import build_specs, load_family_split


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for TAF training."""
    parser = argparse.ArgumentParser(
        description=(
            "Train Transfer Acquisition Function (TAF) by running "
            "bo_scratch_multistart on family tasks and storing final GP states."
        )
    )
    parser.add_argument(
        "--split-path",
        default="configs/family_splits/branin_split.json",
        help="Path to saved train/test family split JSON.",
    )
    parser.add_argument(
        "--subset",
        choices=["train", "test", "all"],
        default="train",
        help="Which subset from split to use for TAF training.",
    )
    parser.add_argument("--n-evals", type=int, default=30, help="Evaluations per task.")
    parser.add_argument(
        "--optimizer-seed",
        type=int,
        default=0,
        help="Base seed for optimizer runs (task index is added).",
    )
    parser.add_argument(
        "--output-root",
        default="meta-bo-training/taf-gps",
        help="Root folder for saved trajectories and GP states.",
    )
    parser.add_argument(
        "--run-id",
        default="default",
        help="Run identifier used for output subfolder.",
    )
    return parser.parse_args()


def _bo_budget(n_evals: int) -> tuple[int, int]:
    """Return teaching-friendly BO budget split."""
    n_init = max(3, int(round(0.2 * n_evals)))
    n_init = min(n_init, max(n_evals - 1, 1))
    n_iter = max(n_evals - n_init, 1)
    return n_init, n_iter


def main() -> None:
    """Execute TAF training and persist per-task trajectories + final GPs."""
    args = parse_args()
    split = load_family_split(args.split_path)

    if args.subset == "train":
        variants = split.train_variants
        prefix = "train_task"
    elif args.subset == "test":
        variants = split.test_variants
        prefix = "test_task"
    else:
        variants = split.train_variants + split.test_variants
        prefix = "task"

    family = build_specs(split.base_name, variants, prefix=prefix)
    n_init, n_iter = _bo_budget(args.n_evals)

    run_dir = Path(args.output_root) / args.run_id
    trajectories_dir = run_dir / "trajectories"
    gp_states_dir = run_dir / "gp_states"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    gp_states_dir.mkdir(parents=True, exist_ok=True)

    summary_tasks: list[dict[str, object]] = []
    for idx, spec in enumerate(family):
        seed = args.optimizer_seed + idx
        result = run_bo_scratch(
            objective=spec.objective,
            bounds=spec.bounds,
            n_init=n_init,
            n_iter=n_iter,
            search_strategy="multistart",
            seed=seed,
        )

        traj_payload = {
            "task_name": spec.name,
            "base_function": split.base_name,
            "subset": args.subset,
            "n_points": int(result.y_obs.shape[0]),
            "optimal_value": None if spec.optimum is None else float(spec.optimum),
            "x_values": [[float(v) for v in row] for row in result.x_obs],
            "y_values": [float(v) for v in result.y_obs],
            "best_so_far": [float(v) for v in np.maximum.accumulate(result.y_obs)],
            "final_best": float(np.max(result.y_obs)),
        }
        gp_payload = {
            "task_name": spec.name,
            "base_function": split.base_name,
            "subset": args.subset,
            "optimizer": "bo_scratch_multistart",
            "n_init": int(n_init),
            "n_iter": int(n_iter),
            "seed": int(seed),
            "gp_state": result.final_gp_state,
        }

        (trajectories_dir / f"{spec.name}.json").write_text(
            json.dumps(traj_payload, indent=2), encoding="utf-8"
        )
        (gp_states_dir / f"{spec.name}.json").write_text(
            json.dumps(gp_payload, indent=2), encoding="utf-8"
        )

        summary_tasks.append(
            {
                "task_name": spec.name,
                "seed": int(seed),
                "final_best": float(np.max(result.y_obs)),
            }
        )
        print(
            f"task={idx:02d} name={spec.name} "
            f"final_best={float(np.max(result.y_obs)):.6f}"
        )

    summary = {
        "run_id": args.run_id,
        "split_path": args.split_path,
        "base_function": split.base_name,
        "subset": args.subset,
        "n_tasks": len(family),
        "n_evals": int(args.n_evals),
        "optimizer": "bo_scratch_multistart",
        "output_root": args.output_root,
        "tasks": summary_tasks,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved_taf_training={run_dir}")


if __name__ == "__main__":
    main()
