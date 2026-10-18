"""Train a MetaBO neural acquisition function with PPO (Volpp et al., ICLR 2020).

Saves to `<output-dir>/<run-id>/` after every iteration: `af.pt` (latest),
`best_af.pt` (highest mean training return, the reference's checkpoint),
`naf_config.json`, `training_history.json` and `training_state.pt`. Pass
`--resume` with a larger `--n-iterations` to continue a run.

Families: `branin`, `goldstein_price` and `hartmann3` (MetaBO's global
benchmarks). `--tasks metabo50` trains on the 50 fixed tasks (the reference's
released setup); `--tasks full` draws a fresh task per episode. The candidate
set follows the family (Hartmann-3 is 3D, so its grid and AF inputs differ).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from openbo.optimizers.naf import NAF_FAMILY_SPECS, get_naf_family, train_naf


def main() -> None:
    """Parse arguments and train."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=tuple(NAF_FAMILY_SPECS), default="branin", help="Function family.")
    parser.add_argument("--tasks", choices=("metabo50", "full"), default="metabo50", help="Training tasks.")
    parser.add_argument("--n-iterations", type=int, default=2000, help="PPO iterations (1200 steps each).")
    parser.add_argument("--workers", type=int, default=7, help="Rollout worker processes.")
    parser.add_argument("--update-threads", type=int, default=8, help="Torch threads for the PPO update.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for initialization, tasks and sampling.")
    parser.add_argument("--run-id", default=None, help="Output subfolder name (default: <family>_metabo50).")
    parser.add_argument("--output-dir", default="meta-bo-training/naf", help="Output root folder.")
    parser.add_argument("--resume", action="store_true", help="Continue the run saved in the output folder.")
    args = parser.parse_args()

    family = get_naf_family(args.family)
    run_id = args.run_id or f"{args.family}_metabo50"
    config = family.config(
        n_iterations=args.n_iterations,
        n_workers=args.workers,
        update_threads=args.update_threads,
        seed=args.seed,
    )
    out = Path(args.output_dir) / run_id
    if (out / "training_state.pt").exists() and not args.resume:
        parser.error(f"{out} already holds a run; pass --resume to continue it, or pick another --run-id.")
    task_pool = family.grid_tasks(50) if args.tasks == "metabo50" else None
    task_sampler = family.sample_tasks if args.tasks == "full" else None
    print(f"training {args.family} ({args.tasks}) -> {out}", flush=True)
    start = time.perf_counter()
    result = train_naf(
        config,
        family.gp_hyperparameters,
        task_pool=task_pool,
        task_sampler=task_sampler,
        checkpoint_dir=out,
        log=lambda line: print(line, flush=True),
        resume=args.resume,
    )
    print(
        f"done: {len(result.mean_return)} iterations, best iteration {result.best_iteration} "
        f"(mean return {result.mean_return[result.best_iteration]:.3f}), "
        f"{time.perf_counter() - start:.0f} s this session; saved {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
