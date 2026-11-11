"""Train NAF+ (Liao et al., CHI 2026): a weight-conditioned neural AF plus its novelty detector.

Fits the novelty detector first (once per run; about 25 minutes with the
paper's 2000 epochs), then trains the neural AF with PPO on fresh synthetic
users with random objective weights. Saves to `<output-dir>/<run-id>/`: the
neural AF files of `train_naf.py` (`af.pt`, `best_af.pt`, `naf_config.json`,
`training_history.json`, `training_state.pt`), `naf_plus_config.json` and the
detector (`detector.pt`, `detector.json`). Pass `--resume` with a larger
`--n-iterations` to continue a run.

`--no-weights` trains the "without weights" ablation: the same users, but the
AF never sees the weights. `--detector-from` copies the detector of an
earlier run with the same family settings instead of fitting it again.

Families: `double_sphere` (the paper's centres (0.4, 0.4) and (0.6, 0.6)) and
`double_sphere_wide` (OpenBO's: (0.2, 0.2) and (0.8, 0.8), so the weights move the
optimum across most of the square; novel users on the other diagonal). The
shift and scale ranges are not stated in the paper; the defaults are MetaBO's.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from openbo.optimizers.naf import NAFConfig
from openbo.optimizers.naf_plus import (
    DETECTOR_EPOCHS,
    DOUBLE_SPHERE_BUDGET,
    NAF_PLUS_FAMILIES,
    NAFPlusConfig,
    train_naf_plus,
)
from openbo.test_functions.naf_plus_families import DEFAULT_SCALE_RANGE, DEFAULT_SHIFT_RANGE


def main() -> None:
    """Parse arguments and train."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=NAF_PLUS_FAMILIES, default="double_sphere", help="User family.")
    parser.add_argument("--no-weights", action="store_true", help="Train the ablation whose AF ignores the weights.")
    parser.add_argument("--n-iterations", type=int, default=500, help="PPO iterations (1200 steps each).")
    parser.add_argument("--budget", type=int, default=DOUBLE_SPHERE_BUDGET, help="Episode length T.")
    parser.add_argument("--shift-range", type=float, default=DEFAULT_SHIFT_RANGE, help="User input shifts U(-r/2, r/2).")
    parser.add_argument("--scale-range", type=float, default=DEFAULT_SCALE_RANGE, help="User output scales U(1-s/2, 1+s/2).")
    parser.add_argument("--detector-epochs", type=int, default=DETECTOR_EPOCHS, help="Novelty detector epochs.")
    parser.add_argument("--detector-from", default=None, help="Copy the detector of this earlier run.")
    parser.add_argument("--workers", type=int, default=7, help="Rollout worker processes.")
    parser.add_argument("--update-threads", type=int, default=8, help="Torch threads for the PPO update.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for initialization, users and sampling.")
    parser.add_argument("--run-id", default="double_sphere", help="Output subfolder name.")
    parser.add_argument("--output-dir", default="meta-bo-training/naf_plus", help="Output root folder.")
    parser.add_argument("--resume", action="store_true", help="Continue the run saved in the output folder.")
    args = parser.parse_args()

    plus_config = NAFPlusConfig(
        family=args.family,
        shift_range=args.shift_range,
        scale_range=args.scale_range,
        detector_epochs=args.detector_epochs,
    )
    naf_config = NAFConfig(
        dim=plus_config.dim,
        budget=args.budget,
        n_iterations=args.n_iterations,
        n_workers=args.workers,
        update_threads=args.update_threads,
        seed=args.seed,
        n_weights=0 if args.no_weights else plus_config.n_objectives,
    )
    out = Path(args.output_dir) / args.run_id
    if (out / "training_state.pt").exists() and not args.resume:
        parser.error(f"{out} already holds a run; pass --resume to continue it, or pick another --run-id.")
    kind = "without weights" if args.no_weights else "with weights"
    print(f"training NAF+ {args.family} ({kind}) -> {out}", flush=True)
    start = time.perf_counter()
    result = train_naf_plus(
        naf_config,
        plus_config,
        out,
        log=lambda line: print(line, flush=True),
        resume=args.resume,
        detector_from=args.detector_from,
    )
    naf = result.naf
    print(
        f"done: {len(naf.mean_return)} iterations, best iteration {naf.best_iteration} "
        f"(mean return {naf.mean_return[naf.best_iteration]:.3f}), "
        f"{time.perf_counter() - start:.0f} s this session; saved {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
