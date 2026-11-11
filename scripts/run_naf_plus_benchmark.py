"""Run one method of the NAF+ benchmark (Liao et al., CHI 2026, App. B, Test 1).

Methods: `naf_plus`, `naf_plus_no_ei`, `naf_plus_sample`, `naf_plus_no_ei_sample`,
`naf_plus_norepeat` and `naf_plus_no_ei_norepeat` (pass `--run-dir`, a `train_naf_plus.py` output
folder), `naf_plus_no_w`, `naf_plus_no_w_sample` and `naf_plus_no_w_norepeat`
(pass `--run-dir-no-w`, a run trained with `--no-weights`), `ei`, `taf_plus_15` and `random`, all on the
same unseen users, each with random objective weights. Tests: `double_sphere`
(users like the training ones, Test 1.1) and `double_sphere_novel` (moved
centres, Test 1.2). The `_sample` methods draw the AF's points from its
softmax, as in training, instead of taking its argmax; the `_norepeat` methods
take the argmax but never suggest a point already evaluated. Writes
`<results-dir>/trajectories/<test-id>_<method>_<test>.json` with the points,
values and simple regret per step, and for the NAF+ methods the fallback's
`p_bar` and `lambda_ei` per step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openbo.benchmarks.naf_plus_runner import NAF_PLUS_METHODS, NAF_PLUS_TESTS, run_naf_plus_method
from openbo.optimizers.naf_plus import DOUBLE_SPHERE_BUDGET


def main() -> None:
    """Parse arguments, run the method and save its trajectories."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=NAF_PLUS_METHODS, required=True, help="Method to run.")
    parser.add_argument("--test", choices=NAF_PLUS_TESTS, default="double_sphere", help="Test users.")
    parser.add_argument("--run-dir", default=None, help="NAF+ training run (the naf_plus methods except naf_plus_no_w*).")
    parser.add_argument("--run-dir-no-w", default=None, help="Training run without weights (naf_plus_no_w*).")
    parser.add_argument("--n-tasks", type=int, default=100, help="Unseen test users.")
    parser.add_argument("--budget", type=int, default=DOUBLE_SPHERE_BUDGET, help="Evaluations per user.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the methods' own randomness.")
    parser.add_argument("--test-seed", type=int, default=100, help="Seed that draws the test users.")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes (one user each).")
    parser.add_argument("--results-dir", default="test_results/naf_plus", help="Output root folder.")
    parser.add_argument("--test-id", default="default", help="Experiment id used in filenames.")
    args = parser.parse_args()
    if args.method.startswith("naf_plus") and not args.method.startswith("naf_plus_no_w") and args.run_dir is None:
        parser.error(f"--method {args.method} needs --run-dir.")
    if args.method.startswith("naf_plus_no_w") and args.run_dir_no_w is None:
        parser.error(f"--method {args.method} needs --run-dir-no-w.")

    start = time.perf_counter()
    result = run_naf_plus_method(
        args.method,
        test=args.test,
        n_tasks=args.n_tasks,
        run_dir=args.run_dir,
        run_dir_no_w=args.run_dir_no_w,
        budget=args.budget,
        seed=args.seed,
        test_seed=args.test_seed,
        n_workers=args.workers,
    )
    out = Path(args.results_dir) / "trajectories" / f"{args.test_id}_{args.method}_{args.test}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"method={args.method} test={args.test} tasks={args.n_tasks} seconds={time.perf_counter() - start:.0f} saved={out}")


if __name__ == "__main__":
    main()
