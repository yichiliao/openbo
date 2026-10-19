"""Run one method of the MetaBO benchmark (Volpp et al., ICLR 2020, Fig. 2).

Methods: `naf` (a trained neural AF; pass `--af-dir`, a `train_naf.py` output
folder), `ei`, `taf_me_50`, `taf_r_50` and `random`, all on the same unseen test
tasks of the family. Writes
`<results-dir>/trajectories/<test-id>_<method>_<family>.json` with the points,
values and simple regret per step for every test task.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openbo.benchmarks.naf_runner import NAF_FAMILIES, NAF_METHODS, run_naf_method


def main() -> None:
    """Parse arguments, run the method and save its trajectories."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=NAF_METHODS, required=True, help="Method to run.")
    parser.add_argument("--family", choices=NAF_FAMILIES, default="branin", help="Function family.")
    parser.add_argument("--af-dir", default=None, help="Training run folder with best_af.pt (naf only).")
    parser.add_argument("--n-tasks", type=int, default=100, help="Unseen test tasks.")
    parser.add_argument("--budget", type=int, default=30, help="Evaluations per task (naf: its training budget).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the methods' own randomness.")
    parser.add_argument("--test-seed", type=int, default=100, help="Seed that draws the test tasks.")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes (one task each).")
    parser.add_argument("--results-dir", default="test_results/naf", help="Output root folder.")
    parser.add_argument("--test-id", default="default", help="Experiment id used in filenames.")
    args = parser.parse_args()
    if args.method == "naf" and args.af_dir is None:
        parser.error("--method naf needs --af-dir.")

    start = time.perf_counter()
    result = run_naf_method(
        args.method,
        family=args.family,
        n_tasks=args.n_tasks,
        af_dir=args.af_dir,
        budget=args.budget,
        seed=args.seed,
        test_seed=args.test_seed,
        n_workers=args.workers,
    )
    out = Path(args.results_dir) / "trajectories" / f"{args.test_id}_{args.method}_{args.family}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"method={args.method} tasks={args.n_tasks} seconds={time.perf_counter() - start:.0f} saved={out}")


if __name__ == "__main__":
    main()
