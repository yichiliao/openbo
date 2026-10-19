"""Run one ConBO benchmark method over a sequence of simulated users.

Writes one JSON per seed to
`<results-dir>/trajectories/<test-id>_<method>_<test>_s<seed>.json`.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from openbo.benchmarks.conbo_runner import CONBO_METHODS, CONBO_TESTS, run_conbo_method


def main() -> None:
    """Parse arguments and run the requested seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=CONBO_METHODS, required=True, help="Method to run.")
    parser.add_argument("--test", choices=sorted(CONBO_TESTS), default="branin", help="User setting.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds (users differ per seed).")
    parser.add_argument("--results-dir", default="test_results/conbo", help="Output root folder.")
    parser.add_argument("--test-id", default="default", help="Experiment id used in filenames.")
    args = parser.parse_args()

    out_dir = Path(args.results_dir) / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as work_dir:
            result = run_conbo_method(args.method, args.test, seed, work_dir)
        path = out_dir / f"{args.test_id}_{args.method}_{args.test}_s{seed}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        print(
            f"method={args.method} test={args.test} seed={seed} "
            f"seconds={time.perf_counter() - start:.1f} saved={path}"
        )


if __name__ == "__main__":
    main()
