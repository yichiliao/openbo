"""Calibrate NAF+'s novelty test for a training run (plan ``docs/plans/naf_plus.md``, "Design stance", item 3).

Simulates fresh users from the run's own (training) family, each with random
weights, runs the run's neural AF alone on them (greedy and sampled), and
records the statistic ``S_n = sum_i z_i^2`` of the detector's standardized
residuals after each evaluation. The result, ``novelty_calibration.json`` in
the run folder, is the null distribution that turns a new user's ``S_n`` into
a calibrated p-value (the share of simulated users at least as extreme); the
``naf_plus_cal`` benchmark methods need it. By construction about ``tau`` (5%)
of training-like users are flagged at any one evaluation.

Example::

    uv run python scripts/calibrate_naf_plus_novelty.py \\
        --run-dir meta-bo-training/naf_plus/double_sphere_wide_v1 --workers 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from openbo.acquisition.naf_plus import CALIBRATED_TAU
from openbo.optimizers.naf_plus import (
    CALIBRATION_POLICIES,
    CALIBRATION_SEED,
    CALIBRATION_USERS,
    calibrate_novelty,
)


def main() -> None:
    """Run the calibration and save it next to the detector."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="A train_naf_plus output folder (with a detector).")
    parser.add_argument("--n-users", type=int, default=CALIBRATION_USERS, help="Simulated users per policy.")
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEED, help="Seed that draws the users.")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes.")
    parser.add_argument("--policies", nargs="+", choices=CALIBRATION_POLICIES, default=list(CALIBRATION_POLICIES))
    args = parser.parse_args()

    start = time.perf_counter()
    calibration = calibrate_novelty(args.run_dir, args.n_users, args.seed, args.workers, args.policies)
    calibration.save(args.run_dir)
    print(f"saved {args.run_dir}/novelty_calibration.json ({time.perf_counter() - start:.0f} s)")
    for policy, per_n in calibration.statistics.items():
        thresholds = [np.quantile(row, 1 - CALIBRATED_TAU) for row in per_n]
        shown = ", ".join(f"n={n}: {thresholds[n - 1]:.1f}" for n in (1, 3, 5, 10, 20) if n <= len(thresholds))
        print(f"{policy}: S_n at the {1 - CALIBRATED_TAU:.0%} quantile: {shown}")


if __name__ == "__main__":
    main()
