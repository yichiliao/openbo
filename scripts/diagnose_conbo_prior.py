"""Judge a ConBO population model as a prior for unseen users (Test 1 setting).

Runs one method over the 15 users of `--seed`, then compares the final population BNN with
30 held-out users on a 50 x 50 grid: log10 regret of the BNN mean's argmax, Spearman
correlation with each user's function, the share of the function inside the BNN's 2 sd
band (calibration), the mean absolute error and the mean predicted sd. Used for the
question "why does replay not win" in
`docs/plans/paper_reproduction.md`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
from scipy.stats import spearmanr

from openbo.benchmarks import conbo_runner as cr
from openbo.optimizers.conbo import run_conbo_sequence


def main() -> None:
    """Train a population and write the held-out diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=sorted(cr.CONBO_VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="JSON file to write.")
    args = parser.parse_args()
    method, seed, out = args.method, args.seed, args.out
    test = cr.CONBO_TESTS["branin"]
    _, users = cr.make_test_users(test, seed)
    cfg = replace(cr.conbo_config(test, users[0].dim, seed), **cr.CONBO_VARIANTS[method])
    res = run_conbo_sequence([u.objective for u in users], cfg, test.n_evals, names=[u.name for u in users])
    pop = res.population
    axis = np.linspace(0, 1, 50)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    mean, var = pop.bnn.predict(grid)
    _, held = cr.make_test_users(replace(test, n_users=30), seed + 100)
    rows = []
    for spec in held:
        f = spec.objective(grid)
        best = grid[int(np.argmax(mean))]
        regret = spec.optimum - float(spec.objective(best[None, :])[0])
        z = np.abs(f - mean) / np.sqrt(var)
        # spread of predicted mean vs spread of the true function, and fit on the top 10% of the user's function
        top = f >= np.quantile(f, 0.9)
        rows.append(dict(log_regret_argmax=float(np.log10(max(regret, 1e-12))), spearman=float(spearmanr(mean, f)[0]),
                         spearman_top=float(spearmanr(mean[top], f[top])[0]), cover2=float(np.mean(z <= 2)),
                         mean_abs_err=float(np.mean(np.abs(f - mean))), mean_sd=float(np.mean(np.sqrt(var)))))
    json.dump(dict(method=method, seed=seed, rows=rows, mean_var=float(var.mean()), mean_range=[float(mean.min()), float(mean.max())]), open(out, "w"))
    print(method, seed, "done")


if __name__ == "__main__":
    main()
