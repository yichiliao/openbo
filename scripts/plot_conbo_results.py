"""Plot and summarize ConBO benchmark runs written by run_conbo_benchmark.py.

Regret is `log10(max(optimum - best_so_far, 1e-12))` per user. Curves and the
summary table average over users from `--from-user` on (later users are where
transfer can help), then show mean +- std over seeds.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLES = {
    "conbo": ("ConBO", "tab:orange"),
    "bo": ("Standard BO", "tab:blue"),
    "taf_m": ("TAF-M", "tab:green"),
    "taf_r": ("TAF-R", "tab:olive"),
    "single_gp": ("Single GP", "tab:purple"),
}


def load(results_dir: Path, test_id: str, test: str) -> dict[str, list[dict]]:
    """Load runs grouped by method."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((results_dir / "trajectories").glob(f"{test_id}_*_{test}_s*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[payload["method"]].append(payload)
    if not runs:
        raise SystemExit(f"No runs found for test_id={test_id} test={test} in {results_dir}.")
    return runs


def log_regret(run: dict) -> np.ndarray:
    """Return best-so-far log10 regret, shape (n_users, n_evals)."""
    rows = []
    for user in run["users"]:
        best = np.maximum.accumulate(np.asarray(user["y"]))
        rows.append(np.log10(np.maximum(user["optimum"] - best, 1e-12)))
    return np.asarray(rows)


def main() -> None:
    """Make the plots and print the summary table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="test_results/conbo", help="Root with trajectories/.")
    parser.add_argument("--test-id", default="default", help="Experiment id used in filenames.")
    parser.add_argument("--test", default="branin", help="User setting.")
    parser.add_argument("--from-user", type=int, default=4, help="First user (1-based) in the averages.")
    parser.add_argument("--evals", type=int, nargs="+", default=[1, 3, 5, 10, 20, 30], help="Table columns.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = load(results_dir, args.test_id, args.test)
    methods = [m for m in STYLES if m in runs] + [m for m in runs if m not in STYLES]
    regrets = {m: np.stack([log_regret(r) for r in runs[m]]) for m in methods}  # (seeds, users, evals)
    first = args.from_user - 1

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    summary: dict[str, dict] = {}
    for m in methods:
        label, color = STYLES.get(m, (m, None))
        r = regrets[m]
        per_seed = r[:, first:, :].mean(axis=1)  # (seeds, evals)
        evals = np.arange(1, r.shape[2] + 1)
        mean, std = per_seed.mean(axis=0), per_seed.std(axis=0)
        axes[0].plot(evals, mean, label=f"{label} (n={r.shape[0]})", color=color)
        axes[0].fill_between(evals, mean - std, mean + std, alpha=0.15, color=color)
        users = np.arange(1, r.shape[1] + 1)
        axes[1].plot(users, r[:, :, 4].mean(axis=0), marker="o", label=label, color=color)
        axes[2].plot(users, r[:, :, -1].mean(axis=0), marker="o", label=label, color=color)
        seconds = np.array([[u["seconds"] for u in run["users"]] for run in runs[m]])
        summary[m] = {
            "n_seeds": int(r.shape[0]),
            "log_regret_mean": {str(k): float(mean[k - 1]) for k in args.evals if k <= len(mean)},
            "log_regret_std": {str(k): float(std[k - 1]) for k in args.evals if k <= len(std)},
            "seconds_per_run_mean": float(seconds.sum(axis=1).mean()),
        }
    axes[0].set(xlabel="Evaluations for the user", ylabel="log10 regret (best so far)",
                title=f"Users {args.from_user}-{regrets[methods[0]].shape[1]}: mean ± std over seeds")
    axes[1].set(xlabel="User (order of arrival)", ylabel="log10 regret", title="After 5 evaluations (mean over seeds)")
    axes[2].set(xlabel="User (order of arrival)", ylabel="log10 regret", title="After all evaluations (mean over seeds)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(f"ConBO benchmark: {args.test} ({args.test_id})")
    fig.tight_layout()
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"{args.test_id}_{args.test}_regret.png"
    fig.savefig(out, dpi=150)
    (plot_dir / f"{args.test_id}_{args.test}_summary.json").write_text(json.dumps(summary, indent=2))

    header = "method".ljust(12) + "".join(f"k={k}".rjust(16) for k in args.evals) + "  sec/run"
    print(header)
    for m in methods:
        s = summary[m]
        cells = "".join(
            f"{s['log_regret_mean'][str(k)]:7.2f} ± {s['log_regret_std'][str(k)]:4.2f}".rjust(16)
            for k in args.evals
            if str(k) in s["log_regret_mean"]
        )
        print(m.ljust(12) + cells + f"  {s['seconds_per_run_mean']:7.0f}")
    print(f"saved_plot={out}")


if __name__ == "__main__":
    main()
