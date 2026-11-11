"""Draw the ConBO paper's simulation figures from runs of run_conbo_benchmark.py.

Liao et al., CHI 2025, Appendix A:
- Figures 4 and 6 (`--test branin_paper` / `mccormick_paper`): mean log10 regret per
  evaluation over all users, in the paper's three groups (a: BNN-based, b: GP-based,
  c: ConBO with and without the user GP) and (d) the computation time of one
  iteration per user. Error bars are one standard deviation over the (seed, user) pairs.
- Figure 5: the BNN's predicted variance after the last user, per BNN-based method.
- Figure 7 (`--test branin_revisit`): re-optimization of users 1-5 and 6-10 from the
  population after the whole sequence, ConBO against BNN without replay.

It also writes `<test_id>_<test>_summary.json` (table numbers) next to the plots.
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

LABELS = {
    "conbo": "ConBO",
    "conbo_no_gp": "ConBO-wo-GP",
    "conbo_no_filter": "ConBO-wo-Filter",
    "bnn_no_replay": "BNN-wo-Replay",
    "bnn_direct_replay": "BNN-Direct-Replay",
    "single_gp": "Single GP",
    "taf_paper": "TAF (decayed)",
    "taf_m": "TAF-M",
    "taf_r": "TAF-R",
    "bo": "Standard BO",
}
GROUPS = {
    "a) BNN-based": ("conbo_no_gp", "conbo_no_filter", "bnn_no_replay", "bnn_direct_replay"),
    "b) GP-based": ("conbo_no_gp", "taf_paper", "taf_m", "taf_r", "single_gp", "bo"),
    "c) With and without the user GP": ("conbo", "conbo_no_gp"),
}
COLORS = {m: f"C{i}" for i, m in enumerate(LABELS)}


def load(results_dir: Path, test_id: str, test: str) -> dict[str, list[dict]]:
    """Load the runs of a test, grouped by method."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((results_dir / "trajectories").glob(f"{test_id}_*_{test}_s*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[payload["method"]].append(payload)
    if not runs:
        raise SystemExit(f"No runs found for test_id={test_id} test={test} in {results_dir}.")
    return runs


def log_regret(user: dict, key: str = "y") -> np.ndarray:
    """Best-so-far log10 regret of one user, shape (n_evals,)."""
    best = np.maximum.accumulate(np.asarray(user[key], dtype=float))
    return np.log10(np.maximum(user["optimum"] - best, 1e-12))


def pooled(runs: list[dict], key: str = "users", users: slice = slice(None)) -> np.ndarray:
    """Stack the log regret of every (seed, user) pair, shape (pairs, n_evals)."""
    return np.stack([log_regret(u) for r in runs for u in r[key][users]])


def curve(ax: plt.Axes, values: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Plot mean +- one std of rows of `values`; return mean and std."""
    mean, std = values.mean(axis=0), values.std(axis=0)
    evals = np.arange(1, values.shape[1] + 1)
    ax.errorbar(evals, mean, yerr=std, marker="o", ms=3, lw=1.2, capsize=2, alpha=0.85,
                label=LABELS.get(method, method), color=COLORS.get(method))
    return mean, std


def regret_figure(runs: dict[str, list[dict]], title: str, out: Path) -> dict[str, object]:
    """Figures 4 / 6: three regret panels and the computation time."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.6))
    summary: dict[str, object] = {}
    for ax, (name, members) in zip(axes[:3], GROUPS.items()):
        for m in members:
            if m in runs:
                curve(ax, pooled(runs[m]), m)
        ax.set(title=name, xlabel="Evaluations for the user", ylabel="log10 regret (best so far)")
        ax.legend(fontsize=8)
    for m, rs in runs.items():
        values = pooled(rs)
        mean, std = values.mean(axis=0), values.std(axis=0)
        times = np.array([[np.mean(u["iteration_seconds"]) for u in r["users"]] for r in rs])
        axes[3].plot(np.arange(1, times.shape[1] + 1), times.mean(axis=0), marker="o", ms=3,
                     label=LABELS.get(m, m), color=COLORS.get(m))
        summary[m] = {
            "n_seeds": len(rs),
            "log_regret_mean": {str(k): float(mean[k - 1]) for k in range(1, len(mean) + 1)},
            "log_regret_std": {str(k): float(std[k - 1]) for k in range(1, len(std) + 1)},
            "seconds_per_iteration_by_user": [float(t) for t in times.mean(axis=0)],
        }
    axes[3].set(title="d) Computation time of one iteration", xlabel="User", ylabel="Seconds", yscale="log")
    axes[3].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return summary


def variance_figure(runs: dict[str, list[dict]], seed: int, title: str, out: Path) -> None:
    """Figure 5: predicted variance of each BNN-based method after the last user."""
    methods = [m for m in GROUPS["a) BNN-based"] + ("conbo",) if m in runs]
    panels = []
    for m in methods:
        chosen = [r for r in runs[m] if r["seed"] == seed] or runs[m][:1]
        if chosen[0].get("variance_map"):
            panels.append((m, chosen[0]["variance_map"]))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), squeeze=False)
    for ax, (m, vm) in zip(axes[0], panels):
        n = vm["points"]
        var = np.asarray(vm["var"]).reshape(n, n)
        image = ax.imshow(var.T, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
        ax.set(title=f"{LABELS[m]}\nvariance range [{var.min():.3g}, {var.max():.3g}]",
               xlabel="Parameter 1", ylabel="Parameter 2")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def revisit_figure(runs: dict[str, list[dict]], title: str, out: Path) -> dict[str, object]:
    """Figure 7: re-optimization of users 1-5 and 6-10 from the final population."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    summary: dict[str, object] = {}
    for ax, (name, users) in zip(axes, (("Re-optimizing users 1-5", slice(0, 5)),
                                        ("Re-optimizing users 6-10", slice(5, 10)))):
        for m in ("conbo", "bnn_no_replay"):
            if m in runs:
                mean, std = curve(ax, pooled(runs[m], "reoptimization", users), m)
                summary.setdefault(m, {})[name] = {"mean": mean.tolist(), "std": std.tolist()}  # type: ignore[union-attr]
        ax.set(title=name, xlabel="Evaluations", ylabel="log10 regret (best so far)")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return summary


def main() -> None:
    """Make the figures that fit the runs found."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="test_results/conbo", help="Root with trajectories/.")
    parser.add_argument("--test-id", default="default", help="Experiment id used in filenames.")
    parser.add_argument("--test", required=True, help="branin_paper, mccormick_paper or branin_revisit.")
    parser.add_argument("--seed", type=int, default=0, help="Seed whose variance maps are drawn (Figure 5).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = load(results_dir, args.test_id, args.test)
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stem = plot_dir / f"{args.test_id}_{args.test}"
    title = f"ConBO simulation {args.test} ({args.test_id})"
    summary: dict[str, object] = {}
    if args.test == "branin_revisit":
        summary = revisit_figure(runs, title + ": revisiting seen users", Path(f"{stem}_revisit.png"))
    else:
        summary = regret_figure(runs, title, Path(f"{stem}_regret.png"))
    variance_figure(runs, args.seed, title + ": predicted variance after the last user", Path(f"{stem}_variance.png"))
    Path(f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={stem}_*.png methods={sorted(runs)}")


if __name__ == "__main__":
    main()
