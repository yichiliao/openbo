"""Plot the MetaBO benchmark written by run_naf_benchmark.py, and a training curve.

Per method: the median log10 simple regret over the test tasks at each step,
with a 30%-70% percentile band (the paper's Fig. 2 statistics). With
`--training-dir`, also plots the mean final log10 regret per PPO iteration of
a `train_naf.py` run. Writes `<results-dir>/plots/<test-id>_<family>.png`
(and `<test-id>_training.png`) and prints a table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Categorical slots 1-5 of the reference palette in fixed order, with line
# styles so identity never rests on colour alone.
STYLES = {
    "naf": ("MetaBO (NAF)", "#2a78d6", "-"),
    "taf_me_50": ("TAF-ME-50", "#eb6834", "--"),
    "taf_r_50": ("TAF-R-50", "#1baf7a", "-."),
    "ei": ("EI", "#eda100", ":"),
    "random": ("Random", "#e87ba4", (0, (5, 1, 1, 1))),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive grid and axes, ink-coloured text."""
    ax.set_title(title, color=INK, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def main() -> None:
    """Make the plots and print the summary table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="test_results/naf", help="Root with trajectories/.")
    parser.add_argument("--test-id", default="default", help="Experiment id used in filenames.")
    parser.add_argument("--family", default="branin", help="Function family.")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 5, 10, 20, 30], help="Table columns.")
    parser.add_argument("--training-dir", default=None, help="A train_naf.py output folder to plot.")
    args = parser.parse_args()

    root = Path(args.results_dir)
    (root / "plots").mkdir(parents=True, exist_ok=True)
    table = [f"median log10 simple regret over test tasks ({args.family})"]
    table.append(f"{'method':<14} " + " ".join(f"{'@' + str(s):>7}" for s in args.steps))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    found = False
    for method, (label, color, dash) in STYLES.items():
        path = root / "trajectories" / f"{args.test_id}_{method}_{args.family}.json"
        if not path.exists():
            continue
        found = True
        run = json.loads(path.read_text(encoding="utf-8"))
        log_regret = np.log10(np.maximum(np.asarray([t["regret"] for t in run["tasks"]]), 1e-12))
        steps = np.arange(1, log_regret.shape[1] + 1)
        median = np.median(log_regret, axis=0)
        low, high = np.percentile(log_regret, [30, 70], axis=0)
        ax.plot(steps, median, color=color, linestyle=dash, linewidth=2, label=label)
        ax.fill_between(steps, low, high, color=color, alpha=0.12, linewidth=0)
        cells = " ".join(f"{median[s - 1]:7.2f}" for s in args.steps if s <= median.size)
        table.append(f"{label:<14} {cells}")
    if found:
        style_axis(ax, f"MetaBO benchmark: {args.family}", "Evaluation", "log10 simple regret (median, 30-70%)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "plots" / f"{args.test_id}_{args.family}.png", dpi=150)
        print("\n".join(table))
    plt.close(fig)

    if args.training_dir is not None:
        history = json.loads((Path(args.training_dir) / "training_history.json").read_text(encoding="utf-8"))
        curve = np.asarray(history["mean_final_log_regret"])
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(np.arange(curve.size), curve, color="#c3c2b7", linewidth=1, label="per iteration")
        window = min(20, curve.size)
        smooth = np.convolve(curve, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, curve.size), smooth, color="#2a78d6", linewidth=2, label=f"{window}-iteration mean")
        best = history["best_iteration"]
        ax.axvline(best, color=MUTED, linestyle=":", linewidth=1)
        ax.annotate(f"best return: iteration {best}", (best, ax.get_ylim()[1]), color=MUTED, fontsize=8, ha="right", va="top")
        style_axis(ax, "NAF training", "PPO iteration", "mean final log10 regret")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "plots" / f"{args.test_id}_training.png", dpi=150)
        plt.close(fig)
        print(f"training: {curve.size} iterations, best return at iteration {best}")


if __name__ == "__main__":
    main()
