"""Plot the NAF+ benchmark written by run_naf_plus_benchmark.py.

Per test and method: the median log10 simple regret over the test users at
each step, with a 30%-70% percentile band (MetaBO's statistics; the paper's
App. B plots log10 regret). Per test, also the fallback's EI weight
`lambda_ei` per iteration of `naf_plus` over the users, as box plots (the
paper's Fig. 7). Writes `<results-dir>/plots/<test-id>_<test>.png` and
`<test-id>_<test>_lambda.png`, and prints a table.
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

# Categorical slots 1-6 of the reference palette in fixed order, with line
# styles so identity never rests on colour alone (slots 3-5 sit below 3:1
# contrast on white, so the legend, line styles and the table carry them).
# A sampled variant shares its greedy method's hue and adds circle markers; a
# no-repeat variant adds square markers. The calibrated-fallback methods take
# slot 4 (yellow) and TAF+ slot 7 (violet): the focused fallback figure
# (--methods naf_plus naf_plus_no_ei naf_plus_cal naf_plus_cal_sample ei)
# then passes the validator (blue, aqua, yellow, magenta; line styles carry the
# aqua-magenta pair under deuteranopia). The core figure (--methods naf_plus
# naf_plus_no_ei naf_plus_no_w ei --suffix _core) recolours naf_plus_no_w to
# yellow: orange fails the normal-vision floor against EI's magenta.
STYLES = {
    "naf_plus": ("NAF+", "#2a78d6", "-"),
    "naf_plus_sample": ("NAF+ (sampled)", "#2a78d6", "-"),
    "naf_plus_no_w": ("NAF+ w/o weights", "#eb6834", "--"),
    "naf_plus_no_w_sample": ("NAF+ w/o weights (sampled)", "#eb6834", "--"),
    "naf_plus_norepeat": ("NAF+ (no repeats)", "#2a78d6", "-"),
    "naf_plus_no_w_norepeat": ("NAF+ w/o weights (no repeats)", "#eb6834", "--"),
    "naf_plus_no_ei_norepeat": ("NAF+ w/o EI (no repeats)", "#1baf7a", "-."),
    "naf_plus_no_ei": ("NAF+ w/o EI", "#1baf7a", "-."),
    "naf_plus_no_ei_sample": ("NAF+ w/o EI (sampled)", "#1baf7a", "-."),
    "naf_plus_mix": ("NAF+ (policy mix)", "#eda100", (0, (6, 2))),
    "naf_plus_cal": ("NAF+ (calibrated)", "#eda100", "-"),
    "naf_plus_cal_sample": ("NAF+ (calibrated, sampled)", "#eda100", "-"),
    "taf_plus_15": ("TAF+ (15 users)", "#6250d6", (0, (5, 1, 1, 1))),
    "ei": ("EI", "#e87ba4", ":"),
    "random": ("Random", "#008300", (0, (1, 2))),
}
TESTS = {"double_sphere": "Test 1.1, training-like users", "double_sphere_novel": "Test 1.2, novel users"}
INK, MUTED, GRID, BOX = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str, integer_x: bool = True) -> None:
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
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_regret(
    root: Path,
    test_id: str,
    test: str,
    steps_shown: list[int],
    methods: list[str] | None = None,
    suffix: str = "",
    colours: dict[str, str] | None = None,
) -> list[str]:
    """The regret figure of one test; returns its table lines (empty if no runs exist).

    ``methods`` limits the figure to those methods; ``suffix`` is added to its file name;
    ``colours`` overrides a method's colour (for a subset that fails the palette check).
    """
    table = [f"median log10 simple regret over test users ({test})"]
    table.append(f"{'method':<30} " + " ".join(f"{'@' + str(s):>7}" for s in steps_shown))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    found = False
    for method, (label, color, dash) in STYLES.items():
        if methods is not None and method not in methods:
            continue
        path = root / "trajectories" / f"{test_id}_{method}_{test}.json"
        if not path.exists():
            continue
        found = True
        color = (colours or {}).get(method, color)
        run = json.loads(path.read_text(encoding="utf-8"))
        log_regret = np.log10(np.maximum(np.asarray([t["regret"] for t in run["tasks"]]), 1e-12))
        steps = np.arange(1, log_regret.shape[1] + 1)
        median = np.median(log_regret, axis=0)
        low, high = np.percentile(log_regret, [30, 70], axis=0)
        marker = {}
        if method.endswith("_sample"):
            marker = {"marker": "o", "markersize": 4, "markevery": 2}
        elif method.endswith("_norepeat"):
            marker = {"marker": "s", "markersize": 4, "markevery": 2}
        ax.plot(steps, median, color=color, linestyle=dash, linewidth=2, label=label, **marker)
        ax.fill_between(steps, low, high, color=color, alpha=0.12, linewidth=0)
        cells = " ".join(f"{median[s - 1]:7.2f}" for s in steps_shown if s <= median.size)
        table.append(f"{label:<30} {cells}")
    if found:
        style_axis(ax, f"NAF+ benchmark: {TESTS.get(test, test)}", "Evaluation", "log10 simple regret (median, 30-70%)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "plots" / f"{test_id}_{test}{suffix}.png", dpi=150)
    plt.close(fig)
    return table if found else []


def plot_lambda(root: Path, test_id: str, test: str) -> list[str]:
    """Box plots of naf_plus's lambda_ei per iteration over the users; returns summary lines."""
    path = root / "trajectories" / f"{test_id}_naf_plus_{test}.json"
    if not path.exists():
        return []
    run = json.loads(path.read_text(encoding="utf-8"))
    lam = np.asarray([[h["lambda_ei"] for h in t["history"]] for t in run["tasks"]])
    iterations = np.arange(1, lam.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.boxplot(
        [lam[:, i] for i in range(lam.shape[1])],
        positions=iterations,
        widths=0.6,
        patch_artist=True,
        boxprops={"facecolor": "#b7d3f6", "edgecolor": BOX, "linewidth": 1},
        medianprops={"color": INK, "linewidth": 1.5},
        whiskerprops={"color": BOX},
        capprops={"color": BOX},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": BOX, "markeredgecolor": "none", "alpha": 0.5},
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.4, lam.shape[1] + 0.6)
    ax.set_xticks(iterations, [str(i) for i in iterations])
    style_axis(ax, f"NAF+ EI weight per iteration: {TESTS.get(test, test)}", "Iteration", "lambda_EI over users", integer_x=False)
    fig.tight_layout()
    fig.savefig(root / "plots" / f"{test_id}_{test}_lambda.png", dpi=150)
    plt.close(fig)
    mean = lam.mean(axis=0)
    share = (lam > 0).mean(axis=0)
    return [
        f"naf_plus lambda_ei ({test}): mean per iteration " + " ".join(f"{v:.2f}" for v in mean),
        "  users with lambda_ei > 0 per iteration " + " ".join(f"{v:.2f}" for v in share),
    ]


def main() -> None:
    """Make the plots and print the summary tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="test_results/naf_plus", help="Root with trajectories/.")
    parser.add_argument("--test-id", default="default", help="Experiment id used in filenames.")
    parser.add_argument("--tests", nargs="+", default=list(TESTS), help="Tests to plot.")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20], help="Table columns.")
    parser.add_argument(
        "--methods", nargs="+", choices=list(STYLES), default=None, help="Only these methods (default: all found)."
    )
    parser.add_argument("--suffix", default="", help="Added to the regret figure's file name, e.g. _fallback.")
    parser.add_argument(
        "--colours", nargs="+", default=[], metavar="METHOD=HEX", help="Override method colours, e.g. naf_plus_no_w=#eda100."
    )
    args = parser.parse_args()
    colours = dict(item.split("=", 1) for item in args.colours)
    unknown = set(colours) - set(STYLES)
    if unknown:
        parser.error(f"unknown methods in --colours: {sorted(unknown)}")

    root = Path(args.results_dir)
    (root / "plots").mkdir(parents=True, exist_ok=True)
    for test in args.tests:
        lines = plot_regret(root, args.test_id, test, args.steps, args.methods, args.suffix, colours)
        if args.methods is None:
            lines += plot_lambda(root, args.test_id, test)
        if lines:
            print("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
