"""Plot and summarize the TAF+ simulations written by run_taf_plus_benchmark.py.

Regret is `log10(max(optimum - value, 1e-12))` of the noise-free weighted sum
`f(x) @ alpha`, per new user. `--metric instant` (default) uses the design
evaluated at each iteration, which is what a user experiences; `--metric best`
uses the best design so far. `--metric hv` instead scores the designs
evaluated so far by their noise-free hypervolume above `v_ref`, as
`log10(max(HV* - HV, 1e-2))` with `HV*` the user's true Pareto hypervolume
(estimated to about 3e-3 of about 7.7, hence the floor; cached in `plots/pareto_hypervolume.json`);
it ignores `alpha`, so it also scores TAF-EHVI and qLogNEHVI on their own
terms. Curves show the mean over new users with a 95% confidence band.
Writes (with the metric as a suffix):

- `sim2_<metric>.png`: one panel per similarity range (Simulation 2).
- `sim1_<metric>.png`: one panel per objective weight vector (Simulation 1).
- `decay.png`: the decay search scores (mean per-iteration value) per range,
  with the chosen pair and the paper's.

and prints a table of mean log regret at selected evaluations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from openbo.benchmarks.taf_plus_runner import hypervolume_curve, pareto_hypervolume
from openbo.test_functions.taf_plus_users import SphereUser

# Categorical slots 1-4 of the reference palette, in fixed order, plus line
# styles so identity never rests on colour alone.
STYLES = {
    "taf_plus": ("TAF+", "#2a78d6", "-"),
    "taf": ("TAF (no decay)", "#eb6834", "--"),
    "bo": ("BO (weighted sum)", "#1baf7a", "-."),
    "random": ("Random", "#eda100", ":"),
    "taf_ehvi": ("TAF-EHVI", "#8a4fc8", "-"),
    "mobo": ("qLogNEHVI", "#52514e", "--"),
}
SIM2 = ("sim2_r005", "sim2_r01", "sim2_r02", "sim2_r03")
SIM2_LABELS = {"sim2_r005": "range 0.05", "sim2_r01": "range 0.1", "sim2_r02": "range 0.2", "sim2_r03": "range 0.3"}
PAPER_DECAY = {"sim2_r005": (7, 0.1), "sim2_r01": (4, 0.1), "sim2_r02": (4, 0.2), "sim2_r03": (3, 0.2)}
SIM1_ALPHAS = ("a100-000-000", "a000-100-000", "a000-000-100", "a033-033-034", "a050-030-020", "a030-050-020")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


METRIC_KEYS = {"instant": "instant_regret", "best": "regret", "hv": None}
METRIC_LABELS = {
    "instant": "log10 regret of each design",
    "best": "log10 regret of best so far",
    "hv": "log10 hypervolume regret",
}
_PARETO_HV: dict[str, float] = {}
HV_FLOOR = 1e-2


def log_regret(run: dict, metric: str) -> np.ndarray:
    """Return log10 regret per new user, shape (n_users, n_evals)."""
    if metric == "hv":
        v_ref = tuple(run["settings"]["v_ref"])
        rows = []
        for u in run["users"]:
            user = SphereUser(tuple(u["shift"]), float(u["scale"]))
            key = json.dumps([list(user.shift), user.scale, list(v_ref)])
            if key not in _PARETO_HV:
                _PARETO_HV[key] = pareto_hypervolume(user, v_ref)
            rows.append(_PARETO_HV[key] - hypervolume_curve(user, np.asarray(u["x"]), v_ref))
        return np.log10(np.maximum(np.asarray(rows), HV_FLOOR))
    key = METRIC_KEYS[metric]
    return np.log10(np.maximum(np.asarray([u[key] for u in run["users"]]), 1e-12))


def load_runs(root: Path, test_id: str, sim: str, alpha: str, seed: int, metric: str) -> dict[str, np.ndarray]:
    """Return log regret per method for one simulation and weight vector."""
    out = {}
    for method in STYLES:
        path = root / "trajectories" / f"{test_id}_{method}_{sim}_{alpha}_s{seed}.json"
        if path.exists():
            out[method] = log_regret(json.loads(path.read_text(encoding="utf-8")), metric)
    return out


def style_axis(ax: plt.Axes, title: str) -> None:
    """Recessive grid and axes, ink-coloured text."""
    ax.set_title(title, color=INK, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_panel(ax: plt.Axes, runs: dict[str, np.ndarray], title: str) -> None:
    """Mean log regret with a 95% band, one line per method."""
    for method, r in runs.items():
        label, color, dash = STYLES[method]
        evals = np.arange(1, r.shape[1] + 1)
        mean = r.mean(axis=0)
        half = 1.96 * r.std(axis=0, ddof=1) / np.sqrt(r.shape[0]) if r.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(evals, mean, color=color, linestyle=dash, linewidth=2, label=label)
        ax.fill_between(evals, mean - half, mean + half, color=color, alpha=0.12, linewidth=0)
    style_axis(ax, title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Evaluation", color=MUTED, fontsize=9)


def summary_rows(name: str, runs: dict[str, np.ndarray], evals: list[int]) -> list[str]:
    """Table rows: mean log regret at the given evaluations."""
    rows = []
    for method, r in runs.items():
        cells = " ".join(f"{r[:, e - 1].mean():7.2f}" for e in evals if e <= r.shape[1])
        rows.append(f"{name:<22} {STYLES[method][0]:<18} {cells}")
    return rows


def main() -> None:
    """Make the plots and print the summary table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="test_results/taf_plus", help="Root with trajectories/ and decay/.")
    parser.add_argument("--test-id", default="default", help="Experiment id used in filenames.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the runs to plot.")
    parser.add_argument("--evals", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20], help="Table columns.")
    parser.add_argument("--metric", choices=sorted(METRIC_KEYS), default="instant", help="Regret of each design or of the best so far.")
    args = parser.parse_args()

    root = Path(args.results_dir)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    hv_cache = plots / "pareto_hypervolume.json"
    if args.metric == "hv" and hv_cache.exists():
        _PARETO_HV.update(json.loads(hv_cache.read_text(encoding="utf-8")))
    table = [f"{METRIC_LABELS[args.metric]}, mean over new users"]
    table.append(f"{'setting':<22} {'method':<18} " + " ".join(f"{'@' + str(e):>7}" for e in args.evals))

    groups = [
        ("sim2", [(sim, "a030-050-020", SIM2_LABELS[sim]) for sim in SIM2], (1, 4)),
        ("sim1", [("sim1", a, "α = " + a[1:].replace("-", " / ")) for a in SIM1_ALPHAS], (2, 3)),
    ]
    for name, panels, (rows, cols) in groups:
        loaded = [(load_runs(root, args.test_id, sim, a, args.seed, args.metric), label) for sim, a, label in panels]
        if not any(runs for runs, _ in loaded):
            continue
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), sharey=True, squeeze=False)
        for ax, (runs, label) in zip(axes.flat, loaded):
            plot_panel(ax, runs, label)
            table += summary_rows(f"{name} {label}", runs, args.evals)
        for ax in axes[:, 0]:
            ax.set_ylabel(METRIC_LABELS[args.metric], color=MUTED, fontsize=9)
        handles, labels = next(
            (handles, labels)
            for handles, labels in (ax.get_legend_handles_labels() for ax in axes.flat)
            if labels
        )
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(plots / f"{name}_{args.metric}.png", dpi=150)
        plt.close(fig)

    decay_files = [root / "decay" / f"{args.test_id}_{sim}_a030-050-020_s{args.seed}.json" for sim in SIM2]
    found = [(sim, json.loads(p.read_text(encoding="utf-8"))) for sim, p in zip(SIM2, decay_files) if p.exists()]
    if found:
        fig, axes = plt.subplots(1, len(found), figsize=(3.6 * len(found), 3.6), squeeze=False)
        for ax, (sim, d) in zip(axes.flat, found):
            scores = np.asarray(d["scores"])
            image = ax.imshow(scores.T, origin="lower", aspect="auto", cmap="Blues")
            ax.set_xticks(range(len(d["d1_values"])), d["d1_values"])
            ax.set_yticks(range(len(d["d2_values"])), d["d2_values"])
            i, k = d["d1_values"].index(d["best"][0]), d["d2_values"].index(d["best"][1])
            ax.plot(i, k, marker="o", markersize=9, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=2)
            p1, p2 = PAPER_DECAY[sim]
            if p1 in d["d1_values"] and p2 in d["d2_values"]:
                ax.plot(d["d1_values"].index(p1), d["d2_values"].index(p2), marker="x", markersize=9, color=INK)
            style_axis(ax, f"{SIM2_LABELS[sim]} (no decay {d['no_decay_score']:.3f})")
            ax.grid(False)
            ax.set_xlabel("d1", color=MUTED, fontsize=9)
            ax.set_ylabel("d2", color=MUTED, fontsize=9)
            fig.colorbar(image, ax=ax, label="mean value per iteration")
        fig.suptitle("Decay search: o = chosen, x = paper's choice", color=INK, fontsize=10)
        fig.tight_layout()
        fig.savefig(plots / "decay.png", dpi=150)
        plt.close(fig)

    if args.metric == "hv":
        hv_cache.write_text(json.dumps(_PARETO_HV, indent=1), encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
