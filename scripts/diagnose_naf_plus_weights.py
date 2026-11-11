"""Does NAF+'s weight conditioning work? Run the same users under every objective weight.

Each of `--n-users` training-like double-sphere users (Test 1.1's users, but every user
under all 11 weights `w = (w1, 1 - w1)`, `w1 = 0, 0.1, ..., 1`) is run with:

- `naf_plus`: the weight-conditioned AF given the true weights, greedy, no EI fallback;
- `naf_plus_no_w`: the AF trained without weight inputs, greedy, no EI fallback;
- `ei`: plain BO, EI on the same fixed GP after one Sobol point (as in the benchmark);
- `naf_plus_cal` (only if asked with `--methods`): NAF+ with the calibrated novelty test
  and the policy mix, greedy (needs `novelty_calibration.json` in `--run-dir`).

`--novel` draws Test 1.2's novel users instead. On the wide family a novel user's
optimum is `|0.6 - 1.2 w1|` from the training user's with the same weights, so the
extreme weights are the far-novel users and `w1 = 0.5` is not novel at all.

A method's randomness is seeded per user, the same for every weight, so runs pair
across weights and methods. Writes to `<out-dir>`:

- `<test-id>_weights_prior.png`: the AF's first-step probabilities relative to uniform
  (softmax over the global grid, no data yet, so they depend only on `x` and `w`) for five weights,
  with the average user's optimum for that weight;
- `<test-id>_weights_regret.png` and `<test-id>_weights_distance.png`: median log10
  simple regret and median distance of each evaluated point to the user's optimum,
  per iteration, for all weights, the extreme weights and the middle ones;
- `<test-id>_weights_by_weight.png`: mean median log10 regret over iterations 1-5
  and 1-20 against `w1`;
- `<test-id>_weights.json`: every trajectory and the summary.

Plan: `docs/plans/naf_plus.md`, "Design stance".
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.stats import qmc, wilcoxon
from threadpoolctl import threadpool_limits

from openbo.acquisition.naf import AdaptiveSet, naf_score_fn
from openbo.benchmarks.naf_runner import ei_score_factory, grid_bo
from openbo.optimizers.naf import load_naf
from openbo.optimizers.naf_plus import NAFPlusConfig, NAFPlusSequentialOptimizer
from openbo.test_functions.naf_plus_families import DoubleSphereUser, WeightedTask

W1_GRID = tuple(round(0.1 * i, 1) for i in range(11))
PRIOR_W1 = (0.0, 0.25, 0.5, 0.75, 1.0)
EXTREME = (0.0, 0.1, 0.2, 0.8, 0.9, 1.0)
MIDDLE = (0.3, 0.4, 0.5, 0.6, 0.7)
METHODS = ("naf_plus", "naf_plus_no_w", "ei", "naf_plus_cal")
DEFAULT_METHODS = METHODS[:3]
# Categorical slots 1-3 of the reference palette in fixed order (validated), plus line
# styles so identity never rests on colour alone. The calibrated fallback is slot 4
# (yellow), as in plot_naf_plus_results.py; yellow fails the normal-vision floor
# against orange, so run it without naf_plus_no_w (blue, aqua, yellow pass).
STYLES = {
    "naf_plus": ("NAF+ (weights, no EI)", "#2a78d6", "-"),
    "naf_plus_no_w": ("NAF+ w/o weights", "#eb6834", "--"),
    "ei": ("EI (plain BO)", "#1baf7a", ":"),
    "naf_plus_cal": ("NAF+ calibrated fallback", "#eda100", "-."),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def weights_of(w1: float) -> tuple[float, float]:
    """The weight vector ``(w1, 1 - w1)``."""
    return (w1, round(1.0 - w1, 10))


def run_user(payload: tuple) -> list[dict[str, object]]:  # type: ignore[type-arg]
    """All weights of one user for one method."""
    method, user, index, run_dir, run_dir_no_w, seed = payload
    with threadpool_limits(1):
        records = []
        config = NAFPlusConfig.load(run_dir)
        for k, w1 in enumerate(W1_GRID):
            task = WeightedTask(user, weights_of(w1))
            if method == "ei":
                x_init = qmc.Sobol(d=2, scramble=True, seed=seed + 1009 * index).random(1)
                adaptive_set = AdaptiveSet.build(2, seed=seed)
                x, y = grid_bo(task, ei_score_factory, config.gp_hyperparameters, adaptive_set, 20, x_init)
            else:
                directory = run_dir_no_w if method == "naf_plus_no_w" else run_dir
                if method == "naf_plus_cal":
                    opt = NAFPlusSequentialOptimizer.from_run(
                        directory, task.weights, novelty_test="calibrated", mix="policy",
                        mix_rng=np.random.default_rng([seed, index, k, 1]),
                    )
                else:
                    opt = NAFPlusSequentialOptimizer.from_run(directory, task.weights, use_fallback=False)
                for _ in range(opt.budget):
                    point = opt.suggest()
                    opt.observe(point, task(point))
                x, y = opt.x_obs, opt.y_obs
            x, y = np.asarray(x), np.asarray(y)
            argmax = user.argmax(task.weights)
            records.append(
                {
                    "method": method,
                    "user": index,
                    "w1": w1,
                    "x": x.tolist(),
                    "regret": (task.optimum - np.maximum.accumulate(y)).tolist(),
                    "distance": np.linalg.norm(x - argmax, axis=1).tolist(),
                    **({"lambda_ei": [h.lambda_ei for h in opt.history]} if method == "naf_plus_cal" else {}),
                }
            )
        return records


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


def present(records: list[dict]) -> list[str]:  # type: ignore[type-arg]
    """The methods with records, in STYLES order."""
    found = {r["method"] for r in records}
    return [m for m in STYLES if m in found]


def prior_figure(run_dir: str, out: Path, novel: bool = False) -> dict[str, object]:
    """Heatmaps of the AF's first-step softmax over the global grid; returns its centres of mass."""
    af, config, hp = load_naf(run_dir, "best_af")
    centres = NAFPlusConfig.load(run_dir).centres(novel)
    af.eval()
    grid = config.adaptive_set().global_grid
    side = int(round(np.sqrt(grid.shape[0])))
    fig, axes = plt.subplots(1, len(PRIOR_W1), figsize=(3.0 * len(PRIOR_W1), 3.3), constrained_layout=True)
    summary: dict[str, object] = {}
    for ax, w1 in zip(axes, PRIOR_W1):
        w = weights_of(w1)
        logits = naf_score_fn(af, hp.new_gp(), 0, config.budget, w)(grid)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        centre = p @ grid
        target = np.asarray(w) @ np.asarray(centres)  # the average user's optimum
        first = grid[int(np.argmax(logits))]
        summary[str(w1)] = {
            "centre_of_mass": centre.tolist(),
            "greedy_first_point": first.tolist(),
            "average_user_optimum": target.tolist(),
            "max_probability": float(p.max()),
        }
        order = np.lexsort((grid[:, 0], grid[:, 1]))
        image = ax.imshow(
            p[order].reshape(side, side) * p.size, origin="lower", extent=(0, 1, 0, 1), cmap=BLUES
        )
        ax.plot(*target, marker="x", color=INK, markersize=9, markeredgewidth=2, linestyle="none", label="average user's optimum")
        ax.plot(*first, marker="o", markerfacecolor="none", markeredgecolor="#eb6834", markersize=9, markeredgewidth=2, linestyle="none", label="greedy first point")
        ax.set_title(f"w = ({w1:g}, {1 - w1:g}); max {p.max() * p.size:.2f}x uniform", color=INK, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=7)
        fig.colorbar(image, ax=ax, shrink=0.75).ax.tick_params(labelsize=7, colors=MUTED)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("NAF+ first-step probability relative to uniform (1 = uniform; no data yet)", color=INK, fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return summary


def curves(records: list[dict], key: str, w1_set: tuple[float, ...], log: bool) -> dict[str, np.ndarray]:  # type: ignore[type-arg]
    """Per method, the (users, 20) array of a quantity averaged over the weights in ``w1_set``."""
    out = {}
    for method in present(records):
        rows = [r for r in records if r["method"] == method and r["w1"] in w1_set]
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        if log:
            values = np.log10(np.maximum(values, 1e-12))
        users = np.asarray([r["user"] for r in rows])
        out[method] = np.stack([values[users == u].mean(axis=0) for u in np.unique(users)])
    return out


def iteration_figure(records: list[dict], key: str, log: bool, ylabel: str, out: Path) -> dict[str, object]:  # type: ignore[type-arg]
    """Medians over users per iteration in three panels; returns the medians and paired tests."""
    groups = {"all weights": W1_GRID, "extreme weights (w1 <= 0.2 or >= 0.8)": EXTREME, "middle weights (0.3-0.7)": MIDDLE}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), sharey=True, constrained_layout=True)
    summary: dict[str, object] = {}
    for ax, (name, w1_set) in zip(axes, groups.items()):
        data = curves(records, key, w1_set, log)
        steps = np.arange(1, 21)
        group: dict[str, object] = {}
        for method in data:
            label, color, dash = STYLES[method]
            median = np.median(data[method], axis=0)
            low, high = np.percentile(data[method], [30, 70], axis=0)
            ax.plot(steps, median, color=color, linestyle=dash, linewidth=2, label=label)
            ax.fill_between(steps, low, high, color=color, alpha=0.12, linewidth=0)
            group[method] = median.tolist()
        pairs = [(m, o) for m in ("naf_plus", "naf_plus_cal") for o in data if m in data and o != m]
        for first, other in pairs:
            if first == "naf_plus" and other == "naf_plus_cal":
                continue
            a, b = data[first], data[other]
            better = [s + 1 for s in range(20) if not np.allclose(a[:, s], b[:, s]) and wilcoxon(a[:, s], b[:, s], alternative="less").pvalue < 0.05]
            worse = [s + 1 for s in range(20) if not np.allclose(a[:, s], b[:, s]) and wilcoxon(a[:, s], b[:, s], alternative="greater").pvalue < 0.05]
            group[f"{first}_vs_{other}"] = {"lower_at": better, "higher_at": worse}
        summary[name] = group
        style_axis(ax, name, "Evaluation", ylabel if ax is axes[0] else "")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return summary


def by_weight_figure(records: list[dict], out: Path) -> dict[str, object]:  # type: ignore[type-arg]
    """Mean median log10 regret over iterations 1-5 and 1-20 against w1."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False, constrained_layout=True)
    summary: dict[str, object] = {}
    for ax, (name, stop) in zip(axes, (("iterations 1-5", 5), ("iterations 1-20", 20))):
        summary[name] = {}
        for method in present(records):
            label, color, dash = STYLES[method]
            values = []
            for w1 in W1_GRID:
                lr = np.log10(np.maximum([r["regret"] for r in records if r["method"] == method and r["w1"] == w1], 1e-12))
                values.append(float(np.median(lr, axis=0)[:stop].mean()))
            ax.plot(W1_GRID, values, color=color, linestyle=dash, linewidth=2, marker="o", markersize=5, label=label)
            summary[name][method] = values
        style_axis(ax, f"Mean median log10 regret, {name}", "w1 (weight of the first objective)", "")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return summary


def main() -> None:
    """Run every method on every user and weight, then write the figures and summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="meta-bo-training/naf_plus/double_sphere_v1")
    parser.add_argument("--run-dir-no-w", default="meta-bo-training/naf_plus/double_sphere_no_w_v1")
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=100, help="Seed that draws the users (Test 1.1's).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the methods' own randomness.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", default="test_results/naf_plus/weights")
    parser.add_argument("--test-id", default="v1")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS), help="Methods to run.")
    parser.add_argument("--novel", action="store_true", help="Test 1.2's novel users instead of training-like ones.")
    parser.add_argument(
        "--reuse-from", default=None, help="Test id whose records supply the methods not run (same users and weights)."
    )
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plus = NAFPlusConfig.load(args.run_dir)
    if NAFPlusConfig.load(args.run_dir_no_w).family != plus.family:
        raise SystemExit("--run-dir and --run-dir-no-w were trained on different families.")
    users: list[DoubleSphereUser] = plus.users(args.n_users, args.test_seed, novel=args.novel)
    payloads = [(m, u, i, args.run_dir, args.run_dir_no_w, args.seed) for m in args.methods for i, u in enumerate(users)]
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        records = [r for batch in pool.map(run_user, payloads) for r in batch]
    seconds = time.perf_counter() - start
    if args.reuse_from is not None:
        old = json.loads((out / f"{args.reuse_from}_weights.json").read_text(encoding="utf-8"))
        settings = old["summary"]["settings"]
        used = (settings["n_users"], settings["test_seed"], settings.get("novel", False))
        if used != (args.n_users, args.test_seed, args.novel):
            raise SystemExit(f"{args.reuse_from} used other users (n_users, test_seed, novel).")
        records += [r for r in old["records"] if r["method"] not in args.methods]

    summary = {
        "settings": {**vars(args), "w1_grid": W1_GRID, "seconds": seconds},
        "prior": prior_figure(args.run_dir, out / f"{args.test_id}_weights_prior.png", args.novel),
        "regret": iteration_figure(records, "regret", True, "log10 simple regret (median, 30-70%)", out / f"{args.test_id}_weights_regret.png"),
        "distance": iteration_figure(records, "distance", False, "distance of x_t to the user's optimum (median, 30-70%)", out / f"{args.test_id}_weights_distance.png"),
        "by_weight": by_weight_figure(records, out / f"{args.test_id}_weights_by_weight.png"),
    }
    (out / f"{args.test_id}_weights.json").write_text(json.dumps({"summary": summary, "records": records}), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=1))
    print(f"runs={len(records)} seconds={seconds:.0f}")


if __name__ == "__main__":
    main()
