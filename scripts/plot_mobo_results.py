"""Plot hypervolume trajectories and Pareto fronts from run_mobo_benchmark.py output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openbo.benchmarks.hypervolume import pareto_front

METHOD_STYLES: dict[str, tuple[str, str]] = {
    "mobo_ceipv_slf": ("tab:red", "CEIPV-SLF"),
    "mobo_ceipv_slf_slice": ("tab:purple", "CEIPV-SLF (slice)"),
    "mobo_ceipv_mt": ("tab:orange", "CEIPV-MT"),
    "mobo_ieipv": ("tab:green", "IEIPV"),
    "mobo_botorch": ("tab:blue", "BoTorch qLogNEHVI"),
    "random": ("tab:gray", "Random"),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot MOBO benchmark results.")
    parser.add_argument("--problem", required=True, help="Problem label used in filenames.")
    parser.add_argument(
        "--test-id",
        nargs="+",
        default=["default"],
        help="Experiment id(s) used in filenames; runs from several ids are merged.",
    )
    parser.add_argument(
        "--output-id", default=None, help="Id for the plot filenames (default: first test id)."
    )
    parser.add_argument("--results-dir", default="test_results/mobo", help="Results root folder.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHOD_STYLES),
        default=None,
        help="Methods to include (default: every method with a results file).",
    )
    return parser.parse_args()


def load_results(
    results_dir: Path, test_ids: list[str], problem: str, methods: list[str] | None
) -> dict[str, dict]:
    """Load results per method in METHOD_STYLES order, merging runs across test ids."""
    loaded: dict[str, dict] = {}
    for method in METHOD_STYLES:
        if methods is not None and method not in methods:
            continue
        for test_id in test_ids:
            path = results_dir / "trajectories" / f"{test_id}_{method}_{problem}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if method in loaded:
                loaded[method]["runs"].extend(payload["runs"])
            else:
                loaded[method] = payload
    if not loaded:
        raise FileNotFoundError(
            f"No results for test_id={test_ids!r} problem={problem!r} in {results_dir}."
        )
    return loaded


def plot_hypervolume(results: dict[str, dict], problem: str, output_path: Path) -> None:
    """Plot mean +/- std hypervolume (and log10 regret if the true value is known)."""
    true_hv = next(iter(results.values()))["true_hypervolume"]
    n_panels = 2 if true_hv is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5), squeeze=False)
    for method, payload in results.items():
        color, label = METHOD_STYLES[method]
        hv = np.array([run["hv_history"] for run in payload["runs"]])
        evals = payload["n_init"] + np.arange(hv.shape[1])
        n_seeds = hv.shape[0]
        panels = [(axes[0, 0], hv)]
        if true_hv is not None:
            panels.append((axes[0, 1], np.log10(np.maximum(true_hv - hv, 1e-12))))
        for ax, values in panels:
            mean, std = values.mean(axis=0), values.std(axis=0)
            ax.plot(evals, mean, color=color, label=f"{label} (n={n_seeds})")
            ax.fill_between(evals, mean - std, mean + std, color=color, alpha=0.15)
    axes[0, 0].set_ylabel("Pareto hypervolume")
    axes[0, 0].set_title(f"Hypervolume, {problem}")
    if true_hv is not None:
        axes[0, 1].set_ylabel("log10(true HV - HV)")
        axes[0, 1].set_title(f"Log hypervolume regret, {problem}")
    for ax in axes[0]:
        ax.set_xlabel("evaluations")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0, 0].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_pareto_fronts(results: dict[str, dict], problem: str, output_path: Path) -> bool:
    """Scatter each method's final Pareto front (first seed); 2-objective problems only."""
    first = next(iter(results.values()))
    if first["n_objectives"] != 2:
        return False
    fig, ax = plt.subplots(figsize=(6, 5))
    for method, payload in results.items():
        color, label = METHOD_STYLES[method]
        front = pareto_front(np.asarray(payload["runs"][0]["y_values"]))
        front = front[np.argsort(front[:, 0])]
        # Between neighbours the attained boundary is the next point's f2 (maximization).
        ax.step(front[:, 0], front[:, 1], where="pre", color=color, lw=1)
        ax.plot(front[:, 0], front[:, 1], "o", color=color, ms=4, label=label)
    ax.set_xlabel("objective 1 (maximized)")
    ax.set_ylabel("objective 2 (maximized)")
    ax.set_title(f"Final Pareto fronts, {problem} (seed {first['runs'][0]['seed']})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    """Load results and write the plots."""
    args = parse_args()
    results_dir = Path(args.results_dir)
    results = load_results(results_dir, args.test_id, args.problem, args.methods)
    plots_dir = results_dir / "plots"
    output_id = args.output_id or args.test_id[0]
    hv_path = plots_dir / f"{output_id}_{args.problem}_hypervolume.png"
    plot_hypervolume(results, args.problem, hv_path)
    print(f"saved_plot={hv_path}")
    front_path = plots_dir / f"{output_id}_{args.problem}_pareto.png"
    if plot_pareto_fronts(results, args.problem, front_path):
        print(f"saved_plot={front_path}")


if __name__ == "__main__":
    main()
