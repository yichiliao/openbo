#!/usr/bin/env python3
"""Plot log10(regret) trajectories from README §4c manual workflow artifacts.

Uses the same regret definition as ``scripts/plot_family_results.py`` (maximization):

``log10(max(optimal_value - y, 1e-12))`` applied to the **best-so-far** curve per task.

**Train (scratch server)** JSON files: ``<scratch_artifacts_dir>/trajectories/*.json``
(field ``best_so_far``). These are **train** tasks; the test curve uses **test** tasks —
overlapping mean regret is expected and does not prove transfer (see README §4c Step 8).

**Test (TAF server)** JSON files from::

    run_manual_family_test_clients.py --save-results-dir <dir>

(field ``best_y_history`` in the terminal server payload).

**Optima** come from the saved family split (same file as ``create_family_split.py`` output),
via ``build_specs`` with prefixes ``train_task`` / ``test_task`` — matching the manual client
scripts.

Plot modes (``--plot-mode``):

- ``mean_std`` (default): mean ± 1 std across tasks, **same figure style** as
  ``scripts/plot_family_results._plot_mean_std`` (single 9×5 panel when train and/or test
  are plotted; train = green, test = purple).
- ``per_task``: one line per task (separate subpanels for train vs test when both exist).

Example::

    uv run python server_scripts/plot_manual_family_trajectories.py \\
      --split-path configs/family_splits/branin_split_15.json \\
      --train-trajectories-dir meta-bo-training/my_manual_taf_run/trajectories \\
      --test-results-dir test_results/manual_family/test_sessions \\
      --output test_results/manual_family/log_regret.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openbo.test_functions.families import build_specs, load_family_split

# Match scripts/plot_family_results.METHOD_COLORS for scratch vs TAF curves.
_SERIES_COLORS = {"train": "tab:green", "test": "tab:purple"}


def _to_log_regret(y: np.ndarray, optimum: float) -> np.ndarray:
    """Convert objective values to log10 regret for maximization (same as plot_family_results)."""
    diff = np.maximum(optimum - y, 1e-12)
    return np.log10(diff)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot log10(regret) trajectories for manual train/test server workflow "
            "(same regret as plot_family_results)."
        )
    )
    p.add_argument(
        "--split-path",
        type=Path,
        required=True,
        help="Family split JSON (same as create_family_split.py / manual Steps 1 & 4–7).",
    )
    p.add_argument(
        "--train-prefix",
        default="train_task",
        help="Must match run_manual_family_train_clients.py --task-prefix.",
    )
    p.add_argument(
        "--test-prefix",
        default="test_task",
        help="Must match run_manual_family_test_clients.py --task-prefix.",
    )
    p.add_argument(
        "--train-trajectories-dir",
        type=Path,
        default=None,
        help="Directory with train_task_*.json from scratch server auto-save.",
    )
    p.add_argument(
        "--test-results-dir",
        type=Path,
        default=None,
        help="Directory with test_task_*_server.json from --save-results-dir.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    p.add_argument(
        "--plot-mode",
        choices=("per_task", "mean_std"),
        default="mean_std",
        help=(
            "mean_std: mean ±1 std per train/test group, same style as plot_family_results "
            "(default); per_task: one curve per task."
        ),
    )
    p.add_argument(
        "--title-prefix",
        default="Manual family workflow",
        help="Figure suptitle prefix.",
    )
    return p.parse_args()


def _optima_map(args: argparse.Namespace) -> dict[str, float]:
    split = load_family_split(args.split_path)
    train_specs = build_specs(split.base_name, split.train_variants, args.train_prefix)
    test_specs = build_specs(split.base_name, split.test_variants, args.test_prefix)
    out: dict[str, float] = {}
    for spec in train_specs + test_specs:
        if spec.optimum is None:
            continue
        out[spec.name] = float(spec.optimum)
    return out


def _align_rows(rows: list[np.ndarray]) -> tuple[np.ndarray, bool]:
    """Stack to (n_tasks, T); trim to common min length if needed."""
    if not rows:
        raise ValueError("No trajectory rows to stack.")
    lengths = {int(r.shape[0]) for r in rows}
    if len(lengths) == 1:
        return np.stack(rows, axis=0), False
    min_len = min(int(r.shape[0]) for r in rows)
    print(
        f"warning: trimming trajectories to common length {min_len} for aggregation.",
        file=sys.stderr,
    )
    return np.stack([r[:min_len] for r in rows], axis=0), True


def _plot_per_task(
    ax: plt.Axes,
    series: list[tuple[str, np.ndarray]],
    *,
    ylabel: str,
    title: str,
) -> None:
    if not series:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    for name, ys in sorted(series, key=lambda t: t[0]):
        xs = np.arange(1, len(ys) + 1)
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, linestyle="--", alpha=0.3)


def _plot_mean_std_panel(
    ax: plt.Axes,
    label_to_mat: dict[str, np.ndarray],
    *,
    ylabel: str,
    title: str,
) -> None:
    """Mean + ±1 std bands — mirrors ``plot_family_results._plot_mean_std``."""
    for key in ("train", "test"):
        if key not in label_to_mat:
            continue
        y_mat = label_to_mat[key]
        y_mean = np.mean(y_mat, axis=0)
        y_std = np.std(y_mat, axis=0)
        iterations = np.arange(1, y_mean.shape[0] + 1)
        color = _SERIES_COLORS[key]
        ax.plot(
            iterations,
            y_mean,
            linewidth=2.0,
            color=color,
            label=f"{key} mean",
        )
        ax.fill_between(
            iterations,
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.2,
            color=color,
            label=f"{key} ±1 std",
        )
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)


def main() -> None:
    args = _parse_args()
    opt_by_name = _optima_map(args)

    regret_ylabel = "log10(optimal_value - best_so_far_y)"

    train_series: list[tuple[str, np.ndarray]] = []
    test_series: list[tuple[str, np.ndarray]] = []

    if args.train_trajectories_dir is not None:
        d = args.train_trajectories_dir
        if not d.is_dir():
            raise ValueError(f"Not a directory: {d}")
        for path in sorted(d.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            name = str(data.get("task_name", path.stem))
            if "best_so_far" not in data:
                continue
            opt = opt_by_name.get(name)
            if opt is None:
                print(
                    f"warning: skip train trajectory '{name}' (unknown optimum in split).",
                    file=sys.stderr,
                )
                continue
            ys = np.asarray(data["best_so_far"], dtype=np.float64)
            train_series.append((name, _to_log_regret(ys, opt)))

    if args.test_results_dir is not None:
        d = args.test_results_dir
        if not d.is_dir():
            raise ValueError(f"Not a directory: {d}")
        for path in sorted(d.glob("*_server.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            hist = data.get("best_y_history")
            if hist is None:
                continue
            name = path.stem.replace("_server", "")
            opt = opt_by_name.get(name)
            if opt is None:
                print(
                    f"warning: skip test session '{name}' (unknown optimum in split).",
                    file=sys.stderr,
                )
                continue
            ys = np.asarray(hist, dtype=np.float64)
            test_series.append((name, _to_log_regret(ys, opt)))

    if not train_series and not test_series:
        raise SystemExit(
            "Nothing to plot: provide --train-trajectories-dir and/or "
            "--test-results-dir with JSON files, and ensure task names match the split."
        )

    if train_series and test_series:
        len_tr = [len(v) for _, v in train_series]
        len_te = [len(v) for _, v in test_series]
        med_tr = float(np.median(len_tr))
        med_te = float(np.median(len_te))
        if med_tr != med_te:
            print(
                "warning: train vs test trajectory lengths differ "
                f"(train median length={int(med_tr)}, test={int(med_te)}). "
                "The x-axis is one step per server observe; TAF batches random init into "
                "one observe, so counts need not match scratch. "
                "Green and purple also use different task cohorts — see README §4c Step 8.",
                file=sys.stderr,
            )

    if args.plot_mode == "mean_std":
        label_to_mat: dict[str, np.ndarray] = {}
        if train_series:
            label_to_mat["train"], _ = _align_rows([v for _, v in train_series])
        if test_series:
            label_to_mat["test"], _ = _align_rows([v for _, v in test_series])
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(1, 1, 1)
        _plot_mean_std_panel(
            ax,
            label_to_mat,
            ylabel=regret_ylabel,
            title=(
                f"{args.title_prefix}: mean best-so-far log10 regret "
                "with standard deviation"
            ),
        )
        fig.tight_layout()
    else:
        nrows = int(bool(train_series)) + int(bool(test_series))
        fig, axes = plt.subplots(nrows, 1, figsize=(9, 3.8 * nrows), squeeze=False)
        ax_flat = axes.ravel()
        i = 0
        if train_series:
            _plot_per_task(
                ax_flat[i],
                train_series,
                ylabel=regret_ylabel,
                title="Train sources (scratch): log10 regret vs iteration",
            )
            i += 1
        if test_series:
            _plot_per_task(
                ax_flat[i],
                test_series,
                ylabel=regret_ylabel,
                title="Test targets (TAF): log10 regret vs iteration",
            )
        fig.suptitle(args.title_prefix, fontsize=12)
        fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"saved_plot={args.output.resolve()}")


if __name__ == "__main__":
    main()
