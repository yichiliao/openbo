#!/usr/bin/env python3
"""Combine several ``run_family_benchmark.py`` trajectory folders into one family plot.

Repeat ``--run DIR:LABEL`` for **each** method you want on the same figure (two, three, or
more—there is no hard cap beyond readability of the legend).

Each input directory must match the layout written by ``run_family_benchmark.py``:
``summary.json`` plus one ``*.json`` per task with ``y_values``, ``best_so_far``, and
``optimal_value``. Plots use the same style as ``plot_family_results.py`` (mean ± 1 std,
log10 regret).

Example (two methods)::

    uv run python scripts/plot_family_from_benchmark_runs.py \\
      --results-dir test_results \\
      --plot-id compare_botorch_random \\
      --run default_bo_botorch_branin_all:botorch \\
      --run default_random_branin_all:random

Example (three methods)::

    uv run python scripts/plot_family_from_benchmark_runs.py \\
      --results-dir test_results \\
      --plot-id compare_three \\
      --run run_a_bo_scratch_ms:scratch \\
      --run run_b_bo_botorch:botorch \\
      --run run_c_random:random

Paths without a leading path separator are resolved as
``<results-dir>/trajectories/<name>``. You can also pass absolute paths::

    /abs/path/to/run_a:method_a /abs/path/to/run_b:method_b
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

# Load sibling module without requiring ``scripts`` on PYTHONPATH.
_scripts_dir = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_plot_family_results",
    _scripts_dir / "plot_family_results.py",
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to load plot_family_results.py")
_pfr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pfr)

load_family_benchmark_run_dir = _pfr.load_family_benchmark_run_dir
_plot_mean_std = _pfr._plot_mean_std
METHOD_COLORS = _pfr.METHOD_COLORS


def _parse_run_spec(spec: str) -> tuple[str, str]:
    """Return (path_or_name, plot_label)."""
    if ":" not in spec:
        raise ValueError(
            f"Invalid --run entry {spec!r}: use 'path_or_folder:plot_label' "
            "(e.g. default_bo_botorch_branin_all:botorch)."
        )
    path_part, label = spec.rsplit(":", 1)
    path_part, label = path_part.strip(), label.strip()
    if not path_part or not label:
        raise ValueError(f"Invalid --run entry {spec!r}: empty path or label.")
    return path_part, label


def _resolve_run_dir(path_part: str, results_dir: Path) -> Path:
    p = Path(path_part)
    if p.is_dir():
        return p.resolve()
    candidate = (results_dir / "trajectories" / path_part).resolve()
    if candidate.is_dir():
        return candidate
    raise ValueError(
        f"Not a directory: {path_part!r} (also tried {candidate}). "
        "Pass a folder name under <results-dir>/trajectories/ or an absolute path."
    )


def _color_for_label(plot_label: str, summary_method: str) -> str | None:
    """Map user-facing legend name to a tab color (same palette as plot_family_results)."""
    for key in (plot_label, summary_method):
        if key in METHOD_COLORS:
            return METHOD_COLORS[key]
    aliases: dict[str, str] = {
        "botorch": "bo_botorch",
        "scratch": "bo_scratch_multistart",
        "scratch_ms": "bo_scratch_multistart",
        "scratch_grid": "bo_scratch_grid",
        "taf": "bo_taf",
        "taf_m": "bo_taf_m",
        "taf_r": "bo_taf_r",
    }
    canon = aliases.get(plot_label.lower())
    if canon and canon in METHOD_COLORS:
        return METHOD_COLORS[canon]
    return METHOD_COLORS.get(summary_method, None)


def _align_task_counts(
    label_to_raw: dict[str, np.ndarray],
    label_to_best: dict[str, np.ndarray],
) -> tuple[int, int]:
    n_tasks_set = {m.shape[0] for m in label_to_raw.values()}
    if len(n_tasks_set) != 1:
        raise ValueError(
            "All runs must have the same number of tasks; got "
            f"{ {k: v.shape[0] for k, v in label_to_raw.items()} }."
        )
    lengths = {m.shape[1] for m in label_to_raw.values()}
    min_len = min(lengths)
    if len(lengths) > 1:
        print(
            f"warning: trimming all trajectories to common length {min_len} for comparison.",
            file=sys.stderr,
        )
    return int(n_tasks_set.pop()), min_len


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot family mean ± std log-regret from multiple "
            "run_family_benchmark.py trajectory directories (same figure style as "
            "plot_family_results.py)."
        )
    )
    p.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="DIR_OR_NAME:LABEL",
        help=(
            "Trajectory folder and legend label. "
            "Example: default_bo_botorch_branin_all:botorch. "
            "Repeat --run once per method (any number of curves). "
            "Folder resolves under <--results-dir>/trajectories/ unless DIR_OR_NAME "
            "is an absolute path."
        ),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("test_results"),
        help="Root used to resolve relative run folder names (default: test_results).",
    )
    p.add_argument(
        "--plot-id",
        default="combined",
        help="String used in default output PNG filenames.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG for per-iteration log-regret plot (default: under results-dir/plots/).",
    )
    p.add_argument(
        "--output-best-so-far",
        type=Path,
        default=None,
        help="Output PNG for best-so-far log-regret plot (default: under results-dir/plots/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("Provide at least one --run DIR:LABEL.")

    label_to_raw: dict[str, np.ndarray] = {}
    label_to_best: dict[str, np.ndarray] = {}
    base_functions: list[str] = []
    line_colors: dict[str, str | None] = {}

    for raw_spec in args.run:
        path_part, plot_label = _parse_run_spec(raw_spec)
        if plot_label in label_to_raw:
            raise ValueError(f"Duplicate plot label: {plot_label!r}")
        run_dir = _resolve_run_dir(path_part, args.results_dir)
        smethod, base, raw_m, best_m = load_family_benchmark_run_dir(run_dir)
        base_functions.append(base)
        label_to_raw[plot_label] = raw_m
        label_to_best[plot_label] = best_m
        line_colors[plot_label] = _color_for_label(plot_label, smethod)

    bases = set(base_functions)
    if len(bases) != 1:
        raise ValueError(
            "All runs must share the same base_function in summary.json; got "
            f"{sorted(bases)}."
        )
    effective_base = base_functions[0]

    n_tasks, min_len = _align_task_counts(label_to_raw, label_to_best)
    label_to_raw = {k: v[:, :min_len] for k, v in label_to_raw.items()}
    label_to_best = {k: v[:, :min_len] for k, v in label_to_best.items()}

    plots_dir = args.results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    raw_out = (
        args.output
        if args.output is not None
        else plots_dir / f"{args.plot_id}_{effective_base}_family_mean_std_plot.png"
    )
    best_out = (
        args.output_best_so_far
        if args.output_best_so_far is not None
        else plots_dir
        / f"{args.plot_id}_{effective_base}_family_best_so_far_mean_std_plot.png"
    )

    _plot_mean_std(
        method_to_matrix=label_to_raw,
        title="Family benchmark: mean log10 regret with standard deviation",
        ylabel="log10(optimal_value - y)",
        output_path=raw_out,
        line_colors=line_colors,
    )
    print(f"saved_plot={raw_out.resolve()}")

    _plot_mean_std(
        method_to_matrix=label_to_best,
        title="Family benchmark: mean best-so-far log10 regret with standard deviation",
        ylabel="log10(optimal_value - best_so_far_y)",
        output_path=best_out,
        line_colors=line_colors,
    )
    print(f"saved_best_so_far_plot={best_out.resolve()}")

    print(
        f"summary n_runs={len(label_to_raw)} n_tasks={n_tasks} "
        f"steps={min_len} base_function={effective_base}"
    )


if __name__ == "__main__":
    main()
