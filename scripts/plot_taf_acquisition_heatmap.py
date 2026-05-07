"""Visualize stored TAF acquisition-query values as heatmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot TAF acquisition query heatmaps from run_benchmark JSON."
    )
    parser.add_argument(
        "--trajectory-json",
        required=True,
        help="Path to single-run trajectory JSON saved by run_benchmark.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for saved heatmaps (default: next to trajectory).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Optional cap on number of iterations to plot.",
    )
    return parser.parse_args()


def _plot_iteration(
    x: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str,
) -> None:
    """Scatter heatmap for one iteration."""
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        x[:, 0],
        x[:, 1],
        c=values,
        cmap=cmap,
        s=18,
        edgecolors="none",
        alpha=0.9,
    )
    plt.colorbar(sc, label="acquisition value")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    """Load acquisition trace and render per-iteration visualizations."""
    args = parse_args()
    trajectory_path = Path(args.trajectory_json)
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))

    metadata = payload.get("metadata", {})
    trace = metadata.get("taf_acquisition_trace", [])
    if not isinstance(trace, list) or len(trace) == 0:
        raise ValueError(
            "No TAF acquisition trace found. Re-run run_benchmark.py with method=bo_taf."
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else trajectory_path.parent / f"{trajectory_path.stem}_taf_acq_heatmaps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    zero_fracs: list[float] = []
    n_iters = len(trace) if args.max_iters is None else min(len(trace), args.max_iters)
    for item in trace[:n_iters]:
        iter_idx = int(item["iteration"])
        x = np.asarray(item["x"], dtype=np.float64)
        values = np.asarray(item["values"], dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(
                f"Only 2D heatmaps supported, got queried x shape {x.shape} at iter {iter_idx}."
            )

        zero_frac = float(np.mean(values <= 1e-12))
        zero_fracs.append(zero_frac)
        _plot_iteration(
            x,
            values,
            output_path=output_dir / f"iter_{iter_idx:03d}_values.png",
            title=f"TAF acquisition values (iter {iter_idx}, zero_frac={zero_frac:.3f})",
            cmap="viridis",
        )
        _plot_iteration(
            x,
            (values <= 1e-12).astype(np.float64),
            output_path=output_dir / f"iter_{iter_idx:03d}_is_zero.png",
            title=f"TAF acquisition zero-mask (iter {iter_idx})",
            cmap="magma",
        )

    summary = {
        "trajectory_json": str(trajectory_path),
        "n_iters_plotted": int(n_iters),
        "mean_zero_fraction": float(np.mean(zero_fracs)) if zero_fracs else None,
        "max_zero_fraction": float(np.max(zero_fracs)) if zero_fracs else None,
        "min_zero_fraction": float(np.min(zero_fracs)) if zero_fracs else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved_acquisition_heatmaps={output_dir}")
    if zero_fracs:
        print(
            "zero_fraction_stats "
            f"mean={float(np.mean(zero_fracs)):.4f} "
            f"min={float(np.min(zero_fracs)):.4f} "
            f"max={float(np.max(zero_fracs)):.4f}"
        )


if __name__ == "__main__":
    main()
