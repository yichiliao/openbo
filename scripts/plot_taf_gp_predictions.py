"""Plot heatmaps from saved TAF GP states and trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metabo.models.gp_scratch import GPScratch


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Load saved TAF GP states + trajectories and render prediction heatmaps."
        )
    )
    parser.add_argument(
        "--run-dir",
        default="meta-bo-training/taf-gps",
        help=(
            "TAF run directory or root folder. If it contains multiple runs, "
            "the script traverses all */gp_states/*.json recursively."
        ),
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=120,
        help="Grid resolution per axis for 2D heatmaps.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional limit on number of task plots.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Number of grid points per posterior call (memory-safe batching).",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, n, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    x_grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    return xx, yy, x_grid


def _posterior_batched(
    gp: GPScratch,
    x_grid: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate posterior on large grids in memory-safe batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    means: list[np.ndarray] = []
    vars_: list[np.ndarray] = []
    for start in range(0, x_grid.shape[0], batch_size):
        stop = min(start + batch_size, x_grid.shape[0])
        mean_b, var_b = gp.posterior(x_grid[start:stop])
        means.append(np.asarray(mean_b, dtype=np.float64))
        vars_.append(np.asarray(var_b, dtype=np.float64))
    return np.concatenate(means, axis=0), np.concatenate(vars_, axis=0)


def main() -> None:
    """Render GP mean/std heatmaps from stored TAF model states."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    single_run_mode = (run_dir / "gp_states").exists()
    if single_run_mode:
        gp_files = sorted((run_dir / "gp_states").glob("*.json"))
    else:
        gp_files = sorted(run_dir.glob("**/gp_states/*.json"))
    if args.max_tasks is not None:
        gp_files = gp_files[: args.max_tasks]
    if not gp_files:
        raise ValueError(f"No GP state files found under {run_dir}")

    xx, yy, x_grid = _make_grid(args.grid_size)

    plotted = 0
    for gp_path in gp_files:
        run_root = gp_path.parents[1]
        trajectories_dir = run_root / "trajectories"
        predictions_dir = run_root / "gp_predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        task_name = gp_path.stem
        traj_path = trajectories_dir / f"{task_name}.json"
        if not traj_path.exists():
            print(
                f"skip_missing_trajectory task={task_name} run={run_root.name} path={traj_path}"
            )
            continue

        gp_payload = _load_json(gp_path)
        traj_payload = _load_json(traj_path)
        gp_state = gp_payload.get("gp_state")
        if not isinstance(gp_state, dict):
            print(f"skip_missing_gp_state task={task_name}")
            continue

        x_obs = np.asarray(traj_payload["x_values"], dtype=np.float64)
        y_obs = np.asarray(traj_payload["y_values"], dtype=np.float64)
        if x_obs.ndim != 2 or x_obs.shape[1] != 2:
            print(f"skip_non_2d task={task_name} dim={x_obs.shape[1] if x_obs.ndim==2 else 'unknown'}")
            continue

        gp = GPScratch(
            lengthscale=np.asarray(gp_state["lengthscale"], dtype=np.float64),
            variance=float(gp_state["variance"]),
            noise=float(gp_state["noise"]),
            kernel_type=str(gp_state["kernel_type"]),
            optimize_hyperparameters=False,
            standardize_targets=bool(gp_state.get("standardize_targets", True)),
            optimize_noise=bool(gp_state.get("optimize_noise", False)),
        )
        gp.fit(x_obs, y_obs)
        mean, var = _posterior_batched(gp, x_grid, batch_size=args.batch_size)
        mean_img = mean.reshape(args.grid_size, args.grid_size)
        std_img = np.sqrt(np.maximum(var, 1e-12)).reshape(args.grid_size, args.grid_size)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        im0 = axes[0].imshow(
            mean_img,
            extent=(0.0, 1.0, 0.0, 1.0),
            origin="lower",
            cmap="coolwarm",
            aspect="equal",
        )
        axes[0].scatter(x_obs[:, 0], x_obs[:, 1], c="k", s=10, alpha=0.6)
        axes[0].set_title(f"{task_name} - GP mean")
        axes[0].set_xlabel("x1")
        axes[0].set_ylabel("x2")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(
            std_img,
            extent=(0.0, 1.0, 0.0, 1.0),
            origin="lower",
            cmap="viridis",
            aspect="equal",
        )
        axes[1].scatter(x_obs[:, 0], x_obs[:, 1], c="white", s=10, alpha=0.7)
        axes[1].set_title(f"{task_name} - GP std")
        axes[1].set_xlabel("x1")
        axes[1].set_ylabel("x2")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        out_path = predictions_dir / f"{task_name}_gp_prediction_heatmap.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        plotted += 1
        print(f"saved_gp_prediction={out_path}")

    print(f"processed_root={run_dir} n_plots={plotted}")


if __name__ == "__main__":
    main()
