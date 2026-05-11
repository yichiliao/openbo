"""Fake client for OpenBO websocket server using Branin evaluations."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import websockets

from openbo.test_functions.registry import get_function_spec


def _plot_x_locations(
    x_values: list[list[float]],
    objective: object,
    output_path: Path,
    method_label: str,
    function_name: str,
) -> None:
    """Plot 2D sampled points over a function-value heatmap (same style as run_benchmark)."""
    x_arr = np.asarray(x_values, dtype=np.float64)
    if x_arr.ndim != 2 or x_arr.shape[1] != 2:
        raise ValueError(f"x_values must be (n, 2), got shape {x_arr.shape}.")
    n_points = x_arr.shape[0]
    tones = np.linspace(0.8, 0.0, max(n_points, 1))
    colors = np.stack([tones, tones, tones], axis=1)

    grid_n = 160
    axis = np.linspace(0.0, 1.0, grid_n, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    x_grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    z = np.asarray(objective(x_grid), dtype=np.float64).reshape(grid_n, grid_n)

    plt.figure(figsize=(6, 6))
    heat = plt.imshow(
        z,
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        cmap="coolwarm",
        alpha=0.75,
        aspect="equal",
    )
    plt.scatter(
        x_arr[:, 0],
        x_arr[:, 1],
        c=colors,
        s=45,
        edgecolors="white",
        linewidths=0.35,
    )
    plt.plot(x_arr[:, 0], x_arr[:, 1], color="0.7", linewidth=0.8, alpha=0.6)
    plt.colorbar(heat, fraction=0.046, pad=0.04, label="objective value")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"2D search trajectory ({method_label}, {function_name})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a fake Branin client against the websocket server."
    )
    parser.add_argument("--uri", default="ws://127.0.0.1:8765", help="Server websocket URI.")
    parser.add_argument(
        "--n-init",
        type=int,
        default=2,
        help="Number of random initialization suggestions to request from server.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=8,
        help="Number of BO iterations to request from server after initialization.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed forwarded to server.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-step progress logs.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save final done payload as JSON.",
    )
    parser.add_argument(
        "--plot-x-locations",
        action="store_true",
        help=(
            "After the run, plot 2D x trajectory on a Branin heatmap "
            "(requires dim=2; same style as scripts/run_benchmark.py --plot-x-locations)."
        ),
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Output PNG for --plot-x-locations (default: next to --save-json or under test_results/plots/).",
    )
    return parser.parse_args()


async def run_fake_client(
    uri: str,
    n_init: int,
    n_iter: int,
    seed: int,
    verbose: bool = True,
) -> dict[str, object]:
    """Run suggest/observe loop with Branin objective and return final done payload."""
    spec = get_function_spec("branin")

    async with websockets.connect(uri) as websocket:
        if verbose:
            print(
                f"[client] connect uri={uri} n_init={n_init} n_iter={n_iter} seed={seed}"
            )
        await websocket.send(
            json.dumps(
                {
                    "type": "start",
                    "n_init": int(n_init),
                    "n_iter": int(n_iter),
                    "seed": int(seed),
                }
            )
        )
        msg = json.loads(await websocket.recv())
        if verbose:
            print(f"[client] recv type={msg.get('type')}")

        while msg.get("type") not in {"done", "stopped"}:
            if msg.get("type") != "suggest":
                raise RuntimeError(f"Unexpected server message: {msg}")

            x = np.asarray(msg["x"], dtype=np.float64)
            y = float(spec.objective(x.reshape(1, -1))[0])
            if verbose:
                phase = msg.get("phase", "unknown")
                iteration = msg.get("iteration", "?")
                print(
                    "[client] suggest "
                    f"phase={phase} iter={iteration} "
                    f"x={[round(float(v), 6) for v in x]} y={y:.6f}"
                )
            await websocket.send(
                json.dumps(
                    {
                        "type": "observe",
                        "x": [float(v) for v in x],
                        "y": y,
                    }
                )
            )
            msg = json.loads(await websocket.recv())
            if verbose:
                print(
                    f"[client] recv type={msg.get('type')} "
                    f"n_observations={msg.get('total_observations', msg.get('n_observations'))}"
                )

        if verbose:
            print(f"[client] terminal type={msg.get('type')}")

        return msg


def main() -> None:
    """Run async fake client and print summary."""
    args = parse_args()
    done = asyncio.run(
        run_fake_client(
            uri=args.uri,
            n_init=args.n_init,
            n_iter=args.n_iter,
            seed=args.seed,
            verbose=not args.quiet,
        )
    )
    print(
        "client_done "
        f"total_observations={done.get('total_observations')} "
        f"best_value={done.get('best_value')} "
        f"best_x={done.get('best_x')}"
    )

    if args.save_json is not None:
        output = Path(args.save_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(done, indent=2), encoding="utf-8")
        print(f"saved_client_result={output}")

    if args.plot_x_locations:
        spec = get_function_spec("branin")
        if spec.dim != 2:
            print(
                f"skip_x_location_plot function=branin dim={spec.dim} only_2d_supported=true"
            )
        else:
            raw_xv = done.get("x_values")
            if not isinstance(raw_xv, list) or len(raw_xv) == 0:
                print("skip_x_location_plot missing_or_empty x_values in server payload")
            else:
                method_label = str(done.get("optimizer", "websocket"))
                if args.plot_output is not None:
                    plot_path = Path(args.plot_output)
                elif args.save_json is not None:
                    p = Path(args.save_json)
                    plot_path = p.parent / f"{p.stem}_x_locations.png"
                else:
                    plot_path = (
                        Path("test_results")
                        / "plots"
                        / "fake_client_branin_x_locations.png"
                    )
                _plot_x_locations(
                    x_values=raw_xv,
                    objective=spec.objective,
                    output_path=plot_path,
                    method_label=method_label,
                    function_name="branin",
                )
                print(f"saved_x_location_plot={plot_path.resolve()}")


if __name__ == "__main__":
    main()
