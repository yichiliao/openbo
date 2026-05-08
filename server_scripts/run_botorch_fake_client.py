"""Fake client for OpenBO BoTorch websocket server using Branin evaluations."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import websockets

from openbo.test_functions.registry import get_function_spec


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a fake Branin client against the BoTorch websocket server."
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
        "--save-json",
        default=None,
        help="Optional path to save final done payload as JSON.",
    )
    return parser.parse_args()


async def run_fake_client(
    uri: str,
    n_init: int,
    n_iter: int,
    seed: int,
) -> dict[str, object]:
    """Run suggest/observe loop with Branin objective and return final done payload."""
    spec = get_function_spec("branin")
    bounds_payload = [[float(lo), float(hi)] for lo, hi in spec.bounds]

    async with websockets.connect(uri) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "start",
                    "bounds": bounds_payload,
                    "n_init": int(n_init),
                    "n_iter": int(n_iter),
                    "seed": int(seed),
                }
            )
        )
        msg = json.loads(await websocket.recv())

        while msg.get("type") != "done":
            if msg.get("type") != "suggest":
                raise RuntimeError(f"Unexpected server message: {msg}")

            x = np.asarray(msg["x"], dtype=np.float64)
            y = float(spec.objective(x.reshape(1, -1))[0])
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


if __name__ == "__main__":
    main()
