"""Drive the MOBO server with a synthetic multi-objective function."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect

from openbo.test_functions.multi_objective import MO_REGISTRY, get_mo_function_spec

DEFAULT_RESULTS_DIR = Path("test_results") / "mobo"


async def run_client(
    url: str, *, function: str, n_init: int, n_iter: int, seed: int
) -> dict[str, object]:
    """Answer every suggestion with the function's objective vector."""
    spec = get_mo_function_spec(function)
    async with connect(url) as websocket:
        await websocket.send(
            json.dumps({"type": "start", "n_init": n_init, "n_iter": n_iter, "seed": seed})
        )
        while True:
            message = json.loads(await websocket.recv())
            if message["type"] == "error":
                raise RuntimeError(message["message"])
            if message["type"] == "done":
                return message
            if message["type"] != "suggest":
                raise RuntimeError(f"unexpected server message: {message['type']}")
            x = np.asarray(message["x"], dtype=float)
            if x.shape != (spec.dim,):
                raise RuntimeError(
                    f"{function} needs {spec.dim}D inputs but the server sent {x.shape}; "
                    "check the server input_dim"
                )
            y = spec.objective(x[None, :])[0]
            print(f"{message['phase']} {message['iteration']}: y={np.round(y, 4).tolist()}", flush=True)
            await websocket.send(json.dumps({"type": "observe", "x": message["x"], "y": y.tolist()}))


def main() -> None:
    """Run one session and save the server's final message."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8768")
    parser.add_argument("--function", choices=sorted(MO_REGISTRY), default="oka2")
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, help="Default: test_results/mobo/<function>_mobo_server_smoke.json"
    )
    args = parser.parse_args()
    output = args.output or DEFAULT_RESULTS_DIR / f"{args.function}_mobo_server_smoke.json"
    result = asyncio.run(
        run_client(args.url, function=args.function, n_init=args.n_init, n_iter=args.n_iter, seed=args.seed)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"completed {result['total_observations']} evaluations; "
        f"hypervolume={result['hv_history'][-1]:.4f}; pareto_size={len(result['pareto_y'])}; "
        f"saved {output}"
    )


if __name__ == "__main__":
    main()
