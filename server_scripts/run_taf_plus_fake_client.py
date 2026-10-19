"""Drive the TAF+ server with a simulated sphere user (CHI 2024 App. B)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect

from openbo.test_functions.taf_plus_users import SphereUser, make_sphere_users

DEFAULT_RESULTS_DIR = Path("test_results") / "taf_plus"


async def run_client(
    url: str,
    *,
    user: SphereUser,
    objective_weights: list[float] | None,
    n_iter: int | None,
    seed: int,
) -> dict[str, object]:
    """Answer every suggestion with the user's noisy objective vector."""
    rng = np.random.default_rng(seed)
    start: dict[str, object] = {"type": "start", "seed": seed}
    if objective_weights is not None:
        start["objective_weights"] = objective_weights
    if n_iter is not None:
        start["n_iter"] = n_iter
    async with connect(url) as websocket:
        await websocket.send(json.dumps(start))
        while True:
            message = json.loads(await websocket.recv())
            if message["type"] == "error":
                raise RuntimeError(message["message"])
            if message["type"] == "done":
                return message
            if message["type"] != "suggest":
                raise RuntimeError(f"unexpected server message: {message['type']}")
            x = np.asarray(message["x"], dtype=float)
            if x.shape != (len(user.shift),):
                raise RuntimeError(
                    f"the sphere users need {len(user.shift)}D inputs but the server sent {x.shape}; "
                    "check the server input_dim"
                )
            y = user.observe(x[None, :], rng)[0]
            decay = message["decay"]
            print(
                f"{message['phase']} {message['iteration']}: decay={'-' if decay is None else f'{decay:.2f}'} "
                f"y={np.round(y, 3).tolist()}",
                flush=True,
            )
            await websocket.send(json.dumps({"type": "observe", "x": message["x"], "y": y.tolist()}))


def main() -> None:
    """Run one session for one simulated new user and save the server's final message."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8770")
    parser.add_argument("--range", type=float, default=0.1, help="Similarity range r of the simulated user.")
    parser.add_argument("--user-seed", type=int, default=100, help="Seed that draws the user's shift and scale.")
    parser.add_argument(
        "--alpha", type=float, nargs="+", default=None, help="Objective weights (default: the server's)."
    )
    parser.add_argument("--n-iter", type=int, default=None, help="Evaluations (default: the server's).")
    parser.add_argument("--seed", type=int, default=0, help="Optimizer and noise seed.")
    parser.add_argument("--output", type=Path, help="Default: test_results/taf_plus/taf_plus_server_smoke.json")
    args = parser.parse_args()
    user = make_sphere_users(1, args.range, args.range, seed=args.user_seed)[0]
    output = args.output or DEFAULT_RESULTS_DIR / "taf_plus_server_smoke.json"
    result = asyncio.run(
        run_client(args.url, user=user, objective_weights=args.alpha, n_iter=args.n_iter, seed=args.seed)
    )
    alpha = np.asarray(result["objective_weights"])
    _, optimum = user.weighted_optimum(alpha)
    noise_free = user.noise_free(np.asarray(result["x_values"])) @ alpha
    regret = optimum - np.maximum.accumulate(noise_free)
    result["user"] = {"shift": list(user.shift), "scale": user.scale, "weighted_optimum": optimum}
    result["noise_free_regret"] = regret.tolist()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"completed {result['total_observations']} evaluations; "
        f"best weighted y={result['best_so_far'][-1]:.4f}; noise-free regret={regret[-1]:.4f}; saved {output}"
    )


if __name__ == "__main__":
    main()
