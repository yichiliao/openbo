"""Drive the TAF-MOBO server with a simulated sphere user (CHI 2024 App. B)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect

from openbo.benchmarks.taf_plus_runner import pareto_hypervolume
from openbo.benchmarks.hypervolume import hypervolume
from openbo.test_functions.taf_plus_users import SphereUser, make_sphere_users

DEFAULT_RESULTS_DIR = Path("test_results") / "taf_mobo"


async def run_client(
    url: str, *, user: SphereUser, method: str | None, n_iter: int | None, seed: int
) -> dict[str, object]:
    """Answer every suggestion with the user's noisy objective vector."""
    rng = np.random.default_rng(seed)
    start: dict[str, object] = {"type": "start", "seed": seed}
    if method is not None:
        start["method"] = method
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
            print(f"{message['method']} {message['phase']} {message['iteration']}: y={np.round(y, 3).tolist()}", flush=True)
            await websocket.send(json.dumps({"type": "observe", "x": message["x"], "y": y.tolist()}))


def main() -> None:
    """Run one session for one simulated new user and save the server's final message."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8772")
    parser.add_argument("--method", choices=("taf_eipv", "taf_ehvi"), default=None, help="Default: the server's.")
    parser.add_argument("--range", type=float, default=0.1, help="Similarity range r of the simulated user.")
    parser.add_argument("--user-seed", type=int, default=100, help="Seed that draws the user's shift and scale.")
    parser.add_argument("--n-iter", type=int, default=None, help="Evaluations (default: the server's).")
    parser.add_argument("--seed", type=int, default=0, help="Optimizer and noise seed.")
    parser.add_argument("--output", type=Path, help="Default: test_results/taf_mobo/taf_mobo_server_smoke.json")
    args = parser.parse_args()
    user = make_sphere_users(1, args.range, args.range, seed=args.user_seed)[0]
    output = args.output or DEFAULT_RESULTS_DIR / "taf_mobo_server_smoke.json"
    result = asyncio.run(run_client(args.url, user=user, method=args.method, n_iter=args.n_iter, seed=args.seed))
    v_ref = tuple(result["v_ref"])
    front = pareto_hypervolume(user, v_ref)
    noise_free = user.noise_free(np.asarray(result["x_values"]))
    coverage = [100.0 * hypervolume(noise_free[: t + 1], v_ref) / front for t in range(noise_free.shape[0])]
    result["user"] = {"shift": list(user.shift), "scale": user.scale, "pareto_hypervolume": front}
    result["noise_free_coverage_percent"] = coverage
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"completed {result['total_observations']} evaluations with {result['method']}; "
        f"Pareto front covered (noise-free hypervolume): {coverage[0]:.0f}% after 1, {coverage[-1]:.0f}% at the end; "
        f"saved {output}"
    )


if __name__ == "__main__":
    main()
