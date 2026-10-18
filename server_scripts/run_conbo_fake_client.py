"""Drive the ConBO server with a sequence of simulated users.

Each user is a shifted, rescaled Branin variant (``generate_user_variants``) and
gets its own WebSocket session, one after another, so the server stores each
finished user and later users start from the population. The client prints
each user's final regret; it keeps the objective and optima to itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect

from openbo.test_functions.families import generate_user_variants, make_user_sequence

DEFAULT_RESULTS_DIR = Path("test_results") / "conbo"


async def run_user(url: str, spec: object, *, name: str, n_evaluations: int) -> dict[str, object]:
    """Run one user's session and return the server's final message."""
    async with connect(url) as websocket:
        await websocket.send(
            json.dumps({"type": "start", "user_name": name, "n_evaluations": n_evaluations})
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
            y = float(spec.objective(x[None, :])[0])  # type: ignore[attr-defined]
            await websocket.send(json.dumps({"type": "observe", "x": message["x"], "y": y}))


async def run_sequence(
    url: str, *, n_users: int, n_evaluations: int, seed: int, prefix: str
) -> list[dict[str, object]]:
    """Run the users in order; return one summary per user."""
    variants = generate_user_variants("branin", n_users, 0.3, 0.2, seed=seed)
    specs = make_user_sequence("branin", variants)
    summaries = []
    for i, spec in enumerate(specs):
        name = f"{prefix}_{i:03d}"
        done = await run_user(url, spec, name=name, n_evaluations=n_evaluations)
        best = np.asarray(done["best_y_history"], dtype=float)
        regret = [math.log10(max(spec.optimum - b, 1e-12)) for b in best]
        summary = {
            "user_name": done["user_name"],
            "user_index": done["user_index"],
            "n_random": done["n_random"],
            "log10_regret_after_5": regret[min(4, len(regret) - 1)],
            "log10_regret_final": regret[-1],
            "n_users_in_memory": done["n_users_in_memory"],
            "replay_seconds": done["replay"]["seconds"],
            "server_result": done,
        }
        print(
            f"{name}: user {done['user_index']}, {done['n_random']} random starts, "
            f"log10 regret after 5 = {summary['log10_regret_after_5']:.2f}, "
            f"final = {summary['log10_regret_final']:.2f}; "
            f"{done['n_users_in_memory']} users in memory",
            flush=True,
        )
        summaries.append(summary)
    return summaries


def main() -> None:
    """Run a user sequence and save the summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8769")
    parser.add_argument("--n-users", type=int, default=4)
    parser.add_argument("--n-evaluations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="seed for the simulated users")
    parser.add_argument(
        "--prefix", default="sim", help="user name prefix; names must be new to the memory"
    )
    parser.add_argument(
        "--output", type=Path, help="Default: test_results/conbo/<prefix>_server_smoke.json"
    )
    args = parser.parse_args()
    output = args.output or DEFAULT_RESULTS_DIR / f"{args.prefix}_server_smoke.json"
    summaries = asyncio.run(
        run_sequence(
            args.url,
            n_users=args.n_users,
            n_evaluations=args.n_evaluations,
            seed=args.seed,
            prefix=args.prefix,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
