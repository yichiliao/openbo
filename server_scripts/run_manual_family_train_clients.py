#!/usr/bin/env python3
"""WebSocket clients: one session per **train** variant from a saved family split.

Used in README §4c Step 4 with ``run_bo_server`` (scratch, auto-save GP artifacts).
See ``server_scripts/run_manual_family_test_clients.py`` for the TAF test phase.

Run from repo root, with the scratch server already running::

    uv run python server_scripts/run_manual_family_train_clients.py \\
      --split-path configs/family_splits/branin_split_15.json \\
      --uri ws://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import websockets

from openbo.test_functions.families import build_specs, load_family_split


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run one websocket session per train variant (family split), "
            "with task_name set so GP artifacts do not overwrite."
        )
    )
    p.add_argument(
        "--split-path",
        type=Path,
        default=Path("configs/family_splits/branin_split_15.json"),
        help="Path to JSON from scripts/create_family_split.py.",
    )
    p.add_argument(
        "--uri",
        default="ws://127.0.0.1:8765",
        help="run_bo_server websocket URI.",
    )
    p.add_argument("--n-init", type=int, default=2, help="start message n_init.")
    p.add_argument("--n-iter", type=int, default=8, help="start message n_iter.")
    p.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Seed for train session i is seed_base + i.",
    )
    p.add_argument(
        "--task-prefix",
        default="train_task",
        help="Prefix for task names (train_task_000, ...).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print finish lines per task.",
    )
    return p.parse_args()


async def _run_one_train(
    uri: str,
    spec,
    *,
    n_init: int,
    n_iter: int,
    seed: int,
    verbose: bool,
) -> None:
    async with websockets.connect(uri) as ws:
        if verbose:
            print(f"[train] connect {uri} task={spec.name} seed={seed}")
        await ws.send(
            json.dumps(
                {
                    "type": "start",
                    "task_name": spec.name,
                    "n_init": int(n_init),
                    "n_iter": int(n_iter),
                    "seed": int(seed),
                }
            )
        )
        msg = json.loads(await ws.recv())
        while msg.get("type") not in {"done", "stopped"}:
            if msg.get("type") == "error":
                raise RuntimeError(str(msg.get("message", msg)))
            if msg.get("type") != "suggest":
                raise RuntimeError(f"Unexpected server message: {msg}")
            x = np.asarray(msg["x"], dtype=np.float64)
            y = float(spec.objective(x.reshape(1, -1))[0])
            if verbose:
                print(f"[train] {spec.name} y={y:.6f}")
            await ws.send(
                json.dumps(
                    {
                        "type": "observe",
                        "x": [float(v) for v in x],
                        "y": y,
                    }
                )
            )
            msg = json.loads(await ws.recv())


async def _async_main(args: argparse.Namespace) -> None:
    split = load_family_split(args.split_path)
    train_specs = build_specs(
        split.base_name, split.train_variants, args.task_prefix
    )
    verbose = not args.quiet
    for i, spec in enumerate(train_specs):
        seed = args.seed_base + i
        await _run_one_train(
            args.uri,
            spec,
            n_init=args.n_init,
            n_iter=args.n_iter,
            seed=seed,
            verbose=verbose,
        )
        print("finished", spec.name)


def main() -> None:
    asyncio.run(_async_main(_parse_args()))


if __name__ == "__main__":
    main()
