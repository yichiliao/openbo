#!/usr/bin/env python3
"""WebSocket clients: one session per **test** variant from a saved family split.

Used in README §4c Step 7 with ``run_taf_server``. Train-phase helper:
``run_manual_family_train_clients.py``.

Run from repo root, with the TAF server already running::

    uv run python server_scripts/run_manual_family_test_clients.py \\
      --split-path configs/family_splits/branin_split_15.json \\
      --uri ws://127.0.0.1:8766
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
            "Run one websocket session per test variant (family split) against "
            "the TAF server."
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
        default="ws://127.0.0.1:8766",
        help="run_taf_server websocket URI.",
    )
    p.add_argument("--n-init", type=int, default=0, help="start message n_init.")
    p.add_argument("--n-iter", type=int, default=8, help="start message n_iter.")
    p.add_argument(
        "--seed-base",
        type=int,
        default=100,
        help="Seed for test session j is seed_base + j.",
    )
    p.add_argument(
        "--task-prefix",
        default="test_task",
        help="Prefix for task names (test_task_000, ...).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print finish lines per task.",
    )
    p.add_argument(
        "--save-results-dir",
        type=Path,
        default=None,
        help=(
            "If set, write each terminal server message (done/stopped) as JSON "
            "for trajectory plots (see README §4c Step 8)."
        ),
    )
    return p.parse_args()


async def _run_one_test(
    uri: str,
    spec,
    *,
    n_init: int,
    n_iter: int,
    seed: int,
    verbose: bool,
) -> dict[str, object]:
    async with websockets.connect(uri) as ws:
        if verbose:
            print(f"[test] connect {uri} task={spec.name} seed={seed}")
        await ws.send(
            json.dumps(
                {
                    "type": "start",
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
                print(f"[test] {spec.name} y={y:.6f}")
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

        return msg


async def _async_main(args: argparse.Namespace) -> None:
    split = load_family_split(args.split_path)
    test_specs = build_specs(split.base_name, split.test_variants, args.task_prefix)
    verbose = not args.quiet
    if args.save_results_dir is not None:
        args.save_results_dir.mkdir(parents=True, exist_ok=True)
    for j, spec in enumerate(test_specs):
        seed = args.seed_base + j
        final_msg = await _run_one_test(
            args.uri,
            spec,
            n_init=args.n_init,
            n_iter=args.n_iter,
            seed=seed,
            verbose=verbose,
        )
        print("finished", spec.name)
        if args.save_results_dir is not None:
            out_path = args.save_results_dir / f"{spec.name}_server.json"
            out_path.write_text(json.dumps(final_msg, indent=2), encoding="utf-8")
            if verbose:
                print(f"[test] saved {out_path}")


def main() -> None:
    asyncio.run(_async_main(_parse_args()))


if __name__ == "__main__":
    main()
