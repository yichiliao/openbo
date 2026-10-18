"""Drive the NAF server with a simulated task from one of MetaBO's function families."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect

from openbo.optimizers.naf import NAF_FAMILY_SPECS, get_naf_family

DEFAULT_RESULTS_DIR = Path("test_results") / "naf"


async def run_client(url: str, *, task: object, dim: int, n_iter: int | None) -> dict[str, object]:
    """Answer every suggestion with the task's value."""
    start: dict[str, object] = {"type": "start"}
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
            if x.shape != (dim,):
                raise RuntimeError(f"This family needs {dim}D inputs but the server sent {x.shape}; check the AF's family")
            y = float(task(x[None, :])[0])
            regret = task.optimum - y
            print(f"step {message['iteration']}: x={np.round(x, 4).tolist()} y={y:.4f} regret={regret:.2e}", flush=True)
            await websocket.send(json.dumps({"type": "observe", "x": message["x"], "y": y}))


def main() -> None:
    """Run one session for one simulated task and save the server's final message."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8771")
    parser.add_argument("--family", choices=tuple(NAF_FAMILY_SPECS), default="branin", help="Family of the simulated task; match the AF the server loaded.")
    parser.add_argument("--task-seed", type=int, default=100, help="Seed that draws the task's translation and scaling.")
    parser.add_argument("--n-iter", type=int, default=None, help="Evaluations (default: the server's).")
    parser.add_argument("--output", type=Path, help="Default: test_results/naf/naf_server_smoke.json")
    args = parser.parse_args()
    family = get_naf_family(args.family)
    task = family.sample_tasks(1, args.task_seed)[0]
    output = args.output or DEFAULT_RESULTS_DIR / "naf_server_smoke.json"
    result = asyncio.run(run_client(args.url, task=task, dim=family.dim, n_iter=args.n_iter))
    regret = task.optimum - np.asarray(result["best_so_far"])
    result["task"] = {
        "translation": list(task.translation),
        "scaling": task.scaling,
        "optimum": task.optimum,
    }
    result["simple_regret"] = regret.tolist()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"completed {result['total_observations']} evaluations; best y={result['best_so_far'][-1]:.4f}; "
        f"log10 regret={np.log10(max(regret[-1], 1e-20)):.2f}; saved {output}"
    )


if __name__ == "__main__":
    main()
