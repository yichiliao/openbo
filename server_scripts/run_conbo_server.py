"""Run the continual BO (ConBO) WebSocket server: one session per user."""

from __future__ import annotations

import argparse
import asyncio

from openbo.server_optimizers.conbo_server import serve_conbo_websocket


def main() -> None:
    """Parse arguments and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument(
        "--config-path",
        default="configs/server_optimizers/conbo_server.yaml",
    )
    args = parser.parse_args()
    print(
        f"starting_conbo_server ws://{args.host}:{args.port} "
        f"config_path={args.config_path}",
        flush=True,
    )
    try:
        asyncio.run(serve_conbo_websocket(args.host, args.port, args.config_path))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
