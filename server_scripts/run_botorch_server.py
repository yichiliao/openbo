"""Run BoTorch optimizer websocket server."""

from __future__ import annotations

import argparse
import asyncio

from openbo.server_optimizers.bo_botorch_server import serve_botorch_websocket


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run OpenBO BoTorch websocket optimizer server."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    return parser.parse_args()


def main() -> None:
    """Run websocket server forever."""
    args = parse_args()
    print(f"starting_botorch_server ws://{args.host}:{args.port}")
    asyncio.run(serve_botorch_websocket(host=args.host, port=args.port))


if __name__ == "__main__":
    main()
