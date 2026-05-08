"""Run generic OpenBO websocket optimizer server."""

from __future__ import annotations

import argparse
import asyncio

from openbo.server_optimizers.bo_server import serve_bo_websocket


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run OpenBO websocket optimizer server."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument(
        "--config-path",
        default="configs/server_optimizers/bo_server.yaml",
        help="Path to server runtime YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    """Run websocket server forever."""
    args = parse_args()
    print(
        f"starting_bo_server ws://{args.host}:{args.port} "
        f"config_path={args.config_path}"
    )
    asyncio.run(
        serve_bo_websocket(
            host=args.host,
            port=args.port,
            config_path=args.config_path,
        )
    )


if __name__ == "__main__":
    main()
