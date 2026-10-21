"""Run the TAF-MOBO WebSocket server (TAF-EIPV or TAF-EHVI)."""

from __future__ import annotations

import argparse
import asyncio

from openbo.server_optimizers.taf_mobo_server import serve_taf_mobo_websocket


def main() -> None:
    """Parse arguments and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument(
        "--config-path",
        default="configs/server_optimizers/taf_mobo_server.yaml",
    )
    args = parser.parse_args()
    print(
        f"starting_taf_mobo_server ws://{args.host}:{args.port} "
        f"config_path={args.config_path}",
        flush=True,
    )
    try:
        asyncio.run(serve_taf_mobo_websocket(args.host, args.port, args.config_path))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
