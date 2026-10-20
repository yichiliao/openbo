"""End-to-end tests for the async WebSocket serve loop (protocol + framing).

The other server tests drive the session state machine directly; these exercise
the real ``serve_bo_websocket`` coroutine over a live socket: JSON framing, the
auto-suggest on connect, the ask/tell loop to ``done``, and error responses for
NaN observations, malformed first messages, and bad config paths.
"""

from __future__ import annotations

import asyncio
import json
import socket

import websockets

from openbo.server_optimizers.bo_server import serve_bo_websocket


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _write_scratch_config(path) -> str:
    path.write_text(
        "optimizer: bo_scratch\n"
        "input_dim: 2\n"
        "y_range: [-100.0, 100.0]\n"
        "n_init_default: 2\n"
        "n_iter_default: 2\n",
        encoding="utf-8",
    )
    return str(path)


async def _connect_with_retry(uri: str, attempts: int = 40):
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return await websockets.connect(uri)
        except OSError as exc:  # server not bound yet
            last = exc
            await asyncio.sleep(0.05)
    raise AssertionError(f"could not connect to {uri}: {last}")


def test_serve_bo_websocket_end_to_end(tmp_path) -> None:
    config_path = _write_scratch_config(tmp_path / "srv.yaml")
    port = _free_port()
    uri = f"ws://127.0.0.1:{port}"

    async def scenario() -> None:
        server_task = asyncio.create_task(
            serve_bo_websocket("127.0.0.1", port, config_path)
        )
        try:
            # --- full run to done ---
            ws = await _connect_with_retry(uri)
            async with ws:
                await ws.send(json.dumps({"type": "start", "n_init": 2, "n_iter": 2, "seed": 0}))
                msg = json.loads(await ws.recv())
                assert msg["type"] == "suggest"
                while msg["type"] != "done":
                    x = msg["x"]
                    y = -float(sum((xi - 0.2) ** 2 for xi in x))
                    await ws.send(json.dumps({"type": "observe", "x": x, "y": y}))
                    msg = json.loads(await ws.recv())
                assert msg["type"] == "done"
                assert msg["total_observations"] == 4
                assert len(msg["x_values"]) == 4

            # --- NaN observation is rejected over the wire ---
            async with await _connect_with_retry(uri) as ws2:
                await ws2.send(json.dumps({"type": "start", "n_init": 2, "n_iter": 1, "seed": 0}))
                msg = json.loads(await ws2.recv())
                assert msg["type"] == "suggest"
                await ws2.send(json.dumps({"type": "observe", "x": msg["x"], "y": float("nan")}))
                err = json.loads(await ws2.recv())
                assert err["type"] == "error"
                assert "finite" in err["message"].lower()

            # --- first message must be 'start' ---
            async with await _connect_with_retry(uri) as ws3:
                await ws3.send(json.dumps({"type": "suggest"}))
                err = json.loads(await ws3.recv())
                assert err["type"] == "error"
                assert "start" in err["message"].lower()
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())


def test_serve_bo_websocket_bad_config_returns_error(tmp_path) -> None:
    """A missing config path yields a clean protocol error, not an abnormal close."""
    missing = str(tmp_path / "does_not_exist.yaml")
    port = _free_port()
    uri = f"ws://127.0.0.1:{port}"

    async def scenario() -> None:
        server_task = asyncio.create_task(serve_bo_websocket("127.0.0.1", port, missing))
        try:
            # The handler fails to load the config on connect and pushes a single
            # error frame before closing, so just receive it (no 'start' needed).
            async with await _connect_with_retry(uri) as ws:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                assert msg["type"] == "error"
                assert "config" in msg["message"].lower()
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())
