"""WebSocket server adapter for BoTorch ask/tell optimization."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from websockets.asyncio.server import serve

from openbo.optimizers.bo_botorch import BoTorchConfig, BoTorchSequentialOptimizer


def _as_vector(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("x must be a 1D vector.")
    return arr


@dataclass
class BoTorchServerSession:
    """Single optimization session state for ask/tell over WebSocket."""

    optimizer: BoTorchSequentialOptimizer
    n_iter: int
    n_init: int
    init_count: int = 0
    bo_count: int = 0
    pending_x: np.ndarray | None = None

    @classmethod
    def from_start_message(cls, payload: dict[str, Any]) -> "BoTorchServerSession":
        """Create a session from a `start` message."""
        bounds_raw = payload.get("bounds")
        if not isinstance(bounds_raw, list) or len(bounds_raw) == 0:
            raise ValueError("bounds must be a non-empty list of [low, high].")
        bounds: list[tuple[float, float]] = []
        for idx, item in enumerate(bounds_raw):
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], (int, float))
                or not isinstance(item[1], (int, float))
            ):
                raise ValueError(f"bounds[{idx}] must be [low, high] numeric pair.")
            low = float(item[0])
            high = float(item[1])
            if high <= low:
                raise ValueError(f"bounds[{idx}] must satisfy high > low.")
            bounds.append((low, high))

        n_init = int(payload.get("n_init", 5))
        n_iter = int(payload.get("n_iter", 25))
        if n_init < 0:
            raise ValueError("n_init must be non-negative.")
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative.")

        config = BoTorchConfig(
            bounds=bounds,
            n_init=n_init,
            num_restarts=int(payload.get("num_restarts", 5)),
            raw_samples=int(payload.get("raw_samples", 64)),
            seed=payload.get("seed", 0),
        )
        optimizer = BoTorchSequentialOptimizer(config)
        return cls(optimizer=optimizer, n_iter=n_iter, n_init=n_init)

    def _next_suggestion(self) -> dict[str, Any]:
        if self.init_count < self.n_init:
            x = self.optimizer.rng.uniform(
                self.optimizer.lower, self.optimizer.upper, size=(1, self.optimizer.d)
            ).astype(np.float64)
            self.init_count += 1
            phase = "init"
            iteration = self.init_count - 1
        else:
            x = self.optimizer.suggest()
            self.bo_count += 1
            phase = "bo"
            iteration = self.bo_count - 1
        self.pending_x = x.reshape(-1)
        return {
            "type": "suggest",
            "x": [float(v) for v in self.pending_x],
            "phase": phase,
            "iteration": int(iteration),
            "n_observations": int(self.optimizer.x_obs.shape[0]),
        }

    def _done_payload(self) -> dict[str, Any]:
        result = self.optimizer.result()
        best_idx = int(np.argmax(result.y_obs))
        return {
            "type": "done",
            "total_observations": int(result.x_obs.shape[0]),
            "best_value": float(result.y_obs[best_idx]),
            "best_x": [float(v) for v in result.x_obs[best_idx]],
            "x_values": [[float(v) for v in row] for row in result.x_obs],
            "y_values": [float(v) for v in result.y_obs],
            "best_y_history": [float(v) for v in result.best_y_history],
        }

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle one inbound message and return one outbound message."""
        msg_type = payload.get("type")
        if not isinstance(msg_type, str):
            raise ValueError("message must include string field 'type'.")

        if msg_type == "suggest":
            if self.pending_x is not None:
                raise ValueError("Cannot suggest again before observe.")
            if self.init_count >= self.n_init and self.bo_count >= self.n_iter:
                return self._done_payload()
            return self._next_suggestion()

        if msg_type == "observe":
            if self.pending_x is None:
                raise ValueError("No pending suggestion. Call suggest first.")
            y = float(payload["y"])
            x_client = payload.get("x")
            if x_client is not None:
                x_vec = _as_vector(x_client)
                if x_vec.shape != self.pending_x.shape or not np.allclose(
                    x_vec, self.pending_x, atol=1e-12
                ):
                    raise ValueError("observe x does not match pending suggestion.")
            self.optimizer.observe(
                self.pending_x.reshape(1, -1),
                np.array([y], dtype=np.float64),
            )
            self.pending_x = None
            if self.init_count >= self.n_init and self.bo_count >= self.n_iter:
                return self._done_payload()
            return self._next_suggestion()

        if msg_type == "status":
            return {
                "type": "status",
                "n_init": int(self.n_init),
                "n_iter": int(self.n_iter),
                "init_count": int(self.init_count),
                "bo_count": int(self.bo_count),
                "n_observations": int(self.optimizer.x_obs.shape[0]),
                "has_pending": bool(self.pending_x is not None),
            }

        raise ValueError(f"Unsupported message type: {msg_type}")


async def _ws_handler(websocket) -> None:
    """Handle one websocket connection with one optimization session."""
    session: BoTorchServerSession | None = None
    async for raw in websocket:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("message must be a JSON object.")

            msg_type = payload.get("type")
            if session is None:
                if msg_type != "start":
                    raise ValueError("First message must be type='start'.")
                session = BoTorchServerSession.from_start_message(payload)
                response = session.handle({"type": "suggest"})
            else:
                if msg_type == "start":
                    raise ValueError("Session already started.")
                response = session.handle(payload)
        except Exception as exc:  # noqa: BLE001
            response = {"type": "error", "message": str(exc)}
        await websocket.send(json.dumps(response))
        if response.get("type") == "done":
            return


async def serve_botorch_websocket(
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Run the BoTorch websocket optimizer server forever."""
    async with serve(_ws_handler, host, port):
        await asyncio.Future()
