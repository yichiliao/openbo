"""Generic WebSocket server adapter for ask/tell optimization backends."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from websockets.asyncio.server import serve

from openbo.optimizers.bo_botorch import BoTorchConfig, BoTorchSequentialOptimizer
from openbo.optimizers.bo_scratch import ScratchConfig, ScratchSequentialOptimizer


def _as_vector(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("x must be a 1D vector.")
    return arr


@dataclass
class BOServerRuntimeConfig:
    """Static server-side config loaded from file."""

    optimizer: str
    input_dim: int
    y_min: float
    y_max: float
    n_init_default: int = 5
    n_iter_default: int = 25
    num_restarts_default: int = 5
    raw_samples_default: int = 64

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "BOServerRuntimeConfig":
        """Load server runtime config from YAML file."""
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Server config must be a YAML mapping/object.")

        input_dim = int(payload["input_dim"])
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        y_range = payload.get("y_range")
        if (
            not isinstance(y_range, list)
            or len(y_range) != 2
            or not isinstance(y_range[0], (int, float))
            or not isinstance(y_range[1], (int, float))
        ):
            raise ValueError("y_range must be [y_min, y_max].")
        y_min = float(y_range[0])
        y_max = float(y_range[1])
        if y_max <= y_min:
            raise ValueError("y_range must satisfy y_max > y_min.")

        optimizer = str(payload.get("optimizer", "bo_botorch"))
        if optimizer not in {"bo_botorch", "bo_scratch"}:
            raise ValueError("optimizer must be 'bo_botorch' or 'bo_scratch'.")

        return cls(
            optimizer=optimizer,
            input_dim=input_dim,
            y_min=y_min,
            y_max=y_max,
            n_init_default=int(payload.get("n_init_default", 5)),
            n_iter_default=int(payload.get("n_iter_default", 25)),
            num_restarts_default=int(payload.get("num_restarts_default", 5)),
            raw_samples_default=int(payload.get("raw_samples_default", 64)),
        )


@dataclass
class BOServerSession:
    """Single optimization session state for ask/tell over WebSocket."""

    optimizer_name: str
    optimizer: Any
    n_iter: int
    n_init: int
    init_count: int = 0
    bo_count: int = 0
    pending_x: np.ndarray | None = None
    y_min: float = float("-inf")
    y_max: float = float("inf")

    @classmethod
    def from_start_message(
        cls,
        payload: dict[str, Any],
        runtime_config: BOServerRuntimeConfig,
    ) -> "BOServerSession":
        """Create a session from a `start` message."""
        # For server mode, bounds are standardized to normalized [0,1]^d from config.
        bounds = [(0.0, 1.0) for _ in range(runtime_config.input_dim)]

        n_init = int(payload.get("n_init", runtime_config.n_init_default))
        n_iter = int(payload.get("n_iter", runtime_config.n_iter_default))
        if n_init < 0:
            raise ValueError("n_init must be non-negative.")
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative.")

        seed = payload.get("seed", 0)
        optimizer_name = str(payload.get("optimizer", runtime_config.optimizer))
        if optimizer_name not in {"bo_botorch", "bo_scratch"}:
            raise ValueError("optimizer must be 'bo_botorch' or 'bo_scratch'.")

        if optimizer_name == "bo_botorch":
            config = BoTorchConfig(
                bounds=bounds,
                n_init=n_init,
                num_restarts=int(
                    payload.get("num_restarts", runtime_config.num_restarts_default)
                ),
                raw_samples=int(
                    payload.get("raw_samples", runtime_config.raw_samples_default)
                ),
                seed=seed,
            )
            optimizer = BoTorchSequentialOptimizer(config)
        else:
            config = ScratchConfig(
                bounds=bounds,
                n_init=n_init,
                n_candidates=int(payload.get("n_candidates", 512)),
                n_starts=int(payload.get("n_starts", 8)),
                search_strategy=str(payload.get("search_strategy", "multistart")),
                kernel_type=str(payload.get("kernel_type", "matern52")),
                optimize_hyperparameters=bool(
                    payload.get("optimize_hyperparameters", True)
                ),
                seed=seed,
            )
            optimizer = ScratchSequentialOptimizer(config)
        return cls(
            optimizer_name=optimizer_name,
            optimizer=optimizer,
            n_iter=n_iter,
            n_init=n_init,
            y_min=runtime_config.y_min,
            y_max=runtime_config.y_max,
        )

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
            "optimizer": self.optimizer_name,
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
            "optimizer": self.optimizer_name,
            "total_observations": int(result.x_obs.shape[0]),
            "best_value": float(result.y_obs[best_idx]),
            "best_x": [float(v) for v in result.x_obs[best_idx]],
            "x_values": [[float(v) for v in row] for row in result.x_obs],
            "y_values": [float(v) for v in result.y_obs],
            "best_y_history": [float(v) for v in result.best_y_history],
        }

    def _stopped_payload(self, reason: str) -> dict[str, Any]:
        """Build stop payload with current partial optimization state."""
        result = self.optimizer.result()
        best_value: float | None = None
        best_x: list[float] | None = None
        if result.y_obs.size > 0:
            best_idx = int(np.argmax(result.y_obs))
            best_value = float(result.y_obs[best_idx])
            best_x = [float(v) for v in result.x_obs[best_idx]]
        return {
            "type": "stopped",
            "optimizer": self.optimizer_name,
            "reason": str(reason),
            "total_observations": int(result.x_obs.shape[0]),
            "best_value": best_value,
            "best_x": best_x,
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
            if y < self.y_min or y > self.y_max:
                raise ValueError(
                    f"Observed y={y} is outside configured y_range=[{self.y_min}, {self.y_max}]."
                )
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
                "optimizer": self.optimizer_name,
                "n_init": int(self.n_init),
                "n_iter": int(self.n_iter),
                "init_count": int(self.init_count),
                "bo_count": int(self.bo_count),
                "n_observations": int(self.optimizer.x_obs.shape[0]),
                "has_pending": bool(self.pending_x is not None),
            }

        if msg_type == "stop":
            return self._stopped_payload(reason=str(payload.get("reason", "client_stop")))

        raise ValueError(f"Unsupported message type: {msg_type}")


async def serve_bo_websocket(
    host: str = "127.0.0.1",
    port: int = 8765,
    config_path: str | Path = "configs/server_optimizers/bo_server.yaml",
) -> None:
    """Run generic websocket optimizer server forever."""
    async def _handler(websocket) -> None:
        runtime_config = BOServerRuntimeConfig.from_yaml_file(config_path)
        session: BOServerSession | None = None
        async for raw in websocket:
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("message must be a JSON object.")

                msg_type = payload.get("type")
                if session is None:
                    if msg_type != "start":
                        raise ValueError("First message must be type='start'.")
                    session = BOServerSession.from_start_message(
                        payload=payload,
                        runtime_config=runtime_config,
                    )
                    response = session.handle({"type": "suggest"})
                else:
                    if msg_type == "start":
                        raise ValueError("Session already started.")
                    response = session.handle(payload)
            except Exception as exc:  # noqa: BLE001
                response = {"type": "error", "message": str(exc)}
            await websocket.send(json.dumps(response))
            if response.get("type") in {"done", "stopped"}:
                return

    async with serve(_handler, host, port):
        await asyncio.Future()
