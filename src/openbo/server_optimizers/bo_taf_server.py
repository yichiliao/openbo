"""Dedicated WebSocket server adapter for ask/tell TAF optimization."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from websockets.asyncio.server import serve

from openbo.optimizers.bo_taf import TAFConfig, TAFSequentialOptimizer


def _as_vector(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("x must be a 1D vector.")
    return arr


@dataclass
class BOTAFServerRuntimeConfig:
    """Static server-side config loaded from file for TAF backend."""

    input_dim: int
    y_min: float
    y_max: float
    taf_run_dir: str
    n_init_default: int = 0
    n_iter_default: int = 25
    n_candidates_default: int = 512
    n_starts_default: int = 8
    search_strategy_default: str = "multistart"
    kernel_type_default: str = "matern52"
    optimize_hyperparameters_default: bool = True
    taf_weight_mode_default: str = "taf_m"
    taf_rho_default: float = 1.0
    source_only_warmup_iters_default: int = 3
    track_acquisition_default: bool = True

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "BOTAFServerRuntimeConfig":
        """Load TAF server runtime config from YAML file."""
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

        taf_run_dir = payload.get("taf_run_dir")
        if not isinstance(taf_run_dir, str) or not taf_run_dir:
            raise ValueError("taf_run_dir must be a non-empty string path.")

        taf_weight_mode_default = str(payload.get("taf_weight_mode_default", "taf_m"))
        if taf_weight_mode_default not in {"taf_m", "taf_r"}:
            raise ValueError("taf_weight_mode_default must be 'taf_m' or 'taf_r'.")

        return cls(
            input_dim=input_dim,
            y_min=y_min,
            y_max=y_max,
            taf_run_dir=taf_run_dir,
            n_init_default=int(payload.get("n_init_default", 0)),
            n_iter_default=int(payload.get("n_iter_default", 25)),
            n_candidates_default=int(payload.get("n_candidates_default", 512)),
            n_starts_default=int(payload.get("n_starts_default", 8)),
            search_strategy_default=str(payload.get("search_strategy_default", "multistart")),
            kernel_type_default=str(payload.get("kernel_type_default", "matern52")),
            optimize_hyperparameters_default=bool(
                payload.get("optimize_hyperparameters_default", True)
            ),
            taf_weight_mode_default=taf_weight_mode_default,
            taf_rho_default=float(payload.get("taf_rho_default", 1.0)),
            source_only_warmup_iters_default=int(
                payload.get("source_only_warmup_iters_default", 3)
            ),
            track_acquisition_default=bool(
                payload.get("track_acquisition_default", True)
            ),
        )


@dataclass
class BOTAFServerSession:
    """Single optimization session state for ask/tell TAF over WebSocket."""

    optimizer: TAFSequentialOptimizer
    n_iter: int
    n_init: int
    init_count: int = 0
    bo_count: int = 0
    pending_x: np.ndarray | None = None
    y_min: float = float("-inf")
    y_max: float = float("inf")
    # Pre-drawn init design (single rng call, same as TAFSequentialOptimizer.bootstrap).
    _init_x_batch: np.ndarray | None = None
    _init_y_buffer: list[float] = field(default_factory=list)

    @classmethod
    def from_start_message(
        cls,
        payload: dict[str, Any],
        runtime_config: BOTAFServerRuntimeConfig,
    ) -> "BOTAFServerSession":
        """Create a TAF session from a `start` message."""
        bounds = [(0.0, 1.0) for _ in range(runtime_config.input_dim)]
        n_init = int(payload.get("n_init", runtime_config.n_init_default))
        n_iter = int(payload.get("n_iter", runtime_config.n_iter_default))
        if n_init < 0:
            raise ValueError("n_init must be non-negative.")
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative.")

        seed = payload.get("seed", 0)
        taf_run_dir = str(payload.get("taf_run_dir", runtime_config.taf_run_dir))
        taf_weight_mode = str(
            payload.get("taf_weight_mode", runtime_config.taf_weight_mode_default)
        )
        if taf_weight_mode not in {"taf_m", "taf_r"}:
            raise ValueError("taf_weight_mode must be 'taf_m' or 'taf_r'.")

        # Match run_bo_taf: after batched init observe, iter_count is 1; suggest()
        # needs iter_count < config.n_iter, so allow one extra slot when n_init > 0.
        taf_loop_n_iter = n_iter + (1 if n_init > 0 else 0)
        config = TAFConfig(
            bounds=bounds,
            taf_run_dir=taf_run_dir,
            n_init=n_init,
            n_iter=taf_loop_n_iter,
            n_candidates=int(
                payload.get("n_candidates", runtime_config.n_candidates_default)
            ),
            n_starts=int(payload.get("n_starts", runtime_config.n_starts_default)),
            search_strategy=str(
                payload.get("search_strategy", runtime_config.search_strategy_default)
            ),
            kernel_type=str(payload.get("kernel_type", runtime_config.kernel_type_default)),
            optimize_hyperparameters=bool(
                payload.get(
                    "optimize_hyperparameters",
                    runtime_config.optimize_hyperparameters_default,
                )
            ),
            rho=float(payload.get("rho", runtime_config.taf_rho_default)),
            taf_weight_mode=taf_weight_mode,
            source_only_warmup_iters=int(
                payload.get(
                    "source_only_warmup_iters",
                    runtime_config.source_only_warmup_iters_default,
                )
            ),
            track_acquisition=bool(
                payload.get("track_acquisition", runtime_config.track_acquisition_default)
            ),
            seed=seed,
        )
        optimizer = TAFSequentialOptimizer(config)
        init_x_batch: np.ndarray | None = None
        if n_init > 0:
            init_x_batch = optimizer.rng.uniform(
                optimizer.lower,
                optimizer.upper,
                size=(n_init, optimizer.d),
            ).astype(np.float64)
        return cls(
            optimizer=optimizer,
            n_iter=n_iter,
            n_init=n_init,
            y_min=runtime_config.y_min,
            y_max=runtime_config.y_max,
            _init_x_batch=init_x_batch,
        )

    def _next_suggestion(self) -> dict[str, Any]:
        if self.init_count < self.n_init:
            if self._init_x_batch is None:
                raise RuntimeError("Internal error: n_init > 0 but init design is missing.")
            x = self._init_x_batch[self.init_count : self.init_count + 1].astype(
                np.float64
            )
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
            "optimizer": "bo_taf",
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
            "optimizer": "bo_taf",
            "total_observations": int(result.x_obs.shape[0]),
            "best_value": float(result.y_obs[best_idx]),
            "best_x": [float(v) for v in result.x_obs[best_idx]],
            "x_values": [[float(v) for v in row] for row in result.x_obs],
            "y_values": [float(v) for v in result.y_obs],
            "best_y_history": [float(v) for v in result.best_y_history],
        }

    def _stopped_payload(self, reason: str) -> dict[str, Any]:
        if self.optimizer.x_obs.shape[0] == 0:
            return {
                "type": "stopped",
                "optimizer": "bo_taf",
                "reason": str(reason),
                "total_observations": 0,
                "best_value": None,
                "best_x": None,
                "x_values": [],
                "y_values": [],
                "best_y_history": [],
            }
        result = self.optimizer.result()
        best_value: float | None = None
        best_x: list[float] | None = None
        if result.y_obs.size > 0:
            best_idx = int(np.argmax(result.y_obs))
            best_value = float(result.y_obs[best_idx])
            best_x = [float(v) for v in result.x_obs[best_idx]]
        return {
            "type": "stopped",
            "optimizer": "bo_taf",
            "reason": str(reason),
            "total_observations": int(result.x_obs.shape[0]),
            "best_value": best_value,
            "best_x": best_x,
            "x_values": [[float(v) for v in row] for row in result.x_obs],
            "y_values": [float(v) for v in result.y_obs],
            "best_y_history": [float(v) for v in result.best_y_history],
        }

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
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

            # Match TAFSequentialOptimizer.bootstrap: one batched observe for all init
            # points so iter_count / best_y_history align with run_bo_taf.
            if self.n_init > 0 and self.optimizer.x_obs.shape[0] == 0:
                self._init_y_buffer.append(y)
                self.pending_x = None
                if len(self._init_y_buffer) < self.n_init:
                    if self.init_count >= self.n_init and self.bo_count >= self.n_iter:
                        return self._done_payload()
                    return self._next_suggestion()
                x_batch = self._init_x_batch
                if x_batch is None:
                    raise RuntimeError("Internal error: init y buffer full but no x batch.")
                y_batch = np.asarray(self._init_y_buffer, dtype=np.float64)
                self._init_y_buffer.clear()
                self.optimizer.observe(x_batch, y_batch)
            else:
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
                "optimizer": "bo_taf",
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


async def serve_bo_taf_websocket(
    host: str = "127.0.0.1",
    port: int = 8766,
    config_path: str | Path = "configs/server_optimizers/bo_taf_server.yaml",
) -> None:
    """Run websocket TAF optimizer server forever."""

    async def _handler(websocket) -> None:
        runtime_config = BOTAFServerRuntimeConfig.from_yaml_file(config_path)
        session: BOTAFServerSession | None = None
        async for raw in websocket:
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("message must be a JSON object.")

                msg_type = payload.get("type")
                if session is None:
                    if msg_type != "start":
                        raise ValueError("First message must be type='start'.")
                    session = BOTAFServerSession.from_start_message(
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
