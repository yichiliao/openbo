"""WebSocket server for NAF / MetaBO sessions (Volpp et al., ICLR 2020).

One WebSocket connection is one task. Each session loads a trained neural AF
(``<af_name>.pt`` plus ``naf_config.json``, written by ``scripts/train_naf.py``)
from the configured ``af_dir``, which is read-only. The AF is greedy, so a
session is deterministic given the observations.

A trained AF only makes sense on tasks from the family it was trained on, with
values on that family's scale: the AF reads the GP's raw posterior mean and
standard deviation, and the GP uses the family's fixed hyperparameters.
The budget ``T`` the AF was trained with is one of its inputs, so a session
can stop early (``n_iter`` < ``T``) but never run longer.

Protocol (inputs are in ``[0, 1]^d``, the objective is maximized):

* client ``{"type": "start", "n_iter": n?}`` -> server ``suggest``.
  ``n_iter`` defaults to the config's ``n_iter_default`` (or ``T``).
* server ``{"type": "suggest", "x": [...], "iteration": i, "budget": T, ...}``.
* client ``{"type": "observe", "y": value, "x": [...]?}`` -> next ``suggest`` or
  ``done``. ``y`` is one number inside the configured ``y_range``.
* client ``{"type": "status"}`` or ``{"type": "stop"}`` at any time.
* ``done``/``stopped`` carry ``x_values``, ``y_values`` and ``best_so_far``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from websockets.asyncio.server import serve

from openbo.optimizers.naf import NAFSequentialOptimizer


@dataclass
class NAFServerRuntimeConfig:
    """Static server-side config loaded from YAML.

    ``y_range`` is ``[min, max]``; observations outside it are rejected. The
    AF's input dimension must match ``input_dim``. ``n_iter_default`` of
    ``None`` means the AF's training budget.
    """

    input_dim: int
    y_range: tuple[float, float]
    af_dir: str
    af_name: str = "best_af"
    n_iter_default: int | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> NAFServerRuntimeConfig:
        """Validate and build a config from a parsed YAML mapping."""
        input_dim = int(payload["input_dim"])
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        raw_range = payload.get("y_range")
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise ValueError("y_range must be [y_min, y_max].")
        y_range = (float(raw_range[0]), float(raw_range[1]))
        if y_range[1] <= y_range[0]:
            raise ValueError("y_range must satisfy y_max > y_min.")
        if "af_dir" not in payload:
            raise ValueError("af_dir is required (a scripts/train_naf.py output folder).")
        n_iter_default = payload.get("n_iter_default")
        return cls(
            input_dim=input_dim,
            y_range=y_range,
            af_dir=str(payload["af_dir"]),
            af_name=str(payload.get("af_name", "best_af")),
            n_iter_default=None if n_iter_default is None else int(n_iter_default),
        )

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> NAFServerRuntimeConfig:
        """Load the server config from a YAML file."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Server config must be a YAML mapping.")
        return cls.from_mapping(payload)

    def load_optimizer(self) -> NAFSequentialOptimizer:
        """Load the trained AF and check that it matches this config."""
        directory = Path(self.af_dir)
        if not (directory / f"{self.af_name}.pt").exists() or not (directory / "naf_config.json").exists():
            raise ValueError(f"No trained NAF at {directory} (needs {self.af_name}.pt and naf_config.json).")
        optimizer = NAFSequentialOptimizer.from_directory(directory, name=self.af_name)
        if optimizer.af.dim != self.input_dim:
            raise ValueError(f"The AF has input_dim={optimizer.af.dim}, but the config has {self.input_dim}.")
        return optimizer


@dataclass
class NAFServerSession:
    """One ask/tell NAF session (one task) driven by WebSocket messages."""

    optimizer: NAFSequentialOptimizer
    n_iter: int
    y_range: tuple[float, float]
    af_dir: str
    pending_x: np.ndarray | None = None

    @classmethod
    def from_start_message(
        cls,
        payload: dict[str, Any],
        runtime: NAFServerRuntimeConfig,
        optimizer: NAFSequentialOptimizer | None = None,
    ) -> NAFServerSession:
        """Create a session from a ``start`` message. ``optimizer`` defaults to loading ``runtime.af_dir``."""
        if optimizer is None:
            optimizer = runtime.load_optimizer()
        default = optimizer.budget if runtime.n_iter_default is None else runtime.n_iter_default
        n_iter = int(payload.get("n_iter", default))
        if not 1 <= n_iter <= optimizer.budget:
            raise ValueError(
                f"n_iter must be between 1 and the AF's training budget {optimizer.budget}, got {n_iter}."
            )
        return cls(optimizer=optimizer, n_iter=n_iter, y_range=runtime.y_range, af_dir=runtime.af_dir)

    @property
    def finished(self) -> bool:
        """Whether all ``n_iter`` evaluations have been observed."""
        return self.optimizer.t >= self.n_iter

    def _best_so_far(self) -> list[float]:
        return np.maximum.accumulate(self.optimizer.y_obs).tolist() if self.optimizer.t else []

    def _next_suggestion(self) -> dict[str, Any]:
        """Ask the optimizer for the next point."""
        x = self.optimizer.suggest()
        self.pending_x = x.reshape(-1)
        return {
            "type": "suggest",
            "optimizer": "naf",
            "x": [float(v) for v in self.pending_x],
            "iteration": int(self.optimizer.t),
            "n_observations": int(self.optimizer.t),
            "n_iter": int(self.n_iter),
            "budget": int(self.optimizer.budget),
        }

    def _result(self, kind: str, **extra: Any) -> dict[str, Any]:
        """Build a ``done``/``stopped`` message with everything observed so far."""
        return {
            "type": kind,
            "optimizer": "naf",
            **extra,
            "total_observations": int(self.optimizer.t),
            "budget": int(self.optimizer.budget),
            "af_dir": self.af_dir,
            "x_values": self.optimizer.x_obs.tolist(),
            "y_values": self.optimizer.y_obs.tolist(),
            "best_so_far": self._best_so_far(),
        }

    def _parse_y(self, raw: Any) -> float:
        """Validate an observed value against the configured range."""
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("y must be a single number.")
        y = float(raw)
        lo, hi = self.y_range
        if not np.isfinite(y):
            raise ValueError("y must be finite.")
        if y < lo or y > hi:
            raise ValueError(f"Observed y={y} is outside the configured y_range=[{lo}, {hi}].")
        return y

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle one inbound message and return one outbound message."""
        kind = payload.get("type")
        if not isinstance(kind, str):
            raise ValueError("message must include string field 'type'.")
        if kind == "suggest":
            if self.pending_x is not None:
                raise ValueError("Cannot suggest again before observe.")
            return self._result("done") if self.finished else self._next_suggestion()
        if kind == "observe":
            if self.pending_x is None:
                raise ValueError("No pending suggestion. Call suggest first.")
            if "y" not in payload:
                raise ValueError("observe needs 'y', the objective value.")
            y = self._parse_y(payload["y"])
            x_client = payload.get("x")
            if x_client is not None:
                x_vec = np.asarray(x_client, dtype=np.float64)
                if x_vec.shape != self.pending_x.shape or not np.allclose(x_vec, self.pending_x, atol=1e-12):
                    raise ValueError("observe x does not match the pending suggestion.")
            self.optimizer.observe(self.pending_x[None, :], np.array([y]))
            self.pending_x = None
            return self._result("done") if self.finished else self._next_suggestion()
        if kind == "status":
            best = self._best_so_far()
            return {
                "type": "status",
                "optimizer": "naf",
                "n_iter": int(self.n_iter),
                "budget": int(self.optimizer.budget),
                "n_observations": int(self.optimizer.t),
                "has_pending": self.pending_x is not None,
                "best_y": best[-1] if best else None,
            }
        if kind == "stop":
            return self._result("stopped", reason=str(payload.get("reason", "client_stop")))
        raise ValueError(f"Unsupported message type: {kind}")


async def _naf_websocket_handler(websocket: Any, config: NAFServerRuntimeConfig) -> None:
    """Run one session on ``websocket`` until done, stopped or disconnected."""
    session: NAFServerSession | None = None
    async for raw in websocket:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise TypeError("message must be a JSON object.")
            if session is None:
                if payload.get("type") != "start":
                    raise ValueError("First message must be type='start'.")
                session = NAFServerSession.from_start_message(payload, config)
                response = session.handle({"type": "suggest"})
            else:
                if payload.get("type") == "start":
                    raise ValueError("Session already started.")
                response = session.handle(payload)
        except Exception as exc:  # noqa: BLE001
            response = {"type": "error", "message": str(exc)}
        await websocket.send(json.dumps(response))
        if response.get("type") in {"done", "stopped"}:
            return


async def _naf_websocket_handler_from_config_file(websocket: Any, config_path: str | Path) -> None:
    """Load the current config for a new connection, then run its session."""
    await _naf_websocket_handler(websocket, NAFServerRuntimeConfig.from_yaml_file(config_path))


async def serve_naf_websocket(
    host: str = "127.0.0.1",
    port: int = 8771,
    config_path: str | Path = "configs/server_optimizers/naf_server.yaml",
) -> None:
    """Serve independent NAF sessions forever."""

    async def handler(websocket: Any) -> None:
        await _naf_websocket_handler_from_config_file(websocket, config_path)

    async with serve(handler, host, port):
        await asyncio.Future()
