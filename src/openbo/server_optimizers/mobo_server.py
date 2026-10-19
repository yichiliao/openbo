"""WebSocket server for multi-objective BO (CEIPV/IEIPV) sessions.

Protocol (one WebSocket connection is one session; inputs are in ``[0, 1]^d``):

* client ``{"type": "start", ...overrides}`` -> server ``suggest``
* server ``{"type": "suggest", "x": [...], "phase": "init" | "bo", ...}``
* client ``{"type": "observe", "y": [y_1, ..., y_L], "x": [...]?}`` -> next
  ``suggest`` or ``done``. ``y`` holds one value per objective (all
  maximized), each inside that objective's configured ``y_range``.
* client ``{"type": "status"}`` or ``{"type": "stop"}`` at any time.
* ``done``/``stopped`` carry ``x_values``, ``y_values`` (n x L), the Pareto
  set and ``hv_history`` (one entry per observation, initial points included).
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

from openbo.optimizers.mobo import MOBOConfig, MOBOSequentialOptimizer

_MODELS = ("independent", "mt", "slf")
_ACQUISITIONS = ("ceipv", "ieipv")
_HYPERPARAMETERS = ("map", "slice")


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    """Return ``value`` if it is one of ``choices``; raise otherwise."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}.")
    return value


@dataclass
class MOBOServerRuntimeConfig:
    """Static server-side config loaded from YAML.

    ``y_range`` gives ``[min, max]`` per objective; observations outside it
    are rejected. ``v_ref``/``w_ref`` default to the range minimums/maximums.
    """

    input_dim: int
    y_range: list[tuple[float, float]]
    v_ref: list[float]
    w_ref: list[float]
    model: str = "slf"
    acquisition: str = "ceipv"
    hyperparameters: str = "map"
    n_init_default: int = 5
    n_iter_default: int = 25
    n_candidates_default: int = 1024

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> MOBOServerRuntimeConfig:
        """Validate and build a config from a parsed YAML mapping."""
        input_dim = int(payload["input_dim"])
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        raw_range = payload.get("y_range")
        if (
            not isinstance(raw_range, list)
            or not raw_range
            or not all(isinstance(r, list) and len(r) == 2 for r in raw_range)
        ):
            raise ValueError("y_range must be a list of [y_min, y_max], one per objective.")
        y_range = [(float(lo), float(hi)) for lo, hi in raw_range]
        if any(hi <= lo for lo, hi in y_range):
            raise ValueError("Every y_range entry must satisfy y_max > y_min.")
        n_objectives = len(y_range)
        v_ref = [float(v) for v in payload.get("v_ref", [lo for lo, _ in y_range])]
        w_ref = [float(w) for w in payload.get("w_ref", [hi for _, hi in y_range])]
        if len(v_ref) != n_objectives or len(w_ref) != n_objectives:
            raise ValueError("v_ref and w_ref need one entry per objective in y_range.")
        if any(w <= v for v, w in zip(v_ref, w_ref)):
            raise ValueError("w_ref must exceed v_ref in every objective.")
        return cls(
            input_dim=input_dim,
            y_range=y_range,
            v_ref=v_ref,
            w_ref=w_ref,
            model=_check_choice("model", str(payload.get("model", "slf")), _MODELS),
            acquisition=_check_choice(
                "acquisition", str(payload.get("acquisition", "ceipv")), _ACQUISITIONS
            ),
            hyperparameters=_check_choice(
                "hyperparameters", str(payload.get("hyperparameters", "map")), _HYPERPARAMETERS
            ),
            n_init_default=int(payload.get("n_init_default", 5)),
            n_iter_default=int(payload.get("n_iter_default", 25)),
            n_candidates_default=int(payload.get("n_candidates_default", 1024)),
        )

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> MOBOServerRuntimeConfig:
        """Load the server config from a YAML file."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Server config must be a YAML mapping.")
        return cls.from_mapping(payload)


@dataclass
class MOBOServerSession:
    """One ask/tell MOBO session driven by WebSocket messages."""

    optimizer: MOBOSequentialOptimizer
    n_init: int
    n_iter: int
    y_range: list[tuple[float, float]]
    init_count: int = 0
    bo_count: int = 0
    pending_x: np.ndarray | None = None

    @classmethod
    def from_start_message(
        cls, payload: dict[str, Any], runtime: MOBOServerRuntimeConfig
    ) -> MOBOServerSession:
        """Create a session from a ``start`` message, applying its overrides."""
        n_init = int(payload.get("n_init", runtime.n_init_default))
        n_iter = int(payload.get("n_iter", runtime.n_iter_default))
        if n_init < 1:
            raise ValueError("n_init must be at least 1 (the model needs data).")
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative.")
        n_rescore = payload.get("n_rescore")
        config = MOBOConfig(
            bounds=[(0.0, 1.0)] * runtime.input_dim,
            v_ref=runtime.v_ref,
            w_ref=runtime.w_ref,
            n_init=n_init,
            model=_check_choice("model", str(payload.get("model", runtime.model)), _MODELS),  # type: ignore[arg-type]
            acquisition=_check_choice(  # type: ignore[arg-type]
                "acquisition", str(payload.get("acquisition", runtime.acquisition)), _ACQUISITIONS
            ),
            hyperparameters=_check_choice(  # type: ignore[arg-type]
                "hyperparameters",
                str(payload.get("hyperparameters", runtime.hyperparameters)),
                _HYPERPARAMETERS,
            ),
            n_candidates=int(payload.get("n_candidates", runtime.n_candidates_default)),
            n_rescore=None if n_rescore is None else int(n_rescore),
            seed=payload.get("seed", 0),
        )
        return cls(
            optimizer=MOBOSequentialOptimizer(config),
            n_init=n_init,
            n_iter=n_iter,
            y_range=runtime.y_range,
        )

    @property
    def finished(self) -> bool:
        """Whether every initial and BO evaluation has been observed."""
        return self.init_count >= self.n_init and self.bo_count >= self.n_iter

    def _next_suggestion(self) -> dict[str, Any]:
        """Draw the next point (random during init, then acquisition-based)."""
        opt = self.optimizer
        if self.init_count < self.n_init:
            x = opt.rng.uniform(opt.lower, opt.upper, size=(1, opt.d))
            self.init_count += 1
            phase, iteration = "init", self.init_count - 1
        else:
            x = opt.suggest()
            self.bo_count += 1
            phase, iteration = "bo", self.bo_count - 1
        self.pending_x = x.reshape(-1)
        return {
            "type": "suggest",
            "optimizer": "mobo",
            "x": [float(v) for v in self.pending_x],
            "phase": phase,
            "iteration": int(iteration),
            "n_observations": int(opt.x_obs.shape[0]),
            "n_objectives": int(opt.n_objectives),
        }

    def _result(self, kind: str, **extra: Any) -> dict[str, Any]:
        """Build a ``done``/``stopped`` message with everything observed so far."""
        opt = self.optimizer
        payload: dict[str, Any] = {"type": kind, "optimizer": "mobo", **extra}
        payload["total_observations"] = int(opt.x_obs.shape[0])
        if opt.x_obs.shape[0] == 0:
            payload.update(x_values=[], y_values=[], pareto_x=[], pareto_y=[], hv_history=[])
            return payload
        result = opt.result()
        payload.update(
            x_values=result.x_obs.tolist(),
            y_values=result.y_obs.tolist(),
            pareto_x=result.pareto_x.tolist(),
            pareto_y=result.pareto_y.tolist(),
            hv_history=result.hv_history.tolist(),
            v_ref=opt.v_ref.tolist(),
            model_state=result.final_model_state,
        )
        return payload

    def _parse_y(self, raw: Any) -> np.ndarray:
        """Validate an observed objective vector against the configured ranges."""
        y = np.asarray(raw, dtype=np.float64)
        n_objectives = len(self.y_range)
        if y.shape != (n_objectives,):
            raise ValueError(f"y must be a list of {n_objectives} objective values.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y values must be finite.")
        for k, (value, (lo, hi)) in enumerate(zip(y, self.y_range)):
            if value < lo or value > hi:
                raise ValueError(
                    f"Observed y[{k}]={value} is outside its configured y_range=[{lo}, {hi}]."
                )
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
                raise ValueError("observe needs 'y', a list with one value per objective.")
            y = self._parse_y(payload["y"])
            x_client = payload.get("x")
            if x_client is not None:
                x_vec = np.asarray(x_client, dtype=np.float64)
                if x_vec.shape != self.pending_x.shape or not np.allclose(
                    x_vec, self.pending_x, atol=1e-12
                ):
                    raise ValueError("observe x does not match the pending suggestion.")
            self.optimizer.observe(self.pending_x[None, :], y[None, :])
            self.pending_x = None
            return self._result("done") if self.finished else self._next_suggestion()
        if kind == "status":
            return {
                "type": "status",
                "optimizer": "mobo",
                "n_init": int(self.n_init),
                "n_iter": int(self.n_iter),
                "init_count": int(self.init_count),
                "bo_count": int(self.bo_count),
                "n_observations": int(self.optimizer.x_obs.shape[0]),
                "has_pending": self.pending_x is not None,
                "hypervolume": (
                    float(self.optimizer.hv_history[-1]) if self.optimizer.hv_history else 0.0
                ),
            }
        if kind == "stop":
            return self._result("stopped", reason=str(payload.get("reason", "client_stop")))
        raise ValueError(f"Unsupported message type: {kind}")


async def _mobo_websocket_handler(websocket: Any, config: MOBOServerRuntimeConfig) -> None:
    """Run one session on ``websocket`` until done, stopped or disconnected."""
    session: MOBOServerSession | None = None
    async for raw in websocket:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise TypeError("message must be a JSON object.")
            if session is None:
                if payload.get("type") != "start":
                    raise ValueError("First message must be type='start'.")
                session = MOBOServerSession.from_start_message(payload, config)
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


async def _mobo_websocket_handler_from_config_file(
    websocket: Any, config_path: str | Path
) -> None:
    """Load the current config for a new connection, then run its session."""
    await _mobo_websocket_handler(websocket, MOBOServerRuntimeConfig.from_yaml_file(config_path))


async def serve_mobo_websocket(
    host: str = "127.0.0.1",
    port: int = 8768,
    config_path: str | Path = "configs/server_optimizers/mobo_server.yaml",
) -> None:
    """Serve independent MOBO sessions forever."""

    async def handler(websocket: Any) -> None:
        await _mobo_websocket_handler_from_config_file(websocket, config_path)

    async with serve(handler, host, port):
        await asyncio.Future()
