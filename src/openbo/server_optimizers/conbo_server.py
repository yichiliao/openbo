"""WebSocket ask/tell server for continual BO (ConBO) across successive users.

Unlike the other servers, ConBO's state outlives a connection: the population
model and the memory of finished users live in ``memory_dir``. Each WebSocket
session is one user. At ``start`` the population is loaded from disk (or
created empty), the session runs ask/tell with scalar ``y`` like the generic BO
server, and when the user's budget is used up the user is stored
(``finish_user``: replay retraining plus save) before ``done`` is sent.

Only one user session may be active at a time, because ConBO requires users to
be finished in the order they were started. ``stop`` discards the user unless
it carries ``"store": true``.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from websockets.asyncio.server import serve

from openbo.optimizers.conbo import (
    ConBOConfig,
    ConBOPopulation,
    ConBOSequentialOptimizer,
    ReplayReport,
)

_USER_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ConBOServerRuntimeConfig:
    """Server-side settings loaded from YAML.

    ``conbo`` holds :class:`ConBOConfig` fields (``input_dim`` is filled in).
    They are used only when ``memory_dir`` holds no population yet; a saved
    population keeps the settings it was created with.
    """

    input_dim: int
    y_min: float
    y_max: float
    memory_dir: str
    n_evaluations_default: int = 20
    conbo: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if not self.y_max > self.y_min:
            raise ValueError("y_range must satisfy y_max > y_min.")
        if self.n_evaluations_default < 1:
            raise ValueError("n_evaluations_default must be at least 1.")
        if "input_dim" in self.conbo:
            raise ValueError("set input_dim at the top level, not under conbo.")
        self.conbo_config()  # fail early on bad ConBO settings

    def conbo_config(self) -> ConBOConfig:
        """The ConBO settings for a new population."""
        return ConBOConfig(input_dim=self.input_dim, **self.conbo)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> ConBOServerRuntimeConfig:
        """Load the server config from YAML."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("ConBO server config must be a YAML mapping.")
        y_range = payload.pop("y_range", None)
        if (
            not isinstance(y_range, list)
            or len(y_range) != 2
            or not all(isinstance(v, (int, float)) for v in y_range)
        ):
            raise ValueError("y_range must be [y_min, y_max].")
        return cls(y_min=float(y_range[0]), y_max=float(y_range[1]), **payload)


def load_population(config: ConBOServerRuntimeConfig) -> ConBOPopulation:
    """Load the population from ``memory_dir``, or create an empty one there."""
    root = Path(config.memory_dir)
    if (root / "conbo_state.json").exists():
        population = ConBOPopulation.load(root)
        if population.config.input_dim != config.input_dim:
            raise ValueError(
                f"memory_dir holds a {population.config.input_dim}D population, "
                f"but the server config has input_dim={config.input_dim}."
            )
        return population
    return ConBOPopulation(config.conbo_config(), root)


def _report_dict(report: ReplayReport) -> dict[str, Any]:
    return asdict(report)


@dataclass
class ConBOServerSession:
    """One user's ask/tell session against the shared population."""

    population: ConBOPopulation
    optimizer: ConBOSequentialOptimizer
    n_evaluations: int
    y_min: float
    y_max: float
    pending_x: np.ndarray | None = None
    finished: bool = False

    @classmethod
    def from_start_message(
        cls, payload: dict[str, Any], runtime_config: ConBOServerRuntimeConfig
    ) -> ConBOServerSession:
        """Load the population and start the next user."""
        n_evaluations = payload.get("n_evaluations", runtime_config.n_evaluations_default)
        if type(n_evaluations) is not int or n_evaluations < 1:
            raise ValueError("n_evaluations must be an integer >= 1.")
        name = payload.get("user_name")
        if name is not None and (not isinstance(name, str) or not _USER_NAME.match(name)):
            raise ValueError("user_name may contain only letters, digits, '_', '.' and '-'.")
        population = load_population(runtime_config)
        optimizer = population.new_user(name)
        return cls(
            population=population,
            optimizer=optimizer,
            n_evaluations=n_evaluations,
            y_min=runtime_config.y_min,
            y_max=runtime_config.y_max,
        )

    def _user(self) -> dict[str, Any]:
        return {
            "optimizer": "conbo",
            "user_name": self.optimizer.name,
            "user_index": self.optimizer.user_index,
        }

    def _suggest(self) -> dict[str, Any]:
        opt = self.optimizer
        step = opt.n_observations
        x = opt.suggest()
        self.pending_x = x.reshape(-1)
        weight = opt.weight_history[-1]
        return {
            "type": "suggest",
            **self._user(),
            "x": [float(v) for v in self.pending_x],
            "phase": "random" if weight is None else "bo",
            "population_weight": weight,
            "iteration": step,
            "n_observations": step,
            "n_evaluations": self.n_evaluations,
        }

    def _trajectory(self) -> dict[str, Any]:
        opt = self.optimizer
        if opt.n_observations == 0:
            best_value, best_x = None, None
        else:
            best = int(np.argmax(opt.y_obs))
            best_value, best_x = float(opt.y_obs[best]), opt.x_obs[best].tolist()
        return {
            "total_observations": opt.n_observations,
            "best_value": best_value,
            "best_x": best_x,
            "x_values": opt.x_obs.tolist(),
            "y_values": opt.y_obs.tolist(),
            "best_y_history": np.maximum.accumulate(opt.y_obs).tolist(),
            "n_random": opt.n_random,
            "population_weights": list(opt.weight_history),
        }

    def _store(self) -> dict[str, Any]:
        """Store the user in the population (replay retraining plus save)."""
        report = self.population.finish_user(self.optimizer)
        self.finished = True
        return {
            "stored": True,
            "n_users_in_memory": self.population.n_users,
            "replay": _report_dict(report),
        }

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Process suggest/observe/status/stop for this user."""
        kind = payload.get("type")
        if self.finished:
            raise ValueError("this user's session is finished")
        if kind == "suggest":
            if self.pending_x is not None:
                raise ValueError("observe the pending x before requesting another")
            return self._suggest()
        if kind == "observe":
            if self.pending_x is None:
                raise ValueError("no pending suggestion to observe")
            if "y" not in payload:
                raise ValueError("observe needs a scalar 'y'")
            y_raw = payload["y"]
            if isinstance(y_raw, bool) or not isinstance(y_raw, (int, float)):
                raise ValueError("y must be a number")
            y = float(y_raw)
            if not np.isfinite(y) or y < self.y_min or y > self.y_max:
                raise ValueError(
                    f"Observed y={y} is outside configured y_range=[{self.y_min}, {self.y_max}]."
                )
            if payload.get("x") is not None:
                x_client = np.asarray(payload["x"], dtype=np.float64)
                if x_client.shape != self.pending_x.shape or not np.allclose(
                    x_client, self.pending_x, atol=1e-12
                ):
                    raise ValueError("observe x does not match pending suggestion.")
            self.optimizer.observe(self.pending_x[None, :], np.array([y]))
            self.pending_x = None
            if self.optimizer.n_observations >= self.n_evaluations:
                trajectory = self._trajectory()
                return {"type": "done", **self._user(), **trajectory, **self._store()}
            return self._suggest()
        if kind == "status":
            opt = self.optimizer
            return {
                "type": "status",
                **self._user(),
                "n_users_in_memory": self.population.n_users,
                "n_evaluations": self.n_evaluations,
                "n_observations": opt.n_observations,
                "n_random": opt.n_random,
                "population_weight": opt.current_weight(),
                "has_pending": self.pending_x is not None,
            }
        if kind == "stop":
            message = {
                "type": "stopped",
                **self._user(),
                "reason": str(payload.get("reason", "client_stop")),
                **self._trajectory(),
            }
            store = payload.get("store", False)
            if not isinstance(store, bool):
                raise ValueError("store must be true or false")
            if store and self.optimizer.n_observations > 0:
                message.update(self._store())
            else:
                self.finished = True
                message.update(stored=False, n_users_in_memory=self.population.n_users)
            return message
        raise ValueError(f"unsupported ConBO message type: {kind}")


@dataclass
class ConBOServerState:
    """Shared across connections: at most one user session at a time."""

    active_user: str | None = None


async def _conbo_websocket_handler(
    websocket: Any, config: ConBOServerRuntimeConfig, state: ConBOServerState
) -> None:
    session: ConBOServerSession | None = None
    try:
        async for raw in websocket:
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise TypeError("message must be a JSON object")
                if session is None:
                    if payload.get("type") != "start":
                        raise ValueError("first message must have type='start'")
                    if state.active_user is not None:
                        raise ValueError(
                            f"another user session is active ({state.active_user}); "
                            "ConBO serves one user at a time"
                        )
                    session = ConBOServerSession.from_start_message(payload, config)
                    state.active_user = session.optimizer.name
                    response = session.handle({"type": "suggest"})
                else:
                    if payload.get("type") == "start":
                        raise ValueError("session already started")
                    response = session.handle(payload)
            except Exception as exc:  # noqa: BLE001
                response = {"type": "error", "message": str(exc)}
            await websocket.send(json.dumps(response))
            if response.get("type") in {"done", "stopped"}:
                return
    finally:
        # A dropped connection abandons the user: nothing is stored.
        if session is not None and state.active_user == session.optimizer.name:
            state.active_user = None


async def serve_conbo_websocket(
    host: str = "127.0.0.1",
    port: int = 8769,
    config_path: str | Path = "configs/server_optimizers/conbo_server.yaml",
) -> None:
    """Serve successive ConBO user sessions forever, one at a time."""
    state = ConBOServerState()

    async def handler(websocket: Any) -> None:
        config = ConBOServerRuntimeConfig.from_yaml_file(config_path)
        await _conbo_websocket_handler(websocket, config, state)

    async with serve(handler, host, port):
        await asyncio.Future()
