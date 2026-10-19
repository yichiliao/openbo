"""WebSocket server for TAF+ sessions (Liao et al., CHI 2024).

One WebSocket connection is one new user. Each session loads the population
(earlier users' per-objective GPs) from the configured ``population_dir``,
which is read-only: finished users are not added to it. Build a population
with ``TAFPlusPopulation.add_user`` + ``save``, or with
``scripts/run_taf_plus_benchmark.py --method population``.

Protocol (inputs are in ``[0, 1]^d``, every objective is maximized):

* client ``{"type": "start", ...overrides}`` -> server ``suggest``. Overrides:
  ``objective_weights`` (one per objective, summing to 1), ``decay``, ``d1``,
  ``d2``, ``n_iter``, ``n_candidates``, ``weight_mode``, ``rho``,
  ``first_point``, ``refine_suggestion``, ``seed``.
* server ``{"type": "suggest", "x": [...], "phase": "population" | "taf_plus",
  "iteration": i, "decay": delta | null, ...}``. The first design comes from
  the population alone; ``decay`` is ``delta(t)`` for TAF+ steps.
* client ``{"type": "observe", "y": [y_1, ..., y_L], "x": [...]?}`` -> next
  ``suggest`` or ``done``. ``y`` holds one value per objective, each inside
  that objective's configured ``y_range``.
* client ``{"type": "status"}`` or ``{"type": "stop"}`` at any time.
* ``done``/``stopped`` carry ``x_values``, ``y_values`` (n x L),
  ``weighted_y`` (``y @ objective_weights``), ``best_so_far`` and
  ``decay_history``.
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

from openbo.acquisition.taf_plus import TAF_PLUS_WEIGHT_MODES
from openbo.optimizers.bo_taf_plus import (
    TAFPlusConfig,
    TAFPlusPopulation,
    TAFPlusSequentialOptimizer,
)

_FIRST_POINTS = ("population", "centre")


@dataclass
class TAFPlusServerRuntimeConfig:
    """Static server-side config loaded from YAML.

    ``y_range`` gives ``[min, max]`` per objective; observations outside it
    are rejected. The population's input dimension and objective count must
    match ``input_dim`` and ``y_range``; its kernel is used for the new user.
    """

    input_dim: int
    y_range: list[tuple[float, float]]
    population_dir: str
    objective_weights_default: tuple[float, ...]
    decay_default: bool = True
    d1_default: int = 2
    d2_default: float = 0.3
    weight_mode_default: str = "poe"
    rho_default: float = 1.0
    n_iter_default: int = 10
    n_candidates_default: int = 1024
    refine_suggestion_default: bool = True
    first_point_default: str = "population"

    @property
    def n_objectives(self) -> int:
        """Number of objectives, from ``y_range``."""
        return len(self.y_range)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> TAFPlusServerRuntimeConfig:
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
        if "population_dir" not in payload:
            raise ValueError("population_dir is required (a directory written by TAFPlusPopulation.save).")
        n_objectives = len(y_range)
        weights = tuple(
            float(w) for w in payload.get("objective_weights_default", [1.0 / n_objectives] * n_objectives)
        )
        if len(weights) != n_objectives:
            raise ValueError("objective_weights_default needs one entry per objective in y_range.")
        runtime = cls(
            input_dim=input_dim,
            y_range=y_range,
            population_dir=str(payload["population_dir"]),
            objective_weights_default=weights,
            decay_default=bool(payload.get("decay_default", True)),
            d1_default=int(payload.get("d1_default", 2)),
            d2_default=float(payload.get("d2_default", 0.3)),
            weight_mode_default=str(payload.get("weight_mode_default", "poe")),
            rho_default=float(payload.get("rho_default", 1.0)),
            n_iter_default=int(payload.get("n_iter_default", 10)),
            n_candidates_default=int(payload.get("n_candidates_default", 1024)),
            refine_suggestion_default=bool(payload.get("refine_suggestion_default", True)),
            first_point_default=str(payload.get("first_point_default", "population")),
        )
        # Validate the defaults the same way a start message is validated.
        runtime.taf_plus_config({})
        return runtime

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> TAFPlusServerRuntimeConfig:
        """Load the server config from a YAML file."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Server config must be a YAML mapping.")
        return cls.from_mapping(payload)

    def taf_plus_config(self, payload: dict[str, Any], kernel_type: str = "rbf") -> TAFPlusConfig:
        """Build a ``TAFPlusConfig`` from the defaults and a start message's overrides."""
        weights = tuple(float(w) for w in payload.get("objective_weights", self.objective_weights_default))
        if len(weights) != self.n_objectives:
            raise ValueError(f"objective_weights needs {self.n_objectives} entries, one per objective.")
        weight_mode = str(payload.get("weight_mode", self.weight_mode_default))
        if weight_mode not in TAF_PLUS_WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {list(TAF_PLUS_WEIGHT_MODES)}, got {weight_mode!r}.")
        first_point = str(payload.get("first_point", self.first_point_default))
        if first_point not in _FIRST_POINTS:
            raise ValueError(f"first_point must be one of {list(_FIRST_POINTS)}, got {first_point!r}.")
        return TAFPlusConfig(
            input_dim=self.input_dim,
            objective_weights=weights,
            decay=bool(payload.get("decay", self.decay_default)),
            d1=int(payload.get("d1", self.d1_default)),
            d2=float(payload.get("d2", self.d2_default)),
            weight_mode=weight_mode,
            rho=float(payload.get("rho", self.rho_default)),
            kernel_type=kernel_type,  # type: ignore[arg-type]
            n_candidates=int(payload.get("n_candidates", self.n_candidates_default)),
            refine_suggestion=bool(payload.get("refine_suggestion", self.refine_suggestion_default)),
            first_point=first_point,  # type: ignore[arg-type]
            seed=int(payload.get("seed", 0)),
        )

    def load_population(self) -> TAFPlusPopulation:
        """Load the population and check that it matches this config."""
        directory = Path(self.population_dir)
        if not (directory / "taf_plus_population.json").exists():
            raise ValueError(f"No TAF+ population at {directory} (missing taf_plus_population.json).")
        population = TAFPlusPopulation.load(directory)
        if population.input_dim != self.input_dim:
            raise ValueError(
                f"The population has input_dim={population.input_dim}, but the config has {self.input_dim}."
            )
        if population.n_objectives != self.n_objectives:
            raise ValueError(
                f"The population has {population.n_objectives} objectives, but y_range has {self.n_objectives}."
            )
        return population


@dataclass
class TAFPlusServerSession:
    """One ask/tell TAF+ session (one new user) driven by WebSocket messages."""

    optimizer: TAFPlusSequentialOptimizer
    n_iter: int
    y_range: list[tuple[float, float]]
    pending_x: np.ndarray | None = None

    @classmethod
    def from_start_message(
        cls,
        payload: dict[str, Any],
        runtime: TAFPlusServerRuntimeConfig,
        population: TAFPlusPopulation | None = None,
    ) -> TAFPlusServerSession:
        """Create a session from a ``start`` message, applying its overrides.

        ``population`` defaults to loading ``runtime.population_dir``.
        """
        n_iter = int(payload.get("n_iter", runtime.n_iter_default))
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1.")
        if population is None:
            population = runtime.load_population()
        config = runtime.taf_plus_config(payload, kernel_type=population.kernel_type)
        return cls(
            optimizer=TAFPlusSequentialOptimizer(config, population),
            n_iter=n_iter,
            y_range=runtime.y_range,
        )

    @property
    def finished(self) -> bool:
        """Whether all ``n_iter`` evaluations have been observed."""
        return self.optimizer.n_observations >= self.n_iter

    def _next_suggestion(self) -> dict[str, Any]:
        """Ask the optimizer for the next design."""
        opt = self.optimizer
        iteration = opt.n_observations
        x = opt.suggest()
        self.pending_x = x.reshape(-1)
        decay = opt.decay_history[-1]
        return {
            "type": "suggest",
            "optimizer": "taf_plus",
            "x": [float(v) for v in self.pending_x],
            "phase": "population" if iteration == 0 else "taf_plus",
            "iteration": int(iteration),
            "decay": None if decay is None else float(decay),
            "n_observations": int(iteration),
            "n_objectives": int(opt.config.n_objectives),
        }

    def _result(self, kind: str, **extra: Any) -> dict[str, Any]:
        """Build a ``done``/``stopped`` message with everything observed so far."""
        result = self.optimizer.result()
        n_obs = int(result.y_obs.shape[0])
        return {
            "type": kind,
            "optimizer": "taf_plus",
            **extra,
            "total_observations": n_obs,
            "objective_weights": list(result.objective_weights),
            "x_values": result.x_obs.tolist(),
            "y_values": result.y_obs.tolist(),
            "weighted_y": result.weighted_y.tolist(),
            "best_so_far": result.best_so_far.tolist(),
            # One entry per suggestion; a pending suggestion's entry has no observation yet.
            "decay_history": [None if d is None else float(d) for d in result.decay_history[:n_obs]],
            "population_users": [u.name for u in self.optimizer.population.users],
        }

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
                raise ValueError(f"Observed y[{k}]={value} is outside its configured y_range=[{lo}, {hi}].")
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
                if x_vec.shape != self.pending_x.shape or not np.allclose(x_vec, self.pending_x, atol=1e-12):
                    raise ValueError("observe x does not match the pending suggestion.")
            self.optimizer.observe(self.pending_x[None, :], y[None, :])
            self.pending_x = None
            return self._result("done") if self.finished else self._next_suggestion()
        if kind == "status":
            result = self.optimizer.result()
            return {
                "type": "status",
                "optimizer": "taf_plus",
                "n_iter": int(self.n_iter),
                "n_observations": int(self.optimizer.n_observations),
                "has_pending": self.pending_x is not None,
                "best_weighted_y": float(result.best_so_far[-1]) if result.best_so_far.size else None,
                "n_population_users": int(self.optimizer.population.n_users),
            }
        if kind == "stop":
            return self._result("stopped", reason=str(payload.get("reason", "client_stop")))
        raise ValueError(f"Unsupported message type: {kind}")


async def _taf_plus_websocket_handler(websocket: Any, config: TAFPlusServerRuntimeConfig) -> None:
    """Run one session on ``websocket`` until done, stopped or disconnected."""
    session: TAFPlusServerSession | None = None
    async for raw in websocket:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise TypeError("message must be a JSON object.")
            if session is None:
                if payload.get("type") != "start":
                    raise ValueError("First message must be type='start'.")
                session = TAFPlusServerSession.from_start_message(payload, config)
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


async def _taf_plus_websocket_handler_from_config_file(websocket: Any, config_path: str | Path) -> None:
    """Load the current config for a new connection, then run its session."""
    await _taf_plus_websocket_handler(websocket, TAFPlusServerRuntimeConfig.from_yaml_file(config_path))


async def serve_taf_plus_websocket(
    host: str = "127.0.0.1",
    port: int = 8770,
    config_path: str | Path = "configs/server_optimizers/taf_plus_server.yaml",
) -> None:
    """Serve independent TAF+ sessions forever."""

    async def handler(websocket: Any) -> None:
        await _taf_plus_websocket_handler_from_config_file(websocket, config_path)

    async with serve(handler, host, port):
        await asyncio.Future()
