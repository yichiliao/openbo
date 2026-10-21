"""WebSocket server for TAF-MOBO sessions: transfer for multi-objective BO.

One WebSocket connection is one new user. The sources (earlier users or tasks)
are read from the configured ``population_dir`` (a saved ``TAFPlusPopulation``)
or ``taf_run_dir`` (the TAF run layout), and are never changed by a session.
Two implementations are served, chosen by ``method``:

* ``taf_eipv`` (default): TAF-EIPV on OpenBO's multi-objective BO
  (``optimizers/mobo_taf_eipv.py``). Its first design comes from the sources.
* ``taf_ehvi``: TAF-EHVI on BoTorch (``optimizers/mobo_taf.py``). It needs at
  least one random design first.

Protocol (inputs in ``[0, 1]^d``, every objective maximized):

* client ``{"type": "start", ...overrides}`` -> server ``suggest``. Overrides:
  ``method``, ``n_iter`` (total evaluations), ``n_init`` (random designs
  first), ``weight_mode``, ``rho``, ``seed``, and per method
  ``n_candidates`` (TAF-EIPV) or ``num_restarts``/``raw_samples`` (TAF-EHVI).
* server ``{"type": "suggest", "x": [...], "phase": "random" | "sources" | "taf",
  "iteration": i, "source_weights": [...] | null, ...}``. ``sources`` is
  TAF-EIPV's first design, from the sources alone.
* client ``{"type": "observe", "y": [y_1, ..., y_L], "x": [...]?}`` -> next
  ``suggest`` or ``done``. ``y`` holds one value per objective, each inside its
  configured ``y_range``.
* client ``{"type": "status"}`` or ``{"type": "stop"}`` at any time.
* ``done``/``stopped`` carry ``x_values``, ``y_values`` (n x L), ``pareto_x``,
  ``pareto_y``, ``hv_history`` (hypervolume above ``v_ref`` after each
  observation) and the source names.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from websockets.asyncio.server import serve

from openbo.benchmarks.hypervolume import hypervolume, pareto_mask
from openbo.optimizers.bo_taf_plus import TAFPlusPopulation
from openbo.optimizers.mobo_taf import MOTAFConfig, MOTAFSequentialOptimizer, write_population_sources
from openbo.optimizers.mobo_taf_eipv import TAFEIPVConfig, TAFEIPVSequentialOptimizer

METHODS = ("taf_eipv", "taf_ehvi")
_WEIGHT_MODES = {"taf_eipv": ("taf_r", "uniform"), "taf_ehvi": ("taf_r", "taf_r_pareto", "taf_m")}


@dataclass
class TAFMOBOServerRuntimeConfig:
    """Static server-side config loaded from YAML.

    ``y_range`` gives ``[min, max]`` per objective; observations outside it are
    rejected, and its corners are the default hypervolume box (``v_ref`` the
    minimums, ``w_ref`` the maximums). Exactly one of ``population_dir`` and
    ``taf_run_dir`` holds the sources.
    """

    input_dim: int
    y_range: list[tuple[float, float]]
    v_ref: list[float]
    w_ref: list[float]
    population_dir: str | None = None
    taf_run_dir: str | None = None
    method_default: str = "taf_eipv"
    n_iter_default: int = 10
    n_init_default: int | None = None  # None: 0 for taf_eipv, 1 for taf_ehvi
    weight_mode_default: str = "taf_r"
    rho_default: float = 1.0
    n_candidates_default: int = 1024
    num_restarts_default: int = 4
    raw_samples_default: int = 256
    _source_cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_objectives(self) -> int:
        """Number of objectives, from ``y_range``."""
        return len(self.y_range)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> TAFMOBOServerRuntimeConfig:
        """Validate and build a config from a parsed YAML mapping."""
        input_dim = int(payload["input_dim"])
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        raw_range = payload.get("y_range")
        if (
            not isinstance(raw_range, list)
            or len(raw_range) < 2
            or not all(isinstance(r, list) and len(r) == 2 for r in raw_range)
        ):
            raise ValueError("y_range must be a list of [y_min, y_max], one per objective (at least two).")
        y_range = [(float(lo), float(hi)) for lo, hi in raw_range]
        if any(hi <= lo for lo, hi in y_range):
            raise ValueError("Every y_range entry must satisfy y_max > y_min.")
        v_ref = [float(v) for v in payload.get("v_ref", [lo for lo, _ in y_range])]
        w_ref = [float(w) for w in payload.get("w_ref", [hi for _, hi in y_range])]
        if len(v_ref) != len(y_range) or len(w_ref) != len(y_range):
            raise ValueError("v_ref and w_ref need one value per objective.")
        if any(w <= v for v, w in zip(v_ref, w_ref)):
            raise ValueError("w_ref must exceed v_ref in every objective.")
        population_dir = payload.get("population_dir")
        taf_run_dir = payload.get("taf_run_dir")
        if (population_dir is None) == (taf_run_dir is None):
            raise ValueError("Set exactly one of population_dir and taf_run_dir (the sources).")
        n_init_default = payload.get("n_init_default")
        runtime = cls(
            input_dim=input_dim,
            y_range=y_range,
            v_ref=v_ref,
            w_ref=w_ref,
            population_dir=None if population_dir is None else str(population_dir),
            taf_run_dir=None if taf_run_dir is None else str(taf_run_dir),
            method_default=str(payload.get("method_default", "taf_eipv")),
            n_iter_default=int(payload.get("n_iter_default", 10)),
            n_init_default=None if n_init_default is None else int(n_init_default),
            weight_mode_default=str(payload.get("weight_mode_default", "taf_r")),
            rho_default=float(payload.get("rho_default", 1.0)),
            n_candidates_default=int(payload.get("n_candidates_default", 1024)),
            num_restarts_default=int(payload.get("num_restarts_default", 4)),
            raw_samples_default=int(payload.get("raw_samples_default", 256)),
        )
        runtime.settings({})  # validate the defaults like a start message
        return runtime

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> TAFMOBOServerRuntimeConfig:
        """Load the server config from a YAML file."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Server config must be a YAML mapping.")
        return cls.from_mapping(payload)

    def settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Session settings from the defaults and a start message's overrides."""
        method = str(payload.get("method", self.method_default))
        if method not in METHODS:
            raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}.")
        n_iter = int(payload.get("n_iter", self.n_iter_default))
        default_init = (0 if method == "taf_eipv" else 1) if self.n_init_default is None else self.n_init_default
        n_init = int(payload.get("n_init", default_init))
        if n_init < 0 or n_iter < 1 or n_init > n_iter:
            raise ValueError("Need n_iter >= 1 and 0 <= n_init <= n_iter.")
        if method == "taf_ehvi" and n_init < 1:
            raise ValueError("taf_ehvi needs n_init >= 1: it cannot suggest before one observation.")
        weight_mode = str(payload.get("weight_mode", self.weight_mode_default))
        if weight_mode not in _WEIGHT_MODES[method]:
            raise ValueError(f"weight_mode for {method} must be one of {list(_WEIGHT_MODES[method])}.")
        rho = float(payload.get("rho", self.rho_default))
        if rho <= 0.0:
            raise ValueError("rho must be positive.")
        return {
            "method": method,
            "n_iter": n_iter,
            "n_init": n_init,
            "weight_mode": weight_mode,
            "rho": rho,
            "seed": int(payload.get("seed", 0)),
            "n_candidates": int(payload.get("n_candidates", self.n_candidates_default)),
            "num_restarts": int(payload.get("num_restarts", self.num_restarts_default)),
            "raw_samples": int(payload.get("raw_samples", self.raw_samples_default)),
        }

    def population(self) -> TAFPlusPopulation:
        """Load ``population_dir`` once per config and check its shape."""
        if "population" not in self._source_cache:
            directory = Path(str(self.population_dir))
            if not (directory / "taf_plus_population.json").exists():
                raise ValueError(f"No TAF+ population at {directory} (missing taf_plus_population.json).")
            population = TAFPlusPopulation.load(directory)
            if population.input_dim != self.input_dim or population.n_objectives != self.n_objectives:
                raise ValueError(
                    f"The population has input_dim={population.input_dim} and {population.n_objectives} "
                    f"objectives; the config has {self.input_dim} and {self.n_objectives}."
                )
            self._source_cache["population"] = population
        return self._source_cache["population"]

    def source_dir(self) -> str:
        """A TAF run directory for TAF-EHVI; a population is converted once."""
        if self.taf_run_dir is not None:
            return self.taf_run_dir
        if "dir" not in self._source_cache:
            target = tempfile.mkdtemp(prefix="taf_mobo_sources_")
            write_population_sources(self.population(), target)
            self._source_cache["dir"] = target
        return self._source_cache["dir"]


@dataclass
class TAFMOBOServerSession:
    """One ask/tell TAF-MOBO session (one new user) driven by WebSocket messages."""

    optimizer: TAFEIPVSequentialOptimizer | MOTAFSequentialOptimizer
    settings: dict[str, Any]
    runtime: TAFMOBOServerRuntimeConfig
    rng: np.random.Generator
    x_obs: list[np.ndarray] = field(default_factory=list)
    y_obs: list[np.ndarray] = field(default_factory=list)
    pending_x: np.ndarray | None = None
    pending_phase: str | None = None

    @classmethod
    def from_start_message(cls, payload: dict[str, Any], runtime: TAFMOBOServerRuntimeConfig) -> TAFMOBOServerSession:
        """Create a session from a ``start`` message, applying its overrides."""
        settings = runtime.settings(payload)
        bounds = [(0.0, 1.0)] * runtime.input_dim
        if settings["method"] == "taf_eipv":
            config = TAFEIPVConfig(
                bounds=bounds,
                v_ref=runtime.v_ref,
                w_ref=runtime.w_ref,
                n_init=0,  # the session draws its own random designs
                n_candidates=settings["n_candidates"],
                weight_mode=settings["weight_mode"],
                rho=settings["rho"],
                taf_run_dir=runtime.taf_run_dir,
                seed=settings["seed"],
            )
            population = runtime.population() if runtime.population_dir is not None else None
            optimizer: TAFEIPVSequentialOptimizer | MOTAFSequentialOptimizer = TAFEIPVSequentialOptimizer(
                config, population
            )
            if not optimizer.sources:
                raise ValueError("No sources found.")
        else:
            optimizer = MOTAFSequentialOptimizer(
                MOTAFConfig(
                    bounds=bounds,
                    v_ref=runtime.v_ref,
                    n_init=0,
                    num_restarts=settings["num_restarts"],
                    raw_samples=settings["raw_samples"],
                    taf_run_dir=runtime.source_dir(),
                    n_iter=settings["n_iter"],
                    taf_weight_mode=settings["weight_mode"],
                    rho=settings["rho"],
                    seed=settings["seed"],
                )
            )
        if optimizer.d != runtime.input_dim:
            raise ValueError("The optimizer's input dimension does not match input_dim.")
        return cls(optimizer, settings, runtime, np.random.default_rng(settings["seed"]))

    @property
    def n_observations(self) -> int:
        """Designs observed so far."""
        return len(self.y_obs)

    @property
    def finished(self) -> bool:
        """Whether all ``n_iter`` evaluations have been observed."""
        return self.n_observations >= self.settings["n_iter"]

    def _source_names(self) -> list[str]:
        sources = getattr(self.optimizer, "sources", None) or getattr(self.optimizer, "source_surrogates", [])
        return [s.name for s in sources]

    def _next_suggestion(self) -> dict[str, Any]:
        """Ask for the next design: a random one during the random starts, else the optimizer's."""
        t = self.n_observations
        if t < self.settings["n_init"]:
            x = self.rng.random(self.runtime.input_dim)
            phase = "random"
            weights = None
        else:
            x = self.optimizer.suggest().reshape(-1)
            phase = "sources" if t == 0 else "taf"
            weights = [float(w) for w in np.asarray(self.optimizer.last_source_weights).reshape(-1)]
        self.pending_x, self.pending_phase = x, phase
        return {
            "type": "suggest",
            "optimizer": "taf_mobo",
            "method": self.settings["method"],
            "x": [float(v) for v in x],
            "phase": phase,
            "iteration": int(t),
            "n_iter": int(self.settings["n_iter"]),
            "source_weights": weights,
            "n_observations": int(t),
            "n_objectives": int(self.runtime.n_objectives),
        }

    def _result(self, kind: str, **extra: Any) -> dict[str, Any]:
        """Build a ``done``/``stopped`` message with everything observed so far."""
        n_obj = self.runtime.n_objectives
        x = np.asarray(self.x_obs, dtype=np.float64).reshape(-1, self.runtime.input_dim)
        y = np.asarray(self.y_obs, dtype=np.float64).reshape(-1, n_obj)
        mask = pareto_mask(y) if y.shape[0] else np.zeros(0, dtype=bool)
        return {
            "type": kind,
            "optimizer": "taf_mobo",
            "method": self.settings["method"],
            **extra,
            "total_observations": int(y.shape[0]),
            "x_values": x.tolist(),
            "y_values": y.tolist(),
            "pareto_x": x[mask].tolist(),
            "pareto_y": y[mask].tolist(),
            "hv_history": [hypervolume(y[: t + 1], self.runtime.v_ref) for t in range(y.shape[0])],
            "v_ref": list(self.runtime.v_ref),
            "sources": self._source_names(),
        }

    def _parse_y(self, raw: Any) -> np.ndarray:
        """Validate an observed objective vector against the configured ranges."""
        y = np.asarray(raw, dtype=np.float64)
        if y.shape != (self.runtime.n_objectives,):
            raise ValueError(f"y must be a list of {self.runtime.n_objectives} objective values.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y values must be finite.")
        for k, (value, (lo, hi)) in enumerate(zip(y, self.runtime.y_range)):
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
            self.x_obs.append(self.pending_x)
            self.y_obs.append(y)
            self.pending_x = self.pending_phase = None
            return self._result("done") if self.finished else self._next_suggestion()
        if kind == "status":
            y = np.asarray(self.y_obs, dtype=np.float64).reshape(-1, self.runtime.n_objectives)
            return {
                "type": "status",
                "optimizer": "taf_mobo",
                "method": self.settings["method"],
                "n_iter": int(self.settings["n_iter"]),
                "n_observations": int(self.n_observations),
                "has_pending": self.pending_x is not None,
                "hypervolume": hypervolume(y, self.runtime.v_ref) if y.shape[0] else 0.0,
                "n_sources": len(self._source_names()),
            }
        if kind == "stop":
            return self._result("stopped", reason=str(payload.get("reason", "client_stop")))
        raise ValueError(f"Unsupported message type: {kind}")


async def _taf_mobo_websocket_handler(websocket: Any, config: TAFMOBOServerRuntimeConfig) -> None:
    """Run one session on ``websocket`` until done, stopped or disconnected."""
    session: TAFMOBOServerSession | None = None
    async for raw in websocket:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise TypeError("message must be a JSON object.")
            if session is None:
                if payload.get("type") != "start":
                    raise ValueError("First message must be type='start'.")
                session = TAFMOBOServerSession.from_start_message(payload, config)
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


async def serve_taf_mobo_websocket(
    host: str = "127.0.0.1",
    port: int = 8772,
    config_path: str | Path = "configs/server_optimizers/taf_mobo_server.yaml",
) -> None:
    """Serve independent TAF-MOBO sessions forever.

    The config (and its sources) is loaded once at start-up; restart the server
    after changing it.
    """
    config = TAFMOBOServerRuntimeConfig.from_yaml_file(config_path)

    async def handler(websocket: Any) -> None:
        await _taf_mobo_websocket_handler(websocket, config)

    async with serve(handler, host, port):
        await asyncio.Future()
