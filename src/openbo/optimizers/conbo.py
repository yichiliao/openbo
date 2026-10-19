"""ConBO: population-informed continual BO across a sequence of users.

Follows Liao et al., *Continual Human-in-the-Loop Optimization* (CHI 2025).
Two classes split the state:

- :class:`ConBOPopulation` lives across users. It holds the population BNN and
  the memory of finished users (one fitted GP each), and can persist both to a
  directory. Finished users are stored in the TAF artifact layout
  (``gp_states/<user>.json`` + ``trajectories/<user>.json``), so a ConBO memory
  directory also works as a TAF source directory.
- :class:`ConBOSequentialOptimizer` runs one user with ``suggest``/``observe``.
  It starts with ``r_u`` random points, then maximizes the mixed EI over a
  candidate grid while adapting a private copy of the BNN online.

After a user, :meth:`ConBOPopulation.finish_user` stores the user's GP and
retrains the BNN by generative replay of every stored user.

Choices the paper leaves open are config fields; see ``docs/plans/conbo.md``.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from openbo.acquisition.conbo import (
    default_incumbents,
    mixed_expected_improvement,
    population_weight,
)
from openbo.models.bnn import BNNConfig, PopulationBNN
from openbo.models.conbo_replay import (
    ReplayTargets,
    replay_inputs,
    replay_targets,
    standardized_variance_threshold,
)
from openbo.models.gp_scratch import GPScratch

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]
Acquisition = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def argmax_on_grid(
    acquisition: Acquisition,
    candidates: NDArray[np.float64],
    refine: bool,
) -> NDArray[np.float64]:
    """Return the best candidate, optionally polished with bounded L-BFGS-B.

    `acquisition` maps `(n, d)` points to `(n,)` values. The polished point is
    kept only if it scores higher than the best grid point.
    """
    values = acquisition(candidates)
    best = int(np.argmax(values))
    x_best = candidates[best].copy()
    if refine and values[best] > 0.0:
        result = minimize(
            lambda z: -float(acquisition(z[None, :])[0]),
            x_best,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * candidates.shape[1],
            options={"maxiter": 50},
        )
        if np.all(np.isfinite(result.x)) and -float(result.fun) > float(values[best]):
            x_best = np.clip(result.x, 0.0, 1.0)
    return x_best[None, :]


@dataclass(frozen=True)
class ConBOConfig:
    """Configuration shared by a :class:`ConBOPopulation` and its users.

    Defaults follow the paper's simulations where it reports them. The replay
    threshold uses ``replay_max_variance`` (raw units, as in the paper) when
    set, otherwise ``replay_variance_fraction`` in standardized units.
    """

    input_dim: int
    # Mixed acquisition schedule (paper: alpha1 = 5, alpha2 = 0.2).
    alpha1: float = 5.0
    alpha2: float = 0.2
    # What the schedule's t counts: every evaluation of the user, or only the BO
    # steps after the random starts. The paper does not say; with many random
    # starts, "evaluations" leaves the population weight at 0 before the first
    # BO step.
    weight_clock: Literal["evaluations", "bo_steps"] = "evaluations"
    # Random starts r_u = max(0, r0 - (u - 1) * dr) for 1-based user u.
    r0: int = 6
    dr: int = 2
    # Candidate grid per axis (paper: 40 x 40); Sobol of this squared above 2D.
    candidate_grid_points: int = 40
    # Polish the best grid candidate with bounded L-BFGS-B on the same
    # acquisition. Not in the paper; without it the grid caps the achievable
    # regret (see benchmark_results/conbo_branin_quick).
    refine_suggestion: bool = True
    # Replay inputs and variance filter.
    replay_grid_points: int = 30
    replay_random_points: int = 50
    replay_filter: bool = True
    replay_max_variance: float | None = None
    replay_variance_fraction: float = 0.05
    # BNN training.
    population_epochs: int = 1200
    online_epochs: int = 15
    online_update: bool = True
    retrain: Literal["reset", "continue"] = "reset"
    hidden_units: int = 100
    n_hidden_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-3
    n_predict_samples: int = 50
    # Use the user GP at all ("ConBO without GP" ablation when False).
    use_user_gp: bool = True
    kernel_type: str = "matern52"
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.r0 < 0 or self.dr < 0:
            raise ValueError("r0 and dr must be non-negative.")
        if self.candidate_grid_points < 2:
            raise ValueError("candidate_grid_points must be at least 2.")
        if self.population_epochs < 0 or self.online_epochs < 0:
            raise ValueError("Epoch counts must be non-negative.")
        if self.replay_max_variance is not None and self.replay_max_variance <= 0.0:
            raise ValueError("replay_max_variance must be positive or None.")
        if self.replay_variance_fraction <= 0.0:
            raise ValueError("replay_variance_fraction must be positive.")
        if self.retrain not in ("reset", "continue"):
            raise ValueError("retrain must be 'reset' or 'continue'.")
        if self.weight_clock not in ("evaluations", "bo_steps"):
            raise ValueError("weight_clock must be 'evaluations' or 'bo_steps'.")
        # Fail early on schedule values population_weight would reject.
        population_weight(1, self.alpha1, self.alpha2)

    def bnn_config(self) -> BNNConfig:
        """Return the BNN configuration derived from this config."""
        return BNNConfig(
            input_dim=self.input_dim,
            hidden_units=self.hidden_units,
            n_hidden_layers=self.n_hidden_layers,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            n_predict_samples=self.n_predict_samples,
            seed=self.seed,
        )


@dataclass
class StoredUser:
    """A finished user kept in the population memory."""

    name: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    gp_state: dict[str, object]


@dataclass
class ReplayReport:
    """Summary of one replay retraining, for logging and tuning."""

    n_users: int
    max_variance: float | None
    n_inputs_kept: int
    n_inputs_total: int
    kept_fraction: list[float]
    excluded_users: list[int]
    seconds: float


def new_user_gp(kernel_type: str, dim: int, seed: int | None) -> GPScratch:
    """Return an unfitted user GP configured like ``bo_scratch``."""
    return GPScratch(
        lengthscale=np.full(dim, 1.5, dtype=np.float64),
        variance=1.0,
        noise=1e-6,
        kernel_type=kernel_type,  # type: ignore[arg-type]
        optimize_hyperparameters=True,
        optimize_noise=False,
        optimizer_seed=seed,
    )


def gp_state_dict(gp: GPScratch) -> dict[str, object]:
    """Serialize fitted GP hyperparameters in the TAF ``gp_state`` format."""
    return {
        "kernel_type": gp.kernel_type,
        "lengthscale": [float(v) for v in np.asarray(gp.lengthscale, dtype=np.float64).ravel()],
        "variance": float(gp.variance),
        "noise": float(gp.noise),
        "standardize_targets": bool(gp.standardize_targets),
        "optimize_noise": bool(gp.optimize_noise),
    }


def gp_from_state(
    state: dict[str, object], x: NDArray[np.float64], y: NDArray[np.float64]
) -> GPScratch:
    """Rebuild a fitted GP with fixed hyperparameters, as the TAF loader does."""
    gp = GPScratch(
        lengthscale=np.asarray(state["lengthscale"], dtype=np.float64),
        variance=float(state["variance"]),  # type: ignore[arg-type]
        noise=float(state["noise"]),  # type: ignore[arg-type]
        kernel_type=str(state["kernel_type"]),  # type: ignore[arg-type]
        optimize_hyperparameters=False,
        standardize_targets=bool(state.get("standardize_targets", True)),
        optimize_noise=bool(state.get("optimize_noise", False)),
    )
    gp.fit(x, y)
    return gp


class ConBOSequentialOptimizer:
    """Ask/tell ConBO optimizer for one user; create it with ``ConBOPopulation.new_user``."""

    def __init__(
        self,
        config: ConBOConfig,
        name: str,
        user_index: int,
        bnn: PopulationBNN | None,
        seed: int,
    ) -> None:
        self.config = config
        self.name = name
        self.user_index = user_index
        # A private copy, so an abandoned session never changes the population model.
        self.bnn = copy.deepcopy(bnn)
        self.rng = np.random.default_rng(seed)
        self.gp = new_user_gp(config.kernel_type, config.input_dim, seed)
        self.candidates = replay_inputs(
            config.input_dim, config.candidate_grid_points, n_random=0, seed=seed
        )
        self.n_random = max(0, config.r0 - (user_index - 1) * config.dr)
        if self.bnn is None:
            self.n_random = max(1, self.n_random)
        self.x_obs = np.empty((0, config.input_dim), dtype=np.float64)
        self.y_obs = np.empty((0,), dtype=np.float64)
        self.weight_history: list[float | None] = []
        self._gp_stale = False

    @property
    def n_observations(self) -> int:
        """Number of evaluations so far for this user."""
        return int(self.y_obs.shape[0])

    def current_weight(self) -> float:
        """Population weight for the next suggestion.

        ``t`` is 1-based: evaluations so far + 1, or BO steps so far + 1 when
        ``config.weight_clock == "bo_steps"``.
        """
        if self.bnn is None:
            return 0.0
        if not self.config.use_user_gp or self.n_observations == 0:
            return 1.0
        t = self.n_observations + 1
        if self.config.weight_clock == "bo_steps":
            t -= min(self.n_random, self.n_observations)
        return population_weight(t, self.config.alpha1, self.config.alpha2)

    def suggest(self) -> NDArray[np.float64]:
        """Return the next point to evaluate, shape `(1, d)`."""
        if self.n_observations < self.n_random:
            self.weight_history.append(None)
            return self.rng.random((1, self.config.input_dim))

        weight = self.current_weight()
        if weight < 1.0 and self._gp_stale:
            self.gp.fit(self.x_obs, self.y_obs)
            self._gp_stale = False
        pop_mean = self.bnn.predict(self.candidates)[0] if weight > 0.0 else np.empty(0)
        best_pop, best_user = default_incumbents(self.y_obs, pop_mean)

        def acquisition(x: NDArray[np.float64]) -> NDArray[np.float64]:
            pop_pred = self.bnn.predict(x) if weight > 0.0 else None
            user_pred = self.gp.posterior(x) if weight < 1.0 else None
            return mixed_expected_improvement(weight, pop_pred, user_pred, best_pop, best_user)

        self.weight_history.append(weight)
        return argmax_on_grid(acquisition, self.candidates, self.config.refine_suggestion)

    def observe(self, x_new: NDArray[np.float64], y_new: NDArray[np.float64]) -> None:
        """Record evaluations and adapt the BNN online; the user GP is refit lazily."""
        x_new = np.asarray(x_new, dtype=np.float64)
        y_new = np.asarray(y_new, dtype=np.float64)
        if x_new.ndim != 2 or x_new.shape[1] != self.config.input_dim:
            raise ValueError(
                f"x_new must have shape (n, {self.config.input_dim}), got {x_new.shape}."
            )
        if y_new.ndim != 1 or y_new.shape[0] != x_new.shape[0]:
            raise ValueError("y_new must have shape (n,) and match x_new rows.")
        if not np.all(np.isfinite(y_new)):
            raise ValueError("y_new must be finite.")
        self.x_obs = np.vstack([self.x_obs, x_new])
        self.y_obs = np.concatenate([self.y_obs, y_new])
        self._gp_stale = True
        if self.bnn is not None and self.config.online_update and self.config.online_epochs > 0:
            self.bnn.fit_nll(self.x_obs, self.y_obs, self.config.online_epochs)


class ConBOPopulation:
    """Population model and memory of finished users, optionally persisted to a directory."""

    def __init__(self, config: ConBOConfig, memory_dir: str | Path | None = None) -> None:
        self.config = config
        self.memory_dir = None if memory_dir is None else Path(memory_dir)
        self.users: list[StoredUser] = []
        self.bnn: PopulationBNN | None = None
        self.replay_reports: list[ReplayReport] = []

    @property
    def n_users(self) -> int:
        """Number of finished users in memory."""
        return len(self.users)

    def new_user(self, name: str | None = None) -> ConBOSequentialOptimizer:
        """Start the next user; it is user ``n_users + 1`` (1-based)."""
        index = self.n_users + 1
        user_name = name or f"user_{index - 1:03d}"
        if any(u.name == user_name for u in self.users):
            raise ValueError(f"User '{user_name}' is already in memory.")
        seed = self.config.seed + 1009 * index
        return ConBOSequentialOptimizer(self.config, user_name, index, self.bnn, seed)

    def user_gps(self) -> list[GPScratch]:
        """Rebuild every stored user's GP from its hyperparameters and data."""
        return [gp_from_state(u.gp_state, u.x, u.y) for u in self.users]

    def finish_user(self, optimizer: ConBOSequentialOptimizer) -> ReplayReport:
        """Store a finished user, retrain the population BNN by replay and persist."""
        if optimizer.n_observations == 0:
            raise ValueError("Cannot store a user without observations.")
        if optimizer.user_index != self.n_users + 1:
            raise ValueError("Users must be finished in the order they were started.")
        gp = new_user_gp(self.config.kernel_type, self.config.input_dim, self.config.seed)
        gp.fit(optimizer.x_obs, optimizer.y_obs)
        self.users.append(
            StoredUser(optimizer.name, optimizer.x_obs.copy(), optimizer.y_obs.copy(), gp_state_dict(gp))
        )
        report = self._retrain()
        self.replay_reports.append(report)
        if self.memory_dir is not None:
            self.save()
        return report

    def _retrain(self) -> ReplayReport:
        """Retrain the BNN on replayed targets from every stored user."""
        start = time.perf_counter()
        gps = self.user_gps()
        x = replay_inputs(
            self.config.input_dim,
            self.config.replay_grid_points,
            self.config.replay_random_points,
            seed=self.config.seed,
        )
        if not self.config.replay_filter:
            threshold = None
        elif self.config.replay_max_variance is not None:
            threshold = self.config.replay_max_variance
        else:
            threshold = standardized_variance_threshold(
                gps, x, self.config.replay_variance_fraction
            )
        targets: ReplayTargets = replay_targets(gps, x, threshold)
        if targets.x.shape[0] > 0:
            if self.bnn is None:
                self.bnn = PopulationBNN(self.config.bnn_config())
            elif self.config.retrain == "reset":
                self.bnn.reset_parameters()
            self.bnn.fit_replay(
                targets.x, targets.mean, targets.var, self.config.population_epochs
            )
        return ReplayReport(
            n_users=self.n_users,
            max_variance=threshold,
            n_inputs_kept=int(targets.x.shape[0]),
            n_inputs_total=int(x.shape[0]),
            kept_fraction=[float(v) for v in targets.kept_fraction],
            excluded_users=list(targets.excluded_users),
            seconds=time.perf_counter() - start,
        )

    def save(self, memory_dir: str | Path | None = None) -> None:
        """Write users (TAF layout), the BNN and the config to a directory."""
        root = Path(memory_dir) if memory_dir is not None else self.memory_dir
        if root is None:
            raise ValueError("No memory_dir given.")
        (root / "gp_states").mkdir(parents=True, exist_ok=True)
        (root / "trajectories").mkdir(parents=True, exist_ok=True)
        for user in self.users:
            (root / "gp_states" / f"{user.name}.json").write_text(
                json.dumps({"gp_state": user.gp_state}, indent=2), encoding="utf-8"
            )
            (root / "trajectories" / f"{user.name}.json").write_text(
                json.dumps(
                    {
                        "x_values": user.x.tolist(),
                        "y_values": user.y.tolist(),
                        "best_so_far": np.maximum.accumulate(user.y).tolist(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if self.bnn is not None:
            self.bnn.save(root / "population_bnn.pt")
        (root / "conbo_state.json").write_text(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "user_order": [u.name for u in self.users],
                    "replay_reports": [asdict(r) for r in self.replay_reports],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, memory_dir: str | Path) -> ConBOPopulation:
        """Load a population written by :meth:`save`."""
        root = Path(memory_dir)
        state = json.loads((root / "conbo_state.json").read_text(encoding="utf-8"))
        population = cls(ConBOConfig(**state["config"]), root)
        for name in state["user_order"]:
            gp_payload = json.loads((root / "gp_states" / f"{name}.json").read_text(encoding="utf-8"))
            traj = json.loads((root / "trajectories" / f"{name}.json").read_text(encoding="utf-8"))
            population.users.append(
                StoredUser(
                    name,
                    np.asarray(traj["x_values"], dtype=np.float64),
                    np.asarray(traj["y_values"], dtype=np.float64),
                    gp_payload["gp_state"],
                )
            )
        population.replay_reports = [ReplayReport(**r) for r in state["replay_reports"]]
        bnn_path = root / "population_bnn.pt"
        if bnn_path.exists():
            population.bnn = PopulationBNN.load(bnn_path)
        return population


@dataclass
class ConBOUserResult:
    """One user's trajectory within a ConBO sequence."""

    name: str
    x_obs: NDArray[np.float64]
    y_obs: NDArray[np.float64]
    best_so_far: NDArray[np.float64]
    optimum: float | None
    n_random: int
    weights: list[float | None]
    seconds: float
    replay: ReplayReport


@dataclass
class ConBORunResult:
    """Results of :func:`run_conbo_sequence`, one entry per user in order."""

    users: list[ConBOUserResult] = field(default_factory=list)


def run_conbo_sequence(
    objectives: list[Objective],
    config: ConBOConfig,
    n_iter_per_user: int,
    *,
    names: list[str] | None = None,
    optima: list[float | None] | None = None,
    memory_dir: str | Path | None = None,
) -> ConBORunResult:
    """Run ConBO over users in order, ``n_iter_per_user`` evaluations each."""
    if n_iter_per_user <= 0:
        raise ValueError("n_iter_per_user must be positive.")
    if names is not None and len(names) != len(objectives):
        raise ValueError("names must match objectives.")
    if optima is not None and len(optima) != len(objectives):
        raise ValueError("optima must match objectives.")
    population = ConBOPopulation(config, memory_dir)
    result = ConBORunResult()
    for i, objective in enumerate(objectives):
        start = time.perf_counter()
        optimizer = population.new_user(None if names is None else names[i])
        for _ in range(n_iter_per_user):
            x = optimizer.suggest()
            optimizer.observe(x, np.asarray(objective(x), dtype=np.float64))
        replay = population.finish_user(optimizer)
        result.users.append(
            ConBOUserResult(
                name=optimizer.name,
                x_obs=optimizer.x_obs,
                y_obs=optimizer.y_obs,
                best_so_far=np.maximum.accumulate(optimizer.y_obs),
                optimum=None if optima is None else optima[i],
                n_random=min(optimizer.n_random, n_iter_per_user),
                weights=optimizer.weight_history,
                seconds=time.perf_counter() - start,
                replay=replay,
            )
        )
    return result
