"""TAF+ optimizer: transfer from earlier users with objective weights and decay.

Follows Liao et al. (CHI 2024) on top of the original TAF (Wistuba et al.,
2018); where the two disagree, the original wins (``docs/plans/taf_plus.md``).

Two classes split the state:

- :class:`TAFPlusPopulation` holds the earlier users ("population models"):
  each user's data and one fitted GP per objective. It saves to and loads from
  a directory in the TAF layout, extended to several objectives.
- :class:`TAFPlusSequentialOptimizer` runs one new user with
  ``suggest``/``observe``. Every point after the first maximizes
  :func:`taf_plus_acquisition` over a Sobol pool.

The first point maximizes the population's product-of-experts mean (or is the
domain centre, by configuration). Objectives are maximized, and ``y`` has one
column per objective. The population is read-only during a session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import qmc

from openbo.acquisition.taf_plus import (
    TAF_PLUS_WEIGHT_MODES,
    decay_factor,
    population_mean,
    source_references,
    taf_plus_acquisition,
    taf_r_weights,
)
from openbo.models.gp_scratch import GPScratch
from openbo.optimizers.conbo import argmax_on_grid, gp_from_state, gp_state_dict

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class TAFPlusConfig:
    """Configuration of one TAF+ adaptation run.

    ``decay=False`` keeps the population weights at full strength (``delta = 1``),
    which is the paper's "TAF (multi-objective)" baseline; with one objective it
    is the original TAF.
    """

    input_dim: int
    objective_weights: tuple[float, ...]
    # Decay delta(t) (CHI eq. 6); the paper's study found d1 = 2, d2 = 0.3.
    decay: bool = True
    d1: int = 2
    d2: float = 0.3
    # "poe": variance-based TAF-PoE weights (what the CHI paper uses).
    # "taf_r": ranking-based TAF-R weights with Epanechnikov bandwidth rho.
    weight_mode: str = "poe"
    rho: float = 1.0
    # SE-ARD ("rbf") follows the original; "matern52" is the CHI paper's choice.
    kernel_type: Literal["rbf", "matern52"] = "rbf"
    n_candidates: int = 1024
    # Draw a fresh scrambled Sobol pool at every step, as in the paper's study
    # ("at each iteration, we generated 1024 parameter settings as candidates").
    # With one fixed pool, a run can never beat the pool's best point.
    resample_candidates: bool = True
    # Polish the best Sobol candidate with L-BFGS-B. False is the paper's pure pool search.
    refine_suggestion: bool = True
    # First point, before the user has data (neither paper defines it):
    # "population" maximizes the population's product-of-experts mean
    # (original eq. 37, sources only); "centre" is the domain centre. With an
    # empty population, "population" falls back to the centre.
    first_point: Literal["population", "centre"] = "population"
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        alpha = np.asarray(self.objective_weights, dtype=np.float64)
        if alpha.ndim != 1 or alpha.size == 0:
            raise ValueError("objective_weights must be a non-empty sequence.")
        if np.any(alpha < 0.0) or abs(float(alpha.sum()) - 1.0) > 1e-9:
            raise ValueError("objective_weights must be non-negative and sum to 1.")
        decay_factor(1, self.d1, self.d2)
        if self.weight_mode not in TAF_PLUS_WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {TAF_PLUS_WEIGHT_MODES}.")
        if self.rho <= 0.0:
            raise ValueError("rho must be positive.")
        if self.kernel_type not in ("rbf", "matern52"):
            raise ValueError("kernel_type must be 'rbf' or 'matern52'.")
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be positive.")
        if self.first_point not in ("population", "centre"):
            raise ValueError("first_point must be 'population' or 'centre'.")

    @property
    def n_objectives(self) -> int:
        """Number of objectives, from the objective weights."""
        return len(self.objective_weights)


def new_objective_gp(kernel_type: str, dim: int, seed: int | None) -> GPScratch:
    """Return an unfitted GP for one objective, with fitted noise."""
    return GPScratch(
        lengthscale=np.full(dim, 1.5, dtype=np.float64),
        variance=1.0,
        noise=1e-4,
        kernel_type=kernel_type,  # type: ignore[arg-type]
        optimize_hyperparameters=True,
        optimize_noise=True,
        optimizer_seed=seed,
    )


def _check_xy(
    x: NDArray[np.float64], y: NDArray[np.float64], dim: int, n_obj: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != dim:
        raise ValueError(f"x must have shape (n, {dim}), got {x.shape}.")
    if y.ndim == 1 and n_obj == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape != (x.shape[0], n_obj):
        raise ValueError(f"y must have shape ({x.shape[0]}, {n_obj}), got {y.shape}.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite.")
    return x, y


@dataclass
class PopulationUser:
    """One earlier user: their data and one fitted GP state per objective."""

    name: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    gp_states: list[dict[str, object]]
    gps: list[GPScratch] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.gps:
            self.gps = [
                gp_from_state(state, self.x, self.y[:, obj])
                for obj, state in enumerate(self.gp_states)
            ]


class TAFPlusPopulation:
    """Earlier users of a TAF+ population, optionally persisted to a directory.

    Layout: ``gp_states/<user>.json`` holds ``{"gp_states": [one per objective]}``,
    ``trajectories/<user>.json`` holds ``x_values`` and ``y_values`` of shape
    ``(n, L)``, and ``taf_plus_population.json`` holds the user order and settings.
    """

    def __init__(self, input_dim: int, n_objectives: int, kernel_type: str = "rbf") -> None:
        if input_dim <= 0 or n_objectives <= 0:
            raise ValueError("input_dim and n_objectives must be positive.")
        self.input_dim = input_dim
        self.n_objectives = n_objectives
        self.kernel_type = kernel_type
        self.users: list[PopulationUser] = []

    @property
    def n_users(self) -> int:
        """Number of earlier users."""
        return len(self.users)

    def add_user(
        self,
        name: str,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        seed: int | None = 0,
    ) -> PopulationUser:
        """Fit one GP per objective to a user's data and add the user."""
        if any(u.name == name for u in self.users):
            raise ValueError(f"User '{name}' is already in the population.")
        x, y = _check_xy(x, y, self.input_dim, self.n_objectives)
        gps = []
        for obj in range(self.n_objectives):
            gp = new_objective_gp(self.kernel_type, self.input_dim, seed)
            gp.fit(x, y[:, obj])
            gps.append(gp)
        user = PopulationUser(name, x.copy(), y.copy(), [gp_state_dict(gp) for gp in gps], gps)
        self.users.append(user)
        return user

    def subset(self, exclude: str) -> TAFPlusPopulation:
        """Return a population without one user (for leave-one-out simulations)."""
        other = TAFPlusPopulation(self.input_dim, self.n_objectives, self.kernel_type)
        other.users = [u for u in self.users if u.name != exclude]
        if len(other.users) == len(self.users):
            raise ValueError(f"User '{exclude}' is not in the population.")
        return other

    def save(self, directory: str | Path) -> None:
        """Write the population to a directory."""
        root = Path(directory)
        (root / "gp_states").mkdir(parents=True, exist_ok=True)
        (root / "trajectories").mkdir(parents=True, exist_ok=True)
        for user in self.users:
            (root / "gp_states" / f"{user.name}.json").write_text(
                json.dumps({"gp_states": user.gp_states}, indent=2), encoding="utf-8"
            )
            (root / "trajectories" / f"{user.name}.json").write_text(
                json.dumps({"x_values": user.x.tolist(), "y_values": user.y.tolist()}, indent=2),
                encoding="utf-8",
            )
        (root / "taf_plus_population.json").write_text(
            json.dumps(
                {
                    "input_dim": self.input_dim,
                    "n_objectives": self.n_objectives,
                    "kernel_type": self.kernel_type,
                    "user_order": [u.name for u in self.users],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> TAFPlusPopulation:
        """Load a population written by :meth:`save`. GPs keep their saved hyperparameters."""
        root = Path(directory)
        state = json.loads((root / "taf_plus_population.json").read_text(encoding="utf-8"))
        population = cls(int(state["input_dim"]), int(state["n_objectives"]), str(state["kernel_type"]))
        for name in state["user_order"]:
            gp_payload = json.loads((root / "gp_states" / f"{name}.json").read_text(encoding="utf-8"))
            traj = json.loads((root / "trajectories" / f"{name}.json").read_text(encoding="utf-8"))
            x, y = _check_xy(
                np.asarray(traj["x_values"], dtype=np.float64),
                np.asarray(traj["y_values"], dtype=np.float64),
                population.input_dim,
                population.n_objectives,
            )
            if len(gp_payload["gp_states"]) != population.n_objectives:
                raise ValueError(f"User '{name}' has the wrong number of GP states.")
            population.users.append(PopulationUser(name, x, y, gp_payload["gp_states"]))
        return population


def _predict_stack(
    gps_per_model: list[list[GPScratch]], x: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return means and variances of shape ``(n_models, n, L)``."""
    n_models = len(gps_per_model)
    n_obj = len(gps_per_model[0]) if n_models else 0
    mean = np.empty((n_models, x.shape[0], n_obj), dtype=np.float64)
    var = np.empty_like(mean)
    for m, gps in enumerate(gps_per_model):
        for obj, gp in enumerate(gps):
            mean[m, :, obj], var[m, :, obj] = gp.posterior(x)
    return mean, var


@dataclass
class TAFPlusRunResult:
    """One adaptation run: observations, weighted-sum values and the decay used per step."""

    x_obs: NDArray[np.float64]
    y_obs: NDArray[np.float64]
    weighted_y: NDArray[np.float64]
    best_so_far: NDArray[np.float64]
    decay_history: list[float | None]
    objective_weights: tuple[float, ...]


class TAFPlusSequentialOptimizer:
    """Ask/tell TAF+ for one new user."""

    def __init__(self, config: TAFPlusConfig, population: TAFPlusPopulation) -> None:
        if population.input_dim != config.input_dim:
            raise ValueError("population.input_dim does not match config.input_dim.")
        if population.n_objectives != config.n_objectives:
            raise ValueError("population.n_objectives does not match the objective weights.")
        self.config = config
        self.population = population
        self.alpha = np.asarray(config.objective_weights, dtype=np.float64)
        self.candidates = self._draw_pool(0)
        self._pool_step = 0
        self.gps = [
            new_objective_gp(config.kernel_type, config.input_dim, config.seed + 1009 * obj)
            for obj in range(config.n_objectives)
        ]
        self.x_obs = np.empty((0, config.input_dim), dtype=np.float64)
        self.y_obs = np.empty((0, config.n_objectives), dtype=np.float64)
        self.decay_history: list[float | None] = []
        self._gp_stale = False
        # The population GPs never change during a run, so their predictions on
        # the candidate pool are computed once per pool, on first use.
        self._pool_sources: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None

    def _draw_pool(self, step: int) -> NDArray[np.float64]:
        sampler = qmc.Sobol(d=self.config.input_dim, scramble=True, seed=self.config.seed + 1009 * step)
        return sampler.random(self.config.n_candidates).astype(np.float64)

    def _refresh_pool(self) -> None:
        """Draw this step's candidate pool when resampling, and drop stale cached predictions."""
        step = self.n_observations
        if self.config.resample_candidates and step != self._pool_step:
            self.candidates = self._draw_pool(step)
            self._pool_step = step
            self._pool_sources = None

    @property
    def n_observations(self) -> int:
        """Number of evaluations so far for this user."""
        return int(self.y_obs.shape[0])

    def current_decay(self) -> float:
        """Decay ``delta(t)`` for the next suggestion; ``t`` counts evaluations from 1."""
        if not self.config.decay:
            return 1.0
        return decay_factor(self.n_observations + 1, self.config.d1, self.config.d2)

    def suggest(self) -> NDArray[np.float64]:
        """Return the next point to evaluate, shape ``(1, d)``."""
        self._refresh_pool()
        if self.n_observations == 0:
            # Neither paper defines TAF before the user has data (plan, open question 3).
            self.decay_history.append(None)
            if self.config.first_point == "centre" or self.population.n_users == 0:
                return np.full((1, self.config.input_dim), 0.5, dtype=np.float64)
            return self._population_first_point()

        if self._gp_stale:
            for obj, gp in enumerate(self.gps):
                # One point says nothing about the hyperparameters (fitting them
                # collapses the signal variance), so keep the prior until two.
                gp.optimize_hyperparameters = self.n_observations >= 2
                gp.fit(self.x_obs, self.y_obs[:, obj])
                if gp.y_std < 1e-9:
                    # GPScratch standardizes one point (or all-equal values) with a
                    # std of 1e-12, which shrinks the predictive variance to ~1e-24
                    # and would give the target an overwhelming TAF-PoE weight.
                    # Its normalized targets are all 0, so a unit scale is exact.
                    gp.y_std = 1.0
            self._gp_stale = False

        decay = self.current_decay()
        source_gps = [u.gps for u in self.population.users] if decay > 0.0 else []
        target_best = np.max(self.y_obs, axis=0)
        mean_at_target, _ = _predict_stack(source_gps, self.x_obs)
        n_obj = self.config.n_objectives
        references = (
            source_references(mean_at_target) if source_gps else np.zeros((0, n_obj))
        )
        ranking = None
        if self.config.weight_mode == "taf_r":
            ranking = taf_r_weights(self.y_obs, mean_at_target.reshape(-1, *self.y_obs.shape), self.config.rho)

        if source_gps and self._pool_sources is None:
            self._pool_sources = _predict_stack(source_gps, self.candidates)

        def acquisition(x: NDArray[np.float64]) -> NDArray[np.float64]:
            target_mean, target_var = _predict_stack([self.gps], x)
            if source_gps and x is self.candidates:
                assert self._pool_sources is not None
                source_mean, source_var = self._pool_sources
            else:
                source_mean, source_var = _predict_stack(source_gps, x)
            return taf_plus_acquisition(
                target_mean[0],
                target_var[0],
                target_best,
                source_mean.reshape(-1, x.shape[0], n_obj),
                source_var.reshape(-1, x.shape[0], n_obj),
                references,
                self.alpha,
                decay=decay,
                weight_mode=self.config.weight_mode,
                ranking_weights=ranking,
            )

        self.decay_history.append(decay)
        return argmax_on_grid(acquisition, self.candidates, self.config.refine_suggestion)

    def _population_first_point(self) -> NDArray[np.float64]:
        """Maximize the population's product-of-experts mean of the weighted objective."""
        source_gps = [u.gps for u in self.population.users]
        if self._pool_sources is None:
            self._pool_sources = _predict_stack(source_gps, self.candidates)

        def mean(x: NDArray[np.float64]) -> NDArray[np.float64]:
            pred = self._pool_sources if x is self.candidates else _predict_stack(source_gps, x)
            assert pred is not None
            return population_mean(pred[0], pred[1], self.alpha, self.config.weight_mode)

        # The mean can be negative, so polish whenever asked, not only when the best value is positive.
        values = mean(self.candidates)
        x_best = self.candidates[int(np.argmax(values))][None, :]
        if not self.config.refine_suggestion:
            return x_best.copy()
        result = minimize(
            lambda z: -float(mean(z[None, :])[0]),
            x_best[0],
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * self.config.input_dim,
            options={"maxiter": 50},
        )
        if np.all(np.isfinite(result.x)) and -float(result.fun) > float(np.max(values)):
            return np.clip(result.x, 0.0, 1.0)[None, :]
        return x_best.copy()

    def observe(self, x_new: NDArray[np.float64], y_new: NDArray[np.float64]) -> None:
        """Record evaluations; ``y_new`` has one column per objective. GPs refit lazily."""
        x_new, y_new = _check_xy(x_new, y_new, self.config.input_dim, self.config.n_objectives)
        self.x_obs = np.vstack([self.x_obs, x_new])
        self.y_obs = np.vstack([self.y_obs, y_new])
        self._gp_stale = True

    def result(self) -> TAFPlusRunResult:
        """Return the run so far, with the weighted-sum objective ``y @ alpha``."""
        weighted = (self.y_obs @ self.alpha).astype(np.float64)
        best = np.maximum.accumulate(weighted) if weighted.size else weighted
        return TAFPlusRunResult(
            x_obs=self.x_obs.copy(),
            y_obs=self.y_obs.copy(),
            weighted_y=weighted,
            best_so_far=best,
            decay_history=list(self.decay_history),
            objective_weights=tuple(self.config.objective_weights),
        )


def run_bo_taf_plus(
    objective: Objective,
    config: TAFPlusConfig,
    population: TAFPlusPopulation,
    n_iter: int = 10,
) -> TAFPlusRunResult:
    """Run TAF+ for ``n_iter`` evaluations (the centre point included).

    ``objective`` maps ``(n, d)`` inputs to ``(n, L)`` values, all maximized.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    optimizer = TAFPlusSequentialOptimizer(config, population)
    for _ in range(n_iter):
        x = optimizer.suggest()
        y = np.asarray(objective(x), dtype=np.float64).reshape(1, config.n_objectives)
        optimizer.observe(x, y)
    return optimizer.result()

