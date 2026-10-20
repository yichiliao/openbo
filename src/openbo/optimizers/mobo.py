"""Multi-objective BO with correlated objectives (Shah and Ghahramani, 2016).

A :class:`~openbo.models.multi_output_gp.MultiOutputGP` models the objectives
jointly, and the next point maximizes CEIPV (correlated expected increase in
Pareto volume) or its independent-objective counterpart IEIPV. All objectives
are maximized; observations ``y`` have shape ``(n, L)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import qmc

from openbo.acquisition.mobo import (
    EIPVForm,
    ceipv_terms,
    ieipv,
    log_ceipv,
)
from openbo.benchmarks.hypervolume import hypervolume, nondominated_cells, pareto_mask
from openbo.models.multi_output_gp import MultiOutputGP

MultiObjective = Callable[[NDArray[np.float64]], NDArray[np.float64]]
AcquisitionFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]

# log-acquisition floor, so L-BFGS-B never sees -inf.
_LOG_FLOOR = -1e4


@dataclass
class MOBORunResult:
    """Observations, hypervolume trajectory and Pareto set of a MOBO run."""

    x_obs: NDArray[np.float64]
    y_obs: NDArray[np.float64]
    hv_history: NDArray[np.float64]
    pareto_x: NDArray[np.float64]
    pareto_y: NDArray[np.float64]
    final_model_state: dict[str, object] = field(default_factory=dict)


@dataclass
class MOBOConfig:
    """Configuration for :class:`MOBOSequentialOptimizer`.

    ``v_ref`` and ``w_ref`` bound the objective space (lower reference point
    and finite upper corner). ``n_rescore``, if set, scores the Sobol pool with
    cheap IEIPV and rescores only its best ``n_rescore`` points with CEIPV,
    which keeps three-objective runs affordable.

    ``hyperparameters="slice"`` follows the paper: each iteration draws
    ``n_hyper_samples`` hyperparameter vectors by slice sampling (continuing
    the chain from the previous iteration, ``slice_burn_in`` sweeps then one
    sample every ``slice_thin`` sweeps) and averages the acquisition over them.
    ``"map"`` uses the single MAP estimate.
    """

    bounds: list[tuple[float, float]]
    v_ref: Sequence[float]
    w_ref: Sequence[float]
    n_init: int = 5
    model: Literal["independent", "mt", "slf"] = "slf"
    acquisition: Literal["ceipv", "ieipv"] = "ceipv"
    eipv_form: EIPVForm = "exact"
    n_candidates: int = 1024
    n_starts: int = 4
    n_rescore: int | None = None
    optimizer_maxiter: int = 50
    hyperparameters: Literal["map", "slice"] = "map"
    n_hyper_samples: int = 10
    slice_burn_in: int = 10
    slice_thin: int = 2
    seed: int | None = 0


def _sobol_in_bounds(
    lower: NDArray[np.float64], upper: NDArray[np.float64], n: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Return ``n`` scrambled Sobol points in ``[lower, upper]``."""
    engine = qmc.Sobol(d=lower.shape[0], scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
    return lower + (upper - lower) * engine.random(n)


def _value_and_grad(
    fn: AcquisitionFn,
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    step: float = 1e-6,
) -> tuple[float, NDArray[np.float64]]:
    """Return ``fn(x)`` and its forward-difference gradient from one batched call.

    Steps go backward where a forward step would leave the bounds.
    """
    d = x.shape[0]
    h = np.where(x + step <= upper, step, -step)
    points = np.vstack([x, x + np.diag(h)])
    points = np.clip(points, lower, upper)
    values = fn(points)
    return float(values[0]), (values[1:] - values[0]) / h[:d]


def _maximize(
    fn: AcquisitionFn,
    starts: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    maxiter: int,
) -> tuple[NDArray[np.float64], float]:
    """Maximize ``fn`` with L-BFGS-B from each start; return the best point and value."""
    best_x = starts[0].copy()
    best_value = float(fn(starts[:1])[0])

    def negative(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        value, grad = _value_and_grad(fn, x, lower, upper)
        return -value, -grad

    for x0 in starts:
        res = minimize(
            negative,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options={"maxiter": maxiter},
        )
        x_opt = np.clip(np.asarray(res.x, dtype=np.float64), lower, upper)
        value = float(fn(x_opt[None])[0])
        if value > best_value:
            best_x, best_value = x_opt, value
    return best_x, best_value


class MOBOSequentialOptimizer:
    """Ask/tell multi-objective BO with a multi-output GP and (C/I)EIPV."""

    def __init__(self, config: MOBOConfig) -> None:
        if config.acquisition not in ("ceipv", "ieipv"):
            raise ValueError(
                f"Unknown acquisition {config.acquisition!r}; expected 'ceipv' or 'ieipv'."
            )
        if config.hyperparameters not in ("map", "slice"):
            raise ValueError(
                f"Unknown hyperparameters {config.hyperparameters!r}; expected 'map' or 'slice'."
            )
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.d = len(config.bounds)
        self.lower = np.array([b[0] for b in config.bounds], dtype=np.float64)
        self.upper = np.array([b[1] for b in config.bounds], dtype=np.float64)
        self.v_ref = np.asarray(config.v_ref, dtype=np.float64)
        self.w_ref = np.asarray(config.w_ref, dtype=np.float64)
        if self.v_ref.shape != self.w_ref.shape or self.v_ref.ndim != 1:
            raise ValueError("v_ref and w_ref must be vectors of the same length.")
        self.n_objectives = self.v_ref.shape[0]

        self.x_obs = np.empty((0, self.d), dtype=np.float64)
        self.y_obs = np.empty((0, self.n_objectives), dtype=np.float64)
        self.hv_history: list[float] = []
        self.gp = MultiOutputGP(mode=config.model, optimizer_seed=config.seed)
        self._chain_state: NDArray[np.float64] | None = None

    def bootstrap(self, objective: MultiObjective) -> None:
        """Evaluate ``n_init`` uniform random points."""
        if self.config.n_init <= 0:
            return
        x_init = self.rng.uniform(self.lower, self.upper, size=(self.config.n_init, self.d))
        self.observe(x_init, np.asarray(objective(x_init), dtype=np.float64))

    def _posterior_models(self) -> list[MultiOutputGP]:
        """Fit the GP and return the models to average the acquisition over."""
        self.gp.fit(self.x_obs, self.y_obs)
        if self.config.hyperparameters == "map":
            return [self.gp]
        start = self._chain_state
        if start is not None and start.shape != self.gp._theta.shape:
            start = None
        thetas = self.gp.sample_hyperparameters(
            self.config.n_hyper_samples,
            self.rng,
            start=start,
            burn_in=self.config.slice_burn_in,
            thin=self.config.slice_thin,
        )
        self._chain_state = thetas[-1]
        return [self.gp.with_hyperparameters(theta) for theta in thetas]

    def _log_acquisition(self, models: list[MultiOutputGP]) -> tuple[AcquisitionFn, AcquisitionFn]:
        """Return the cheap pool-scoring and the main log-acquisition functions.

        With several models (hyperparameter samples) each returns the log of
        the acquisition averaged over them.
        """
        front = self.y_obs[pareto_mask(self.y_obs)]
        lower, upper = nondominated_cells(front, self.v_ref, self.w_ref)
        form = self.config.eipv_form
        log_n_models = np.log(len(models))

        def averaged(log_values: Callable[[MultiOutputGP, NDArray[np.float64]], NDArray[np.float64]]) -> AcquisitionFn:
            """Average ``exp(log_values)`` over the models, in log space."""

            def fn(x: NDArray[np.float64]) -> NDArray[np.float64]:
                stacked = np.stack([log_values(gp, x) for gp in models])
                return np.maximum(logsumexp(stacked, axis=0) - log_n_models, _LOG_FLOOR)

            return fn

        def log_ieipv(gp: MultiOutputGP, x: NDArray[np.float64]) -> NDArray[np.float64]:
            mean, cov = gp.posterior(x)
            with np.errstate(divide="ignore"):
                return np.log(ieipv(mean, cov, lower, upper, form=form))

        independent = averaged(log_ieipv)
        if self.config.acquisition == "ieipv":
            return independent, independent

        terms = ceipv_terms(front, self.v_ref, self.w_ref, form=form)

        def log_correlated(gp: MultiOutputGP, x: NDArray[np.float64]) -> NDArray[np.float64]:
            mean, cov = gp.posterior(x)
            return log_ceipv(mean, cov, terms)

        correlated = averaged(log_correlated)
        return (independent if self.config.n_rescore else correlated), correlated

    def suggest(self) -> NDArray[np.float64]:
        """Return the next point to evaluate, shape ``(1, d)``."""
        if self.x_obs.shape[0] == 0:
            raise ValueError("Cannot suggest without observations. Call bootstrap() first.")
        screen, acquisition = self._log_acquisition(self._posterior_models())

        pool = _sobol_in_bounds(self.lower, self.upper, self.config.n_candidates, self.rng)
        scores = screen(pool)
        if self.config.n_rescore and self.config.acquisition == "ceipv":
            top = np.argsort(scores)[-self.config.n_rescore :]
            pool, scores = pool[top], acquisition(pool[top])
        n_starts = max(1, min(self.config.n_starts, pool.shape[0]))
        starts = pool[np.argsort(scores)[-n_starts:][::-1]]
        best_x, _ = _maximize(
            acquisition, starts, self.lower, self.upper, self.config.optimizer_maxiter
        )
        return best_x[None, :]

    def observe(self, x_new: NDArray[np.float64], y_new: NDArray[np.float64]) -> None:
        """Add observations ``x_new`` (n, d) with objective values ``y_new`` (n, L)."""
        x_new = np.asarray(x_new, dtype=np.float64)
        y_new = np.asarray(y_new, dtype=np.float64)
        if x_new.ndim != 2 or x_new.shape[1] != self.d:
            raise ValueError(f"x_new must have shape (n, {self.d}), got {x_new.shape}.")
        if y_new.shape != (x_new.shape[0], self.n_objectives):
            raise ValueError(
                f"y_new must have shape ({x_new.shape[0]}, {self.n_objectives}), got {y_new.shape}."
            )
        if x_new.shape[0] == 0 or not np.all(np.isfinite(x_new)) or not np.all(np.isfinite(y_new)):
            raise ValueError("observations must be non-empty and finite.")
        self.x_obs = np.vstack([self.x_obs, x_new])
        self.y_obs = np.vstack([self.y_obs, y_new])
        self.hv_history.append(hypervolume(self.y_obs, self.v_ref))

    def result(self) -> MOBORunResult:
        """Return observations, the hypervolume history and the fitted model state."""
        if self.x_obs.shape[0] == 0:
            raise ValueError("MOBO finished without any observations.")
        self.gp.fit(self.x_obs, self.y_obs)
        mask = pareto_mask(self.y_obs)
        state: dict[str, object] = {
            "model": self.config.model,
            "acquisition": self.config.acquisition,
            "eipv_form": self.config.eipv_form,
            "hyperparameters": self.config.hyperparameters,
            "lengthscales": self.gp.lengthscales.tolist(),
            "coregionalization": self.gp.coregionalization.tolist(),
            "noise_variances": self.gp.noise_variances.tolist(),
            "output_correlation": self.gp.output_correlation().tolist(),
            "n_observations": int(self.x_obs.shape[0]),
            "dim": int(self.d),
        }
        return MOBORunResult(
            x_obs=self.x_obs.copy(),
            y_obs=self.y_obs.copy(),
            hv_history=np.asarray(self.hv_history, dtype=np.float64),
            pareto_x=self.x_obs[mask].copy(),
            pareto_y=self.y_obs[mask].copy(),
            final_model_state=state,
        )


def run_mobo(objective: MultiObjective, config: MOBOConfig, n_iter: int = 25) -> MOBORunResult:
    """Run ``n_iter`` MOBO iterations after the random initial design."""
    optimizer = MOBOSequentialOptimizer(config)
    optimizer.bootstrap(objective)
    for _ in range(n_iter):
        x_next = optimizer.suggest()
        optimizer.observe(x_next, np.asarray(objective(x_next), dtype=np.float64))
    return optimizer.result()


def run_mo_random(
    objective: MultiObjective,
    bounds: list[tuple[float, float]],
    v_ref: Sequence[float],
    n_init: int = 5,
    n_iter: int = 25,
    seed: int | None = 0,
) -> MOBORunResult:
    """Random-search floor with the same result format and history layout as MOBO."""
    rng = np.random.default_rng(seed)
    lower = np.array([b[0] for b in bounds], dtype=np.float64)
    upper = np.array([b[1] for b in bounds], dtype=np.float64)
    v = np.asarray(v_ref, dtype=np.float64)
    x_obs = rng.uniform(lower, upper, size=(n_init + n_iter, lower.shape[0]))
    y_obs = np.asarray(objective(x_obs), dtype=np.float64)
    history = [hypervolume(y_obs[: n_init + i], v) for i in range(n_iter + 1)]
    mask = pareto_mask(y_obs)
    return MOBORunResult(
        x_obs=x_obs,
        y_obs=y_obs,
        hv_history=np.asarray(history, dtype=np.float64),
        pareto_x=x_obs[mask],
        pareto_y=y_obs[mask],
    )
