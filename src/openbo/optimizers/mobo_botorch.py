"""Simple sequential multi-objective BO loop implemented with BoTorch.

Multi-objective sibling of ``bo_botorch``: same ask/tell state-machine shape, but the
scalar objective becomes an objective VECTOR in R^M, LogExpectedImprovement becomes
qLogNoisyExpectedHypervolumeImprovement, and "best so far" becomes the hypervolume of the
non-dominated set relative to a fixed reference point.

Conventions (shared with the rest of openbo): every objective is MAXIMIZED, and the
reference point must be dominated by any point that should contribute hypervolume.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood
from numpy.typing import NDArray

from openbo.optimizers.mobo import MOBORunResult

# A multi-objective objective maps (n, d) -> (n, M).
MOObjective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class MORunResult:
    """Container for MO observations and the hypervolume trajectory.

    Multi-objective counterpart of ``BORunResult``: ``y_obs`` gains an objective axis and
    ``best_y_history`` (a running max) becomes ``hypervolume_history`` (a running
    hypervolume of the non-dominated set), with one entry per evaluation so it aligns
    row-for-row with ``x_obs`` and ``y_obs``.
    """

    x_obs: NDArray[np.float64]
    y_obs: NDArray[np.float64]
    hypervolume_history: NDArray[np.float64]
    pareto_front: NDArray[np.float64]
    final_state: dict[str, object] | None = None


def compute_hypervolume(
    y: NDArray[np.float64],
    ref_point: NDArray[np.float64],
) -> float:
    """Hypervolume of the non-dominated subset of ``y`` w.r.t. ``ref_point`` (maximization).

    Points that do not strictly dominate the reference point contribute nothing, so an
    empty or entirely-dominated set yields 0.0 rather than raising.
    """
    y = np.asarray(y, dtype=np.float64)
    ref_point = np.asarray(ref_point, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("y must have shape (n, M).")
    if ref_point.ndim != 1 or ref_point.shape[0] != y.shape[1]:
        raise ValueError("ref_point must have shape (M,) matching y columns.")
    if y.shape[0] == 0:
        return 0.0

    finite = y[np.all(np.isfinite(y), axis=1)]
    if finite.shape[0] == 0:
        return 0.0
    above = finite[np.all(finite > ref_point, axis=1)]
    if above.shape[0] == 0:
        return 0.0

    y_t = torch.tensor(above, dtype=torch.double)
    front = y_t[is_non_dominated(y_t)]
    return float(Hypervolume(ref_point=torch.tensor(ref_point, dtype=torch.double)).compute(front))


def pareto_front(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Non-dominated rows of ``y`` (maximization)."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("y must have shape (n, M).")
    if y.shape[0] == 0:
        return y.reshape(0, y.shape[1] if y.ndim == 2 else 0)
    y_t = torch.tensor(y, dtype=torch.double)
    return y_t[is_non_dominated(y_t)].cpu().numpy().astype(np.float64)


@dataclass
class MOBoTorchConfig:
    """Configuration for the BoTorch multi-objective sequential optimizer."""

    bounds: list[tuple[float, float]]
    ref_point: Sequence[float] | None = None
    n_init: int = 5
    num_restarts: int = 5
    raw_samples: int = 64
    mc_samples: int = 128
    seed: int | None = 0
    v_ref: Sequence[float] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Accept the reference-point names used by both OpenBO APIs."""
        if self.ref_point is None:
            self.ref_point = self.v_ref
        elif self.v_ref is not None and not np.array_equal(self.ref_point, self.v_ref):
            raise ValueError("ref_point and v_ref must agree when both are provided.")
        if self.ref_point is None:
            raise ValueError("Provide ref_point (or v_ref).")
        self.v_ref = self.ref_point


class MOBoTorchSequentialOptimizer:
    """Ask/tell-style BoTorch multi-objective optimizer state machine."""

    def __init__(self, config: MOBoTorchConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # torch seeding is scoped inside suggest() via fork_rng so it never mutates
        # the process-global CPU torch RNG (which would let concurrent server
        # sessions perturb each other). Everything here is CPU double precision.
        # A None seed stays nondeterministic rather than being silently pinned to 0.
        self._torch_seed = config.seed
        self._suggest_calls = 0

        self.d = len(config.bounds)
        if self.d == 0:
            raise ValueError("bounds must describe at least one dimension.")
        self.lower = np.array([b[0] for b in config.bounds], dtype=np.float64)
        self.upper = np.array([b[1] for b in config.bounds], dtype=np.float64)
        self.scale = np.maximum(self.upper - self.lower, 1e-12)

        self.ref_point = np.asarray(config.ref_point, dtype=np.float64)
        if self.ref_point.ndim != 1 or self.ref_point.shape[0] < 2:
            raise ValueError("ref_point must have shape (M,) with M >= 2.")
        self.m = int(self.ref_point.shape[0])

        self.x_obs = np.empty((0, self.d), dtype=np.float64)
        self.y_obs = np.empty((0, self.m), dtype=np.float64)
        self.hypervolume_history: list[float] = []

    def bootstrap(self, objective: MOObjective) -> None:
        """Collect random initial observations."""
        if self.config.n_init <= 0:
            return
        x_init = self.rng.uniform(
            self.lower, self.upper, size=(self.config.n_init, self.d)
        ).astype(np.float64)
        y_init = np.asarray(objective(x_init), dtype=np.float64)
        if y_init.shape != (self.config.n_init, self.m):
            raise ValueError(
                f"Objective must return shape ({self.config.n_init}, {self.m}), got {y_init.shape}."
            )
        self.observe(x_init, y_init)

    def _unit_train_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_obs_unit = (self.x_obs - self.lower) / self.scale
        train_x = torch.tensor(x_obs_unit, dtype=torch.double)
        train_y = torch.tensor(self.y_obs, dtype=torch.double)
        bounds_t = torch.tensor(
            np.array([[0.0] * self.d, [1.0] * self.d], dtype=np.float64),
            dtype=torch.double,
        )
        return train_x, train_y, bounds_t

    def _fit_model(self, train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
        model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=self.m))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def _build_acquisition(self, model: SingleTaskGP, train_x: torch.Tensor):
        """Target-only qLogNEHVI. Overridden by the TAF variant to blend in sources."""
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.config.mc_samples]),
            seed=None if self._torch_seed is None else int(self._torch_seed),
        )
        return qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point.tolist(),
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,
        )

    def suggest(self) -> NDArray[np.float64]:
        """Suggest next point batch of shape (1, d)."""
        if self.x_obs.shape[0] == 0:
            raise ValueError("Cannot suggest without observations. Call bootstrap() first.")

        train_x, train_y, bounds_t = self._unit_train_tensors()
        # Isolate torch RNG use to this call: fork_rng saves/restores the CPU
        # generator, and a per-call seed (base + call index) keeps runs reproducible
        # and each iteration distinct without leaking into other sessions.
        with torch.random.fork_rng(devices=[]):
            if self._torch_seed is not None:
                torch.default_generator.manual_seed(
                    int(self._torch_seed) + self._suggest_calls
                )
            self._suggest_calls += 1

            model = self._fit_model(train_x, train_y)
            acq = self._build_acquisition(model, train_x)
            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds_t,
                q=1,
                num_restarts=self.config.num_restarts,
                raw_samples=self.config.raw_samples,
            )
        x_next_unit = candidate.detach().cpu().numpy().astype(np.float64)
        return x_next_unit * self.scale + self.lower

    def observe(
        self,
        x_new: NDArray[np.float64],
        y_new: NDArray[np.float64],
    ) -> None:
        """Tell optimizer new observations."""
        x_new = np.asarray(x_new, dtype=np.float64)
        y_new = np.asarray(y_new, dtype=np.float64)
        if x_new.ndim != 2 or x_new.shape[1] != self.d:
            raise ValueError(f"x_new must have shape (n, {self.d}), got {x_new.shape}.")
        if y_new.ndim != 2 or y_new.shape[0] != x_new.shape[0] or y_new.shape[1] != self.m:
            raise ValueError(
                f"y_new must have shape (n, {self.m}) and match x_new rows, got {y_new.shape}."
            )

        self.x_obs = np.vstack([self.x_obs, x_new])
        self.y_obs = np.vstack([self.y_obs, y_new])

        # One entry per EVALUATION, not per observe() call. This deliberately differs from
        # bo_botorch's best_y_history, which appends once per call and therefore collapses a
        # whole bootstrap batch into a single point. Hypervolume is normally reported against
        # the evaluation budget, so the trace has to be aligned with x_obs rows to be
        # plottable; a batched bootstrap would otherwise silently shift the curve.
        first_new = self.y_obs.shape[0] - y_new.shape[0]
        for k in range(y_new.shape[0]):
            prefix = self.y_obs[: first_new + k + 1]
            self.hypervolume_history.append(compute_hypervolume(prefix, self.ref_point))

    def result(self) -> MORunResult:
        """Build run result from current state."""
        return MORunResult(
            x_obs=self.x_obs.astype(np.float64),
            y_obs=self.y_obs.astype(np.float64),
            hypervolume_history=np.asarray(self.hypervolume_history, dtype=np.float64),
            pareto_front=pareto_front(self.y_obs),
        )


def run_mobo_botorch(
    objective: MOObjective,
    bounds: list[tuple[float, float]] | MOBoTorchConfig,
    ref_point: Sequence[float] | None = None,
    n_init: int = 5,
    n_iter: int = 25,
    seed: int | None = 0,
) -> MORunResult | MOBORunResult:
    """Run qLogNEHVI using a config or the original bounds/ref_point API.

    A config returns the upstream ``MOBORunResult`` (history per ask/tell
    round). Bounds/ref_point returns ``MORunResult`` (history per evaluation).
    """
    config_api = isinstance(bounds, MOBoTorchConfig)
    config = bounds if config_api else MOBoTorchConfig(
        bounds=bounds, ref_point=ref_point, n_init=n_init, seed=seed
    )
    if config.n_init <= 0:
        raise ValueError(
            "n_init must be >= 1 for mobo_botorch: at least one random observation is "
            "needed before the first suggest()."
        )
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative.")
    optimizer = MOBoTorchSequentialOptimizer(config)
    optimizer.bootstrap(objective)

    for _ in range(n_iter):
        x_next = optimizer.suggest()
        y_next = np.asarray(objective(x_next), dtype=np.float64)
        optimizer.observe(x_next, y_next)

    result = optimizer.result()
    if config_api:
        mask = is_non_dominated(torch.tensor(result.y_obs, dtype=torch.double)).numpy()
        return MOBORunResult(
            x_obs=result.x_obs,
            y_obs=result.y_obs,
            hv_history=result.hypervolume_history[config.n_init - 1 :],
            pareto_x=result.x_obs[mask].copy(),
            pareto_y=result.y_obs[mask].copy(),
            final_model_state={
                "backend": "botorch",
                "model": "SingleTaskGP (independent output batches)",
                "acquisition": "qLogNEHVI",
                "n_observations": int(result.x_obs.shape[0]),
                "dim": optimizer.d,
            },
        )
    return result
