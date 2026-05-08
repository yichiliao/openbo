"""Simple sequential BO loop implemented with BoTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from numpy.typing import NDArray

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class BORunResult:
    """Container for BO observations and best-so-far trajectory."""

    x_obs: NDArray[np.float64]
    y_obs: NDArray[np.float64]
    best_y_history: NDArray[np.float64]


@dataclass
class BoTorchConfig:
    """Configuration for BoTorch sequential optimizer."""

    bounds: list[tuple[float, float]]
    n_init: int = 5
    num_restarts: int = 5
    raw_samples: int = 64
    seed: int | None = 0


class BoTorchSequentialOptimizer:
    """Ask/tell-style BoTorch optimizer state machine."""

    def __init__(self, config: BoTorchConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(0 if config.seed is None else config.seed)

        self.d = len(config.bounds)
        self.lower = np.array([b[0] for b in config.bounds], dtype=np.float64)
        self.upper = np.array([b[1] for b in config.bounds], dtype=np.float64)
        self.scale = np.maximum(self.upper - self.lower, 1e-12)

        self.x_obs = np.empty((0, self.d), dtype=np.float64)
        self.y_obs = np.empty((0,), dtype=np.float64)
        self.best_y_history: list[float] = []

    def bootstrap(self, objective: Objective) -> None:
        """Collect random initial observations."""
        if self.config.n_init <= 0:
            return
        x_init = self.rng.uniform(
            self.lower, self.upper, size=(self.config.n_init, self.d)
        ).astype(np.float64)
        y_init = np.asarray(objective(x_init), dtype=np.float64)
        if y_init.shape != (self.config.n_init,):
            raise ValueError(
                f"Objective must return shape ({self.config.n_init},), got {y_init.shape}."
            )
        self.observe(x_init, y_init)

    def suggest(self) -> NDArray[np.float64]:
        """Suggest next point batch of shape (1, d)."""
        if self.x_obs.shape[0] == 0:
            raise ValueError("Cannot suggest without observations. Call bootstrap() first.")

        x_obs_unit = (self.x_obs - self.lower) / self.scale
        train_x = torch.tensor(x_obs_unit, dtype=torch.double)
        train_y = torch.tensor(self.y_obs, dtype=torch.double).unsqueeze(-1)
        model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        best_f = torch.max(train_y).item()
        acq = LogExpectedImprovement(model=model, best_f=best_f)
        bounds_t = torch.tensor(
            np.array([[0.0] * self.d, [1.0] * self.d], dtype=np.float64),
            dtype=torch.double,
        )
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
        if y_new.ndim != 1 or y_new.shape[0] != x_new.shape[0]:
            raise ValueError("y_new must have shape (n,) and match x_new rows.")

        self.x_obs = np.vstack([self.x_obs, x_new])
        self.y_obs = np.concatenate([self.y_obs, y_new])
        self.best_y_history.append(float(np.max(self.y_obs)))

    def result(self) -> BORunResult:
        """Build run result from current state."""
        return BORunResult(
            x_obs=self.x_obs.astype(np.float64),
            y_obs=self.y_obs.astype(np.float64),
            best_y_history=np.asarray(self.best_y_history, dtype=np.float64),
        )


def run_bo_botorch(
    objective: Objective,
    bounds: list[tuple[float, float]],
    n_init: int = 5,
    n_iter: int = 25,
    seed: int | None = 0,
) -> BORunResult:
    """Run simple BoTorch BO for a maximization objective."""
    optimizer = BoTorchSequentialOptimizer(
        BoTorchConfig(bounds=bounds, n_init=n_init, seed=seed)
    )
    optimizer.bootstrap(objective)

    for _ in range(n_iter):
        x_next = optimizer.suggest()
        y_next = np.asarray(objective(x_next), dtype=np.float64)
        optimizer.observe(x_next, y_next)

    return optimizer.result()
