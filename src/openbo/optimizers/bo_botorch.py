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


def run_bo_botorch(
    objective: Objective,
    bounds: list[tuple[float, float]],
    n_init: int = 5,
    n_iter: int = 25,
    seed: int | None = 0,
) -> BORunResult:
    """Run simple BoTorch BO for a maximization objective."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(0 if seed is None else seed)

    d = len(bounds)
    lower = np.array([b[0] for b in bounds], dtype=np.float64)
    upper = np.array([b[1] for b in bounds], dtype=np.float64)

    x_obs = rng.uniform(lower, upper, size=(n_init, d)).astype(np.float64)
    y_obs = np.asarray(objective(x_obs), dtype=np.float64)
    best_y_history: list[float] = [float(np.max(y_obs))]

    scale = np.maximum(upper - lower, 1e-12)

    for _ in range(n_iter):
        x_obs_unit = (x_obs - lower) / scale
        train_x = torch.tensor(x_obs_unit, dtype=torch.double)
        train_y = torch.tensor(y_obs, dtype=torch.double).unsqueeze(-1)
        model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        best_f = torch.max(train_y).item()
        acq = LogExpectedImprovement(model=model, best_f=best_f)
        bounds_t = torch.tensor(np.array([[0.0] * d, [1.0] * d], dtype=np.float64), dtype=torch.double)
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds_t,
            q=1,
            num_restarts=5,
            raw_samples=64,
        )
        x_next_unit = candidate.detach().cpu().numpy().astype(np.float64)
        x_next = x_next_unit * scale + lower
        y_next = np.asarray(objective(x_next), dtype=np.float64)

        x_obs = np.vstack([x_obs, x_next])
        y_obs = np.concatenate([y_obs, y_next])
        best_y_history.append(float(np.max(y_obs)))

    return BORunResult(
        x_obs=x_obs.astype(np.float64),
        y_obs=y_obs.astype(np.float64),
        best_y_history=np.asarray(best_y_history, dtype=np.float64),
    )
