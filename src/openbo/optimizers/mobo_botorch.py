"""BoTorch baseline for multi-objective BO: independent GPs with qLogNEHVI.

Each objective gets its own ``SingleTaskGP`` (inside a ``ModelListGP``), and
the next point maximizes ``qLogNoisyExpectedHypervolumeImprovement`` with
``q = 1`` and reference point ``v_ref``. Results use the same
:class:`~openbo.optimizers.mobo.MOBORunResult` as the NumPy MOBO.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import SumMarginalLogLikelihood
from numpy.typing import NDArray

from openbo.benchmarks.hypervolume import hypervolume, pareto_mask
from openbo.optimizers.mobo import MOBORunResult, MultiObjective


@dataclass
class MOBoTorchConfig:
    """Configuration for :class:`MOBoTorchSequentialOptimizer`."""

    bounds: list[tuple[float, float]]
    v_ref: Sequence[float]
    n_init: int = 5
    num_restarts: int = 10
    raw_samples: int = 512
    mc_samples: int = 128
    seed: int | None = 0


class MOBoTorchSequentialOptimizer:
    """Ask/tell qLogNEHVI optimizer with one independent GP per objective."""

    def __init__(self, config: MOBoTorchConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(0 if config.seed is None else config.seed)
        self.d = len(config.bounds)
        self.lower = np.array([b[0] for b in config.bounds], dtype=np.float64)
        self.upper = np.array([b[1] for b in config.bounds], dtype=np.float64)
        self.scale = np.maximum(self.upper - self.lower, 1e-12)
        self.v_ref = np.asarray(config.v_ref, dtype=np.float64)
        self.n_objectives = self.v_ref.shape[0]

        self.x_obs = np.empty((0, self.d), dtype=np.float64)
        self.y_obs = np.empty((0, self.n_objectives), dtype=np.float64)
        self.hv_history: list[float] = []

    def _fit_model(self) -> tuple[ModelListGP, torch.Tensor]:
        """Fit one ``SingleTaskGP`` per objective on unit-scaled inputs."""
        train_x = torch.tensor((self.x_obs - self.lower) / self.scale, dtype=torch.double)
        models = [
            SingleTaskGP(
                train_x,
                torch.tensor(self.y_obs[:, [k]], dtype=torch.double),
                outcome_transform=Standardize(m=1),
            )
            for k in range(self.n_objectives)
        ]
        model = ModelListGP(*models)
        fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
        return model, train_x

    def bootstrap(self, objective: MultiObjective) -> None:
        """Evaluate ``n_init`` uniform random points."""
        if self.config.n_init <= 0:
            return
        x_init = self.rng.uniform(self.lower, self.upper, size=(self.config.n_init, self.d))
        self.observe(x_init, np.asarray(objective(x_init), dtype=np.float64))

    def suggest(self) -> NDArray[np.float64]:
        """Return the next point to evaluate, shape ``(1, d)``."""
        if self.x_obs.shape[0] == 0:
            raise ValueError("Cannot suggest without observations. Call bootstrap() first.")
        model, train_x = self._fit_model()
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.v_ref.tolist(),
            X_baseline=train_x,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([self.config.mc_samples])),
            prune_baseline=True,
        )
        bounds = torch.stack(
            [torch.zeros(self.d, dtype=torch.double), torch.ones(self.d, dtype=torch.double)]
        )
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=self.config.num_restarts,
            raw_samples=self.config.raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )
        return candidate.detach().cpu().numpy().astype(np.float64) * self.scale + self.lower

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
        self.x_obs = np.vstack([self.x_obs, x_new])
        self.y_obs = np.vstack([self.y_obs, y_new])
        self.hv_history.append(hypervolume(self.y_obs, self.v_ref))

    def result(self) -> MOBORunResult:
        """Return observations, the hypervolume history and the Pareto set."""
        if self.x_obs.shape[0] == 0:
            raise ValueError("mobo_botorch finished without any observations.")
        mask = pareto_mask(self.y_obs)
        return MOBORunResult(
            x_obs=self.x_obs.copy(),
            y_obs=self.y_obs.copy(),
            hv_history=np.asarray(self.hv_history, dtype=np.float64),
            pareto_x=self.x_obs[mask].copy(),
            pareto_y=self.y_obs[mask].copy(),
            final_model_state={
                "backend": "botorch",
                "model": "ModelListGP[SingleTaskGP]",
                "acquisition": "qLogNEHVI",
                "n_observations": int(self.x_obs.shape[0]),
                "dim": int(self.d),
            },
        )


def run_mobo_botorch(
    objective: MultiObjective, config: MOBoTorchConfig, n_iter: int = 25
) -> MOBORunResult:
    """Run ``n_iter`` qLogNEHVI iterations after the random initial design."""
    optimizer = MOBoTorchSequentialOptimizer(config)
    optimizer.bootstrap(objective)
    for _ in range(n_iter):
        x_next = optimizer.suggest()
        optimizer.observe(x_next, np.asarray(objective(x_next), dtype=np.float64))
    return optimizer.result()
