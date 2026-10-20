"""MC-dropout Bayesian neural network used as ConBO's population model.

Follows Liao et al., *Continual Human-in-the-Loop Optimization* (CHI 2025):
an MLP with ReLU hidden layers, dropout after the last hidden layer that stays
on at prediction time, and two linear heads for the mean and the log-variance.
The predictive variance combines both sources (Kendall & Gal, 2017):
``mean_m sigma2_m + var_m mu_m`` over ``M`` stochastic forward passes.

Training settings the paper does not report (optimizer, learning rate, batch
size, number of prediction samples) are config fields with documented
defaults; see ``docs/plans/conbo.md``.

Targets are standardized internally. The offset and scale are set the first
time the model sees data and are then kept fixed, so online updates and
replay share one output scale; ``fit_replay(..., renormalize=True)`` resets them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn

_DTYPE = torch.float64
_LOGVAR_MIN = math.log(1e-6)
_LOGVAR_MAX = math.log(1e3)


@dataclass(frozen=True)
class BNNConfig:
    """Configuration for :class:`PopulationBNN`."""

    input_dim: int
    hidden_units: int = 100
    n_hidden_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int | None = None  # None trains full-batch.
    n_predict_samples: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.hidden_units <= 0 or self.n_hidden_layers <= 0:
            raise ValueError("hidden_units and n_hidden_layers must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")
        if self.n_predict_samples <= 0:
            raise ValueError("n_predict_samples must be positive.")


class _MLP(nn.Module):
    """ReLU MLP with dropout after the last hidden layer and mean/log-variance heads."""

    def __init__(self, config: BNNConfig) -> None:
        super().__init__()
        widths = [config.input_dim] + [config.hidden_units] * config.n_hidden_layers
        self.hidden = nn.ModuleList(
            nn.Linear(w_in, w_out, dtype=_DTYPE) for w_in, w_out in zip(widths[:-1], widths[1:])
        )
        self.mean_head = nn.Linear(config.hidden_units, 1, dtype=_DTYPE)
        self.logvar_head = nn.Linear(config.hidden_units, 1, dtype=_DTYPE)
        self.dropout = config.dropout

    def forward(
        self, x: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and clamped log-variance for one stochastic pass."""
        h = x
        for layer in self.hidden:
            h = torch.relu(layer(h))
        if self.dropout > 0.0:
            keep = 1.0 - self.dropout
            mask = torch.bernoulli(torch.full_like(h, keep), generator=generator)
            h = h * mask / keep
        mean = self.mean_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1).clamp(_LOGVAR_MIN, _LOGVAR_MAX)
        return mean, logvar


def _as_inputs(x: ArrayLike, dim: int) -> NDArray[np.float64]:
    """Return `x` as a float array of shape `(n, dim)`."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"Expected x shape (n, {dim}) or ({dim},), got {arr.shape}.")
    return arr


def _as_targets(y: ArrayLike, n: int, name: str) -> NDArray[np.float64]:
    """Return `y` as a finite float array of shape `(n,)`."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


class PopulationBNN:
    """MC-dropout BNN with a mean head and a log-variance head."""

    def __init__(self, config: BNNConfig) -> None:
        self.config = config
        self.y_offset: float | None = None
        self.y_scale: float | None = None
        self._train_generator = torch.Generator().manual_seed(config.seed)
        self.reset_parameters()

    @property
    def is_normalized(self) -> bool:
        """Whether the output standardization has been set."""
        return self.y_offset is not None

    def reset_parameters(self) -> None:
        """Re-initialize weights (seeded by ``config.seed``); keep the normalization."""
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(self.config.seed)
            self.net = _MLP(self.config)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.learning_rate)

    def set_normalization(self, y: ArrayLike) -> None:
        """Set the output offset and scale from raw target values."""
        arr = np.asarray(y, dtype=np.float64).ravel()
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("Normalization needs at least one finite value.")
        std = float(np.std(arr))
        self.y_offset = float(np.mean(arr))
        self.y_scale = std if std > 1e-12 else 1.0

    def fit_nll(self, x: ArrayLike, y: ArrayLike, epochs: int) -> list[float]:
        """Train on observations with the Gaussian negative log-likelihood.

        Returns the loss per epoch (in standardized units).
        """
        x_arr = _as_inputs(x, self.config.input_dim)
        y_arr = _as_targets(y, x_arr.shape[0], "y")
        if not self.is_normalized:
            self.set_normalization(y_arr)
        x_t = torch.as_tensor(x_arr, dtype=_DTYPE)
        y_t = torch.as_tensor(self._normalize(y_arr), dtype=_DTYPE)

        def loss_fn(idx: torch.Tensor) -> torch.Tensor:
            mean, logvar = self.net(x_t[idx], self._train_generator)
            return 0.5 * torch.mean(logvar + (y_t[idx] - mean) ** 2 * torch.exp(-logvar))

        return self._train(x_arr.shape[0], epochs, loss_fn)

    def fit_replay(
        self,
        x: ArrayLike,
        mean_target: ArrayLike,
        var_target: ArrayLike,
        epochs: int,
        *,
        renormalize: bool = True,
    ) -> list[float]:
        """Train the heads to match replayed means and variances (paper eq. 7).

        The loss is ``mean_s[(mu_s - mean_target_s)^2 + (sigma2_s - var_target_s)^2]``
        in standardized units, where ``sigma2`` is the variance head. Small target
        variances contribute little to this loss, so the head matches them only
        loosely (see ``docs/plans/conbo.md``, Phase 2). With
        ``renormalize=True`` the output scale is reset from ``mean_target``.
        Returns the loss per epoch.
        """
        x_arr = _as_inputs(x, self.config.input_dim)
        mu = _as_targets(mean_target, x_arr.shape[0], "mean_target")
        var = _as_targets(var_target, x_arr.shape[0], "var_target")
        if np.any(var < 0.0):
            raise ValueError("var_target must be non-negative.")
        if renormalize or not self.is_normalized:
            self.set_normalization(mu)
        assert self.y_scale is not None
        x_t = torch.as_tensor(x_arr, dtype=_DTYPE)
        mu_t = torch.as_tensor(self._normalize(mu), dtype=_DTYPE)
        var_t = torch.as_tensor(var / self.y_scale**2, dtype=_DTYPE)

        def loss_fn(idx: torch.Tensor) -> torch.Tensor:
            mean, logvar = self.net(x_t[idx], self._train_generator)
            return torch.mean((mean - mu_t[idx]) ** 2 + (torch.exp(logvar) - var_t[idx]) ** 2)

        return self._train(x_arr.shape[0], epochs, loss_fn)

    def predict(
        self, x: ArrayLike, n_samples: int | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the predictive mean and variance in raw output units.

        Uses ``n_samples`` MC-dropout passes (default ``config.n_predict_samples``).
        Dropout masks are seeded per call, so the same inputs give the same output.
        """
        x_arr = _as_inputs(x, self.config.input_dim)
        n = self.config.n_predict_samples if n_samples is None else n_samples
        if n <= 0:
            raise ValueError("n_samples must be positive.")
        generator = torch.Generator().manual_seed(self.config.seed + 1)
        x_t = torch.as_tensor(x_arr, dtype=_DTYPE)
        with torch.no_grad():
            passes = [self.net(x_t, generator) for _ in range(n)]
        means = torch.stack([m for m, _ in passes]).numpy()
        variances = np.exp(torch.stack([lv for _, lv in passes]).numpy())
        mean = means.mean(axis=0)
        var = variances.mean(axis=0) + means.var(axis=0)
        offset = 0.0 if self.y_offset is None else self.y_offset
        scale = 1.0 if self.y_scale is None else self.y_scale
        return offset + scale * mean, scale**2 * var

    def save(self, path: str | Path) -> None:
        """Save weights, optimizer state, normalization and config to `path`."""
        torch.save(
            {
                "config": asdict(self.config),
                "y_offset": self.y_offset,
                "y_scale": self.y_scale,
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_generator": self._train_generator.get_state(),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> PopulationBNN:
        """Load a model written by :meth:`save`."""
        payload = torch.load(Path(path), weights_only=False)
        model = cls(BNNConfig(**payload["config"]))
        model.y_offset = payload["y_offset"]
        model.y_scale = payload["y_scale"]
        model.net.load_state_dict(payload["net"])
        model.optimizer.load_state_dict(payload["optimizer"])
        model._train_generator.set_state(payload["train_generator"])
        return model

    def _normalize(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map raw targets to standardized units."""
        assert self.y_offset is not None and self.y_scale is not None
        return (y - self.y_offset) / self.y_scale

    def _train(self, n: int, epochs: int, loss_fn) -> list[float]:  # type: ignore[no-untyped-def]
        """Run `epochs` passes of Adam over `n` points; return the mean loss per epoch."""
        if epochs < 0:
            raise ValueError("epochs must be non-negative.")
        if n == 0:
            raise ValueError("Training needs at least one point.")
        batch = n if self.config.batch_size is None else min(self.config.batch_size, n)
        history: list[float] = []
        self.net.train()
        for _ in range(epochs):
            order = torch.randperm(n, generator=self._train_generator)
            total = 0.0
            for start in range(0, n, batch):
                idx = order[start : start + batch]
                self.optimizer.zero_grad()
                loss = loss_fn(idx)
                loss.backward()
                self.optimizer.step()
                total += float(loss.detach()) * len(idx)
            history.append(total / n)
        return history
