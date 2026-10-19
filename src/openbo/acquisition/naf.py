"""Neural acquisition function of MetaBO (Volpp et al., ICLR 2020).

The AF is an MLP on per-point features ``[mu, sigma, x, t, T]`` that outputs
one logit per candidate point; in training, the policy is the categorical
distribution over the candidates' logits, and at test time the AF's argmax is
evaluated. Candidates come from a hierarchical grid (``AdaptiveSet``).

Details follow the authors' reference code (plan: ``docs/plans/naf.md``):
raw, unstandardized features (``t`` counts evaluations from 0), weights
initialized from ``N(0, 0.01^2)`` with zero biases (a near-uniform initial
policy), a regular global grid, and local Sobol clouds around the best grid
points instead of gradient-based maximization.

An optional extension appends objective weights ``w_1..w_N`` (on the simplex)
to every point's features, giving a weight-conditioned AF. With
``n_weights = 0`` (the default) everything is MetaBO's NAF.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc
from torch import nn

# float32, as in the reference (PyTorch's default); about twice as fast as float64 here.
_DTYPE = torch.float32
NAF_HIDDEN = (200, 200, 200, 200)
INIT_STD = 0.01
SIMPLEX_TOLERANCE = 1e-9


def naf_features(
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    x: NDArray[np.float64],
    t: int,
    budget: int,
    use_x: bool = True,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Per-point features ``[mu, sigma, x, t, T]`` (``x`` omitted when ``use_x`` is False).

    ``mean``/``std`` are the GP posterior at ``x`` (shape ``(n,)``), ``t`` the
    number of evaluations so far and ``budget`` the episode length ``T``.
    All values are raw, as in the reference. A weight-conditioned AF passes
    the objective ``weights``, which are appended to every row.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if mean.shape != (n,) or std.shape != (n,):
        raise ValueError("mean and std must have shape (n,), matching x.")
    if not 0 <= t <= budget:
        raise ValueError("t must be in [0, budget].")
    columns = [mean[:, None], std[:, None]]
    if use_x:
        columns.append(x)
    columns.append(np.full((n, 1), float(t)))
    columns.append(np.full((n, 1), float(budget)))
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(1, -1)
        columns.append(np.repeat(w, n, axis=0))
    return np.hstack(columns)


def n_naf_features(dim: int, use_x: bool = True, n_weights: int = 0) -> int:
    """Number of features per point."""
    return 4 + (dim if use_x else 0) + n_weights


def mlp(n_inputs: int, hidden: tuple[int, ...] = NAF_HIDDEN) -> nn.Sequential:
    """ReLU MLP with one linear output, initialized like the reference (N(0, 0.01^2), zero biases)."""
    layers: list[nn.Module] = []
    width = n_inputs
    for h in hidden:
        layers += [nn.Linear(width, h, dtype=_DTYPE), nn.ReLU()]
        width = h
    layers.append(nn.Linear(width, 1, dtype=_DTYPE))
    net = nn.Sequential(*layers)
    with torch.no_grad():
        for module in net:
            if isinstance(module, nn.Linear):
                module.weight.normal_(0.0, INIT_STD)
                module.bias.zero_()
    return net


class NeuralAF(nn.Module):
    """``alpha_theta(x) = MLP[mu(x), sigma(x), x, t, T(, w)]``: one logit per point.

    ``n_weights > 0`` makes it a weight-conditioned AF.
    """

    def __init__(
        self,
        dim: int,
        use_x: bool = True,
        hidden: tuple[int, ...] = NAF_HIDDEN,
        seed: int = 0,
        n_weights: int = 0,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive.")
        if n_weights < 0:
            raise ValueError("n_weights must be non-negative.")
        self.dim = dim
        self.use_x = use_x
        self.hidden = tuple(hidden)
        self.n_weights = n_weights
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.net = mlp(n_naf_features(dim, use_x, n_weights), self.hidden)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Logits of shape ``(..., n)`` for features of shape ``(..., n, F)``."""
        return self.net(features).squeeze(-1)

    def logits(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """NumPy convenience wrapper around :meth:`forward`, without gradients."""
        with torch.no_grad():
            return self.forward(torch.as_tensor(features, dtype=_DTYPE)).numpy().astype(np.float64)


ScoreFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def regular_grid(points_per_dim: int, dim: int) -> NDArray[np.float64]:
    """A regular grid of ``points_per_dim^dim`` points on ``[0, 1]^dim``, endpoints included."""
    if points_per_dim < 2 or dim < 1:
        raise ValueError("points_per_dim must be at least 2 and dim positive.")
    axes = [np.linspace(0.0, 1.0, points_per_dim)] * dim
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)


@dataclass
class AdaptiveSet:
    """The candidate set ``xi_t``: a fixed global grid plus ``k`` local maxima of the AF.

    Each step, the AF is scored on the global grid; around each of its ``k``
    best grid points, a fixed Sobol cloud of ``n_local`` points is placed in a
    cube of side ``local_side`` (clipped to ``[0, 1]^d``), and the ``k`` best of
    all those local points are added. As in the reference: a 31 x 31 grid,
    ``k = 5``, ``n_local = 1000`` and ``local_side = 2/31`` for 2D problems.
    """

    global_grid: NDArray[np.float64]
    local_offsets: NDArray[np.float64]
    k: int = 5

    @classmethod
    def build(
        cls,
        dim: int,
        points_per_dim: int = 31,
        k: int = 5,
        n_local: int = 1000,
        local_side: float | None = None,
        seed: int = 0,
    ) -> AdaptiveSet:
        """The reference's configuration for a ``dim``-dimensional problem."""
        if k < 1 or n_local < 1:
            raise ValueError("k and n_local must be positive.")
        side = 2.0 / points_per_dim if local_side is None else float(local_side)
        if side <= 0.0:
            raise ValueError("local_side must be positive.")
        cloud = qmc.Sobol(d=dim, scramble=True, seed=seed).random(n_local)
        return cls(global_grid=regular_grid(points_per_dim, dim), local_offsets=side * (cloud - 0.5), k=k)

    @property
    def size(self) -> int:
        """Number of candidates per step: the global grid plus ``k`` local points."""
        return self.global_grid.shape[0] + self.k

    def local_points(self, centres: NDArray[np.float64]) -> NDArray[np.float64]:
        """The local clouds around ``centres`` (shape ``(c, d)``), clipped to the unit cube."""
        points = centres[:, None, :] + self.local_offsets[None, :, :]
        return np.clip(points.reshape(-1, centres.shape[1]), 0.0, 1.0)

    def candidates(self, score: ScoreFn) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``xi_t`` (``k`` local maxima first, then the global grid) and their scores."""
        grid_scores = np.asarray(score(self.global_grid), dtype=np.float64)
        best = np.argsort(-grid_scores, kind="stable")[: self.k]
        local = self.local_points(self.global_grid[best])
        local_scores = np.asarray(score(local), dtype=np.float64)
        top = np.argsort(-local_scores, kind="stable")[: self.k]
        points = np.vstack([local[top], self.global_grid])
        return points, np.concatenate([local_scores[top], grid_scores])

    def maximize(self, score: ScoreFn) -> NDArray[np.float64]:
        """The argmax of ``score`` over ``xi_t``, shape ``(1, d)`` (test-time AF maximization)."""
        points, scores = self.candidates(score)
        return points[int(np.argmax(scores))][None, :].copy()

    def sample(self, score: ScoreFn, rng: np.random.Generator) -> NDArray[np.float64]:
        """A point of ``xi_t`` drawn from ``softmax(score)``, shape ``(1, d)`` (the training policy)."""
        points, scores = self.candidates(score)
        p = np.exp(scores - scores.max())
        return points[int(rng.choice(p.size, p=p / p.sum()))][None, :].copy()


def gp_mean_std(gp: object, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """GP posterior mean and std at ``x``; the prior (0, sqrt(variance)) before any data.

    ``gp`` is a ``GPScratch``; an unfitted one has no ``x_train`` yet.
    """
    x = np.asarray(x, dtype=np.float64)
    if getattr(gp, "x_train", None) is None or gp.x_train.shape[0] == 0:  # type: ignore[attr-defined]
        n = x.shape[0]
        return np.zeros(n), np.full(n, float(np.sqrt(gp.variance)))  # type: ignore[attr-defined]
    mean, var = gp.posterior(x)  # type: ignore[attr-defined]
    return mean, np.sqrt(var)


def validate_weights(weights: ArrayLike, n_objectives: int) -> NDArray[np.float64]:
    """Return ``weights`` as a float array after checking it lies on the simplex."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != n_objectives:
        raise ValueError(f"weights must have {n_objectives} entries, got {w.shape[0]}.")
    if np.any(w < 0.0) or abs(w.sum() - 1.0) > SIMPLEX_TOLERANCE:
        raise ValueError("weights must be non-negative and sum to 1.")
    return w


def af_weights(af: NeuralAF, weights: ArrayLike | None) -> NDArray[np.float64] | None:
    """The weights ``af`` reads: ``None`` for a plain NAF (any given weights are ignored).

    A weight-conditioned AF needs exactly ``af.n_weights`` of them.
    """
    if af.n_weights == 0:
        return None
    if weights is None:
        raise ValueError(f"This AF reads {af.n_weights} objective weights; pass weights.")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != af.n_weights:
        raise ValueError(f"This AF reads {af.n_weights} objective weights, got {w.shape[0]}.")
    return w


def naf_score_fn(af: NeuralAF, gp: object, t: int, budget: int, weights: ArrayLike | None = None) -> ScoreFn:
    """A score function for :class:`AdaptiveSet`: the AF's logits from the GP's posterior.

    ``weights`` are the objective weights; only a weight-conditioned AF reads them.
    """
    w = af_weights(af, weights)

    def score(x: NDArray[np.float64]) -> NDArray[np.float64]:
        mean, std = gp_mean_std(gp, x)
        return af.logits(naf_features(mean, std, x, t, budget, af.use_x, w))

    return score
