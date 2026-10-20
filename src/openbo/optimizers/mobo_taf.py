"""Multi-objective BO loop with the Transfer Acquisition Function (TAF-EHVI).

Multi-objective sibling of ``openbo.optimizers.bo_taf``, and deliberately the same state
machine: source surrogates are reconstructed from a saved run directory, weighted per
iteration by TAF-M (meta-feature similarity) or TAF-R (objective-wise pairwise ranking
agreement; the Pareto-dominance variant stays available as mode "taf_r_pareto" for
ablation), and blended with the target's own acquisition through the weighted-average TAF
form.

The target term is inherited unchanged from ``MOBoTorchSequentialOptimizer`` -- this class
only overrides how the acquisition is built. That inheritance is what makes the degenerate
case exact: with no sources, or with every source rejected, the acquisition IS the plain
qLogNEHVI that ``mobo_botorch`` would have used.

Source artifacts use the same on-disk layout as ``bo_taf``::

    <taf_run_dir>/gp_states/<task>.json     {"gp_state": {...}}
    <taf_run_dir>/trajectories/<task>.json  {"x_values": [[...]], "y_values": [[...]]}

with ``y_values`` extended to shape (n, M). A malformed or unpaired artifact is warned about
and skipped rather than aborting the run.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from numpy.typing import NDArray

from openbo.acquisition.taf_mo_ehvi import (
    MOSourceTaskSurrogate,
    build_source_hvi_term,
    compute_taf_m_weights,
    compute_taf_r_pareto_weights,
    compute_taf_r_ranking_weights,
    mo_meta_features,
    taf_mo_ehvi_acquisition,
)
from openbo.optimizers.mobo_botorch import (
    MOBoTorchConfig,
    MOBoTorchSequentialOptimizer,
    MORunResult,
    MOObjective,
    pareto_front,
)

_KERNELS = {"matern52", "rbf"}


@dataclass
class MOTAFConfig(MOBoTorchConfig):
    """Configuration for the ask/tell multi-objective TAF optimizer.

    Extends ``MOBoTorchConfig`` with the transfer settings; field names mirror ``TAFConfig``.
    ``taf_weight_mode`` selects how source weights are computed each iteration: "taf_r"
    (objective-wise pairwise ranking agreement; the default, because it scores sources on
    actual predictive evidence and can zero out a contradicting source, at negligible
    cost), "taf_m" (meta-feature similarity), or "taf_r_pareto" (Pareto-dominance
    agreement, kept for ablation).
    """

    taf_run_dir: str | Path = ""
    n_iter: int = 25
    rho: float = 1.0
    taf_weight_mode: str = "taf_r"
    target_weight: float = 1.0
    source_meta_features: dict[str, NDArray[np.float64]] | None = None
    target_meta_features: NDArray[np.float64] | None = None
    source_reference_mode: str = "front"
    source_reference_quantile: float = 0.9
    source_only_warmup_iters: int = 0
    min_informative_pairs: int = 1
    # Population-weight decay (TAF+ Extension 2, Liao et al. CHI'24 Eq. 6). d1 is the
    # iteration after which the decay starts; d2 is the per-iteration decay rate. The
    # default d1=0, d2=0.0 means "never decay", i.e. plain TAF behaviour.
    decay_start_iter: int = 0
    decay_rate: float = 0.0


def _materialize_source_mean(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    gp_state: dict,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Rebuild a source's posterior-mean function with FIXED hyperparameters.

    Mirrors ``bo_taf``'s use of ``GPScratch(optimize_hyperparameters=False).fit(...)``: the
    stored hyperparameters are replayed and the model is conditioned on the trajectory, but
    never re-fitted. The kernel is constructed explicitly because BoTorch's default
    ``SingleTaskGP`` covariance is a bare kernel with no outputscale to set. The fitted
    constant mean is replayed too when the artifact carries ``mean_constant`` (in
    standardized-target space); artifacts predating that field default to 0, matching
    the previous behavior.
    """
    x_t = torch.tensor(np.asarray(x_values, dtype=np.float64), dtype=torch.double)
    y_t = torch.tensor(np.asarray(y_values, dtype=np.float64), dtype=torch.double)
    n, d = x_t.shape
    m = int(y_t.shape[1])

    def _validated(entry: dict, label: str) -> tuple[str, NDArray[np.float64], float, float, float]:
        kernel_type = str(entry.get("kernel_type", "matern52")).lower()
        if kernel_type not in _KERNELS:
            raise ValueError(
                f"{label}: kernel_type must be one of {sorted(_KERNELS)}, got {kernel_type!r}"
            )
        lengthscale = np.asarray(entry["lengthscale"], dtype=np.float64).reshape(-1)
        if lengthscale.size not in (1, d):
            raise ValueError(
                f"{label}: lengthscale must be scalar or length {d}, got length {lengthscale.size}"
            )
        if np.any(lengthscale <= 0):
            raise ValueError(f"{label}: lengthscale entries must be positive.")
        variance = float(entry["variance"])
        noise = float(entry["noise"])
        if variance <= 0 or noise <= 0:
            raise ValueError(f"{label}: variance and noise must be positive.")
        mean_constant = float(entry.get("mean_constant", 0.0))
        if not np.isfinite(mean_constant):
            raise ValueError(f"{label}: mean_constant must be finite.")
        return kernel_type, lengthscale, variance, noise, mean_constant

    # Hyperparameters are either one flat set shared by every objective (the original
    # single-objective-style schema) or a per-objective list under "objectives" -- real
    # sources fit each objective separately, and forcing one lengthscale onto all M
    # objectives would distort every objective but one.
    per_objective = gp_state.get("objectives")
    if per_objective is not None:
        if not isinstance(per_objective, list) or len(per_objective) != m:
            raise ValueError(
                f"gp_state['objectives'] must be a list of length {m}, "
                f"got {type(per_objective).__name__} of length "
                f"{len(per_objective) if isinstance(per_objective, list) else 'n/a'}"
            )
        hyper = [_validated(entry, f"objective {j}") for j, entry in enumerate(per_objective)]
    else:
        hyper = [_validated(gp_state, "gp_state")] * m

    models = []
    for j in range(m):
        kernel_type, lengthscale, variance, noise, mean_constant = hyper[j]
        base = (
            MaternKernel(nu=2.5, ard_num_dims=d)
            if kernel_type == "matern52"
            else RBFKernel(ard_num_dims=d)
        )
        covar = ScaleKernel(base)
        gp = SingleTaskGP(
            x_t,
            y_t[:, j : j + 1],
            covar_module=covar,
            outcome_transform=Standardize(m=1),
        )
        with torch.no_grad():
            gp.covar_module.base_kernel.lengthscale = torch.tensor(
                np.broadcast_to(lengthscale, (d,)).copy(), dtype=torch.double
            )
            gp.covar_module.outputscale = torch.tensor(variance, dtype=torch.double)
            gp.likelihood.noise = torch.tensor(noise, dtype=torch.double)
            gp.mean_module.constant = torch.tensor(mean_constant, dtype=torch.double)
        gp.eval()
        models.append(gp)

    model_list = ModelListGP(*models)

    def mean_fn(x: torch.Tensor) -> torch.Tensor:
        return model_list.posterior(x).mean

    return mean_fn


def _load_mo_source_surrogates(
    taf_run_dir: str | Path,
    expected_m: int | None = None,
    expected_d: int | None = None,
) -> list[MOSourceTaskSurrogate]:
    """Reconstruct MO source surrogates from saved gp_states + trajectories.

    Mirrors ``bo_taf._load_source_surrogates``, including its warn-and-skip contract: a
    missing pair member or a present-but-malformed file is skipped with a warning rather
    than raising an opaque error mid-load.
    """
    run_dir = Path(taf_run_dir)
    gp_states_dir = run_dir / "gp_states"
    trajectories_dir = run_dir / "trajectories"
    gp_files = sorted(gp_states_dir.glob("*.json"))
    if not gp_files:
        return []

    surrogates: list[MOSourceTaskSurrogate] = []
    for gp_path in gp_files:
        task_name = gp_path.stem
        traj_path = trajectories_dir / f"{task_name}.json"

        if not traj_path.exists():
            warnings.warn(
                f"MO TAF source '{task_name}': missing trajectory file {traj_path}; "
                "skipping this source."
            )
            continue
        try:
            gp_payload = json.loads(gp_path.read_text(encoding="utf-8"))
            traj_payload = json.loads(traj_path.read_text(encoding="utf-8"))
            gp_state = gp_payload.get("gp_state")
            if not isinstance(gp_state, dict):
                warnings.warn(
                    f"MO TAF source '{task_name}': gp_state missing or not an object; "
                    "skipping this source."
                )
                continue

            x_values = np.asarray(traj_payload["x_values"], dtype=np.float64)
            y_values = np.asarray(traj_payload["y_values"], dtype=np.float64)
            if x_values.ndim != 2 or y_values.ndim != 2:
                raise ValueError("x_values must be (n, d) and y_values must be (n, M)")
            if x_values.shape[0] != y_values.shape[0] or x_values.shape[0] == 0:
                raise ValueError("x_values and y_values must share a non-zero row count")
            if expected_d is not None and x_values.shape[1] != expected_d:
                raise ValueError(
                    f"source has d={x_values.shape[1]}, target expects d={expected_d}"
                )
            if expected_m is not None and y_values.shape[1] != expected_m:
                raise ValueError(
                    f"source has M={y_values.shape[1]}, target expects M={expected_m}"
                )

            mean_fn = _materialize_source_mean(x_values, y_values, gp_state)
            front = traj_payload.get("pareto_front")
            front_arr = (
                np.asarray(front, dtype=np.float64)
                if front is not None
                else pareto_front(y_values)
            )
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            # A present-but-malformed file is treated the same as a missing one
            # (warn + skip) instead of raising an opaque error mid-load.
            warnings.warn(
                f"MO TAF source '{task_name}': malformed artifacts ({exc!r}); "
                "skipping this source."
            )
            continue

        surrogates.append(
            MOSourceTaskSurrogate(
                name=task_name,
                mean_fn=mean_fn,
                pareto_front=front_arr,
                meta_features=mo_meta_features(x_values, y_values),
            )
        )

    return surrogates


class MOTAFSequentialOptimizer(MOBoTorchSequentialOptimizer):
    """Ask/tell-style multi-objective TAF optimizer state machine."""

    def __init__(self, config: MOTAFConfig) -> None:
        super().__init__(config)
        self.config: MOTAFConfig = config

        if config.taf_weight_mode not in {"taf_m", "taf_r", "taf_r_pareto"}:
            raise ValueError(
                "taf_weight_mode must be 'taf_m', 'taf_r', or 'taf_r_pareto'."
            )

        # Number of suggest() calls made so far. Drives the source-only warmup window
        # independently of the observation count, which bootstrap() advances when
        # n_init > 0 (that coupling is what made bo_taf's warmup off by one).
        self.n_suggestions = 0
        self.last_source_weights: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self.last_target_weight: float = float(config.target_weight)

        self.source_surrogates = _load_mo_source_surrogates(
            config.taf_run_dir, expected_m=self.m, expected_d=self.d
        )

        source_meta_map = {} if config.source_meta_features is None else {
            str(k): np.asarray(v, dtype=np.float64).reshape(-1)
            for k, v in config.source_meta_features.items()
        }
        if source_meta_map:
            # Overriding only SOME sources leaves meta vectors of mixed length, which fails
            # later with an opaque ragged stack error. Fail fast and say why.
            missing = [s.name for s in self.source_surrogates if s.name not in source_meta_map]
            if missing:
                raise ValueError(
                    "source_meta_features must cover every source surrogate; no entry for "
                    f"{missing}. Provide meta-features for all sources or pass "
                    "source_meta_features=None to use the built-in defaults."
                )
        for source in self.source_surrogates:
            if source.name in source_meta_map:
                source.meta_features = source_meta_map[source.name]
            if config.source_reference_mode == "quantile":
                source.reference_front = _quantile_front(
                    source.pareto_front, config.source_reference_quantile
                )
            elif config.source_reference_mode != "front":
                raise ValueError("source_reference_mode must be 'front' or 'quantile'.")

        # A source whose front cannot strictly dominate the reference point offers zero
        # hypervolume improvement, and build_source_hvi_term raises for it. Honor the
        # loader's warn-and-skip contract instead of aborting the whole run at
        # construction, dropping the surrogate as well so weights stay aligned with
        # terms; the existing zero-source degradation handles the all-dropped case.
        kept_surrogates = []
        source_terms = []
        for s in self.source_surrogates:
            try:
                term = build_source_hvi_term(s, self.ref_point)
            except ValueError as exc:
                warnings.warn(
                    f"MO TAF source '{s.name}': cannot build its hypervolume term "
                    f"({exc}); skipping this source."
                )
                continue
            kept_surrogates.append(s)
            source_terms.append(term)
        self.source_surrogates = kept_surrogates
        self._source_terms = source_terms

    def _current_source_weights(self) -> NDArray[np.float64]:
        """Weights for this iteration, mirroring bo_taf's taf_m / taf_r selection."""
        n_sources = len(self.source_surrogates)
        if n_sources == 0:
            return np.zeros(0, dtype=np.float64)

        if self.config.taf_weight_mode in {"taf_r", "taf_r_pareto"}:
            weight_fn = (
                compute_taf_r_ranking_weights
                if self.config.taf_weight_mode == "taf_r"
                else compute_taf_r_pareto_weights
            )
            x_unit = (self.x_obs - self.lower) / self.scale
            return weight_fn(
                source_surrogates=self.source_surrogates,
                x_obs=x_unit,
                y_obs=self.y_obs,
                rho=self.config.rho,
                min_informative_pairs=self.config.min_informative_pairs,
            )

        if self.config.target_meta_features is not None:
            target_meta = np.asarray(
                self.config.target_meta_features, dtype=np.float64
            ).reshape(-1)
        elif self.y_obs.shape[0] == 0:
            # Stable fallback before any target observation exists.
            target_meta = np.asarray(
                [float(self.d)] + [0.0] * (2 * self.m), dtype=np.float64
            )
        else:
            x_unit = (self.x_obs - self.lower) / self.scale
            target_meta = mo_meta_features(x_unit, self.y_obs)

        source_meta = np.stack([s.meta_features for s in self.source_surrogates], axis=0)
        return compute_taf_m_weights(source_meta, target_meta, rho=self.config.rho)

    def _decay_factor(self) -> float:
        """Current gamma(t); counts suggest() calls, matching the warmup counter."""
        return population_decay(
            self.n_suggestions, self.config.decay_start_iter, self.config.decay_rate
        )

    def _build_acquisition(self, model: SingleTaskGP, train_x: torch.Tensor):
        """Blend the inherited qLogNEHVI target term with the weighted source terms."""
        target_term = super()._build_acquisition(model, train_x)

        source_weights = self._current_source_weights()
        # TAF+ Extension 2: shrink the population's influence as the target accumulates data.
        decay = self._decay_factor()
        source_weights = source_weights * decay
        target_weight = float(self.config.target_weight)

        # Source-only warmup: follow the prior alone while inside the window. Only ever
        # entered when at least one source is active, so the acquisition can never be empty.
        # n_suggestions is the 1-based index of the in-flight suggestion (suggest()
        # increments before delegating), so "first k suggestions are source-only" is <=,
        # not < -- the same off-by-one bo_taf's n_suggestions counter was introduced to fix.
        in_warmup = self.n_suggestions <= int(max(self.config.source_only_warmup_iters, 0))
        if in_warmup and float(np.sum(source_weights)) > 0.0:
            target_weight = 0.0

        self.last_source_weights = source_weights
        self.last_target_weight = target_weight

        if source_weights.size == 0 or float(np.sum(source_weights)) <= 0.0:
            # No usable source: the acquisition IS plain qLogNEHVI, bit for bit.
            return target_term

        source_terms = list(self._source_terms)
        return _CompositeTAFEHVI(
            target_term=target_term,
            source_terms=source_terms,
            source_weights=source_weights,
            target_weight=target_weight,
        )

    def suggest(self) -> NDArray[np.float64]:
        """Suggest next point batch of shape (1, d)."""
        self.n_suggestions += 1
        try:
            return super().suggest()
        except Exception:
            # A failed suggestion (e.g. called before bootstrap) must not consume a
            # warmup/decay slot.
            self.n_suggestions -= 1
            raise

    def result(self) -> MORunResult:
        """Build run result, recording the transfer state alongside the trajectory."""
        base = super().result()
        base.final_state = {
            "n_sources": int(len(self.source_surrogates)),
            "source_names": [s.name for s in self.source_surrogates],
            "taf_weight_mode": self.config.taf_weight_mode,
            "taf_rho": float(self.config.rho),
            "last_source_weights": [float(w) for w in self.last_source_weights],
            "last_target_weight": float(self.last_target_weight),
            "source_only_warmup_iters": int(self.config.source_only_warmup_iters),
            "n_suggestions": int(self.n_suggestions),
            "decay_start_iter": int(self.config.decay_start_iter),
            "decay_rate": float(self.config.decay_rate),
            "last_decay_factor": self._decay_factor(),
        }
        return base


class _CompositeTAFEHVI(torch.nn.Module):
    """BoTorch-compatible acquisition wrapping the TAF-EHVI blend."""

    def __init__(
        self,
        target_term,
        source_terms: list,
        source_weights: NDArray[np.float64],
        target_weight: float,
    ) -> None:
        super().__init__()
        self.target_term = target_term
        self.source_terms = torch.nn.ModuleList(
            [t for t in source_terms if isinstance(t, torch.nn.Module)]
        )
        self._terms = source_terms
        self.source_weights = np.asarray(source_weights, dtype=np.float64)
        self.target_weight = float(target_weight)
        # optimize_acqf inspects X_pending on the acquisition it is handed.
        self.X_pending = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return taf_mo_ehvi_acquisition(
            x=X,
            target_term=self.target_term,
            source_terms=self._terms,
            source_weights=self.source_weights,
            target_weight=self.target_weight,
        )


def population_decay(iteration: int, decay_start_iter: int, decay_rate: float) -> float:
    """Decay factor applied to every source (population) weight.

    Implements TAF+'s gamma(t) (Liao et al., CHI'24, Eq. 6): stay at 1 while
    ``iteration <= decay_start_iter``, then fall linearly at ``decay_rate`` per iteration,
    and clamp at 0 so the run ends up relying purely on the target (adaptation) model.

    This exists because a source pool that never yields cannot be overridden by the target
    even once the target has ample data: the sources keep an O(1) acquisition contribution
    while the target's improvement shrinks toward zero as it converges, so the blended
    argmax stays source-driven. Decaying the weights is what hands control back.
    """
    if decay_rate <= 0.0:
        return 1.0
    if iteration <= decay_start_iter:
        return 1.0
    decayed = 1.0 - (iteration - decay_start_iter) * decay_rate
    return float(max(0.0, min(1.0, decayed)))


def _quantile_front(
    front: NDArray[np.float64], quantile: float
) -> NDArray[np.float64]:
    """Prune a source front to its stronger points.

    Multi-objective reading of ``bo_taf``'s ``source_reference_mode='quantile'``: instead of
    "beat this source's own 90th percentile" on a scalar, keep only front points that clear
    the per-objective quantile, so the source rewards genuinely strong regions.
    """
    front = np.asarray(front, dtype=np.float64)
    if front.ndim != 2 or front.shape[0] == 0:
        return front
    q = float(np.clip(quantile, 0.0, 1.0))
    thresholds = np.quantile(front, q, axis=0)
    keep = np.any(front >= thresholds, axis=1)
    pruned = front[keep]
    return pruned if pruned.shape[0] > 0 else front


def run_mobo_taf(
    objective: MOObjective,
    bounds: list[tuple[float, float]],
    ref_point: list[float],
    taf_run_dir: str | Path,
    n_init: int = 5,
    n_iter: int = 25,
    rho: float = 1.0,
    taf_weight_mode: str = "taf_r",
    target_weight: float = 1.0,
    source_only_warmup_iters: int = 0,
    seed: int | None = 0,
) -> MORunResult:
    """Run multi-objective BO with the TAF-EHVI acquisition and saved source surrogates."""
    optimizer = MOTAFSequentialOptimizer(
        MOTAFConfig(
            bounds=bounds,
            ref_point=ref_point,
            taf_run_dir=taf_run_dir,
            n_init=n_init,
            n_iter=n_iter,
            rho=rho,
            taf_weight_mode=taf_weight_mode,
            target_weight=target_weight,
            source_only_warmup_iters=source_only_warmup_iters,
            seed=seed,
        )
    )
    optimizer.bootstrap(objective)
    for _ in range(n_iter):
        x_next = optimizer.suggest()
        y_next = np.asarray(objective(x_next), dtype=np.float64)
        optimizer.observe(x_next, y_next)
    return optimizer.result()
