"""BO loop with Transfer Acquisition Function with meta-features (TAF-M)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import qmc

from openbo.acquisition.taf import (
    SourceTaskSurrogate,
    compute_taf_m_weights,
    compute_taf_r_weights,
    taf_m_acquisition,
)
from openbo.models.gp_scratch import GPScratch
from openbo.optimizers.bo_scratch import BORunResult

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class TAFConfig:
    """Configuration for ask/tell-style TAF optimizer."""

    bounds: list[tuple[float, float]]
    taf_run_dir: str | Path
    n_init: int = 0
    n_iter: int = 25
    n_candidates: int = 512
    n_starts: int = 8
    search_strategy: str = "multistart"
    kernel_type: str = "matern52"
    optimize_hyperparameters: bool = True
    rho: float = 1.0
    taf_weight_mode: str = "taf_m"
    target_weight: float = 1.0
    source_meta_features: dict[str, NDArray[np.float64]] | None = None
    target_meta_features: NDArray[np.float64] | None = None
    source_reference_mode: str = "quantile"
    source_reference_quantile: float = 0.9
    source_improvement_mode: str = "softplus"
    source_improvement_temperature: float = 0.05
    target_ei_floor: float = 1e-12
    source_only_warmup_iters: int = 3
    track_acquisition: bool = True
    seed: int | None = 0


def _sobol_in_bounds(
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    d = lower.shape[0]
    seed = int(rng.integers(0, 2**31 - 1))
    engine = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = engine.random(n).astype(np.float64)
    return lower + (upper - lower) * u


def _default_meta_features(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Simple v1 meta-features from trajectory statistics."""
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if x_values.ndim != 2:
        raise ValueError("x_values must have shape (n, d).")
    if y_values.ndim != 1 or y_values.shape[0] != x_values.shape[0]:
        raise ValueError("y_values must have shape (n,) matching x_values rows.")
    return np.array(
        [
            float(x_values.shape[1]),  # dimension
            float(np.mean(y_values)),
            float(np.std(y_values)),
        ],
        dtype=np.float64,
    )


def _default_target_meta_when_empty(d: int) -> NDArray[np.float64]:
    """Stable fallback target meta-features before any target observations."""
    return np.array([float(d), 0.0, 0.0], dtype=np.float64)


def _load_source_surrogates(taf_run_dir: str | Path) -> list[SourceTaskSurrogate]:
    """Reconstruct source surrogates from saved gp_states + trajectories."""
    run_dir = Path(taf_run_dir)
    gp_states_dir = run_dir / "gp_states"
    trajectories_dir = run_dir / "trajectories"
    gp_files = sorted(gp_states_dir.glob("*.json"))
    if not gp_files:
        raise ValueError(f"No source GP states found in: {gp_states_dir}")

    surrogates: list[SourceTaskSurrogate] = []
    for gp_path in gp_files:
        task_name = gp_path.stem
        traj_path = trajectories_dir / f"{task_name}.json"
        
        if not traj_path.exists():
            continue
        gp_payload = json.loads(gp_path.read_text(encoding="utf-8"))
        traj_payload = json.loads(traj_path.read_text(encoding="utf-8"))
        gp_state = gp_payload.get("gp_state")
        if not isinstance(gp_state, dict):
            continue

        x_values = np.asarray(traj_payload["x_values"], dtype=np.float64)
        y_values = np.asarray(traj_payload["y_values"], dtype=np.float64)
        gp = GPScratch(
            lengthscale=np.asarray(gp_state["lengthscale"], dtype=np.float64),
            variance=float(gp_state["variance"]),
            noise=float(gp_state["noise"]),
            kernel_type=str(gp_state["kernel_type"]),
            optimize_hyperparameters=False,
            standardize_targets=bool(gp_state.get("standardize_targets", True)),
            optimize_noise=bool(gp_state.get("optimize_noise", False)),
        )
        gp.fit(x_values, y_values)
        surrogates.append(
            SourceTaskSurrogate(
                name=task_name,
                gp=gp,
                best_y=float(np.max(y_values)),
                meta_features=_default_meta_features(x_values, y_values),
                reference_y=float(np.max(y_values)),
            )
        )

    if not surrogates:
        raise ValueError(
            "Failed to reconstruct any source surrogates from saved trajectories."
        )
    return surrogates


def _normalize_meta_map(
    source_meta_features: dict[str, NDArray[np.float64]] | None,
) -> dict[str, NDArray[np.float64]]:
    if source_meta_features is None:
        return {}
    return {
        str(k): np.asarray(v, dtype=np.float64).reshape(-1)
        for k, v in source_meta_features.items()
    }


def _source_reference_value(
    y_values: NDArray[np.float64],
    mode: str,
    quantile: float,
) -> float:
    if mode == "best":
        return float(np.max(y_values))
    if mode == "quantile":
        q = float(np.clip(quantile, 0.0, 1.0))
        return float(np.quantile(y_values, q))
    raise ValueError("source_reference_mode must be 'best' or 'quantile'.")


def _taf_scalar(
    x: NDArray[np.float64],
    target_gp: GPScratch | None,
    target_best_y: float,
    source_surrogates: list[SourceTaskSurrogate],
    source_weights: NDArray[np.float64],
    target_weight: float,
    source_improvement_mode: str,
    source_improvement_temperature: float,
    target_ei_floor: float,
) -> float:
    return float(
        taf_m_acquisition(
            x=x,
            target_gp=target_gp,
            target_best_y=target_best_y,
            source_surrogates=source_surrogates,
            source_weights=source_weights,
            target_weight=target_weight,
            source_improvement_mode=source_improvement_mode,
            source_improvement_temperature=source_improvement_temperature,
            target_ei_floor=target_ei_floor,
        )
    )


def _maximize_taf_lbfgsb(
    x0: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    target_gp: GPScratch | None,
    target_best_y: float,
    source_surrogates: list[SourceTaskSurrogate],
    source_weights: NDArray[np.float64],
    target_weight: float,
    source_improvement_mode: str,
    source_improvement_temperature: float,
    target_ei_floor: float,
    query_x: list[NDArray[np.float64]] | None = None,
    query_v: list[float] | None = None,
) -> tuple[NDArray[np.float64], float]:
    def objective(x: NDArray[np.float64]) -> float:
        value = _taf_scalar(
            x.astype(np.float64),
            target_gp,
            target_best_y,
            source_surrogates,
            source_weights,
            target_weight,
            source_improvement_mode,
            source_improvement_temperature,
            target_ei_floor,
        )
        if query_x is not None and query_v is not None:
            query_x.append(np.asarray(x, dtype=np.float64).copy())
            query_v.append(float(value))
        return -value

    bounds = [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]
    result = minimize(
        objective,
        x0.astype(np.float64),
        method="L-BFGS-B",
        bounds=bounds,
    )
    x_opt = np.asarray(result.x, dtype=np.float64)
    taf_val = _taf_scalar(
        x_opt,
        target_gp,
        target_best_y,
        source_surrogates,
        source_weights,
        target_weight,
        source_improvement_mode,
        source_improvement_temperature,
        target_ei_floor,
    )
    return x_opt, taf_val


class TAFSequentialOptimizer:
    """Ask/tell-style TAF optimizer state machine."""

    def __init__(self, config: TAFConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.d = len(config.bounds)
        self.lower = np.array([b[0] for b in config.bounds], dtype=np.float64)
        self.upper = np.array([b[1] for b in config.bounds], dtype=np.float64)
        self.x_obs = np.empty((0, self.d), dtype=np.float64)
        self.y_obs = np.empty((0,), dtype=np.float64)
        self.best_y_history: list[float] = []
        self.iter_count = 0
        self.pending_x: NDArray[np.float64] | None = None

        self.source_surrogates = _load_source_surrogates(config.taf_run_dir)
        source_meta_map = _normalize_meta_map(config.source_meta_features)
        for source in self.source_surrogates:
            if source.name in source_meta_map:
                source.meta_features = source_meta_map[source.name]
            source.reference_y = _source_reference_value(
                y_values=source.gp.y_train,
                mode=config.source_reference_mode,
                quantile=config.source_reference_quantile,
            )

        self.target_gp = GPScratch(
            lengthscale=np.full(self.d, 1.5, dtype=np.float64),
            variance=1.0,
            noise=1e-6,
            kernel_type=config.kernel_type,
            optimize_hyperparameters=config.optimize_hyperparameters,
            optimize_noise=False,
            optimizer_seed=config.seed,
        )
        self.acquisition_trace: list[dict[str, object]] = []

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
        if self.pending_x is not None:
            raise ValueError("Cannot suggest again before observe.")
        if self.iter_count >= self.config.n_iter:
            raise ValueError("No iterations remaining.")

        iter_idx = self.iter_count
        source_only_phase = iter_idx < int(max(self.config.source_only_warmup_iters, 0))
        if source_only_phase:
            print("source_only_phase") # FOR DEBUGGING
            target_gp_eval: GPScratch | None = None
            target_weight_eval = 0.0
            target_best_y = float("-inf") if iter_idx == 0 else (
                float(np.max(self.y_obs)) if self.y_obs.size > 0 else float("-inf")
            )
        else:
            print("target_phase") # FOR DEBUGGING
            if self.y_obs.size == 0:
                raise ValueError(
                    "No target observations available when leaving source-only warmup."
                )
            self.target_gp.fit(self.x_obs, self.y_obs)
            target_gp_eval = self.target_gp
            target_weight_eval = float(self.config.target_weight)
            target_best_y = float(np.max(self.y_obs))

        if self.config.target_meta_features is not None:
            target_meta = np.asarray(
                self.config.target_meta_features, dtype=np.float64
            ).reshape(-1)
        elif self.y_obs.size == 0:
            target_meta = _default_target_meta_when_empty(self.d)
        else:
            target_meta = _default_meta_features(self.x_obs, self.y_obs)

        print("target_meta") # FOR DEBUGGING
        print(target_meta) # FOR DEBUGGING

        if self.config.taf_weight_mode == "taf_m":
            source_meta = np.stack(
                [s.meta_features for s in self.source_surrogates], axis=0
            )
            source_weights = compute_taf_m_weights(
                source_meta, target_meta, rho=self.config.rho
            )
        elif self.config.taf_weight_mode == "taf_r":
            source_weights = compute_taf_r_weights(
                source_surrogates=self.source_surrogates,
                x_obs=self.x_obs,
                y_obs=self.y_obs,
                rho=self.config.rho,
            )
        else:
            raise ValueError("taf_weight_mode must be 'taf_m' or 'taf_r'.")
        
        print("source_weights") # FOR DEBUGGING
        print(source_weights) # FOR DEBUGGING

        if self.config.search_strategy == "multistart":
            x_pool = _sobol_in_bounds(
                self.lower, self.upper, self.config.n_candidates, self.rng
            )
            taf_pool = np.asarray(
                taf_m_acquisition(
                    x_pool,
                    target_gp=target_gp_eval,
                    target_best_y=target_best_y,
                    source_surrogates=self.source_surrogates,
                    source_weights=source_weights,
                    target_weight=target_weight_eval,
                    source_improvement_mode=self.config.source_improvement_mode,
                    source_improvement_temperature=self.config.source_improvement_temperature,
                    target_ei_floor=self.config.target_ei_floor,
                ),
                dtype=np.float64,
            )
            n_starts_eff = int(max(1, min(self.config.n_starts, self.config.n_candidates)))
            start_indices = np.argsort(taf_pool)[-n_starts_eff:]
            x_starts = x_pool[start_indices]
            best_start_idx = int(np.argmax(taf_pool[start_indices]))
            best_x = x_starts[best_start_idx].copy()
            best_val = float(taf_pool[start_indices][best_start_idx])
            queried_x: list[NDArray[np.float64]] = []
            queried_v: list[float] = []
            if self.config.track_acquisition:
                queried_x.extend([row.copy() for row in x_pool])
                queried_v.extend([float(v) for v in taf_pool])

            for x_start in x_starts:
                x_refined, taf_refined = _maximize_taf_lbfgsb(
                    x0=x_start,
                    lower=self.lower,
                    upper=self.upper,
                    target_gp=target_gp_eval,
                    target_best_y=target_best_y,
                    source_surrogates=self.source_surrogates,
                    source_weights=source_weights,
                    target_weight=target_weight_eval,
                    source_improvement_mode=self.config.source_improvement_mode,
                    source_improvement_temperature=self.config.source_improvement_temperature,
                    target_ei_floor=self.config.target_ei_floor,
                    query_x=queried_x if self.config.track_acquisition else None,
                    query_v=queried_v if self.config.track_acquisition else None,
                )
                if taf_refined > best_val:
                    best_val = taf_refined
                    best_x = x_refined.copy()
            x_next = best_x[None, :]
            if self.config.track_acquisition and queried_v:
                queried_vals_arr = np.asarray(queried_v, dtype=np.float64)
                queried_x_arr = np.asarray(queried_x, dtype=np.float64)
                self.acquisition_trace.append(
                    {
                        "iteration": int(iter_idx),
                        "strategy": "multistart",
                        "x": queried_x_arr.tolist(),
                        "values": queried_vals_arr.tolist(),
                        "zero_fraction": float(np.mean(queried_vals_arr <= 1e-12)),
                        "mean": float(np.mean(queried_vals_arr)),
                        "min": float(np.min(queried_vals_arr)),
                        "max": float(np.max(queried_vals_arr)),
                        "n_queries": int(queried_vals_arr.size),
                    }
                )
        elif self.config.search_strategy == "grid":
            x_grid = _sobol_in_bounds(
                self.lower, self.upper, self.config.n_candidates, self.rng
            )
            taf_grid = np.asarray(
                taf_m_acquisition(
                    x_grid,
                    target_gp=target_gp_eval,
                    target_best_y=target_best_y,
                    source_surrogates=self.source_surrogates,
                    source_weights=source_weights,
                    target_weight=target_weight_eval,
                    source_improvement_mode=self.config.source_improvement_mode,
                    source_improvement_temperature=self.config.source_improvement_temperature,
                    target_ei_floor=self.config.target_ei_floor,
                ),
                dtype=np.float64,
            )
            best_idx = int(np.argmax(taf_grid))
            x_next = x_grid[best_idx : best_idx + 1]
            if self.config.track_acquisition:
                self.acquisition_trace.append(
                    {
                        "iteration": int(iter_idx),
                        "strategy": "grid",
                        "x": x_grid.tolist(),
                        "values": taf_grid.tolist(),
                        "zero_fraction": float(np.mean(taf_grid <= 1e-12)),
                        "mean": float(np.mean(taf_grid)),
                        "min": float(np.min(taf_grid)),
                        "max": float(np.max(taf_grid)),
                        "n_queries": int(taf_grid.size),
                    }
                )
        else:
            raise ValueError(
                f"Unknown search_strategy '{self.config.search_strategy}'. "
                "Use 'multistart' or 'grid'."
            )

        self.pending_x = x_next.copy()
        return x_next

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
        if self.pending_x is not None and (
            self.pending_x.shape != x_new.shape or not np.allclose(self.pending_x, x_new)
        ):
            raise ValueError("Observed x does not match pending suggestion.")

        self.x_obs = np.vstack([self.x_obs, x_new])
        self.y_obs = np.concatenate([self.y_obs, y_new])
        self.best_y_history.append(float(np.max(self.y_obs)))
        self.pending_x = None
        if self.iter_count < self.config.n_iter:
            self.iter_count += 1

    def result(self) -> BORunResult:
        """Build run result from current state."""
        if self.x_obs.shape[0] == 0:
            raise ValueError("bo_taf finished without any observations.")
        self.target_gp.fit(self.x_obs, self.y_obs)
        lengthscale_arr = np.asarray(self.target_gp.lengthscale, dtype=np.float64).reshape(-1)
        final_gp_state: dict[str, object] = {
            "kernel_type": self.target_gp.kernel_type,
            "lengthscale": [float(v) for v in lengthscale_arr],
            "variance": float(self.target_gp.variance),
            "noise": float(self.target_gp.noise),
            "standardize_targets": bool(self.target_gp.standardize_targets),
            "optimize_hyperparameters": bool(self.target_gp.optimize_hyperparameters),
            "optimize_noise": bool(self.target_gp.optimize_noise),
            "y_mean": float(self.target_gp.y_mean),
            "y_std": float(self.target_gp.y_std),
            "n_observations": int(self.x_obs.shape[0]),
            "dim": int(self.d),
            "taf_rho": float(self.config.rho),
            "taf_weight_mode": self.config.taf_weight_mode,
            "n_sources": int(len(self.source_surrogates)),
            "source_reference_mode": self.config.source_reference_mode,
            "source_reference_quantile": float(self.config.source_reference_quantile),
            "source_improvement_mode": self.config.source_improvement_mode,
            "source_improvement_temperature": float(
                self.config.source_improvement_temperature
            ),
            "target_ei_floor": float(self.config.target_ei_floor),
            "taf_acquisition_trace": self.acquisition_trace,
        }
        return BORunResult(
            x_obs=self.x_obs.astype(np.float64),
            y_obs=self.y_obs.astype(np.float64),
            best_y_history=np.asarray(self.best_y_history, dtype=np.float64),
            final_gp_state=final_gp_state,
        )


def run_bo_taf(
    objective: Objective,
    bounds: list[tuple[float, float]],
    taf_run_dir: str | Path,
    n_init: int = 0,
    n_iter: int = 25,
    n_candidates: int = 512,
    n_starts: int = 8,
    search_strategy: str = "multistart",
    kernel_type: str = "matern52",
    optimize_hyperparameters: bool = True,
    rho: float = 1.0,
    taf_weight_mode: str = "taf_m",
    target_weight: float = 1.0,
    source_meta_features: dict[str, NDArray[np.float64]] | None = None,
    target_meta_features: NDArray[np.float64] | None = None,
    source_reference_mode: str = "quantile",
    source_reference_quantile: float = 0.9,
    source_improvement_mode: str = "softplus",
    source_improvement_temperature: float = 0.05,
    target_ei_floor: float = 1e-12,
    source_only_warmup_iters: int = 3,
    track_acquisition: bool = True,
    seed: int | None = 0,
) -> BORunResult:
    """Run BO with TAF acquisition and source surrogates from saved TAF run."""
    # After bootstrap, iter_count is 1; suggest() requires iter_count < n_iter, so an
    # extra budget slot is needed to complete n_iter BO steps when n_init > 0.
    taf_loop_n_iter = n_iter + (1 if n_init > 0 else 0)
    optimizer = TAFSequentialOptimizer(
        TAFConfig(
            bounds=bounds,
            taf_run_dir=taf_run_dir,
            n_init=n_init,
            n_iter=taf_loop_n_iter,
            n_candidates=n_candidates,
            n_starts=n_starts,
            search_strategy=search_strategy,
            kernel_type=kernel_type,
            optimize_hyperparameters=optimize_hyperparameters,
            rho=rho,
            taf_weight_mode=taf_weight_mode,
            target_weight=target_weight,
            source_meta_features=source_meta_features,
            target_meta_features=target_meta_features,
            source_reference_mode=source_reference_mode,
            source_reference_quantile=source_reference_quantile,
            source_improvement_mode=source_improvement_mode,
            source_improvement_temperature=source_improvement_temperature,
            target_ei_floor=target_ei_floor,
            source_only_warmup_iters=source_only_warmup_iters,
            track_acquisition=track_acquisition,
            seed=seed,
        )
    )
    optimizer.bootstrap(objective)
    for _ in range(n_iter):
        x_next = optimizer.suggest()
        y_next = np.asarray(objective(x_next), dtype=np.float64)
        optimizer.observe(x_next, y_next)
    return optimizer.result()
