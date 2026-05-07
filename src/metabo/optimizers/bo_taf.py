"""BO loop with Transfer Acquisition Function with meta-features (TAF-M)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import qmc

from metabo.acquisition.taf import (
    SourceTaskSurrogate,
    compute_taf_m_weights,
    taf_m_acquisition,
)
from metabo.models.gp_scratch import GPScratch
from metabo.optimizers.bo_scratch import BORunResult

Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


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
    """Run BO with TAF-M acquisition and source surrogates from saved TAF run."""
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lower = np.array([b[0] for b in bounds], dtype=np.float64)
    upper = np.array([b[1] for b in bounds], dtype=np.float64)

    source_surrogates = _load_source_surrogates(taf_run_dir)
    source_meta_map = _normalize_meta_map(source_meta_features)
    for source in source_surrogates:
        if source.name in source_meta_map:
            source.meta_features = source_meta_map[source.name]
    for source in source_surrogates:
        source.reference_y = _source_reference_value(
            y_values=source.gp.y_train,
            mode=source_reference_mode,
            quantile=source_reference_quantile,
        )

    if n_init < 0:
        raise ValueError("n_init must be non-negative.")
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative.")

    if n_init > 0:
        x_obs = rng.uniform(lower, upper, size=(n_init, d)).astype(np.float64)
        y_obs = np.asarray(objective(x_obs), dtype=np.float64)
        if y_obs.shape != (n_init,):
            raise ValueError(
                f"Objective must return shape ({n_init},), got {y_obs.shape}."
            )
    else:
        x_obs = np.empty((0, d), dtype=np.float64)
        y_obs = np.empty((0,), dtype=np.float64)
    best_y_history: list[float] = []

    target_gp = GPScratch(
        lengthscale=np.full(d, 1.5, dtype=np.float64),
        variance=1.0,
        noise=1e-6,
        kernel_type=kernel_type,
        optimize_hyperparameters=optimize_hyperparameters,
        optimize_noise=False,
        optimizer_seed=seed,
    )
    acquisition_trace: list[dict[str, object]] = []

    for iter_idx in range(n_iter):
        source_only_phase = iter_idx < int(max(source_only_warmup_iters, 0))
        if source_only_phase:
            target_gp_eval: GPScratch | None = None
            target_weight_eval = 0.0
            target_best_y = float("-inf") if iter_idx == 0 else (
                float(np.max(y_obs)) if y_obs.size > 0 else float("-inf")
            )
        else:
            if y_obs.size == 0:
                raise ValueError(
                    "No target observations available when leaving source-only warmup."
                )
            target_gp.fit(x_obs, y_obs)
            target_gp_eval = target_gp
            target_weight_eval = float(target_weight)
            target_best_y = float(np.max(y_obs))
        if target_meta_features is not None:
            target_meta = np.asarray(target_meta_features, dtype=np.float64).reshape(-1)
        elif y_obs.size == 0:
            target_meta = _default_target_meta_when_empty(d)
        else:
            target_meta = _default_meta_features(x_obs, y_obs)
        source_meta = np.stack([s.meta_features for s in source_surrogates], axis=0)
        source_weights = compute_taf_m_weights(source_meta, target_meta, rho=rho)

        if search_strategy == "multistart":
            x_pool = _sobol_in_bounds(lower, upper, n_candidates, rng)
            taf_pool = np.asarray(
                taf_m_acquisition(
                    x_pool,
                    target_gp=target_gp_eval,
                    target_best_y=target_best_y,
                    source_surrogates=source_surrogates,
                    source_weights=source_weights,
                    target_weight=target_weight_eval,
                    source_improvement_mode=source_improvement_mode,
                    source_improvement_temperature=source_improvement_temperature,
                    target_ei_floor=target_ei_floor,
                ),
                dtype=np.float64,
            )
            n_starts_eff = int(max(1, min(n_starts, n_candidates)))
            start_indices = np.argsort(taf_pool)[-n_starts_eff:]
            x_starts = x_pool[start_indices]
            best_start_idx = int(np.argmax(taf_pool[start_indices]))
            best_x = x_starts[best_start_idx].copy()
            best_val = float(taf_pool[start_indices][best_start_idx])
            queried_x: list[NDArray[np.float64]] = []
            queried_v: list[float] = []
            if track_acquisition:
                queried_x.extend([row.copy() for row in x_pool])
                queried_v.extend([float(v) for v in taf_pool])

            for x_start in x_starts:
                x_refined, taf_refined = _maximize_taf_lbfgsb(
                    x0=x_start,
                    lower=lower,
                    upper=upper,
                    target_gp=target_gp_eval,
                    target_best_y=target_best_y,
                    source_surrogates=source_surrogates,
                    source_weights=source_weights,
                    target_weight=target_weight_eval,
                    source_improvement_mode=source_improvement_mode,
                    source_improvement_temperature=source_improvement_temperature,
                    target_ei_floor=target_ei_floor,
                    query_x=queried_x if track_acquisition else None,
                    query_v=queried_v if track_acquisition else None,
                )
                if taf_refined > best_val:
                    best_val = taf_refined
                    best_x = x_refined.copy()
            x_next = best_x[None, :]
            if track_acquisition and queried_v:
                queried_vals_arr = np.asarray(queried_v, dtype=np.float64)
                queried_x_arr = np.asarray(queried_x, dtype=np.float64)
                acquisition_trace.append(
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
            
        elif search_strategy == "grid":
            x_grid = _sobol_in_bounds(lower, upper, n_candidates, rng)
            taf_grid = np.asarray(
                taf_m_acquisition(
                    x_grid,
                    target_gp=target_gp_eval,
                    target_best_y=target_best_y,
                    source_surrogates=source_surrogates,
                    source_weights=source_weights,
                    target_weight=target_weight_eval,
                    source_improvement_mode=source_improvement_mode,
                    source_improvement_temperature=source_improvement_temperature,
                    target_ei_floor=target_ei_floor,
                ),
                dtype=np.float64,
            )
            best_idx = int(np.argmax(taf_grid))
            x_next = x_grid[best_idx : best_idx + 1]
            if track_acquisition:
                acquisition_trace.append(
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
                f"Unknown search_strategy '{search_strategy}'. "
                "Use 'multistart' or 'grid'."
            )

        y_next = np.asarray(objective(x_next), dtype=np.float64)
        x_obs = np.vstack([x_obs, x_next])
        y_obs = np.concatenate([y_obs, y_next])
        best_y_history.append(float(np.max(y_obs)))

    if x_obs.shape[0] == 0:
        raise ValueError("bo_taf finished without any observations.")
    target_gp.fit(x_obs, y_obs)
    lengthscale_arr = np.asarray(target_gp.lengthscale, dtype=np.float64).reshape(-1)
    final_gp_state: dict[str, object] = {
        "kernel_type": target_gp.kernel_type,
        "lengthscale": [float(v) for v in lengthscale_arr],
        "variance": float(target_gp.variance),
        "noise": float(target_gp.noise),
        "standardize_targets": bool(target_gp.standardize_targets),
        "optimize_hyperparameters": bool(target_gp.optimize_hyperparameters),
        "optimize_noise": bool(target_gp.optimize_noise),
        "y_mean": float(target_gp.y_mean),
        "y_std": float(target_gp.y_std),
        "n_observations": int(x_obs.shape[0]),
        "dim": int(d),
        "taf_rho": float(rho),
        "n_sources": int(len(source_surrogates)),
        "source_reference_mode": source_reference_mode,
        "source_reference_quantile": float(source_reference_quantile),
        "source_improvement_mode": source_improvement_mode,
        "source_improvement_temperature": float(source_improvement_temperature),
        "target_ei_floor": float(target_ei_floor),
        "taf_acquisition_trace": acquisition_trace,
    }

    return BORunResult(
        x_obs=x_obs.astype(np.float64),
        y_obs=y_obs.astype(np.float64),
        best_y_history=np.asarray(best_y_history, dtype=np.float64),
        final_gp_state=final_gp_state,
    )
