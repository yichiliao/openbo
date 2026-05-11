"""Transfer Acquisition Function with meta-feature weighting (TAF-M)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from openbo.acquisition.ei import expected_improvement_maximization
from openbo.models.gp_scratch import GPScratch


@dataclass
class SourceTaskSurrogate:
    """Reconstructed source-task surrogate used by TAF-M."""

    name: str
    gp: GPScratch
    best_y: float
    meta_features: NDArray[np.float64]
    reference_y: float | None = None


def epanechnikov_weight(distance: float, rho: float) -> float:
    """Epanechnikov kernel value for one distance scalar."""
    if rho <= 0:
        raise ValueError("rho must be positive.")

    t = distance / rho
    if t <= 1.0:
        return float(0.75 * (1.0 - t**2))
    return 0.0


def compute_taf_m_weights(
    source_meta_features: NDArray[np.float64],
    target_meta_features: NDArray[np.float64],
    rho: float,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Compute source-task TAF-M weights from meta-feature distance."""
    source_meta_features = np.asarray(source_meta_features, dtype=np.float64)
    target_meta_features = np.asarray(target_meta_features, dtype=np.float64)

    if source_meta_features.ndim != 2:
        raise ValueError("source_meta_features must have shape (n_sources, m).")
    if target_meta_features.ndim != 1:
        raise ValueError("target_meta_features must have shape (m,).")
    if source_meta_features.shape[1] != target_meta_features.shape[0]:
        raise ValueError("source and target meta-features must have same dimension.")

    
    #distances = np.linalg.norm(
    #    source_meta_features - target_meta_features[None, :],
    #    axis=1,
    #)
    #return np.array(
    #    [epanechnikov_weight(float(dist), rho) for dist in distances],
    #    dtype=np.float64,
    #)
    
    # Normalize meta-features before distance.
    all_features = np.vstack([source_meta_features, target_meta_features[None, :]])
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    std[std < eps] = 1.0

    source_scaled = (source_meta_features - mean) / std
    target_scaled = (target_meta_features - mean) / std

    distances = np.linalg.norm(
        source_scaled - target_scaled[None, :],
        axis=1,
    )

    weights = np.array(
        [epanechnikov_weight(float(dist), rho) for dist in distances],
        dtype=np.float64,
    )

    # Fallback if every source is outside the Epanechnikov bandwidth.
    if weights.sum() <= eps:
        return np.ones(source_meta_features.shape[0], dtype=np.float64) / source_meta_features.shape[0]

    return weights / weights.sum()


def compute_taf_r_weights(
    source_surrogates: list[SourceTaskSurrogate],
    x_obs: NDArray[np.float64],
    y_obs: NDArray[np.float64],
    rho: float,
) -> NDArray[np.float64]:
    """Compute TAF-R weights from ranking agreement on current target observations."""
    x_obs = np.asarray(x_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    if x_obs.ndim != 2:
        raise ValueError("x_obs must have shape (n, d).")
    if y_obs.ndim != 1 or y_obs.shape[0] != x_obs.shape[0]:
        raise ValueError("y_obs must have shape (n,) and match x_obs rows.")

    n_sources = len(source_surrogates)
    n = y_obs.shape[0]
    if n_sources == 0:
        return np.zeros(0, dtype=np.float64)
    if n < 2:
        return np.ones(n_sources, dtype=np.float64)

    weights: list[float] = []
    for source in source_surrogates:
        mu_source, _ = source.gp.posterior(x_obs)
        disagreements = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                target_diff = float(y_obs[i] - y_obs[j])
                source_diff = float(mu_source[i] - mu_source[j])
                if abs(target_diff) <= 1e-12 or abs(source_diff) <= 1e-12:
                    continue
                if (target_diff > 0.0) != (source_diff > 0.0):
                    disagreements += 1
                total += 1
        distance = 0.0 if total == 0 else float(disagreements / total)
        weights.append(epanechnikov_weight(distance, rho))
    return np.asarray(weights, dtype=np.float64)


def taf_m_acquisition(
    x: NDArray[np.float64],
    target_gp: GPScratch | None,
    target_best_y: float,
    source_surrogates: list[SourceTaskSurrogate],
    source_weights: NDArray[np.float64],
    target_weight: float = 1.0,
    source_improvement_mode: str = "relu",
    source_improvement_temperature: float = 0.05,
    target_ei_floor: float = 0.0,
) -> float | NDArray[np.float64]:
    """Compute TAF-M acquisition for one point or a batch of points."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x_eval = x[None, :]
        single_input = True
    elif x.ndim == 2:
        x_eval = x
        single_input = False
    else:
        raise ValueError("x must have shape (d,) or (n, d).")

    source_weights = np.asarray(source_weights, dtype=np.float64)
    if source_weights.ndim != 1:
        raise ValueError("source_weights must have shape (n_sources,).")
    if len(source_surrogates) != source_weights.shape[0]:
        raise ValueError("source_surrogates and source_weights length mismatch.")

    if float(target_weight) > 0.0:
        if target_gp is None:
            raise ValueError("target_gp is required when target_weight > 0.")
        mean_t, var_t = target_gp.posterior(x_eval)
        target_ei = expected_improvement_maximization(mean_t, var_t, target_best_y)
        if target_ei_floor > 0.0:
            target_ei = np.maximum(target_ei, float(target_ei_floor))
    else:
        target_ei = np.zeros(x_eval.shape[0], dtype=np.float64)

    numerator = float(target_weight) * target_ei
    denominator = np.full(
        x_eval.shape[0],
        fill_value=float(target_weight),
        dtype=np.float64,
    )

    for source, w_i in zip(source_surrogates, source_weights):
        w = float(w_i)
        if w <= 0.0:
            continue
        mean_i, _ = source.gp.posterior(x_eval)
        source_ref = (
            float(source.reference_y)
            if source.reference_y is not None
            else float(source.best_y)
        )
        delta = mean_i - source_ref
        if source_improvement_mode == "relu":
            source_imp = np.maximum(delta, 0.0)
        elif source_improvement_mode == "softplus":
            tau = float(max(source_improvement_temperature, 1e-12))
            # Smooth non-negative approximation of ReLU to reduce sparsity.
            source_imp = tau * np.log1p(np.exp(delta / tau))
        else:
            raise ValueError(
                "source_improvement_mode must be 'relu' or 'softplus'."
            )
        numerator = numerator + w * source_imp
        denominator = denominator + w

    value = numerator / np.maximum(denominator, 1e-12)
    if single_input:
        return float(value[0])
    return value.astype(np.float64)
