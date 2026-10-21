//! Original TAF blended in Pareto-volume units (upstream TAF-EIPV).
//!
//! Each source measures capped hypervolume gain relative to its own predictions
//! at designs already observed by the target. Source uncertainty is discarded.
//! Ranking weights use the original ordered-pair descriptors, averaged across
//! objectives; these deliberately differ from the TAF-EHVI ranking estimator.

use super::{
    epanechnikov_weight, hypervolume_gain, matrix_dim, nondominated_cells, pareto_front,
    ranking_distance, weighted_log_blend,
};
use crate::{Matrix, Result};
use anyhow::ensure;

/// Log-acquisition floor used by the upstream MOBO and TAF-EIPV optimizers.
pub const LOG_FLOOR: f64 = -1e4;

/// Finite cells not dominated by a source's reference front. With no target
/// observations, the empty reference leaves the entire box available.
pub fn reference_cells(
    reference_front: &Matrix,
    v_ref: &[f64],
    w_ref: &[f64],
) -> Result<(Matrix, Matrix)> {
    ensure!(
        !v_ref.is_empty() && v_ref.len() == w_ref.len(),
        "reference corners must have one value per objective"
    );
    ensure!(
        v_ref
            .iter()
            .zip(w_ref)
            .all(|(lo, hi)| lo.is_finite() && hi.is_finite() && hi >= lo && (hi - lo).is_finite()),
        "reference corners must be finite increasing vectors with a finite span"
    );
    matrix_dim(reference_front, Some(v_ref.len()))?;
    if reference_front.is_empty() {
        return Ok((vec![v_ref.to_vec()], vec![w_ref.to_vec()]));
    }
    // Dominated observations cannot change the gain and must not inflate the
    // coordinate grid. Upstream nondominated_cells performs this reduction.
    nondominated_cells(&pareto_front(reference_front)?, v_ref, w_ref)
}

/// Hypervolume gained by each source posterior mean. Values above the upper
/// reference are capped; already-evaluated reference points have zero gain.
pub fn source_improvement(mean: &Matrix, cells: &(Matrix, Matrix)) -> Result<Vec<f64>> {
    let (lower, upper) = cells;
    ensure!(
        lower.len() == upper.len(),
        "reference cell counts must agree"
    );
    let d = mean
        .first()
        .or_else(|| lower.first())
        .or_else(|| upper.first())
        .map(Vec::len)
        .unwrap_or(0);
    matrix_dim(mean, Some(d))?;
    matrix_dim(lower, Some(d))?;
    matrix_dim(upper, Some(d))?;
    ensure!(
        lower.iter().zip(upper).all(|(lo, hi)| lo
            .iter()
            .zip(hi)
            .all(|(a, b)| b >= a && (b - a).is_finite())),
        "reference cells must have finite nonnegative spans"
    );
    mean.iter()
        .map(|row| {
            let value = hypervolume_gain(row, lower, upper)?;
            ensure!(
                value.is_finite(),
                "source hypervolume improvement overflowed"
            );
            Ok(value)
        })
        .collect()
}

/// Unnormalized TAF-R source weights. Each source's distance is the mean of
/// its per-objective ordered-pair distances. Ties carry the same half-disagreement
/// behavior as scalar TAF+, and fewer than two observations give weight 3/4.
pub fn taf_r_source_weights(
    y_target: &Matrix,
    source_mean_at_target: &[Matrix],
    rho: f64,
) -> Result<Vec<f64>> {
    epanechnikov_weight(0.0, rho)?;
    let d = matrix_dim(y_target, None)?;
    ensure!(
        y_target.is_empty() || d > 0,
        "target needs at least one objective"
    );
    for source in source_mean_at_target {
        ensure!(
            source.len() == y_target.len(),
            "source predictions must match target observation rows"
        );
        matrix_dim(source, Some(d))?;
    }
    if y_target.len() < 2 {
        return Ok(vec![0.75; source_mean_at_target.len()]);
    }
    source_mean_at_target
        .iter()
        .map(|source| {
            let mut distance = 0.0;
            for k in 0..d {
                let target: Vec<f64> = y_target.iter().map(|r| r[k]).collect();
                let prediction: Vec<f64> = source.iter().map(|r| r[k]).collect();
                distance += ranking_distance(&target, &prediction)?;
            }
            epanechnikov_weight(distance / d as f64, rho)
        })
        .collect()
}

/// Logarithm of the weighted average, floored at [`LOG_FLOOR`]. `source_terms`
/// is source-major `(M,n)` in ordinary volume units; target values are already
/// logarithmic `(n,)`. An absent target contributes neither value nor weight.
pub fn log_taf_eipv(
    log_target: Option<&[f64]>,
    source_terms: &Matrix,
    source_weights: &[f64],
    target_weight: f64,
) -> Result<Vec<f64>> {
    ensure!(
        source_terms.len() == source_weights.len(),
        "source terms and weights must have matching source counts"
    );
    ensure!(
        target_weight.is_finite()
            && target_weight >= 0.0
            && source_weights.iter().all(|v| v.is_finite() && *v >= 0.0),
        "TAF-EIPV weights must be finite and nonnegative"
    );
    let n = source_terms
        .first()
        .map(Vec::len)
        .or_else(|| log_target.map(<[f64]>::len))
        .unwrap_or(0);
    matrix_dim(source_terms, Some(n))?;
    ensure!(
        source_terms.iter().flatten().all(|v| *v >= 0.0),
        "source improvements must be nonnegative"
    );
    if let Some(target) = log_target {
        ensure!(
            target.len() == n && target.iter().all(|v| !v.is_nan() && *v != f64::INFINITY),
            "log target must match candidate count and contain finite values or negative infinity"
        );
    }
    let effective_target_weight = if log_target.is_some() {
        target_weight
    } else {
        0.0
    };
    ensure!(
        effective_target_weight > 0.0 || source_weights.iter().any(|v| *v > 0.0),
        "TAF-EIPV needs an active target or source"
    );
    (0..n)
        .map(|i| {
            let logs: Vec<f64> = source_terms.iter().map(|row| row[i].ln()).collect();
            Ok(weighted_log_blend(
                log_target.map(|target| target[i]),
                &logs,
                source_weights,
                effective_target_weight,
            )?
            .max(LOG_FLOOR))
        })
        .collect()
}
