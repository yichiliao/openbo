//! Acquisition functions and exact Pareto geometry for maximization.
//!
//! Zero predictive variance has its deterministic mathematical meaning. Negative
//! variances, nonfinite inputs and inconsistent dimensions are rejected.

mod mobo;
mod multi_output;
mod sampling;
mod taf_eipv;
pub use mobo::*;
pub use multi_output::*;
pub use sampling::*;
pub use taf_eipv::*;

use crate::{Matrix, Result};
use anyhow::{bail, ensure};
use statrs::function::erf::erfc;

pub(crate) fn matrix_dim(x: &Matrix, expected: Option<usize>) -> Result<usize> {
    let d = expected.or_else(|| x.first().map(Vec::len)).unwrap_or(0);
    ensure!(
        x.iter()
            .all(|r| r.len() == d && r.iter().all(|v| v.is_finite())),
        "matrix must be rectangular and finite with {d} columns"
    );
    Ok(d)
}

pub(crate) fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}
pub(crate) fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Analytic expected improvement in maximization form.
pub fn expected_improvement(mean: f64, variance: f64, best: f64) -> Result<f64> {
    ensure!(
        mean.is_finite() && best.is_finite() && variance.is_finite() && variance >= 0.0,
        "EI requires finite mean/incumbent and nonnegative variance"
    );
    let delta = mean - best;
    ensure!(
        delta != f64::INFINITY,
        "EI exceeds the representable f64 range"
    );
    if variance == 0.0 {
        return Ok(delta.max(0.0));
    }
    let std = variance.sqrt();
    let z = delta / std;
    // The asymptotic expression avoids catastrophic cancellation far below best.
    if z < -10.0 {
        return Ok(log_expected_improvement(mean, variance, best)?.exp());
    }
    Ok((delta * normal_cdf(z) + std * normal_pdf(z)).max(0.0))
}

pub fn expected_improvement_minimization(mean: f64, variance: f64, best: f64) -> Result<f64> {
    expected_improvement(-mean, variance, -best)
}

/// Probability of strict improvement in maximization form. A deterministic
/// prediction equal to the incumbent has zero probability of strict gain.
pub fn probability_of_improvement(mean: f64, variance: f64, best: f64) -> Result<f64> {
    ensure!(
        mean.is_finite() && best.is_finite() && variance.is_finite() && variance >= 0.0,
        "PI requires finite mean/incumbent and nonnegative variance"
    );
    if variance == 0.0 {
        return Ok(if mean > best { 1.0 } else { 0.0 });
    }
    Ok(normal_cdf((mean - best) / variance.sqrt()))
}

/// GP upper confidence bound `mean + sqrt(beta * variance)`.
pub fn upper_confidence_bound(mean: f64, variance: f64, beta: f64) -> Result<f64> {
    ensure!(
        mean.is_finite()
            && variance.is_finite()
            && variance >= 0.0
            && beta.is_finite()
            && beta >= 0.0,
        "UCB requires finite inputs and nonnegative variance/beta"
    );
    let value = mean + beta.sqrt() * variance.sqrt();
    ensure!(value.is_finite(), "UCB overflowed");
    Ok(value)
}

/// Stable logarithm of EI, including negative-tail values too small to represent.
pub fn log_expected_improvement(mean: f64, variance: f64, best: f64) -> Result<f64> {
    ensure!(
        mean.is_finite() && best.is_finite() && variance.is_finite() && variance >= 0.0,
        "log EI requires finite inputs and nonnegative variance"
    );
    let delta = mean - best;
    if delta == f64::INFINITY {
        // Opposite-sign finite extremes can have an overflowing difference
        // while its logarithm is still perfectly representable.
        let scale = mean.abs().max(best.abs());
        return Ok(scale.ln() + (mean / scale - best / scale).ln());
    }
    if variance == 0.0 {
        return Ok(delta.max(0.0).ln());
    }
    let std = variance.sqrt();
    let z = delta / std;
    if z < -10.0 {
        // phi(z) + z Phi(z) = phi(z)/z² (1 - 3/z² + 15/z⁴ - ...).
        let q = 1.0 / (z * z);
        let mut sum = 1.0;
        let mut term = 1.0_f64;
        for k in 1..100 {
            let next = -term * (2 * k + 1) as f64 * q;
            if next.abs() >= term.abs() {
                break;
            }
            sum += next;
            term = next;
            if next.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        return Ok(std.ln()
            - 0.5 * z * z
            - 0.5 * (2.0 * std::f64::consts::PI).ln()
            - 2.0 * (-z).ln()
            + sum.ln());
    }
    Ok((delta * normal_cdf(z) + std * normal_pdf(z)).max(0.0).ln())
}

/// Pareto relation (+1 if a dominates, -1 if b dominates, zero otherwise).
pub fn pareto_relation(a: &[f64], b: &[f64]) -> Result<i8> {
    ensure!(
        !a.is_empty() && a.len() == b.len() && a.iter().chain(b).all(|v| v.is_finite()),
        "objective vectors must be finite and equally sized"
    );
    let a_ge = a.iter().zip(b).all(|(x, y)| *x >= *y - 1e-12);
    let b_ge = b.iter().zip(a).all(|(x, y)| *x >= *y - 1e-12);
    if a_ge && a.iter().zip(b).any(|(x, y)| *x > *y + 1e-12) {
        Ok(1)
    } else if b_ge && b.iter().zip(a).any(|(x, y)| *x > *y + 1e-12) {
        Ok(-1)
    } else {
        Ok(0)
    }
}

/// Distinct nondominated rows, preserving the first duplicate and original order.
pub fn pareto_front(y: &Matrix) -> Result<Matrix> {
    let d = matrix_dim(y, None)?;
    ensure!(y.is_empty() || d > 0, "at least one objective is required");
    let mut out = Vec::new();
    for (i, row) in y.iter().enumerate() {
        if y.iter().enumerate().any(|(j, other)| {
            j != i && other.iter().zip(row).all(|(a, b)| a >= b) && (j < i || other != row)
        }) {
            continue;
        }
        out.push(row.clone());
    }
    Ok(out)
}

/// Exact union volume of axis-aligned reference-to-point boxes in any dimension.
/// Uses a recursive sweep rather than allocating an exponential full grid.
pub fn hypervolume(y: &Matrix, reference: &[f64]) -> Result<f64> {
    ensure!(
        !reference.is_empty() && reference.iter().all(|v| v.is_finite()),
        "reference must be finite and nonempty"
    );
    matrix_dim(y, Some(reference.len()))?;
    let front = pareto_front(
        &y.iter()
            .filter(|row| row.iter().zip(reference).all(|(a, b)| a > b))
            .cloned()
            .collect(),
    )?;
    fn sweep(points: &Matrix, r: &[f64]) -> f64 {
        if points.is_empty() {
            return 0.0;
        }
        if r.len() == 1 {
            return points.iter().map(|p| p[0] - r[0]).fold(0.0, f64::max);
        }
        let mut levels: Vec<f64> = points.iter().map(|p| p[0]).collect();
        levels.push(r[0]);
        levels.sort_by(f64::total_cmp);
        levels.dedup();
        levels
            .windows(2)
            .map(|w| {
                let active: Matrix = points
                    .iter()
                    .filter(|p| p[0] >= w[1])
                    .map(|p| p[1..].to_vec())
                    .collect();
                (w[1] - w[0]) * sweep(&active, &r[1..])
            })
            .sum()
    }
    let value = sweep(&front, reference);
    ensure!(value.is_finite(), "hypervolume overflowed");
    Ok(value)
}

/// Exact front-aligned nondominated cells inside a finite box.
pub fn nondominated_cells(
    front: &Matrix,
    lower: &[f64],
    upper: &[f64],
) -> Result<(Matrix, Matrix)> {
    ensure!(
        !lower.is_empty()
            && lower.len() == upper.len()
            && lower
                .iter()
                .zip(upper)
                .all(|(a, b)| a.is_finite() && b.is_finite() && b >= a && (b - a).is_finite()),
        "invalid cell reference box"
    );
    matrix_dim(front, Some(lower.len()))?;
    let clipped: Matrix = front
        .iter()
        .map(|p| {
            p.iter()
                .enumerate()
                .map(|(d, v)| v.clamp(lower[d], upper[d]))
                .collect()
        })
        .collect();
    let coords: Vec<Vec<f64>> = (0..lower.len())
        .map(|d| {
            let mut c: Vec<f64> = clipped.iter().map(|p| p[d]).collect();
            c.push(lower[d]);
            c.push(upper[d]);
            c.sort_by(f64::total_cmp);
            c.dedup();
            c
        })
        .collect();
    let count = coords
        .iter()
        .try_fold(1usize, |n, c| n.checked_mul(c.len().saturating_sub(1)))
        .ok_or_else(|| anyhow::anyhow!("cell count overflow"))?;
    ensure!(
        count <= 2_000_000,
        "cell grid exceeds 2 million cells; reduce front or objectives"
    );
    let mut lows = Vec::new();
    let mut highs = Vec::new();
    for mut index in 0..count {
        let mut lo = vec![0.0; lower.len()];
        let mut hi = lo.clone();
        for d in (0..lower.len()).rev() {
            let width = coords[d].len() - 1;
            let j = index % width;
            index /= width;
            lo[d] = coords[d][j];
            hi[d] = coords[d][j + 1];
        }
        if !clipped
            .iter()
            .any(|p| p.iter().zip(&hi).all(|(a, b)| a >= b))
        {
            lows.push(lo);
            highs.push(hi);
        }
    }
    Ok((lows, highs))
}

pub fn hypervolume_gain(y: &[f64], lower: &Matrix, upper: &Matrix) -> Result<f64> {
    ensure!(
        !y.is_empty() && y.iter().all(|v| v.is_finite()) && lower.len() == upper.len(),
        "invalid gain inputs"
    );
    matrix_dim(lower, Some(y.len()))?;
    matrix_dim(upper, Some(y.len()))?;
    ensure!(
        lower
            .iter()
            .zip(upper)
            .all(|(l, u)| l.iter().zip(u).all(|(a, b)| b >= a)),
        "cell upper corners must exceed lower corners"
    );
    Ok(lower
        .iter()
        .zip(upper)
        .map(|(l, u)| {
            y.iter()
                .enumerate()
                .map(|(d, v)| (v.min(u[d]) - l[d]).max(0.0))
                .product::<f64>()
        })
        .sum())
}

pub fn epanechnikov_weight(distance: f64, rho: f64) -> Result<f64> {
    ensure!(
        rho.is_finite() && rho > 0.0 && distance.is_finite() && distance >= 0.0,
        "bandwidth must be positive and distance nonnegative"
    );
    Ok(if distance <= rho {
        0.75 * (1.0 - (distance / rho).powi(2))
    } else {
        0.0
    })
}

/// TAF-M weights after joint population standardization of source and target metadata.
pub fn meta_weights(source: &Matrix, target: &[f64], rho: f64) -> Result<Vec<f64>> {
    epanechnikov_weight(0.0, rho)?;
    ensure!(
        !target.is_empty() && target.iter().all(|v| v.is_finite()),
        "target metadata must be finite and nonempty"
    );
    matrix_dim(source, Some(target.len()))?;
    if source.is_empty() {
        return Ok(Vec::new());
    }
    let n = (source.len() + 1) as f64;
    let std: Vec<f64> = (0..target.len())
        .map(|d| {
            let mean = (target[d] + source.iter().map(|r| r[d]).sum::<f64>()) / n;
            let v = ((target[d] - mean).powi(2)
                + source.iter().map(|r| (r[d] - mean).powi(2)).sum::<f64>())
                / n;
            if v.sqrt() < 1e-12 { 1.0 } else { v.sqrt() }
        })
        .collect();
    let mut w: Vec<f64> = source
        .iter()
        .map(|row| {
            epanechnikov_weight(
                row.iter()
                    .enumerate()
                    .map(|(d, v)| ((v - target[d]) / std[d]).powi(2))
                    .sum::<f64>()
                    .sqrt(),
                rho,
            )
        })
        .collect::<Result<_>>()?;
    let total: f64 = w.iter().sum();
    if total <= 1e-12 {
        w.fill(1.0 / source.len() as f64);
    } else {
        w.iter_mut().for_each(|v| *v /= total);
    }
    Ok(w)
}

/// Ranking weights: `scalar` preserves scalar TAF's raw kernel weights;
/// `taf_r`/`objective` scores objective-wise strict rankings and normalizes;
/// `taf_r_pareto` scores informative Pareto relations and normalizes.
pub fn ranking_weights(
    target: &Matrix,
    sources: &[Matrix],
    rho: f64,
    mode: &str,
) -> Result<Vec<f64>> {
    epanechnikov_weight(0.0, rho)?;
    ensure!(
        matches!(
            mode,
            "scalar" | "taf_r" | "objective" | "taf_r_pareto" | "pareto"
        ),
        "unknown ranking mode {mode}"
    );
    let d = matrix_dim(target, None)?;
    for s in sources {
        ensure!(
            s.len() == target.len(),
            "source prediction row count mismatch"
        );
        matrix_dim(s, Some(d))?;
    }
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    if target.len() < 2 {
        return Ok(vec![
            if mode == "scalar" {
                1.0
            } else {
                1.0 / sources.len() as f64
            };
            sources.len()
        ]);
    }
    ensure!(
        d > 0 && (mode != "scalar" || d == 1),
        "invalid objective dimensions for ranking"
    );
    let mut weights = Vec::new();
    for source in sources {
        let mut disagreements = 0.0;
        let mut total = 0usize;
        let mut strict = 0usize;
        for i in 0..target.len() {
            for j in i + 1..target.len() {
                if mode == "taf_r_pareto" || mode == "pareto" {
                    let t = pareto_relation(&target[i], &target[j])?;
                    let s = pareto_relation(&source[i], &source[j])?;
                    if s != 0 {
                        strict += 1;
                    }
                    if t == 0 && s == 0 {
                        continue;
                    }
                    total += 1;
                    if t != s {
                        disagreements += if t != 0 && s != 0 { 1.0 } else { 0.5 };
                    }
                } else {
                    for k in 0..d {
                        let t = target[i][k] - target[j][k];
                        let s = source[i][k] - source[j][k];
                        if t.abs() <= 1e-12 || (mode == "scalar" && s.abs() <= 1e-12) {
                            continue;
                        }
                        total += 1;
                        if s.abs() > 1e-12 {
                            strict += 1;
                        }
                        if s.abs() <= 1e-12 || t.signum() != s.signum() {
                            disagreements += 1.0;
                        }
                    }
                }
            }
        }
        if mode != "scalar" && (strict == 0 || total == 0) {
            weights.push(0.0);
        } else {
            weights.push(epanechnikov_weight(
                if total == 0 {
                    0.0
                } else {
                    disagreements / total as f64
                },
                rho,
            )?);
        }
    }
    if mode != "scalar" {
        let total: f64 = weights.iter().sum();
        if total > 1e-12 {
            weights.iter_mut().for_each(|v| *v /= total);
        } else {
            weights.fill(0.0);
        }
    }
    Ok(weights)
}

/// Original TAF+ ordered-pair distance. A source tie against strict target order
/// contributes one disagreement among the two ordered comparisons.
pub fn ranking_distance(target: &[f64], source: &[f64]) -> Result<f64> {
    ensure!(
        target.len() == source.len() && target.iter().chain(source).all(|v| v.is_finite()),
        "ranking vectors must be finite and same length"
    );
    if target.len() < 2 {
        return Ok(0.0);
    }
    let mut n = 0usize;
    for i in 0..target.len() {
        for j in 0..target.len() {
            if i != j && (target[i] > target[j]) != (source[i] > source[j]) {
                n += 1;
            }
        }
    }
    Ok(n as f64 / (target.len() * (target.len() - 1)) as f64)
}

/// Per-source, per-objective references: the largest predicted source mean at
/// configurations already evaluated on the target.
pub fn source_references(source_mean_at_target: &[Matrix]) -> Result<Matrix> {
    let mut references = Vec::new();
    let mut shape = None;
    for mean in source_mean_at_target {
        let d = matrix_dim(mean, None)?;
        ensure!(
            !mean.is_empty() && d > 0,
            "references need target observations"
        );
        if let Some((n, columns)) = shape {
            ensure!(
                mean.len() == n && d == columns,
                "source reference shapes disagree"
            );
        }
        shape = Some((mean.len(), d));
        references.push(
            (0..d)
                .map(|k| {
                    mean.iter()
                        .map(|row| row[k])
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .collect(),
        );
    }
    Ok(references)
}

/// TAF+ ranking weights use the original ordered-pair descriptor, independently
/// for every objective. The target's self-weight is 3/4.
pub fn taf_plus_ranking_weights(
    target: &Matrix,
    sources: &[Matrix],
    rho: f64,
) -> Result<(Matrix, f64)> {
    let d = matrix_dim(target, None)?;
    ensure!(
        !target.is_empty() && d > 0,
        "target observations are required"
    );
    let target_weight = epanechnikov_weight(0.0, rho)?;
    let mut weights = Vec::new();
    for source in sources {
        ensure!(source.len() == target.len(), "ranking row counts disagree");
        matrix_dim(source, Some(d))?;
        let mut row = Vec::new();
        for k in 0..d {
            let y: Vec<f64> = target.iter().map(|r| r[k]).collect();
            let mu: Vec<f64> = source.iter().map(|r| r[k]).collect();
            row.push(epanechnikov_weight(ranking_distance(&y, &mu)?, rho)?);
        }
        weights.push(row);
    }
    Ok((weights, target_weight))
}

/// Raw TAF-PoE precision weights, including every source and target model.
pub fn poe_weights(variance: &[Matrix]) -> Result<Vec<Matrix>> {
    ensure!(!variance.is_empty(), "at least one model is required");
    let d = matrix_dim(&variance[0], None)?;
    let n = variance[0].len();
    for var in variance {
        ensure!(var.len() == n, "variance row counts disagree");
        matrix_dim(var, Some(d))?;
        ensure!(
            var.iter().flatten().all(|v| *v >= 0.0),
            "variances must be nonnegative"
        );
    }
    Ok(variance
        .iter()
        .map(|model| {
            model
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|v| 1.0 / v.max(1e-300) / variance.len() as f64)
                        .collect()
                })
                .collect()
        })
        .collect())
}

/// Population mean before any target observations, combining objectives within
/// each source before combining models, as specified in the TAF+ paper.
pub fn population_mean(
    source_mean: &[Matrix],
    source_variance: &[Matrix],
    objective_weights: &[f64],
    weight_mode: &str,
) -> Result<Vec<f64>> {
    ensure!(
        !source_mean.is_empty() && source_mean.len() == source_variance.len(),
        "source models and variances must agree and be nonempty"
    );
    ensure!(
        matches!(weight_mode, "poe" | "taf_r"),
        "unknown population weight mode"
    );
    let d = matrix_dim(&source_mean[0], None)?;
    let n = source_mean[0].len();
    ensure!(
        objective_weights.len() == d
            && d > 0
            && objective_weights.iter().all(|v| v.is_finite() && *v >= 0.0)
            && (objective_weights.iter().sum::<f64>() - 1.0).abs() < 1e-9,
        "objective weights must be nonnegative and sum to one"
    );
    for (mean, var) in source_mean.iter().zip(source_variance) {
        ensure!(
            mean.len() == n && var.len() == n,
            "source candidate counts disagree"
        );
        matrix_dim(mean, Some(d))?;
        matrix_dim(var, Some(d))?;
        ensure!(
            var.iter().flatten().all(|v| *v >= 0.0),
            "variances must be nonnegative"
        );
    }
    let mut out = Vec::new();
    for i in 0..n {
        let scale = source_variance
            .iter()
            .flat_map(|v| v[i].iter())
            .copied()
            .fold(f64::INFINITY, f64::min)
            .max(1e-300);
        let mut total = 0.0;
        let mut denominator = 0.0;
        for (mean, var) in source_mean.iter().zip(source_variance) {
            let weight = if weight_mode == "poe" {
                (0..d)
                    .map(|k| objective_weights[k] * scale / var[i][k].max(1e-300))
                    .sum::<f64>()
            } else {
                1.0
            };
            total += weight
                * mean[i]
                    .iter()
                    .zip(objective_weights)
                    .map(|(m, a)| m * a)
                    .sum::<f64>();
            denominator += weight;
        }
        ensure!(denominator > 0.0, "population has no active precision");
        out.push(total / denominator);
    }
    Ok(out)
}

pub fn default_incumbents(
    user_observations: &[f64],
    population_predictions: &[f64],
) -> Result<(f64, f64)> {
    ensure!(
        user_observations
            .iter()
            .chain(population_predictions)
            .all(|v| v.is_finite()),
        "observations and population predictions must be finite"
    );
    let values = if user_observations.is_empty() {
        population_predictions
    } else {
        user_observations
    };
    ensure!(
        !values.is_empty(),
        "observations or population predictions are required"
    );
    let best = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok((best, best))
}

pub fn population_weight(t: usize, alpha1: f64, alpha2: f64) -> Result<f64> {
    ensure!(
        t >= 1 && alpha1.is_finite() && alpha1 >= 0.0 && alpha2.is_finite() && alpha2 > 0.0,
        "invalid population decay schedule"
    );
    Ok((1.0 - (t as f64 - alpha1) * alpha2).clamp(0.0, 1.0))
}

pub fn decay_factor(t: usize, d1: usize, d2: f64) -> Result<f64> {
    ensure!(d2 <= 1.0, "decay rate must be at most one");
    population_weight(t, d1 as f64, d2)
}

pub fn mixed_expected_improvement(
    weight: f64,
    population: Option<(f64, f64)>,
    user: Option<(f64, f64)>,
    best_pop: f64,
    best_user: f64,
) -> Result<f64> {
    ensure!(
        weight.is_finite() && (0.0..=1.0).contains(&weight),
        "population weight must be in [0,1]"
    );
    let mut value = 0.0;
    if weight > 0.0 {
        let (m, v) =
            population.ok_or_else(|| anyhow::anyhow!("population prediction is required"))?;
        value += weight * expected_improvement(m, v, best_pop)?;
    }
    if weight < 1.0 {
        let (m, v) = user.ok_or_else(|| anyhow::anyhow!("user prediction is required"))?;
        value += (1.0 - weight) * expected_improvement(m, v, best_user)?;
    }
    Ok(value)
}

/// Combine target EI and deterministic source improvements at a single point.
pub fn taf_acquisition(
    target_ei: Option<f64>,
    source_means: &[f64],
    references: &[f64],
    weights: &[f64],
    target_weight: f64,
    temperature: Option<f64>,
) -> Result<f64> {
    ensure!(
        source_means.len() == references.len() && weights.len() == references.len(),
        "source vectors must agree"
    );
    ensure!(
        target_weight.is_finite()
            && source_means
                .iter()
                .chain(references)
                .chain(weights)
                .all(|v| v.is_finite()),
        "TAF inputs must be finite"
    );
    if let Some(t) = temperature {
        ensure!(
            t.is_finite() && t > 0.0,
            "softplus temperature must be positive"
        );
    }
    let mut sum = 0.0;
    let mut denominator = 0.0;
    if target_weight > 0.0 {
        let target = target_ei.ok_or_else(|| anyhow::anyhow!("target EI required"))?;
        ensure!(target.is_finite() && target >= 0.0, "invalid target EI");
        sum += target_weight * target;
        denominator += target_weight;
    }
    for ((mu, r), w) in source_means.iter().zip(references).zip(weights) {
        if *w <= 0.0 {
            continue;
        }
        let delta = mu - r;
        let imp = if let Some(t) = temperature {
            delta.max(0.0) + t * (-(delta / t).abs()).exp().ln_1p()
        } else {
            delta.max(0.0)
        };
        sum += w * imp;
        denominator += w;
    }
    ensure!(denominator > 0.0, "TAF has no active terms");
    Ok(sum / denominator)
}

/// Stable TAF-EHVI blend, with all terms already in logarithmic units.
pub fn weighted_log_blend(
    target: Option<f64>,
    sources: &[f64],
    weights: &[f64],
    target_weight: f64,
) -> Result<f64> {
    ensure!(
        sources.len() == weights.len()
            && target_weight.is_finite()
            && weights.iter().all(|v| v.is_finite()),
        "invalid log blend weights"
    );
    let mut terms = Vec::new();
    let mut log_weights = Vec::new();
    if target_weight > 0.0 {
        let v = target.ok_or_else(|| anyhow::anyhow!("target term required"))?;
        ensure!(!v.is_nan() && v != f64::INFINITY, "invalid log acquisition");
        terms.push(target_weight.ln() + v);
        log_weights.push(target_weight.ln());
    }
    for (v, w) in sources.iter().zip(weights) {
        if *w > 0.0 {
            ensure!(
                !v.is_nan() && *v != f64::INFINITY,
                "invalid source log term"
            );
            terms.push(w.ln() + v);
            log_weights.push(w.ln());
        }
    }
    ensure!(!terms.is_empty(), "log blend has no active terms");
    let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return Ok(max);
    }
    let weight_max = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let log_denominator = weight_max
        + log_weights
            .iter()
            .map(|w| (w - weight_max).exp())
            .sum::<f64>()
            .ln();
    Ok(max + terms.iter().map(|v| (v - max).exp()).sum::<f64>().ln() - log_denominator)
}

pub struct TafPlusInput<'a> {
    pub target_mean: &'a Matrix,
    pub target_variance: &'a Matrix,
    pub target_best: &'a [f64],
    pub source_mean: &'a [Matrix],
    pub source_variance: &'a [Matrix],
    pub source_reference: &'a Matrix,
    pub objective_weights: &'a [f64],
    pub decay: f64,
    pub weight_mode: &'a str,
    pub ranking: Option<(&'a Matrix, f64)>,
}

pub fn taf_plus_acquisition(p: TafPlusInput<'_>) -> Result<Vec<f64>> {
    let d = matrix_dim(p.target_mean, None)?;
    ensure!(
        d > 0 && p.target_mean.len() == p.target_variance.len() && p.target_best.len() == d,
        "invalid target shapes"
    );
    matrix_dim(p.target_variance, Some(d))?;
    ensure!(
        p.objective_weights.len() == d
            && p.objective_weights
                .iter()
                .all(|a| a.is_finite() && *a >= 0.0)
            && (p.objective_weights.iter().sum::<f64>() - 1.0).abs() < 1e-9,
        "objective weights must be nonnegative and sum to one"
    );
    ensure!(
        p.decay.is_finite() && (0.0..=1.0).contains(&p.decay),
        "decay must be in [0,1]"
    );
    let m = p.source_mean.len();
    let n = p.target_mean.len();
    ensure!(
        p.source_variance.len() == m && p.source_reference.len() == m,
        "source shape mismatch"
    );
    matrix_dim(p.source_reference, Some(d))?;
    for (mean, var) in p.source_mean.iter().zip(p.source_variance) {
        ensure!(
            mean.len() == n && var.len() == n,
            "source candidate count mismatch"
        );
        matrix_dim(mean, Some(d))?;
        matrix_dim(var, Some(d))?;
    }
    ensure!(
        p.source_variance
            .iter()
            .flat_map(|m| m.iter().flatten())
            .chain(p.target_variance.iter().flatten())
            .all(|v| *v >= 0.0),
        "variances must be nonnegative"
    );
    match p.weight_mode {
        "poe" => {}
        "taf_r" => {
            let (w, t) = p
                .ranking
                .ok_or_else(|| anyhow::anyhow!("ranking weights required"))?;
            ensure!(
                w.len() == m && t.is_finite() && t >= 0.0,
                "invalid ranking weights"
            );
            matrix_dim(w, Some(d))?;
            ensure!(
                w.iter().flatten().all(|v| *v >= 0.0),
                "negative ranking weight"
            );
        }
        _ => bail!("unknown TAF+ weight mode"),
    }
    let mut out = Vec::new();
    for i in 0..n {
        let target_ei = (0..d)
            .map(|k| {
                Ok(p.objective_weights[k]
                    * expected_improvement(
                        p.target_mean[i][k],
                        p.target_variance[i][k],
                        p.target_best[k],
                    )?)
            })
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum::<f64>();
        // Scale every precision by a common minimum variance to avoid 1/0 or
        // overflow when several model variances are nearly deterministic.
        let scale = (0..d)
            .map(|k| p.target_variance[i][k])
            .chain(p.source_variance.iter().flat_map(|v| v[i].iter().copied()))
            .fold(f64::INFINITY, f64::min)
            .max(1e-300);
        let weight = |j: usize, k: usize| -> f64 {
            if p.weight_mode == "poe" {
                scale
                    / (if j == m {
                        p.target_variance[i][k]
                    } else {
                        p.source_variance[j][i][k]
                    })
                    .max(1e-300)
            } else {
                let (w, t) = p.ranking.unwrap();
                if j == m { t } else { w[j][k] }
            }
        };
        let tw = (0..d)
            .map(|k| p.objective_weights[k] * weight(m, k))
            .sum::<f64>();
        let mut numerator = tw * target_ei;
        let mut denom = tw;
        for j in 0..m {
            let sw = p.decay
                * (0..d)
                    .map(|k| p.objective_weights[k] * weight(j, k))
                    .sum::<f64>();
            let imp = (0..d)
                .map(|k| {
                    p.objective_weights[k]
                        * (p.source_mean[j][i][k] - p.source_reference[j][k]).max(0.0)
                })
                .sum::<f64>();
            numerator += sw * imp;
            denom += sw;
        }
        ensure!(denom > 0.0, "TAF+ has no active terms");
        out.push(numerator / denom);
    }
    Ok(out)
}

/// Simplex projection useful for learning nonnegative objective weights.
pub fn project_objective_weights(values: &[f64]) -> Result<Vec<f64>> {
    ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "weights must be finite and nonempty"
    );
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let mut sum = 0.0;
    let mut theta = 0.0;
    for (i, v) in sorted.iter().enumerate() {
        sum += v;
        let t = (sum - 1.0) / (i + 1) as f64;
        if *v > t {
            theta = t;
        }
    }
    Ok(values.iter().map(|v| (v - theta).max(0.0)).collect())
}

/// Fit objective weights to scalar user ratings by projected least squares.
pub fn learn_objective_weights(
    objectives: &Matrix,
    ratings: &[f64],
    iterations: usize,
) -> Result<Vec<f64>> {
    let d = matrix_dim(objectives, None)?;
    ensure!(
        d > 0 && objectives.len() == ratings.len() && ratings.iter().all(|v| v.is_finite()),
        "invalid objective ratings"
    );
    let mut w = vec![1.0 / d as f64; d];
    let lipschitz = 2.0
        * objectives
            .iter()
            .flatten()
            .map(|v| v * v)
            .sum::<f64>()
            .max(1e-12)
        / objectives.len() as f64;
    for _ in 0..iterations {
        let mut gradient = vec![0.0; d];
        for (row, y) in objectives.iter().zip(ratings) {
            let error = row.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() - y;
            for k in 0..d {
                gradient[k] += 2.0 * error * row[k] / objectives.len() as f64;
            }
        }
        w = project_objective_weights(
            &w.iter()
                .zip(gradient)
                .map(|(a, g)| a - g / lipschitz)
                .collect::<Vec<_>>(),
        )?;
    }
    Ok(w)
}
