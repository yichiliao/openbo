//! Native Monte Carlo EHVI/NEHVI and independent/correlated EIPV.
//!
//! EHVI integrates the actual unsmoothed gain. It intentionally differs from
//! BoTorch's differentiable qLogEHVI smoothing and autograd implementation.

use super::{
    expected_improvement, hypervolume, hypervolume_gain, matrix_dim, nondominated_cells, normal_cdf,
};
use crate::{Matrix, Result};
use anyhow::ensure;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

pub(crate) fn covariance_sqrt(cov: &Matrix, d: usize) -> Result<DMatrix<f64>> {
    ensure!(cov.len() == d, "covariance must have {d} rows");
    matrix_dim(cov, Some(d))?;
    let magnitude = cov.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max);
    if magnitude == 0.0 {
        return Ok(DMatrix::zeros(d, d));
    }
    ensure!(
        (0..d)
            .all(|i| (0..d).all(|j| (cov[i][j] / magnitude - cov[j][i] / magnitude).abs() <= 1e-9)),
        "covariance must be symmetric"
    );
    // Normalize before symmetrization/eigendecomposition: even two finite
    // diagonal entries can otherwise overflow while being averaged.
    let m = DMatrix::from_fn(d, d, |i, j| {
        0.5 * (cov[i][j] / magnitude) + 0.5 * (cov[j][i] / magnitude)
    });
    let eig = SymmetricEigen::new(m);
    let scale = eig.eigenvalues.iter().copied().fold(1.0, f64::max);
    ensure!(
        eig.eigenvalues.iter().all(|v| *v >= -1e-9 * scale),
        "covariance must be positive semidefinite"
    );
    Ok(eig.eigenvectors
        * DMatrix::from_diagonal(
            &eig.eigenvalues
                .map(|v| v.max(0.0).sqrt() * magnitude.sqrt()),
        ))
}

pub fn ehvi(
    mean: &[f64],
    cov: &Matrix,
    front: &Matrix,
    reference: &[f64],
    samples: usize,
    seed: u64,
) -> Result<f64> {
    ensure!(
        samples > 0
            && !mean.is_empty()
            && mean.len() == reference.len()
            && mean.iter().all(|v| v.is_finite()),
        "invalid EHVI mean/reference or sample count"
    );
    let sqrt = covariance_sqrt(cov, mean.len())?;
    let baseline = hypervolume(front, reference)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut total = 0.0;
    let mut augmented = front.clone();
    augmented.push(mean.to_vec());
    for _ in 0..samples {
        let z = DVector::from_fn(mean.len(), |_, _| StandardNormal.sample(&mut rng));
        let noise = &sqrt * z;
        *augmented.last_mut().unwrap() =
            mean.iter().enumerate().map(|(k, v)| v + noise[k]).collect();
        total += (hypervolume(&augmented, reference)? - baseline).max(0.0);
    }
    Ok(total / samples as f64)
}

/// Noisy expected hypervolume improvement from one joint posterior draw over
/// baseline plus q candidate rows. The covariance is point-major; baseline
/// uncertainty and its correlation with candidates are preserved.
pub fn nehvi(
    mean: &Matrix,
    joint_cov: &Matrix,
    n_baseline: usize,
    reference: &[f64],
    samples: usize,
    seed: u64,
) -> Result<f64> {
    let d = matrix_dim(mean, Some(reference.len()))?;
    ensure!(
        d > 0 && samples > 0 && mean.len() > n_baseline,
        "NEHVI needs candidates, objectives and positive samples"
    );
    let sqrt = covariance_sqrt(joint_cov, mean.len() * d)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut total = 0.0;
    for _ in 0..samples {
        let z = DVector::from_fn(mean.len() * d, |_, _| StandardNormal.sample(&mut rng));
        let noise = &sqrt * z;
        let draw: Matrix = mean
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(k, v)| v + noise[i * d + k])
                    .collect()
            })
            .collect();
        let baseline = hypervolume(&draw[..n_baseline].to_vec(), reference)?;
        total += (hypervolume(&draw, reference)? - baseline).max(0.0);
    }
    Ok(total / samples as f64)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EipvForm {
    Exact,
    Paper,
}

fn validate_cells(mean: &[f64], cov: &Matrix, lower: &Matrix, upper: &Matrix) -> Result<()> {
    ensure!(
        !mean.is_empty() && mean.iter().all(|v| v.is_finite()),
        "invalid predictive mean"
    );
    covariance_sqrt(cov, mean.len())?;
    ensure!(lower.len() == upper.len(), "cell counts must agree");
    matrix_dim(lower, Some(mean.len()))?;
    matrix_dim(upper, Some(mean.len()))?;
    ensure!(
        lower
            .iter()
            .zip(upper)
            .all(|(l, u)| l.iter().zip(u).all(|(a, b)| b >= a)),
        "cell upper corners must exceed lower corners"
    );
    Ok(())
}

/// Analytic EIPV using marginal variances. Exact for independent objectives.
pub fn ieipv(
    mean: &[f64],
    cov: &Matrix,
    lower: &Matrix,
    upper: &Matrix,
    form: EipvForm,
) -> Result<f64> {
    validate_cells(mean, cov, lower, upper)?;
    let mut total = 0.0;
    for (l, u) in lower.iter().zip(upper) {
        let mut product = 1.0;
        for k in 0..mean.len() {
            let variance = cov[k][k].max(0.0);
            let mut factor = if mean[k] >= 0.5 * l[k] + 0.5 * u[k] {
                // Re-express the difference with lower-tail EIs when the
                // predictive mean is above the cell. Subtracting two huge EIs
                // otherwise loses the cell's entire small, capped gain.
                (u[k] - l[k]) + expected_improvement(-mean[k], variance, -l[k])?
                    - expected_improvement(-mean[k], variance, -u[k])?
            } else {
                expected_improvement(mean[k], variance, l[k])?
                    - expected_improvement(mean[k], variance, u[k])?
            };
            if form == EipvForm::Paper {
                let tail = if cov[k][k] == 0.0 {
                    if mean[k] > u[k] { 1.0 } else { 0.0 }
                } else {
                    normal_cdf((mean[k] - u[k]) / cov[k][k].sqrt())
                };
                factor -= (u[k] - l[k]) * tail;
            }
            product *= factor.max(0.0);
        }
        total += product;
    }
    Ok(total)
}

/// Monte Carlo EIPV and its standard error; supports singular PSD covariance.
#[allow(clippy::too_many_arguments)]
pub fn eipv_mc(
    mean: &[f64],
    cov: &Matrix,
    lower: &Matrix,
    upper: &Matrix,
    samples: usize,
    seed: u64,
    form: EipvForm,
) -> Result<(f64, f64)> {
    validate_cells(mean, cov, lower, upper)?;
    ensure!(samples > 0, "positive MC sample count required");
    let sqrt = covariance_sqrt(cov, mean.len())?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut running = 0.0;
    let mut m2 = 0.0;
    for i in 1..=samples {
        let z = DVector::from_fn(mean.len(), |_, _| StandardNormal.sample(&mut rng));
        let noise = &sqrt * z;
        let y: Vec<f64> = mean.iter().enumerate().map(|(k, v)| v + noise[k]).collect();
        let gain = if form == EipvForm::Exact {
            hypervolume_gain(&y, lower, upper)?
        } else {
            lower
                .iter()
                .zip(upper)
                .filter(|(l, u)| y.iter().enumerate().all(|(k, v)| *v > l[k] && *v <= u[k]))
                .map(|(l, _)| y.iter().zip(l).map(|(a, b)| a - b).product::<f64>())
                .sum()
        };
        let delta = gain - running;
        running += delta / i as f64;
        m2 += delta * (gain - running);
    }
    Ok((
        running,
        if samples > 1 {
            (m2 / (samples - 1) as f64 / samples as f64).sqrt()
        } else {
            f64::INFINITY
        },
    ))
}

#[derive(Debug, Clone)]
pub struct CeipvTerms {
    pub log_weight: Vec<f64>,
    pub center: Matrix,
    pub tau2: Matrix,
}

/// Moment-matched Gaussian approximation used in the upstream CEIPV method.
/// `overflow` contains increasing positive fractions of the box span; the
/// upstream default is `[0.1, 0.25, 0.5]`. An empty slice disables overflow.
pub fn ceipv_terms(
    front: &Matrix,
    v_ref: &[f64],
    w_ref: &[f64],
    form: EipvForm,
    overflow: &[f64],
) -> Result<CeipvTerms> {
    ensure!(
        overflow.iter().all(|v| v.is_finite() && *v > 0.0)
            && overflow.windows(2).all(|w| w[1] > w[0]),
        "overflow layers must be positive and increasing"
    );
    let (lower, upper) = nondominated_cells(front, v_ref, w_ref)?;
    let d = v_ref.len();
    let mut terms = CeipvTerms {
        log_weight: Vec::new(),
        center: Vec::new(),
        tau2: Vec::new(),
    };
    if lower.is_empty() {
        return Ok(terms);
    }
    let coords: Vec<Vec<f64>> = (0..d)
        .map(|k| {
            let mut c: Vec<f64> = front
                .iter()
                .map(|p| p[k].clamp(v_ref[k], w_ref[k]))
                .chain([v_ref[k], w_ref[k]])
                .collect();
            c.sort_by(f64::total_cmp);
            c.dedup();
            if form == EipvForm::Exact {
                for layer in overflow {
                    c.push(w_ref[k] + layer * (w_ref[k] - v_ref[k]));
                }
            }
            c
        })
        .collect();
    // Expand the exact capped gain as a sum of tensor products. Each base
    // nondominated cell contributes a ramp in its own interval and a constant
    // box in every later interval. Repeated terms add linearly, exactly matching
    // the prefix-sum grouping in the Python implementation.
    for (base_lo, base_hi) in lower.iter().zip(&upper) {
        let starts: Vec<usize> = (0..d)
            .map(|k| coords[k].iter().position(|v| *v == base_lo[k]).unwrap())
            .collect();
        let lengths: Vec<usize> = (0..d)
            .map(|k| {
                if form == EipvForm::Paper {
                    1
                } else {
                    coords[k].len() - 1 - starts[k]
                }
            })
            .collect();
        let count = lengths
            .iter()
            .try_fold(1usize, |n, l| n.checked_mul(*l))
            .ok_or_else(|| anyhow::anyhow!("CEIPV term count overflow"))?;
        ensure!(
            terms.center.len().saturating_add(count) <= 2_000_000,
            "CEIPV exceeds two million terms"
        );
        for mut idx in 0..count {
            let mut center = vec![0.0; d];
            let mut tau = vec![0.0; d];
            let mut log_weight = 0.0;
            for k in (0..d).rev() {
                let j = starts[k] + idx % lengths[k];
                idx /= lengths[k];
                let lo = coords[k][j];
                let width = coords[k][j + 1] - lo;
                if j == starts[k] {
                    log_weight += (0.5 * width * width).ln();
                    center[k] = lo + 2.0 * width / 3.0;
                    tau[k] = width * width / 18.0;
                } else {
                    log_weight += ((base_hi[k] - base_lo[k]) * width).ln();
                    center[k] = lo + 0.5 * width;
                    tau[k] = width * width / 12.0;
                }
            }
            terms.log_weight.push(log_weight);
            terms.center.push(center);
            terms.tau2.push(tau);
        }
    }
    Ok(terms)
}

pub fn log_ceipv(mean: &[f64], cov: &Matrix, terms: &CeipvTerms) -> Result<f64> {
    ensure!(
        !mean.is_empty() && mean.iter().all(|v| v.is_finite()),
        "predictive mean must be finite"
    );
    covariance_sqrt(cov, mean.len())?;
    matrix_dim(&terms.center, Some(mean.len()))?;
    matrix_dim(&terms.tau2, Some(mean.len()))?;
    ensure!(
        terms.center.len() == terms.log_weight.len()
            && terms.center.len() == terms.tau2.len()
            && terms.log_weight.iter().all(|v| v.is_finite())
            && terms.tau2.iter().flatten().all(|v| *v > 0.0),
        "invalid CEIPV terms"
    );
    let mut out = f64::NEG_INFINITY;
    for t in 0..terms.center.len() {
        let total = DMatrix::from_fn(mean.len(), mean.len(), |i, j| {
            cov[i][j] + if i == j { terms.tau2[t][i] } else { 0.0 }
        });
        let chol = total
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("CEIPV covariance is not positive definite"))?;
        let diff = DVector::from_fn(mean.len(), |i, _| terms.center[t][i] - mean[i]);
        let quadratic = diff.dot(&chol.solve(&diff));
        let log_det = 2.0 * chol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();
        let value = terms.log_weight[t]
            - 0.5 * (quadratic + log_det + mean.len() as f64 * (2.0 * std::f64::consts::PI).ln());
        if out == f64::NEG_INFINITY {
            out = value;
        } else {
            let max = out.max(value);
            out = max + ((out - max).exp() + (value - max).exp()).ln();
        }
    }
    Ok(out)
}

pub fn ceipv(mean: &[f64], cov: &Matrix, terms: &CeipvTerms) -> Result<f64> {
    Ok(log_ceipv(mean, cov, terms)?.exp())
}
