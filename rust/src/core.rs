//! Gaussian processes and benchmark objectives, all evaluated in double precision.
//!
//! Noise parameters are **variances**. Predictions describe the latent function,
//! excluding observation noise. All supplied benchmark objectives are maximized.

use crate::{Matrix, Point, Result};
use anyhow::{Context, bail, ensure};
use nalgebra::{DMatrix, DVector};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use std::f64::consts::{E, PI};
use std::path::Path;

fn deserialize_lengthscale<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<f64>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Lengthscale {
        Scalar(f64),
        Ard(Vec<f64>),
    }
    Ok(match Lengthscale::deserialize(d)? {
        Lengthscale::Scalar(x) => vec![x],
        Lengthscale::Ard(x) => x,
    })
}

/// Configuration of a scalar GP; a one-element lengthscale broadcasts to all inputs.
/// Saved Python states may supply a scalar lengthscale or an ARD array.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GpConfig {
    #[serde(deserialize_with = "deserialize_lengthscale")]
    pub lengthscale: Vec<f64>,
    pub variance: f64,
    pub noise: f64,
    pub kernel_type: String,
    pub standardize_targets: bool,
    pub optimize_hyperparameters: bool,
    pub mean_constant: f64,
    #[serde(alias = "y_mean")]
    pub fixed_target_mean: Option<f64>,
    #[serde(alias = "y_std")]
    pub fixed_target_std: Option<f64>,
    pub optimize_noise: bool,
    pub optimizer_maxiter: usize,
    pub optimizer_restarts: usize,
    pub optimizer_seed: Option<u64>,
}

impl Default for GpConfig {
    fn default() -> Self {
        Self {
            lengthscale: vec![1.0],
            variance: 1.0,
            noise: 1e-6,
            kernel_type: "matern52".into(),
            standardize_targets: true,
            optimize_hyperparameters: false,
            mean_constant: 0.0,
            fixed_target_mean: None,
            fixed_target_std: None,
            optimize_noise: false,
            optimizer_maxiter: 75,
            optimizer_restarts: 8,
            optimizer_seed: Some(0),
        }
    }
}

impl GpConfig {
    /// Reject invalid model parameters before factorizing a covariance matrix.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.lengthscale.is_empty()
                && self.lengthscale.iter().all(|v| v.is_finite() && *v > 0.0),
            "lengthscale must contain finite positive values"
        );
        ensure!(
            self.variance.is_finite() && self.variance > 0.0,
            "variance must be finite and positive"
        );
        ensure!(
            self.noise.is_finite() && self.noise >= 0.0,
            "noise must be finite and non-negative"
        );
        ensure!(
            self.mean_constant.is_finite(),
            "mean_constant must be finite"
        );
        ensure!(
            matches!(self.kernel_type.as_str(), "rbf" | "matern52"),
            "unknown kernel_type '{}'",
            self.kernel_type
        );
        match (self.fixed_target_mean, self.fixed_target_std) {
            (None, None) => (),
            (Some(m), Some(s)) => ensure!(
                m.is_finite() && s.is_finite() && s > 0.0,
                "fixed target mean/std must be finite, with std > 0"
            ),
            _ => bail!("fixed target mean/std must be supplied together"),
        }
        Ok(())
    }
}

/// Validate a rectangular, finite matrix with at least one column.
/// Empty query matrices are supported when `allow_empty` is true.
pub fn validate_matrix(x: &Matrix, dim: Option<usize>, allow_empty: bool) -> Result<usize> {
    if x.is_empty() {
        ensure!(allow_empty, "need at least one point");
        return Ok(dim.unwrap_or(0));
    }
    let d = x[0].len();
    ensure!(d > 0, "points must have at least one coordinate");
    ensure!(
        dim.is_none_or(|wanted| wanted == d),
        "input dimension mismatch: expected {:?}, got {d}",
        dim
    );
    ensure!(
        x.iter()
            .all(|r| r.len() == d && r.iter().all(|v| v.is_finite())),
        "matrix must be rectangular and finite"
    );
    Ok(d)
}

fn resolve_lengthscale(ls: &[f64], d: usize) -> Result<Vec<f64>> {
    ensure!(
        d > 0 && (ls.len() == 1 || ls.len() == d),
        "lengthscale must be scalar or have {d} entries"
    );
    ensure!(
        ls.iter().all(|v| v.is_finite() && *v > 0.0),
        "lengthscale must be finite and positive"
    );
    Ok(if ls.len() == 1 {
        vec![ls[0]; d]
    } else {
        ls.to_vec()
    })
}

fn kernel_value(a: &[f64], b: &[f64], ls: &[f64], variance: f64, kind: &str) -> f64 {
    let sq: f64 = a
        .iter()
        .zip(b)
        .zip(ls)
        .map(|((a, b), l)| {
            let difference = a - b;
            let scaled = if difference.is_finite() {
                difference / l
            } else {
                a / l - b / l
            };
            scaled.powi(2)
        })
        .sum();
    if kind == "rbf" {
        if sq > 1200.0 {
            (variance.ln() - 0.5 * sq).exp()
        } else {
            variance * (-0.5 * sq).exp()
        }
    } else {
        let r = (5.0 * sq).sqrt();
        if r.is_infinite() {
            0.0
        } else {
            if r > 600.0 {
                (variance.ln() + (1.0 + r + 5.0 / 3.0 * sq).ln() - r).exp()
            } else {
                variance * ((1.0 + r + 5.0 / 3.0 * sq) * (-r).exp())
            }
        }
    }
}

fn gram(a: &Matrix, b: &Matrix, ls: &[f64], variance: f64, kind: &str) -> DMatrix<f64> {
    DMatrix::from_fn(a.len(), b.len(), |i, j| {
        kernel_value(&a[i], &b[j], ls, variance, kind)
    })
}

/// Evaluate a validated RBF or Matérn-5/2 kernel matrix, with ARD lengthscales.
pub fn kernel_matrix(
    a: &Matrix,
    b: &Matrix,
    lengthscale: &[f64],
    variance: f64,
    kernel_type: &str,
) -> Result<Matrix> {
    let d = validate_matrix(a, None, false)?;
    validate_matrix(b, Some(d), true)?;
    let config = GpConfig {
        lengthscale: lengthscale.to_vec(),
        variance,
        kernel_type: kernel_type.into(),
        ..Default::default()
    };
    config.validate()?;
    let ls = resolve_lengthscale(lengthscale, d)?;
    Ok((0..a.len())
        .map(|i| {
            (0..b.len())
                .map(|j| kernel_value(&a[i], &b[j], &ls, variance, kernel_type))
                .collect()
        })
        .collect())
}

fn robust_factor(matrix: DMatrix<f64>) -> Result<nalgebra::linalg::Cholesky<f64, nalgebra::Dyn>> {
    let mut jitter = 1e-10;
    for _ in 0..8 {
        let mut candidate = matrix.clone();
        for i in 0..candidate.nrows() {
            candidate[(i, i)] += jitter;
        }
        if let Some(chol) = candidate.cholesky() {
            return Ok(chol);
        }
        jitter *= 10.0;
    }
    bail!("Cholesky failed even after jitter escalation")
}

#[derive(Clone, Debug)]
struct Fitted {
    x: Matrix,
    y: Vec<f64>,
    centered: DVector<f64>,
    mean: f64,
    std: f64,
    ls: Vec<f64>,
    chol: nalgebra::linalg::Cholesky<f64, nalgebra::Dyn>,
    alpha: DVector<f64>,
}

/// Exact Gaussian process regression with bounded marginal likelihood fitting.
/// A failed refit preserves the last usable model.
#[derive(Clone, Debug)]
pub struct GaussianProcess {
    config: GpConfig,
    initial_lengthscale: Vec<f64>,
    fitted: Option<Fitted>,
}

impl GaussianProcess {
    pub fn new(config: GpConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            initial_lengthscale: config.lengthscale.clone(),
            config,
            fitted: None,
        })
    }

    pub fn config(&self) -> &GpConfig {
        &self.config
    }

    /// Fit finite observations. Constant or singleton targets keep unit scaling,
    /// preserving uncertainty away from the observations.
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        let d = validate_matrix(x, None, false)?;
        ensure!(
            y.len() == x.len() && y.iter().all(|v| v.is_finite()),
            "y must be finite and match x rows"
        );
        let mut config = self.config.clone();
        let base_ls = if config.lengthscale.len() == 1 || config.lengthscale.len() == d {
            &config.lengthscale
        } else {
            &self.initial_lengthscale
        };
        config.lengthscale = resolve_lengthscale(base_ls, d)?;
        let (mean, std) = match (config.fixed_target_mean, config.fixed_target_std) {
            (Some(m), Some(s)) => (m, s),
            _ if config.standardize_targets => {
                // Scale before summation to avoid overflow for large finite targets.
                let m: f64 = y.iter().map(|v| v / y.len() as f64).sum();
                let raw = y
                    .iter()
                    .map(|v| (v - m) / (y.len() as f64).sqrt())
                    .fold(0.0_f64, f64::hypot);
                (m, if raw > 1e-8 { raw } else { 1.0 })
            }
            _ => (0.0, 1.0),
        };
        ensure!(
            mean.is_finite() && std.is_finite(),
            "target scaling overflow; rescale observations"
        );
        let centered = DVector::from_iterator(
            y.len(),
            y.iter().map(|v| (v - mean) / std - config.mean_constant),
        );
        ensure!(
            centered.iter().all(|v| v.is_finite()),
            "target normalization overflow"
        );
        if config.optimize_hyperparameters {
            optimize_gp(&mut config, x, &centered);
        }
        let mut matrix = gram(
            x,
            x,
            &config.lengthscale,
            config.variance,
            &config.kernel_type,
        );
        for i in 0..x.len() {
            matrix[(i, i)] += config.noise + 1e-10;
        }
        let chol = robust_factor(matrix)?;
        let alpha = chol.solve(&centered);
        ensure!(
            alpha.iter().all(|v| v.is_finite()),
            "GP solve produced non-finite coefficients"
        );
        self.fitted = Some(Fitted {
            x: x.clone(),
            y: y.to_vec(),
            centered,
            mean,
            std,
            ls: config.lengthscale.clone(),
            chol,
            alpha,
        });
        self.config = config;
        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> Result<(Vec<f64>, Vec<f64>)> {
        let f = self
            .fitted
            .as_ref()
            .context("fit the GP before prediction")?;
        validate_matrix(x, Some(f.ls.len()), true)?;
        if x.is_empty() {
            return Ok((vec![], vec![]));
        }
        let cross = gram(
            &f.x,
            x,
            &f.ls,
            self.config.variance,
            &self.config.kernel_type,
        );
        let solved = f
            .chol
            .l()
            .solve_lower_triangular(&cross)
            .context("triangular solve failed")?;
        let m = cross.transpose() * &f.alpha;
        let means: Vec<f64> = m
            .iter()
            .map(|v| (v + self.config.mean_constant) * f.std + f.mean)
            .collect();
        let vars: Vec<f64> = (0..x.len())
            .map(|j| {
                (self.config.variance - solved.column(j).norm_squared()).max(1e-12) * f.std.powi(2)
            })
            .collect();
        ensure!(
            means.iter().chain(&vars).all(|v| v.is_finite()),
            "prediction overflow; rescale observations"
        );
        Ok((means, vars))
    }

    /// Predict a full covariance across query points (latent function units).
    pub fn predict_joint(&self, x: &Matrix) -> Result<(Vec<f64>, Matrix)> {
        let f = self
            .fitted
            .as_ref()
            .context("fit the GP before prediction")?;
        validate_matrix(x, Some(f.ls.len()), true)?;
        if x.is_empty() {
            return Ok((vec![], vec![]));
        }
        let cross = gram(
            &f.x,
            x,
            &f.ls,
            self.config.variance,
            &self.config.kernel_type,
        );
        let solved = f
            .chol
            .l()
            .solve_lower_triangular(&cross)
            .context("triangular solve failed")?;
        let m = cross.transpose() * &f.alpha;
        let covariance = gram(x, x, &f.ls, self.config.variance, &self.config.kernel_type)
            - solved.transpose() * solved;
        let means: Vec<f64> = m
            .iter()
            .map(|v| (v + self.config.mean_constant) * f.std + f.mean)
            .collect();
        let cov: Matrix = (0..x.len())
            .map(|i| {
                (0..x.len())
                    .map(|j| {
                        let c = 0.5 * (covariance[(i, j)] + covariance[(j, i)]);
                        if i == j {
                            c.max(1e-12) * f.std.powi(2)
                        } else {
                            c * f.std.powi(2)
                        }
                    })
                    .collect()
            })
            .collect();
        ensure!(
            means
                .iter()
                .chain(cov.iter().flatten())
                .all(|v| v.is_finite()),
            "prediction overflow; rescale observations"
        );
        Ok((means, cov))
    }

    /// Negative log marginal likelihood on normalized, mean-centered targets.
    pub fn negative_log_marginal_likelihood(&self) -> Result<f64> {
        let f = self.fitted.as_ref().context("fit the GP first")?;
        let value = 0.5 * f.centered.dot(&f.alpha)
            + f.chol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>()
            + 0.5 * f.x.len() as f64 * (2.0 * PI).ln();
        ensure!(
            value.is_finite(),
            "marginal likelihood overflow; rescale observations"
        );
        Ok(value)
    }

    pub fn training_inputs(&self) -> Option<&Matrix> {
        self.fitted.as_ref().map(|f| &f.x)
    }
    pub fn training_targets(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|f| f.y.as_slice())
    }

    /// Portable state: Python-compatible `gp_state` and raw training arrays.
    /// Outcome scaling and constant prior mean are saved explicitly.
    pub fn source_state(&self) -> Result<Value> {
        let f = self
            .fitted
            .as_ref()
            .context("fit the GP before serialization")?;
        let mut state = serde_json::to_value(&self.config)?;
        let object = state
            .as_object_mut()
            .context("GP config must be an object")?;
        object.remove("fixed_target_mean");
        object.remove("fixed_target_std");
        object.insert("y_mean".into(), json!(f.mean));
        object.insert("y_std".into(), json!(f.std));
        object.insert("dim".into(), json!(f.ls.len()));
        object.insert("n_observations".into(), json!(f.x.len()));
        object.insert("optimize_hyperparameters".into(), json!(false));
        Ok(json!({"schema_version": 1, "gp_state": state, "x_train": f.x, "y_train": f.y}))
    }

    /// Reconstruct a portable state without refitting hyperparameters.
    /// Python run files with separate trajectories can be joined by adding
    /// `x_train` / `y_train` to the GP payload before calling this method.
    pub fn from_state(state: &Value) -> Result<Self> {
        let x = state
            .get("x_train")
            .or_else(|| state.get("x"))
            .or_else(|| state.get("X"))
            .context("state needs x_train")?;
        let y = state
            .get("y_train")
            .or_else(|| state.get("y"))
            .or_else(|| state.get("Y"))
            .context("state needs y_train")?;
        let x: Matrix = serde_json::from_value(x.clone()).context("invalid x_train")?;
        let y: Vec<f64> = serde_json::from_value(y.clone()).context("invalid y_train")?;
        Self::from_source_state(state, &x, &y)
    }

    /// Rebuild a Python source state accompanied by its separate trajectory.
    pub fn from_source_state(state: &Value, x: &Matrix, y: &[f64]) -> Result<Self> {
        let gp = state.get("gp_state").unwrap_or(state);
        ensure!(gp.is_object(), "gp_state must be a JSON object");
        for key in ["lengthscale", "variance", "noise"] {
            ensure!(gp.get(key).is_some(), "gp_state is missing {key}");
        }
        if let Some(dim) = gp.get("dim") {
            ensure!(
                dim.as_u64()
                    .is_some_and(|d| x.first().is_some_and(|r| r.len() as u64 == d)),
                "source metadata dimension mismatch"
            );
        }
        if let Some(n) = gp.get("n_observations") {
            ensure!(
                n.as_u64() == Some(x.len() as u64),
                "source metadata observation count mismatch"
            );
        }
        let mut config: GpConfig =
            serde_json::from_value(gp.clone()).context("invalid gp_state")?;
        config.optimize_hyperparameters = false;
        let mut model = Self::new(config)?;
        model.fit(x, y)?;
        Ok(model)
    }
}

fn objective_gradient(
    config: &GpConfig,
    x: &Matrix,
    y: &DVector<f64>,
    theta: &DVector<f64>,
) -> Option<(f64, DVector<f64>)> {
    let d = x[0].len();
    let n = x.len();
    let ls: Vec<_> = theta.iter().take(d).map(|v| v.exp()).collect();
    let variance = theta[d].exp();
    let noise = if config.optimize_noise {
        theta[d + 1].exp()
    } else {
        config.noise
    };
    let kernel = gram(x, x, &ls, variance, &config.kernel_type);
    let mut covariance = kernel.clone();
    for i in 0..n {
        covariance[(i, i)] += noise + 1e-10;
    }
    let chol = robust_factor(covariance).ok()?;
    let alpha = chol.solve(y);
    let value = 0.5 * y.dot(&alpha)
        + chol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>()
        + 0.5 * n as f64 * (2.0 * PI).ln();
    if !value.is_finite() {
        return None;
    }
    let q = chol.inverse() - &alpha * alpha.transpose();
    let mut gradient = DVector::zeros(theta.len());
    for i in 0..n {
        for j in 0..n {
            let weight = 0.5 * q[(i, j)];
            gradient[d] += weight * kernel[(i, j)];
            let sq: f64 = (0..d).map(|k| ((x[i][k] - x[j][k]) / ls[k]).powi(2)).sum();
            let r = (5.0 * sq).sqrt();
            let factor = if config.kernel_type == "rbf" {
                kernel[(i, j)]
            } else if r.is_finite() {
                variance * (5.0 / 3.0) * (1.0 + r) * (-r).exp()
            } else {
                0.0
            };
            for k in 0..d {
                gradient[k] += weight * factor * ((x[i][k] - x[j][k]) / ls[k]).powi(2);
            }
        }
    }
    if config.optimize_noise {
        gradient[d + 1] = 0.5 * noise * q.trace();
    }
    if gradient.iter().all(|v| v.is_finite()) {
        Some((value, gradient))
    } else {
        None
    }
}

fn optimize_gp(config: &mut GpConfig, x: &Matrix, y: &DVector<f64>) {
    let original = config.clone();
    let d = x[0].len();
    let mut bounds = vec![(-6.0, 4.0); d];
    bounds.push((-8.0, 6.0));
    if config.optimize_noise {
        bounds.push((-12.0, 0.0));
    }
    let mut values: Vec<_> = config.lengthscale.iter().map(|v| v.ln()).collect();
    values.push(config.variance.ln());
    if config.optimize_noise {
        values.push(config.noise.max((-12.0f64).exp()).ln());
    }
    let initial = DVector::from_iterator(
        values.len(),
        values
            .iter()
            .zip(&bounds)
            .map(|(v, (lo, hi))| v.clamp(*lo, *hi)),
    );
    let Some((mut best_value, _)) = objective_gradient(config, x, y, &initial) else {
        return;
    };
    let mut best = initial.clone();
    let mut rng = ChaCha8Rng::seed_from_u64(config.optimizer_seed.unwrap_or(0));
    for restart in 0..config.optimizer_restarts.max(1) {
        let mut theta = if restart == 0 {
            initial.clone()
        } else {
            DVector::from_iterator(
                bounds.len(),
                bounds.iter().map(|(a, b)| rng.gen_range(*a..*b)),
            )
        };
        let Some((mut value, mut gradient)) = objective_gradient(config, x, y, &theta) else {
            continue;
        };
        let mut inverse_hessian = DMatrix::identity(theta.len(), theta.len());
        for _ in 0..config.optimizer_maxiter {
            if value < best_value {
                best_value = value;
                best = theta.clone();
            }
            let mut projected = gradient.clone();
            for i in 0..theta.len() {
                if (theta[i] <= bounds[i].0 + 1e-10 && projected[i] > 0.0)
                    || (theta[i] >= bounds[i].1 - 1e-10 && projected[i] < 0.0)
                {
                    projected[i] = 0.0;
                }
            }
            if projected.amax() < 1e-6 {
                break;
            }
            let mut direction = -&inverse_hessian * &projected;
            if direction.dot(&gradient) >= -1e-12 {
                direction = -projected;
                inverse_hessian.fill(0.0);
                inverse_hessian.set_diagonal(&DVector::from_element(theta.len(), 1.0));
            }
            let max_step = direction.amax();
            if max_step > 2.0 {
                direction *= 2.0 / max_step;
            }
            let mut step = 1.0;
            let mut next = None;
            for _ in 0..25 {
                let candidate = DVector::from_iterator(
                    theta.len(),
                    (0..theta.len())
                        .map(|i| (theta[i] + step * direction[i]).clamp(bounds[i].0, bounds[i].1)),
                );
                let delta = &candidate - &theta;
                if delta.amax() < 1e-10 {
                    break;
                }
                if let Some((v, g)) = objective_gradient(config, x, y, &candidate) {
                    if v <= value + 1e-4 * gradient.dot(&delta) {
                        next = Some((candidate, v, g));
                        break;
                    }
                }
                step *= 0.5;
            }
            let Some((new_theta, new_value, new_gradient)) = next else {
                break;
            };
            let s = &new_theta - &theta;
            let z = &new_gradient - &gradient;
            let sz = s.dot(&z);
            if sz > 1e-10 {
                let rho = 1.0 / sz;
                let identity = DMatrix::identity(theta.len(), theta.len());
                let left = &identity - rho * &s * z.transpose();
                inverse_hessian =
                    &left * inverse_hessian * left.transpose() + rho * &s * s.transpose();
            }
            theta = new_theta;
            value = new_value;
            gradient = new_gradient;
        }
        if value < best_value {
            best_value = value;
            best = theta;
        }
    }
    config.lengthscale = best.iter().take(d).map(|v| v.exp()).collect();
    config.variance = best[d].exp();
    if config.optimize_noise {
        config.noise = best[d + 1].exp();
    }
    // Clipping a user's starting parameters into the optimizer's search box
    // must never silently worsen the model. Retain the original if necessary.
    let nll = |c: &GpConfig| -> Option<f64> {
        let mut k = gram(x, x, &c.lengthscale, c.variance, &c.kernel_type);
        for i in 0..x.len() {
            k[(i, i)] += c.noise + 1e-10;
        }
        let chol = robust_factor(k).ok()?;
        Some(
            0.5 * y.dot(&chol.solve(y))
                + chol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>()
                + 0.5 * x.len() as f64 * (2.0 * PI).ln(),
        )
    };
    if let (Some(before), Some(after)) = (nll(&original), nll(config)) {
        if after > before {
            *config = original;
        }
    }
}

/// Draw independent uniform points using a deterministic ChaCha8 RNG.
/// Rust seeds are reproducible within this port; NumPy uses a different RNG.
pub fn sample(bounds: &[(f64, f64)], n: usize, seed: u64) -> Result<Matrix> {
    ensure!(
        !bounds.is_empty()
            && bounds
                .iter()
                .all(|(a, b)| a.is_finite() && b.is_finite() && a < b && (b - a).is_finite()),
        "bounds must be finite, nonempty and strictly increasing"
    );
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Ok((0..n)
        .map(|_| bounds.iter().map(|(a, b)| rng.gen_range(*a..*b)).collect())
        .collect())
}

/// Metadata for a scalar benchmark on the unit cube.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionSpec {
    pub name: String,
    pub dim: usize,
    pub optimum: f64,
    pub bounds: Vec<(f64, f64)>,
}

pub fn function_spec(name: &str) -> Result<FunctionSpec> {
    let (dim, optimum) = match name {
        "branin" => (2, -0.39788735772973816),
        "sphere" | "ackley" | "rastrigin" | "rosenbrock" => (2, 0.0),
        "hartmann6" => (6, 3.322368011415515),
        "mccormick" => (2, 3.0_f64.sqrt() / 2.0 + PI / 3.0),
        "normalized_branin" => (2, (-0.39788735772973816 + 54.44) / 51.44),
        "goldstein_price" | "normalized_goldstein_price" => (2, (8.693 - 3.0_f64.ln()) / 2.427),
        "hartmann3" | "normalized_hartmann3" => (
            3,
            (hartmann(
                &[0.11458887075439354, 0.5556488906008059, 0.8525469799283214],
                3,
            ) - 0.93)
                / 0.95,
        ),
        _ => bail!("unknown test function: {name}"),
    };
    Ok(FunctionSpec {
        name: name.into(),
        dim,
        optimum,
        bounds: vec![(0.0, 1.0); dim],
    })
}

fn hartmann(x: &[f64], dim: usize) -> f64 {
    let alpha = [1.0, 1.2, 3.0, 3.2];
    let a6 = [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ];
    let p6 = [
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
    ];
    let a3 = [
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ];
    let p3 = [
        [3689.0, 1170.0, 2673.0],
        [4699.0, 4387.0, 7470.0],
        [1091.0, 8732.0, 5547.0],
        [381.0, 5743.0, 8828.0],
    ];
    (0..4)
        .map(|i| {
            let sum: f64 = (0..dim)
                .map(|j| {
                    if dim == 6 {
                        a6[i][j] * (x[j] - 1e-4 * p6[i][j]).powi(2)
                    } else {
                        a3[i][j] * (x[j] - 1e-4 * p3[i][j]).powi(2)
                    }
                })
                .sum();
            alpha[i] * (-sum).exp()
        })
        .sum()
}

/// Evaluate deterministic scalar test functions. Coordinates may extend outside
/// the unit cube, matching the translated MetaBO objective definitions.
pub fn synthetic(name: &str, x: &Matrix) -> Result<Vec<f64>> {
    let spec = function_spec(name)?;
    validate_matrix(x, Some(spec.dim), true)?;
    let values: Vec<f64> = x
        .iter()
        .map(|v| match name {
            "branin" | "normalized_branin" => {
                let a = -5.0 + 15.0 * v[0];
                let b = 15.0 * v[1];
                let y = -(b - 5.1 / (4.0 * PI * PI) * a * a + 5.0 / PI * a - 6.0).powi(2)
                    - 10.0 * (1.0 - 1.0 / (8.0 * PI)) * a.cos()
                    - 10.0;
                if name == "normalized_branin" {
                    (y + 54.44) / 51.44
                } else {
                    y
                }
            }
            "sphere" => -v.iter().map(|a| (10.0 * a - 5.0).powi(2)).sum::<f64>(),
            "ackley" => {
                let a = 10.0 * v[0] - 5.0;
                let b = 10.0 * v[1] - 5.0;
                20.0 * (-0.2 * ((a * a + b * b) / 2.0).sqrt()).exp()
                    + ((2.0 * PI * a).cos() / 2.0 + (2.0 * PI * b).cos() / 2.0).exp()
                    - 20.0
                    - E
            }
            "rastrigin" => {
                -20.0
                    - v.iter()
                        .map(|a| {
                            let t = 10.24 * a - 5.12;
                            t * t - 10.0 * (2.0 * PI * t).cos()
                        })
                        .sum::<f64>()
            }
            "rosenbrock" => {
                let a = 4.0 * v[0] - 2.0;
                let b = 4.0 * v[1] - 2.0;
                -(1.0 - a).powi(2) - 100.0 * (b - a * a).powi(2)
            }
            "hartmann6" => hartmann(v, 6),
            "mccormick" => {
                let a = 5.5 * v[0] - 1.5;
                let b = 7.0 * v[1] - 3.0;
                -(a + b).sin() - (a - b).powi(2) + 1.5 * a - 2.5 * b - 1.0
            }
            "goldstein_price" | "normalized_goldstein_price" => {
                let a = 4.0 * v[0] - 2.0;
                let b = 4.0 * v[1] - 2.0;
                let first = 1.0
                    + (a + b + 1.0).powi(2)
                        * (19.0 - 14.0 * a + 3.0 * a * a - 14.0 * b + 6.0 * a * b + 3.0 * b * b);
                let second = 30.0
                    + (2.0 * a - 3.0 * b).powi(2)
                        * (18.0 - 32.0 * a + 12.0 * a * a + 48.0 * b - 36.0 * a * b + 27.0 * b * b);
                (8.693 - (first * second).ln()) / 2.427
            }
            "hartmann3" | "normalized_hartmann3" => (hartmann(v, 3) - 0.93) / 0.95,
            _ => unreachable!(),
        })
        .collect();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "objective overflow; use inputs near the benchmark domain"
    );
    Ok(values)
}

/// Add independent Gaussian observation noise; optionally cap at the analytic optimum.
pub fn synthetic_noisy(
    name: &str,
    x: &Matrix,
    noise_std: f64,
    seed: u64,
    cap_at_optimum: bool,
) -> Result<Vec<f64>> {
    ensure!(
        noise_std.is_finite() && noise_std >= 0.0,
        "noise_std must be finite and non-negative"
    );
    let mut y = synthetic(name, x)?;
    let cap = function_spec(name)?.optimum;
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x0b0);
    if noise_std > 0.0 {
        for value in &mut y {
            let z: f64 = StandardNormal.sample(&mut rng);
            *value += noise_std * z;
            if cap_at_optimum {
                *value = value.min(cap);
            }
        }
    }
    ensure!(
        y.iter().all(|v| v.is_finite()),
        "noisy objective overflow; reduce noise_std"
    );
    Ok(y)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MOFunctionSpec {
    pub name: String,
    pub dim: usize,
    pub n_objectives: usize,
    pub bounds: Vec<(f64, f64)>,
    pub v_ref: Vec<f64>,
    pub w_ref: Vec<f64>,
    pub true_hypervolume: Option<f64>,
}

pub fn mo_function_spec(name: &str) -> Result<MOFunctionSpec> {
    let (dim, objectives, v, w, hv) = match name {
        "oka2" => (
            3,
            2,
            vec![-3.2, -5.1],
            vec![PI, 0.0],
            Some(2.0 * PI * (5.1 - 2.0 / 3.0) + 5.1 * (3.2 - PI)),
        ),
        "vlmop3" => (2, 3, vec![-8.3, -62.0, -0.2], vec![0.0, -15.0, 0.1], None),
        "dtlz1a" => (
            6,
            2,
            vec![-570.0, -570.0],
            vec![0.0, 0.0],
            Some(570.0_f64.powi(2) - 0.125),
        ),
        "three_spheres" => (4, 3, vec![-6.0; 3], vec![1.0; 3], None),
        _ => bail!("unknown multi-objective test function: {name}"),
    };
    Ok(MOFunctionSpec {
        name: name.into(),
        dim,
        n_objectives: objectives,
        bounds: vec![(0.0, 1.0); dim],
        v_ref: v,
        w_ref: w,
        true_hypervolume: hv,
    })
}

/// Evaluate the Python maximization forms of OKA2, VLMOP3, DTLZ1a and the TAF+ spheres.
pub fn multi_objective(name: &str, x: &Matrix) -> Result<Matrix> {
    let spec = mo_function_spec(name)?;
    validate_matrix(x, Some(spec.dim), true)?;
    let y: Matrix = x
        .iter()
        .map(|v| match name {
            "oka2" => {
                let a = -PI + 2.0 * PI * v[0];
                let b = -5.0 + 10.0 * v[1];
                let c = -5.0 + 10.0 * v[2];
                vec![
                    -a,
                    -(1.0 - (a + PI).powi(2) / (4.0 * PI * PI)
                        + (b - 5.0 * a.cos()).abs().cbrt()
                        + (c - 5.0 * a.sin()).abs().cbrt()),
                ]
            }
            "vlmop3" => {
                let a = 6.0 * v[0] - 3.0;
                let b = 6.0 * v[1] - 3.0;
                let r = a * a + b * b;
                vec![
                    -0.5 * r - r.sin(),
                    -(3.0 * a - 2.0 * b + 4.0).powi(2) / 8.0 - (a - b + 1.0).powi(2) / 27.0 - 15.0,
                    -1.0 / (r + 1.0) + 1.1 * (-r).exp(),
                ]
            }
            "dtlz1a" => {
                let g = 100.0
                    * (5.0
                        + v[1..]
                            .iter()
                            .map(|a| {
                                let t = a - 0.5;
                                t * t - (2.0 * PI * t).cos()
                            })
                            .sum::<f64>());
                vec![-0.5 * v[0] * (1.0 + g), -0.5 * (1.0 - v[0]) * (1.0 + g)]
            }
            "three_spheres" => sphere_objectives(v),
            _ => unreachable!(),
        })
        .collect();
    ensure!(
        y.iter().flatten().all(|v| v.is_finite()),
        "objective overflow"
    );
    Ok(y)
}

fn sphere_objectives(x: &[f64]) -> Vec<f64> {
    [(0.55, 0.4), (0.6, 0.45), (0.65, 0.35)]
        .iter()
        .enumerate()
        .map(|(i, (a, b))| 1.0 - 8.0 * ((x[i] - a).powi(2) + (x[i + 1] - b).powi(2)))
        .collect()
}

/// Affine variants in normalized input coordinates; transformed inputs are clipped.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TaskVariantSpec {
    pub input_shift: Vec<f64>,
    pub input_scale: Vec<f64>,
    pub output_scale: f64,
    pub noise_std: f64,
    pub cap_at_optimum: bool,
    pub seed: Option<u64>,
}

impl Default for TaskVariantSpec {
    fn default() -> Self {
        Self {
            input_shift: vec![],
            input_scale: vec![],
            output_scale: 1.0,
            noise_std: 0.0,
            cap_at_optimum: false,
            seed: None,
        }
    }
}

impl TaskVariantSpec {
    pub fn validate_for_dim(&self, dim: usize) -> Result<()> {
        ensure!(
            self.input_shift.len() == dim && self.input_scale.len() == dim,
            "variant shift and scale must have {dim} entries"
        );
        ensure!(
            self.input_shift
                .iter()
                .chain(&self.input_scale)
                .all(|v| v.is_finite())
                && self.output_scale.is_finite(),
            "variant transform must be finite"
        );
        ensure!(
            self.noise_std.is_finite() && self.noise_std >= 0.0,
            "noise_std must be finite and non-negative"
        );
        Ok(())
    }
    /// Evaluate a batch with a caller-owned random stream. Repeated calls advance
    /// the stream, so noisy objective evaluation is independent of the optimizer.
    pub fn evaluate(
        &self,
        base_name: &str,
        x: &Matrix,
        rng: &mut impl Rng,
        cap_value: Option<f64>,
    ) -> Result<Vec<f64>> {
        let spec = function_spec(base_name)?;
        self.validate_for_dim(spec.dim)?;
        validate_matrix(x, Some(spec.dim), true)?;
        let transformed: Matrix = x
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| (self.input_scale[j] * v + self.input_shift[j]).clamp(0.0, 1.0))
                    .collect()
            })
            .collect();
        let mut y = synthetic(base_name, &transformed)?;
        for value in &mut y {
            *value *= self.output_scale;
            if self.noise_std > 0.0 {
                let z: f64 = StandardNormal.sample(rng);
                *value += self.noise_std * z;
                if self.cap_at_optimum && self.output_scale >= 0.0 {
                    *value = value.min(cap_value.unwrap_or(self.output_scale * spec.optimum));
                }
            }
        }
        ensure!(
            y.iter().all(|v| v.is_finite()),
            "variant objective overflow"
        );
        Ok(y)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FamilySplit {
    pub base_name: String,
    pub train_variants: Vec<TaskVariantSpec>,
    pub test_variants: Vec<TaskVariantSpec>,
}

/// Family ranges; deltas are nonnegative distances from identity transforms.
#[derive(Clone, Debug)]
pub struct FamilyConfig {
    pub max_input_shift: f64,
    pub max_input_scale_delta: f64,
    pub max_output_scale_delta: f64,
    pub noise_std: f64,
    pub cap_at_optimum: bool,
}
impl Default for FamilyConfig {
    fn default() -> Self {
        Self {
            max_input_shift: 0.05,
            max_input_scale_delta: 0.1,
            max_output_scale_delta: 0.1,
            noise_std: 0.0,
            cap_at_optimum: false,
        }
    }
}

pub fn generate_variants(
    base_name: &str,
    n_tasks: usize,
    seed: u64,
    config: &FamilyConfig,
) -> Result<Vec<TaskVariantSpec>> {
    ensure!(n_tasks > 0, "n_tasks must be positive");
    ensure!(
        [
            config.max_input_shift,
            config.max_input_scale_delta,
            config.max_output_scale_delta,
            config.noise_std
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0),
        "family ranges and noise must be finite and non-negative"
    );
    let dim = function_spec(base_name)?.dim;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut draw = |delta: f64| {
        if delta == 0.0 {
            0.0
        } else {
            rng.gen_range(-delta..delta)
        }
    };
    Ok((0..n_tasks)
        .map(|i| TaskVariantSpec {
            input_shift: (0..dim).map(|_| draw(config.max_input_shift)).collect(),
            input_scale: (0..dim)
                .map(|_| 1.0 + draw(config.max_input_scale_delta))
                .collect(),
            output_scale: 1.0 + draw(config.max_output_scale_delta),
            noise_std: config.noise_std,
            cap_at_optimum: config.cap_at_optimum,
            seed: Some(seed.wrapping_add(i as u64)),
        })
        .collect())
}

pub fn generate_user_variants(
    base_name: &str,
    n_users: usize,
    shift_range: f64,
    scale_range: f64,
    seed: u64,
) -> Result<Vec<TaskVariantSpec>> {
    ensure!(
        shift_range.is_finite()
            && shift_range >= 0.0
            && scale_range.is_finite()
            && (0.0..2.0).contains(&scale_range),
        "shift_range must be non-negative and scale_range in [0,2)"
    );
    generate_variants(
        base_name,
        n_users,
        seed,
        &FamilyConfig {
            max_input_shift: shift_range / 2.0,
            max_input_scale_delta: 0.0,
            max_output_scale_delta: scale_range / 2.0,
            ..Default::default()
        },
    )
}

pub fn split_variants(
    base_name: &str,
    variants: &[TaskVariantSpec],
    train_ratio: f64,
    seed: u64,
) -> Result<FamilySplit> {
    let dim = function_spec(base_name)?.dim;
    ensure!(
        train_ratio.is_finite() && train_ratio > 0.0 && train_ratio < 1.0,
        "train_ratio must be in (0,1)"
    );
    ensure!(variants.len() >= 2, "need at least two variants");
    for variant in variants {
        variant.validate_for_dim(dim)?;
    }
    let mut indices: Vec<_> = (0..variants.len()).collect();
    indices.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    let n = ((train_ratio * variants.len() as f64).round_ties_even() as usize)
        .clamp(1, variants.len() - 1);
    let mut training = vec![false; variants.len()];
    for &idx in &indices[..n] {
        training[idx] = true;
    }
    Ok(FamilySplit {
        base_name: base_name.into(),
        train_variants: variants
            .iter()
            .enumerate()
            .filter(|(i, _)| training[*i])
            .map(|(_, v)| v.clone())
            .collect(),
        test_variants: variants
            .iter()
            .enumerate()
            .filter(|(i, _)| !training[*i])
            .map(|(_, v)| v.clone())
            .collect(),
    })
}

/// Generate variants and split them in one reproducible operation.
pub fn create_family_split(
    base_name: &str,
    n_tasks: usize,
    train_ratio: f64,
    seed: u64,
    config: &FamilyConfig,
) -> Result<FamilySplit> {
    let variants = generate_variants(base_name, n_tasks, seed, config)?;
    split_variants(base_name, &variants, train_ratio, seed)
}

pub fn save_family_split(split: &FamilySplit, path: impl AsRef<Path>) -> Result<()> {
    validate_split(split)?;
    let path = path.as_ref();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(split)?)?;
    Ok(())
}
pub fn load_family_split(path: impl AsRef<Path>) -> Result<FamilySplit> {
    let split: FamilySplit = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    validate_split(&split)?;
    Ok(split)
}
fn validate_split(split: &FamilySplit) -> Result<()> {
    let dim = function_spec(&split.base_name)?.dim;
    ensure!(
        !split.train_variants.is_empty() && !split.test_variants.is_empty(),
        "family split must have train and test tasks"
    );
    for v in split.train_variants.iter().chain(&split.test_variants) {
        v.validate_for_dim(dim)?;
    }
    Ok(())
}

/// MetaBO task `scaling * normalized_base(x - translation)`; no input clipping.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MetaBOTask {
    pub base_name: String,
    pub translation: Point,
    pub scaling: f64,
}
impl MetaBOTask {
    fn validate(&self) -> Result<FunctionSpec> {
        ensure!(
            matches!(
                self.base_name.as_str(),
                "normalized_branin" | "normalized_goldstein_price" | "normalized_hartmann3"
            ),
            "MetaBO requires a normalized base"
        );
        let spec = function_spec(&self.base_name)?;
        ensure!(
            self.translation.len() == spec.dim
                && self.translation.iter().all(|v| v.is_finite())
                && self.scaling.is_finite()
                && self.scaling > 0.0,
            "invalid MetaBO translation/scaling"
        );
        Ok(spec)
    }
    pub fn evaluate(&self, x: &Matrix) -> Result<Vec<f64>> {
        let spec = self.validate()?;
        validate_matrix(x, Some(spec.dim), true)?;
        let shifted: Matrix = x
            .iter()
            .map(|r| {
                r.iter()
                    .zip(&self.translation)
                    .map(|(a, t)| a - t)
                    .collect()
            })
            .collect();
        Ok(synthetic(&self.base_name, &shifted)?
            .into_iter()
            .map(|v| self.scaling * v)
            .collect())
    }
    pub fn argmax(&self) -> Result<Point> {
        self.validate()?;
        let base: Vec<f64> = match self.base_name.as_str() {
            "normalized_branin" => vec![(-PI + 5.0) / 15.0, 12.275 / 15.0],
            "normalized_goldstein_price" => vec![0.5, 0.25],
            _ => vec![0.11458887075439354, 0.5556488906008059, 0.8525469799283214],
        };
        Ok(base
            .iter()
            .zip(&self.translation)
            .map(|(a, t)| a + t)
            .collect())
    }
    /// Analytic optimum is valid only if the translated maximizer is in the domain.
    pub fn optimum(&self) -> Result<f64> {
        let spec = self.validate()?;
        ensure!(
            self.argmax()?.iter().all(|v| (0.0..=1.0).contains(v)),
            "translated optimum falls outside unit cube"
        );
        Ok(self.scaling * spec.optimum)
    }
}

fn metabo_base(name: &str) -> Result<&'static str> {
    match name {
        "branin" | "normalized_branin" => Ok("normalized_branin"),
        "goldstein_price" | "normalized_goldstein_price" => Ok("normalized_goldstein_price"),
        "hartmann3" | "normalized_hartmann3" => Ok("normalized_hartmann3"),
        _ => bail!("unknown MetaBO family: {name}"),
    }
}

pub fn sample_metabo_tasks(name: &str, n: usize, seed: u64) -> Result<Vec<MetaBOTask>> {
    ensure!(n > 0, "n_tasks must be positive");
    let base = metabo_base(name)?;
    let dim = function_spec(base)?.dim;
    let points = sample(&vec![(0.0, 1.0); dim + 1], n, seed)?;
    Ok(points
        .iter()
        .map(|p| MetaBOTask {
            base_name: base.into(),
            translation: p[..dim].iter().map(|u| -0.1 + 0.2 * u).collect(),
            scaling: 0.9 + 0.2 * p[dim],
        })
        .collect())
}

/// Unscrambled Sobol grid, matching SciPy's first four dimensions, with the zero
/// corner omitted as in the Python MetaBO-50 family.
pub fn metabo_grid_tasks(name: &str, n: usize) -> Result<Vec<MetaBOTask>> {
    ensure!(
        n > 0 && n < u32::MAX as usize,
        "n_tasks must be positive and below 2^32"
    );
    let base = metabo_base(name)?;
    let dim = function_spec(base)?.dim;
    let points = sobol_points(dim + 1, n + 1)?;
    Ok(points
        .into_iter()
        .skip(1)
        .map(|p| MetaBOTask {
            base_name: base.into(),
            translation: p[..dim].iter().map(|u| -0.1 + 0.2 * u).collect(),
            scaling: 0.9 + 0.2 * p[dim],
        })
        .collect())
}

/// First `n` points of an unscrambled Sobol sequence in up to four dimensions.
/// This matches the corresponding SciPy sequence exactly.
pub fn sobol(dim: usize, n: usize) -> Result<Matrix> {
    sobol_points(dim, n)
}

/// Seeded low-discrepancy design: a random digital shift of a Sobol sequence
/// through dimension four, or a shifted Halton sequence in higher dimensions.
/// It is deterministic within Rust, but does not reproduce SciPy's scrambling.
pub fn low_discrepancy(dim: usize, n: usize, seed: u64) -> Result<Matrix> {
    ensure!(
        (1..=32).contains(&dim),
        "low-discrepancy designs support dimensions 1 through 32"
    );
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    if dim <= 4 {
        let shifts: Vec<u32> = (0..dim).map(|_| rng.r#gen()).collect();
        return Ok(sobol_points(dim, n)?
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&shifts)
                    .map(|(u, s)| ((u * 4294967296.0) as u32 ^ s) as f64 / 4294967296.0)
                    .collect()
            })
            .collect());
    }
    let primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 101, 103, 107, 109, 113, 127, 131,
    ];
    let shifts: Vec<f64> = (0..dim).map(|_| rng.r#gen()).collect();
    Ok((1..=n)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let mut idx = i;
                    let mut factor = 1.0;
                    let mut value = 0.0;
                    while idx > 0 {
                        factor /= primes[j] as f64;
                        value += (idx % primes[j]) as f64 * factor;
                        idx /= primes[j];
                    }
                    (value + shifts[j]).fract()
                })
                .collect()
        })
        .collect())
}

fn sobol_points(dim: usize, n: usize) -> Result<Matrix> {
    ensure!(
        (1..=4).contains(&dim),
        "Sobol implementation supports dimensions 1 through 4"
    );
    let mut directions = vec![[0u32; 32]; dim];
    for (j, direction) in directions[0].iter_mut().enumerate() {
        *direction = 1u32 << (31 - j);
    }
    for (k, (degree, a, initial)) in [(1, 0, vec![1]), (2, 1, vec![1, 3]), (3, 1, vec![1, 3, 1])]
        .iter()
        .enumerate()
        .take(dim - 1)
    {
        let s = *degree as usize;
        for (j, m) in initial.iter().enumerate() {
            directions[k + 1][j] = (*m as u32) << (31 - j);
        }
        for j in s..32 {
            let mut v = directions[k + 1][j - s] ^ (directions[k + 1][j - s] >> s);
            for l in 1..s {
                if (a >> (s - 1 - l)) & 1 == 1 {
                    v ^= directions[k + 1][j - l];
                }
            }
            directions[k + 1][j] = v;
        }
    }
    if n == 0 {
        return Ok(vec![]);
    }
    ensure!(
        n <= u32::MAX as usize,
        "Sobol design must have fewer than 2^32 points"
    );
    let mut state = vec![0u32; dim];
    let mut out = Vec::with_capacity(n);
    out.push(vec![0.0; dim]);
    for i in 1..n {
        let bit = (i as u32).trailing_zeros() as usize;
        for j in 0..dim {
            state[j] ^= directions[j][bit];
        }
        out.push(state.iter().map(|v| *v as f64 / 4294967296.0).collect());
    }
    Ok(out)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SphereUser {
    pub shift: Point,
    pub scale: f64,
    pub noise_std: f64,
}
impl SphereUser {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.shift.len() == 4
                && self.shift.iter().all(|v| v.is_finite())
                && self.scale.is_finite()
                && self.scale > 0.0
                && self.noise_std.is_finite()
                && self.noise_std >= 0.0,
            "invalid SphereUser parameters"
        );
        Ok(())
    }
    pub fn noise_free(&self, x: &Matrix) -> Result<Matrix> {
        self.validate()?;
        validate_matrix(x, Some(4), true)?;
        let values: Matrix = x
            .iter()
            .map(|r| {
                let shifted: Vec<_> = r.iter().zip(&self.shift).map(|(a, s)| a + s).collect();
                sphere_objectives(&shifted)
                    .iter()
                    .map(|v| self.scale * v)
                    .collect()
            })
            .collect();
        ensure!(
            values.iter().flatten().all(|v| v.is_finite()),
            "sphere user objective overflow"
        );
        Ok(values)
    }
    pub fn observe(&self, x: &Matrix, rng: &mut impl Rng) -> Result<Matrix> {
        let mut y = self.noise_free(x)?;
        for v in y.iter_mut().flatten() {
            let z: f64 = StandardNormal.sample(rng);
            *v += self.noise_std * z;
        }
        ensure!(
            y.iter().flatten().all(|v| v.is_finite()),
            "sphere user observation overflow"
        );
        Ok(y)
    }
    /// Exact maximizer of a non-negative weighted sum (no numerical optimization).
    pub fn weighted_optimum(&self, weights: &[f64]) -> Result<(Point, f64)> {
        self.validate()?;
        ensure!(
            weights.len() == 3
                && weights.iter().all(|v| v.is_finite() && *v >= 0.0)
                && weights.iter().any(|v| *v > 0.0),
            "weights must be three finite non-negative values with positive sum"
        );
        let mut sums = [0.0; 4];
        let mut totals = [0.0; 4];
        let largest = weights.iter().copied().fold(0.0, f64::max);
        for (i, (a, b)) in [(0.55, 0.4), (0.6, 0.45), (0.65, 0.35)].iter().enumerate() {
            let weight = weights[i] / largest;
            sums[i] += weight * (a - self.shift[i]);
            sums[i + 1] += weight * (b - self.shift[i + 1]);
            totals[i] += weight;
            totals[i + 1] += weight;
        }
        let x: Point = (0..4)
            .map(|j| {
                if totals[j] > 0.0 {
                    (sums[j] / totals[j]).clamp(0.0, 1.0)
                } else {
                    0.5
                }
            })
            .collect();
        let value: f64 = self.noise_free(&vec![x.clone()])?[0]
            .iter()
            .zip(weights)
            .map(|(y, w)| y * w)
            .sum();
        ensure!(
            value.is_finite(),
            "weighted optimum overflow; normalize objective weights"
        );
        Ok((x, value))
    }
}

pub fn make_sphere_users(
    n: usize,
    shift_range: f64,
    scale_range: f64,
    seed: u64,
) -> Result<Vec<SphereUser>> {
    ensure!(n > 0, "n_users must be positive");
    ensure!(
        shift_range.is_finite()
            && shift_range >= 0.0
            && scale_range.is_finite()
            && (0.0..2.0).contains(&scale_range),
        "invalid user ranges"
    );
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Ok((0..n)
        .map(|_| SphereUser {
            shift: (0..4)
                .map(|_| {
                    if shift_range == 0.0 {
                        0.0
                    } else {
                        rng.gen_range(-shift_range / 2.0..shift_range / 2.0)
                    }
                })
                .collect(),
            scale: if scale_range == 0.0 {
                1.0
            } else {
                rng.gen_range(1.0 - scale_range / 2.0..1.0 + scale_range / 2.0)
            },
            noise_std: 0.05,
        })
        .collect())
}

/// Configuration of the correlated SLF-GP benchmark. Reference points and
/// hypervolume are estimates from a finite design, not analytic global bounds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlfGpConfig {
    pub means: Vec<f64>,
    pub lengthscales: Matrix,
    pub n_support: usize,
    pub n_reference: usize,
}
impl Default for SlfGpConfig {
    fn default() -> Self {
        Self {
            means: vec![2.0, -2.0],
            lengthscales: vec![vec![0.7, 0.4, 1.0], vec![0.34, 0.9, 0.5]],
            n_support: 1024,
            n_reference: 8192,
        }
    }
}

/// Deterministic pair of correlated objectives sampled from latent Matérn GPs.
/// Random designs/draws are reproducible in Rust but use a different random
/// stream from NumPy. The covariance definition is the upstream SLF benchmark.
#[derive(Clone, Debug, Serialize)]
pub struct SlfGpProblem {
    support: Matrix,
    weights: Matrix,
    config: SlfGpConfig,
    rho: f64,
    spec: MOFunctionSpec,
}
impl SlfGpProblem {
    /// Construct the upstream two-objective benchmark, with a configurable input
    /// dimension. The original 3D problem retains its published lengthscales.
    pub fn new(dim: usize, n_objectives: usize, rho: f64, seed: u64) -> Result<Self> {
        ensure!(
            n_objectives == 2,
            "SLF-GP benchmark has exactly two objectives"
        );
        let mut config = SlfGpConfig::default();
        if dim != 3 {
            config.lengthscales = vec![vec![0.5; dim]; 2];
        }
        Self::with_config(rho, seed, config)
    }

    pub fn with_config(rho: f64, seed: u64, config: SlfGpConfig) -> Result<Self> {
        ensure!(
            rho.is_finite() && (-1.0..=1.0).contains(&rho),
            "rho must lie in [-1,1]"
        );
        ensure!(
            config.means.len() == 2 && config.means.iter().all(|v| v.is_finite()),
            "SLF needs two finite means"
        );
        ensure!(
            config.lengthscales.len() == 2,
            "SLF needs two latent lengthscales"
        );
        let dim = config.lengthscales[0].len();
        ensure!(
            dim > 0
                && dim <= 32
                && config
                    .lengthscales
                    .iter()
                    .all(|l| l.len() == dim && l.iter().all(|v| v.is_finite() && *v > 0.0)),
            "latent lengthscales must be finite positive vectors of equal dimension"
        );
        ensure!(
            config.n_support > 0 && config.n_reference > 0,
            "SLF support and reference sizes must be positive"
        );
        let support = low_discrepancy(dim, config.n_support, seed)?;
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x534c465f4750);
        let mut weights = Vec::with_capacity(2);
        for ls in &config.lengthscales {
            let mut k = gram(&support, &support, ls, 1.0, "matern52");
            for i in 0..support.len() {
                k[(i, i)] += 1e-8;
            }
            let chol = k
                .cholesky()
                .context("SLF support covariance was not positive definite")?;
            let normal = DVector::from_iterator(
                support.len(),
                (0..support.len()).map(|_| StandardNormal.sample(&mut rng)),
            );
            let draw = chol.l() * normal;
            weights.push(chol.solve(&draw).as_slice().to_vec());
        }
        let mut problem = Self {
            support,
            weights,
            config,
            rho,
            spec: MOFunctionSpec {
                name: format!("slf_gp_rho{rho}_seed{seed}"),
                dim,
                n_objectives: 2,
                bounds: vec![(0.0, 1.0); dim],
                v_ref: vec![],
                w_ref: vec![],
                true_hypervolume: None,
            },
        };
        let reference = low_discrepancy(dim, problem.config.n_reference, seed.wrapping_add(1))?;
        let values = problem.evaluate(&reference)?;
        for objective in 0..2 {
            let low = values
                .iter()
                .map(|v| v[objective])
                .fold(f64::INFINITY, f64::min);
            let high = values
                .iter()
                .map(|v| v[objective])
                .fold(f64::NEG_INFINITY, f64::max);
            let padding = 0.1 * (high - low).max(1e-12);
            problem.spec.v_ref.push(low - padding);
            problem.spec.w_ref.push(high + padding);
        }
        problem.spec.true_hypervolume = Some(crate::acquisition::hypervolume(
            &values,
            &problem.spec.v_ref,
        )?);
        Ok(problem)
    }

    pub fn evaluate(&self, x: &Matrix) -> Result<Matrix> {
        validate_matrix(x, Some(self.spec.dim), true)?;
        let residual = (1.0 - self.rho * self.rho).max(0.0).sqrt();
        let mut out = Vec::with_capacity(x.len());
        for row in x {
            let mut latent = [0.0; 2];
            for (s, value) in latent.iter_mut().enumerate() {
                *value = self
                    .support
                    .iter()
                    .zip(&self.weights[s])
                    .map(|(support, w)| {
                        kernel_value(row, support, &self.config.lengthscales[s], 1.0, "matern52")
                            * w
                    })
                    .sum();
            }
            out.push(vec![
                self.config.means[0] + latent[0],
                self.config.means[1] + self.rho * latent[0] + residual * latent[1],
            ]);
        }
        ensure!(
            out.iter().flatten().all(|v| v.is_finite()),
            "SLF evaluation overflow"
        );
        Ok(out)
    }
    pub fn spec(&self) -> &MOFunctionSpec {
        &self.spec
    }
}

/// Controls deterministic numerical optimum estimation. Full Cartesian grids
/// are streamed in batches to keep three-dimensional searches memory bounded.
#[derive(Clone, Debug)]
pub struct OptimumConfig {
    pub grid_points_per_dim: usize,
    pub n_low_discrepancy: usize,
    pub n_polish: usize,
    pub seed: u64,
}
impl Default for OptimumConfig {
    fn default() -> Self {
        Self {
            grid_points_per_dim: 201,
            n_low_discrepancy: 16384,
            n_polish: 5,
            seed: 0,
        }
    }
}

/// Estimate a deterministic maximum over the unit cube; no global guarantee is
/// possible for an arbitrary objective. Noise must be disabled by the caller.
pub fn estimate_optimum<F>(
    objective: F,
    dim: usize,
    options: &OptimumConfig,
) -> Result<(Point, f64)>
where
    F: Fn(&Matrix) -> Result<Vec<f64>>,
{
    ensure!(dim > 0 && dim <= 32, "dimension must be in 1..=32");
    ensure!(
        options.grid_points_per_dim >= 2 && options.n_low_discrepancy > 0,
        "optimum scan sizes must be positive; grid needs at least two points"
    );
    let mut best: Vec<(Point, f64)> = vec![];
    let mut visit = |batch: Matrix| -> Result<()> {
        let values = objective(&batch)?;
        ensure!(
            values.len() == batch.len() && values.iter().all(|v| v.is_finite()),
            "objective must return a finite value for each point"
        );
        for (point, value) in batch.into_iter().zip(values) {
            if best.len() < options.n_polish.max(1) || value > best.last().unwrap().1 {
                best.push((point, value));
                best.sort_by(|a, b| b.1.total_cmp(&a.1));
                best.truncate(options.n_polish.max(1));
            }
        }
        Ok(())
    };
    if dim <= 3 {
        let n = options.grid_points_per_dim;
        let total = n
            .checked_pow(dim as u32)
            .context("optimum grid size overflow")?;
        let mut batch = Vec::with_capacity(2048);
        for mut idx in 0..total {
            let mut row = vec![0.0; dim];
            for j in (0..dim).rev() {
                row[j] = (idx % n) as f64 / (n - 1) as f64;
                idx /= n;
            }
            batch.push(row);
            if batch.len() == 2048 {
                visit(std::mem::take(&mut batch))?;
            }
        }
        if !batch.is_empty() {
            visit(batch)?;
        }
    } else {
        visit(low_discrepancy(
            dim,
            options.n_low_discrepancy,
            options.seed,
        )?)?;
    }
    let mut result = best[0].clone();
    for (start, value) in best.into_iter().take(options.n_polish) {
        let mut x = start;
        let mut value = value;
        let mut step = 0.05;
        // Deterministic bounded coordinate pattern search.
        for _ in 0..250 {
            let mut improved = false;
            for j in 0..dim {
                for sign in [-1.0, 1.0] {
                    let mut point = x.clone();
                    point[j] = (point[j] + sign * step).clamp(0.0, 1.0);
                    let next = objective(&vec![point.clone()])?;
                    ensure!(
                        next.len() == 1 && next[0].is_finite(),
                        "invalid polish evaluation"
                    );
                    if next[0] > value {
                        x = point;
                        value = next[0];
                        improved = true;
                    }
                }
            }
            if !improved {
                step *= 0.5;
                if step < 1e-9 {
                    break;
                }
            }
        }
        if value > result.1 {
            result = (x, value);
        }
    }
    Ok(result)
}

/// Executable affine family task with its own observation-noise random stream.
#[derive(Clone, Debug)]
pub struct VariantFunction {
    pub name: String,
    pub base_name: String,
    pub variant: TaskVariantSpec,
    pub optimum: Option<f64>,
    rng: ChaCha8Rng,
}
impl VariantFunction {
    /// Compute reachable optima up to 3D, keeping the analytic upper bound in
    /// higher dimensions where a scan can miss a narrow global optimum.
    pub fn new(base_name: &str, variant: TaskVariantSpec, name: Option<&str>) -> Result<Self> {
        let spec = function_spec(base_name)?;
        variant.validate_for_dim(spec.dim)?;
        let optimum = if variant.output_scale >= 0.0 {
            let upper = variant.output_scale * spec.optimum;
            if spec.dim <= 3 {
                let clean = TaskVariantSpec {
                    noise_std: 0.0,
                    cap_at_optimum: false,
                    ..variant.clone()
                };
                let (_, estimate) = estimate_optimum(
                    |x| clean.evaluate(base_name, x, &mut ChaCha8Rng::seed_from_u64(0), None),
                    spec.dim,
                    &OptimumConfig::default(),
                )?;
                Some(if upper - estimate <= 1e-6 * upper.abs().max(1.0) {
                    upper
                } else {
                    estimate.min(upper)
                })
            } else {
                Some(upper)
            }
        } else {
            None
        };
        let seed = variant.seed.unwrap_or(0) ^ 0x0b0;
        Ok(Self {
            name: name
                .map(String::from)
                .unwrap_or_else(|| format!("{base_name}_variant")),
            base_name: base_name.into(),
            variant,
            optimum,
            rng: ChaCha8Rng::seed_from_u64(seed),
        })
    }
    pub fn evaluate(&mut self, x: &Matrix) -> Result<Vec<f64>> {
        self.variant
            .evaluate(&self.base_name, x, &mut self.rng, self.optimum)
    }
}

pub fn build_specs(
    base_name: &str,
    variants: &[TaskVariantSpec],
    prefix: &str,
) -> Result<Vec<VariantFunction>> {
    variants
        .iter()
        .enumerate()
        .map(|(i, v)| VariantFunction::new(base_name, v.clone(), Some(&format!("{prefix}_{i:03}"))))
        .collect()
}

#[cfg(test)]
mod mathematical_checks {
    use super::*;

    #[test]
    fn marginal_likelihood_gradient_matches_finite_differences() {
        let x = vec![
            vec![0.1, 0.4],
            vec![0.4, 0.8],
            vec![0.9, 0.2],
            vec![0.7, 0.6],
        ];
        let y = DVector::from_vec(vec![-0.4, 1.2, 0.6, -0.7]);
        for kernel in ["rbf", "matern52"] {
            for optimize_noise in [false, true] {
                let config = GpConfig {
                    kernel_type: kernel.into(),
                    noise: 0.03,
                    optimize_noise,
                    ..Default::default()
                };
                let mut values = vec![0.3_f64.ln(), 0.8_f64.ln(), 1.7_f64.ln()];
                if optimize_noise {
                    values.push(0.03_f64.ln());
                }
                let theta = DVector::from_vec(values);
                let (_, gradient) = objective_gradient(&config, &x, &y, &theta).unwrap();
                for i in 0..theta.len() {
                    let mut left = theta.clone();
                    let mut right = theta.clone();
                    left[i] -= 1e-5;
                    right[i] += 1e-5;
                    let finite = (objective_gradient(&config, &x, &y, &right).unwrap().0
                        - objective_gradient(&config, &x, &y, &left).unwrap().0)
                        / 2e-5;
                    assert!(
                        (finite - gradient[i]).abs() < 2e-7,
                        "{kernel} parameter {i}: {finite} vs {}",
                        gradient[i]
                    );
                }
            }
        }
    }
}
