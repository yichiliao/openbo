//! Multi-output Matern-5/2 GPs with actual coregionalization.
//!
//! Independent, shared-kernel multitask (`mt`), and semiparametric latent
//! factor (`slf`) modes match the Python covariance definitions. Hyperparameter
//! fitting uses bounded deterministic coordinate MAP search, rather than SciPy
//! L-BFGS-B; fixed-hyperparameter posterior calculations are equivalent.

use super::{SliceOptions, matrix_dim, slice_sample};
use crate::{Matrix, Result};
use anyhow::ensure;
use nalgebra::{DMatrix, DVector};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone)]
pub struct MultiOutputGP {
    pub mode: String,
    pub noise: f64,
    pub optimize_hyperparameters: bool,
    pub optimize_noise: bool,
    pub max_iterations: usize,
    pub optimizer_restarts: usize,
    pub seed: u64,
    pub lengthscales: Matrix,
    pub coregionalization: Vec<Matrix>,
    pub noise_variances: Vec<f64>,
    pub y_mean: Vec<f64>,
    pub y_std: Vec<f64>,
    x: Matrix,
    y: Matrix,
    theta: Vec<f64>,
    inverse: DMatrix<f64>,
    alpha: DVector<f64>,
    dim: usize,
    objectives: usize,
    latent: usize,
}

impl MultiOutputGP {
    pub fn new(mode: &str, noise: f64) -> Result<Self> {
        ensure!(
            matches!(mode, "independent" | "mt" | "slf"),
            "unknown multi-output GP mode {mode}"
        );
        ensure!(
            noise.is_finite() && noise > 0.0,
            "noise must be finite and positive"
        );
        Ok(Self {
            mode: mode.to_owned(),
            noise,
            optimize_hyperparameters: true,
            optimize_noise: true,
            max_iterations: 40,
            optimizer_restarts: 1,
            seed: 0,
            lengthscales: Vec::new(),
            coregionalization: Vec::new(),
            noise_variances: Vec::new(),
            y_mean: Vec::new(),
            y_std: Vec::new(),
            x: Vec::new(),
            y: Vec::new(),
            theta: Vec::new(),
            inverse: DMatrix::zeros(0, 0),
            alpha: DVector::zeros(0),
            dim: 0,
            objectives: 0,
            latent: 0,
        })
    }

    pub fn fit(&mut self, x: &Matrix, y: &Matrix) -> Result<()> {
        let dim = matrix_dim(x, None)?;
        let objectives = matrix_dim(y, None)?;
        ensure!(
            !x.is_empty() && dim > 0 && objectives > 0 && x.len() == y.len(),
            "fit requires nonempty matching input/objective matrices"
        );
        ensure!(
            matches!(self.mode.as_str(), "independent" | "mt" | "slf")
                && self.noise.is_finite()
                && self.noise > 0.0,
            "invalid multi-output configuration"
        );
        // Fit on a temporary copy so failed refits preserve a usable posterior.
        let mut next = self.clone();
        let same_shape = next.dim == dim && next.objectives == objectives;
        next.dim = dim;
        next.objectives = objectives;
        next.latent = if next.mode == "mt" { 1 } else { objectives };
        next.x = x.clone();
        next.y = y.clone();
        next.y_mean = Vec::with_capacity(objectives);
        next.y_std = Vec::with_capacity(objectives);
        for k in 0..objectives {
            // Rescale before sums and squares so finite, large-magnitude
            // observations cannot overflow during target standardization.
            let scale = y.iter().map(|row| row[k].abs()).fold(0.0, f64::max);
            let mean_scaled = if scale == 0.0 {
                0.0
            } else {
                y.iter().map(|row| row[k] / scale).sum::<f64>() / y.len() as f64
            };
            let variance_scaled = if scale == 0.0 {
                0.0
            } else {
                y.iter()
                    .map(|row| (row[k] / scale - mean_scaled).powi(2))
                    .sum::<f64>()
                    / y.len() as f64
            };
            let sd = scale * variance_scaled.sqrt().min(1.0);
            next.y_mean.push(mean_scaled * scale);
            next.y_std.push(if sd > 1e-8 { sd } else { 1.0 });
        }
        let theta = if same_shape && next.theta.len() == next.n_params() {
            next.theta.clone()
        } else {
            next.default_theta()
        };
        let theta = if next.optimize_hyperparameters {
            next.optimize(&theta)?
        } else {
            theta
        };
        next.set_state(&theta)?;
        *self = next;
        Ok(())
    }

    fn n_coreg(&self) -> usize {
        if self.mode == "independent" {
            self.objectives
        } else {
            self.objectives * (self.objectives + 1) / 2
        }
    }
    fn n_params(&self) -> usize {
        self.latent * self.dim
            + self.n_coreg()
            + if self.optimize_noise {
                self.objectives
            } else {
                0
            }
    }
    fn default_theta(&self) -> Vec<f64> {
        let mut p = vec![0.5_f64.ln(); self.latent * self.dim];
        if self.mode == "independent" {
            p.extend(vec![0.0; self.objectives]);
        } else {
            for i in 0..self.objectives {
                for j in 0..=i {
                    p.push(if i == j {
                        1.0
                    } else if self.mode == "slf" {
                        0.1
                    } else {
                        0.0
                    });
                }
            }
        }
        if self.optimize_noise {
            p.extend(vec![self.noise.ln(); self.objectives]);
        }
        p
    }
    pub fn hyperparameters(&self) -> &[f64] {
        &self.theta
    }
    pub fn hyperparameter_bounds(&self) -> Vec<(f64, f64)> {
        let mut b = vec![(0.01_f64.ln(), 10.0_f64.ln()); self.latent * self.dim];
        b.extend(vec![(-5.0, 5.0); self.n_coreg()]);
        if self.optimize_noise {
            b.extend(vec![(1e-8_f64.ln(), 0.0); self.objectives]);
        }
        b
    }

    fn unpack(&self, p: &[f64]) -> Result<(Matrix, Vec<Matrix>, Vec<f64>)> {
        ensure!(
            p.len() == self.n_params() && p.iter().all(|v| v.is_finite()),
            "invalid multi-output hyperparameter vector"
        );
        let n_ls = self.latent * self.dim;
        let ls: Matrix = p[..n_ls]
            .chunks(self.dim)
            .map(|r| r.iter().map(|v| v.exp()).collect())
            .collect();
        let mut b = vec![vec![vec![0.0; self.objectives]; self.objectives]; self.latent];
        if self.mode == "independent" {
            for k in 0..self.objectives {
                b[k][k][k] = p[n_ls + k].exp();
            }
        } else {
            let mut a = vec![vec![0.0; self.objectives]; self.objectives];
            let mut index = n_ls;
            for (i, row) in a.iter_mut().enumerate() {
                for value in row.iter_mut().take(i + 1) {
                    *value = p[index];
                    index += 1;
                }
            }
            for i in 0..self.objectives {
                for j in 0..self.objectives {
                    if self.mode == "mt" {
                        b[0][i][j] = (0..self.objectives).map(|k| a[i][k] * a[j][k]).sum();
                    } else {
                        for (k, latent) in b.iter_mut().enumerate() {
                            latent[i][j] = a[i][k] * a[j][k];
                        }
                    }
                }
            }
        }
        let noise = if self.optimize_noise {
            p[n_ls + self.n_coreg()..].iter().map(|v| v.exp()).collect()
        } else {
            vec![self.noise; self.objectives]
        };
        ensure!(
            ls.iter().flatten().all(|v| v.is_finite() && *v > 0.0)
                && b.iter().flatten().flatten().all(|v| v.is_finite())
                && noise.iter().all(|v| v.is_finite() && *v > 0.0),
            "hyperparameters overflowed"
        );
        Ok((ls, b, noise))
    }

    fn kernel(x: &[f64], z: &[f64], ls: &[f64]) -> f64 {
        let r2 = x
            .iter()
            .zip(z)
            .zip(ls)
            .map(|((a, b), l)| ((a - b) / l).powi(2))
            .sum::<f64>();
        let r5 = (5.0 * r2).sqrt();
        if !r5.is_finite() {
            return 0.0;
        }
        (1.0 + r5 + 5.0 * r2 / 3.0) * (-r5).exp()
    }
    fn covariance(&self, p: &[f64]) -> Result<DMatrix<f64>> {
        let (ls, b, noise) = self.unpack(p)?;
        let n = self.x.len();
        let q = n * self.objectives;
        Ok(DMatrix::from_fn(q, q, |i, j| {
            let a = i / n;
            let c = j / n;
            let xi = i % n;
            let xj = j % n;
            let value = (0..self.latent)
                .map(|s| b[s][a][c] * Self::kernel(&self.x[xi], &self.x[xj], &ls[s]))
                .sum::<f64>();
            value + if i == j { noise[a] + 1e-8 } else { 0.0 }
        }))
    }
    fn targets(&self) -> DVector<f64> {
        DVector::from_fn(self.x.len() * self.objectives, |i, _| {
            let k = i / self.x.len();
            let delta = self.y[i % self.x.len()][k] - self.y_mean[k];
            if delta.is_finite() {
                delta / self.y_std[k]
            } else {
                self.y[i % self.x.len()][k] / self.y_std[k] - self.y_mean[k] / self.y_std[k]
            }
        })
    }
    fn negative_log_posterior(&self, p: &[f64]) -> Result<f64> {
        let cov = self.covariance(p)?;
        let Some(chol) = cov.cholesky() else {
            return Ok(f64::INFINITY);
        };
        let y = self.targets();
        let alpha = chol.solve(&y);
        let mut nll = 0.5 * y.dot(&alpha)
            + chol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>()
            + 0.5 * y.len() as f64 * (2.0 * std::f64::consts::PI).ln();
        let n_ls = self.latent * self.dim;
        nll += 0.5
            * p[..n_ls]
                .iter()
                .map(|v| (v - 0.5_f64.ln()).powi(2))
                .sum::<f64>();
        nll += 0.5
            * p[n_ls..n_ls + self.n_coreg()]
                .iter()
                .map(|v| v * v)
                .sum::<f64>();
        if self.optimize_noise {
            nll += 0.5
                * p[n_ls + self.n_coreg()..]
                    .iter()
                    .map(|v| ((v - 1e-3_f64.ln()) / 2.0).powi(2))
                    .sum::<f64>();
        }
        Ok(nll)
    }
    pub fn log_posterior(&self, p: &[f64]) -> Result<f64> {
        ensure!(!self.x.is_empty(), "fit before evaluating posterior");
        Ok(-self.negative_log_posterior(p)?)
    }

    fn optimize(&self, start: &[f64]) -> Result<Vec<f64>> {
        let bounds = self.hyperparameter_bounds();
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut best = start.to_vec();
        let mut best_value = f64::INFINITY;
        for restart in 0..self.optimizer_restarts.max(1) {
            let mut p: Vec<f64> = start
                .iter()
                .zip(&bounds)
                .map(|(v, (lo, hi))| {
                    let value = if restart == 0 {
                        *v
                    } else {
                        v + rng.gen_range(-0.75..0.75)
                    };
                    value.clamp(*lo, *hi)
                })
                .collect();
            let mut value = self.negative_log_posterior(&p)?;
            let mut step = 0.5;
            for _ in 0..self.max_iterations {
                let mut improved = false;
                for k in 0..p.len() {
                    let current = p[k];
                    let mut candidate = current;
                    let mut candidate_value = value;
                    for direction in [-1.0, 1.0] {
                        p[k] = (current + direction * step).clamp(bounds[k].0, bounds[k].1);
                        let trial = self.negative_log_posterior(&p)?;
                        if trial < candidate_value {
                            candidate = p[k];
                            candidate_value = trial;
                        }
                    }
                    p[k] = candidate;
                    if candidate_value < value - 1e-10 {
                        improved = true;
                    }
                    value = candidate_value;
                }
                if !improved {
                    step *= 0.5;
                    if step < 1e-3 {
                        break;
                    }
                }
            }
            if value < best_value {
                best = p;
                best_value = value;
            }
        }
        Ok(best)
    }

    fn set_state(&mut self, p: &[f64]) -> Result<()> {
        let (ls, b, noise) = self.unpack(p)?;
        let covariance = self.covariance(p)?;
        let chol = covariance
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("multi-output covariance factorization failed"))?;
        self.alpha = chol.solve(&self.targets());
        self.inverse = chol.inverse();
        self.theta = p.to_vec();
        self.lengthscales = ls;
        self.coregionalization = b;
        self.noise_variances = noise;
        Ok(())
    }

    pub fn with_hyperparameters(&self, p: &[f64]) -> Result<Self> {
        ensure!(!self.x.is_empty(), "fit before copying hyperparameters");
        let mut model = self.clone();
        model.set_state(p)?;
        Ok(model)
    }

    pub fn output_correlation(&self) -> Result<Matrix> {
        ensure!(!self.x.is_empty(), "fit before accessing correlations");
        let c = |i: usize, j: usize| self.coregionalization.iter().map(|b| b[i][j]).sum::<f64>();
        ensure!(
            (0..self.objectives).all(|i| c(i, i) > 0.0),
            "prior correlation is undefined for a zero output variance"
        );
        Ok((0..self.objectives)
            .map(|i| {
                (0..self.objectives)
                    .map(|j| c(i, j) / (c(i, i) * c(j, j)).sqrt())
                    .collect()
            })
            .collect())
    }

    /// Joint latent covariance in point-major order: `(point * L + objective)`.
    pub fn predict_joint(&self, x: &Matrix) -> Result<(Matrix, Matrix)> {
        ensure!(!self.x.is_empty(), "fit before prediction");
        matrix_dim(x, Some(self.dim))?;
        if x.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let n = self.x.len();
        let l = self.objectives;
        let q = x.len() * l;
        let cross = DMatrix::from_fn(q, n * l, |i, j| {
            (0..self.latent)
                .map(|s| {
                    self.coregionalization[s][i % l][j / n]
                        * Self::kernel(&x[i / l], &self.x[j % n], &self.lengthscales[s])
                })
                .sum::<f64>()
        });
        let mu = &cross * &self.alpha;
        let reduction = &cross * &self.inverse * cross.transpose();
        let mean: Matrix = (0..x.len())
            .map(|i| {
                (0..l)
                    .map(|k| mu[i * l + k] * self.y_std[k] + self.y_mean[k])
                    .collect()
            })
            .collect();
        let covariance: Matrix = (0..q)
            .map(|i| {
                (0..q)
                    .map(|j| {
                        let prior = (0..self.latent)
                            .map(|s| {
                                self.coregionalization[s][i % l][j % l]
                                    * Self::kernel(&x[i / l], &x[j / l], &self.lengthscales[s])
                            })
                            .sum::<f64>();
                        let value = (prior - 0.5 * (reduction[(i, j)] + reduction[(j, i)]))
                            * self.y_std[i % l]
                            * self.y_std[j % l];
                        if i == j { value.max(0.0) } else { value }
                    })
                    .collect()
            })
            .collect();
        ensure!(
            mean.iter()
                .flatten()
                .chain(covariance.iter().flatten())
                .all(|v| v.is_finite()),
            "posterior mean or covariance overflowed"
        );
        Ok((mean, covariance))
    }

    /// Mean `(n,L)` and one objective covariance `(L,L)` per candidate.
    pub fn predict(&self, x: &Matrix) -> Result<(Matrix, Vec<Matrix>)> {
        ensure!(!self.x.is_empty(), "fit before prediction");
        matrix_dim(x, Some(self.dim))?;
        let n = self.x.len();
        let l = self.objectives;
        let mut mean: Matrix = Vec::with_capacity(x.len());
        let mut covariance: Vec<Matrix> = Vec::with_capacity(x.len());
        // Marginal predictions need only one L x L covariance per point.
        // Constructing the entire (mL) x (mL) joint covariance here would make
        // scoring a large candidate pool unnecessarily quadratic in pool size.
        for point in x {
            let cross = DMatrix::from_fn(l, n * l, |a, j| {
                (0..self.latent)
                    .map(|s| {
                        self.coregionalization[s][a][j / n]
                            * Self::kernel(point, &self.x[j % n], &self.lengthscales[s])
                    })
                    .sum::<f64>()
            });
            let mu = &cross * &self.alpha;
            let reduction = &cross * &self.inverse * cross.transpose();
            mean.push(
                (0..l)
                    .map(|a| mu[a] * self.y_std[a] + self.y_mean[a])
                    .collect(),
            );
            covariance.push(
                (0..l)
                    .map(|a| {
                        (0..l)
                            .map(|b| {
                                let prior =
                                    self.coregionalization.iter().map(|s| s[a][b]).sum::<f64>();
                                let value = (prior - 0.5 * (reduction[(a, b)] + reduction[(b, a)]))
                                    * self.y_std[a]
                                    * self.y_std[b];
                                if a == b { value.max(0.0) } else { value }
                            })
                            .collect()
                    })
                    .collect(),
            );
        }
        ensure!(
            mean.iter().flatten().all(|v| v.is_finite())
                && covariance.iter().flatten().flatten().all(|v| v.is_finite()),
            "posterior mean or covariance overflowed"
        );
        Ok((mean, covariance))
    }

    pub fn sample_hyperparameters(
        &self,
        n_samples: usize,
        burn_in: usize,
        thin: usize,
        seed: u64,
    ) -> Result<Matrix> {
        ensure!(
            !self.theta.is_empty(),
            "fit before sampling hyperparameters"
        );
        let bounds = self.hyperparameter_bounds();
        let options = SliceOptions {
            widths: vec![1.0; self.theta.len()],
            lower: bounds.iter().map(|b| b.0).collect(),
            upper: bounds.iter().map(|b| b.1).collect(),
            burn_in,
            thin,
            ..Default::default()
        };
        let start: Vec<f64> = self
            .theta
            .iter()
            .zip(&bounds)
            .map(|(v, (lo, hi))| v.clamp(*lo, *hi))
            .collect();
        slice_sample(
            |p| self.log_posterior(p).unwrap_or(f64::NEG_INFINITY),
            &start,
            n_samples,
            seed,
            &options,
        )
    }
}
