use super::network::{Adam, Network, inputs, targets, write_json};
use crate::core::GaussianProcess;
use crate::{Matrix, Result};
use anyhow::{Context, ensure};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

const LOGVAR_MIN: f64 = -13.815510557964274;
const LOGVAR_MAX: f64 = 6.907755278982137;

/// MC-dropout population model settings. Full-batch training is the default.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct BnnConfig {
    pub input_dim: usize,
    pub hidden_units: usize,
    pub n_hidden_layers: usize,
    pub dropout: f64,
    pub learning_rate: f64,
    pub batch_size: Option<usize>,
    pub n_predict_samples: usize,
    pub seed: u64,
}

impl Default for BnnConfig {
    fn default() -> Self {
        Self {
            input_dim: 2,
            hidden_units: 100,
            n_hidden_layers: 3,
            dropout: 0.1,
            learning_rate: 1e-3,
            batch_size: None,
            n_predict_samples: 50,
            seed: 0,
        }
    }
}

impl BnnConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.input_dim > 0 && self.hidden_units > 0 && self.n_hidden_layers > 0,
            "BNN dimensions must be positive"
        );
        ensure!(
            self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout),
            "dropout must be in [0,1)"
        );
        ensure!(
            self.learning_rate.is_finite() && self.learning_rate > 0.0,
            "learning rate must be positive and finite"
        );
        ensure!(
            self.batch_size != Some(0) && self.n_predict_samples > 0,
            "batch and prediction sample counts must be positive"
        );
        ensure!(
            self.n_hidden_layers <= 128
                && self.hidden_units <= 100_000
                && self.input_dim <= 100_000,
            "network dimensions exceed supported resource limits"
        );
        let widths = self.widths();
        let parameters = widths.windows(2).try_fold(0usize, |total, w| {
            w[0].checked_mul(w[1])
                .and_then(|n| n.checked_add(w[1]))
                .and_then(|n| total.checked_add(n))
        });
        ensure!(
            parameters.is_some_and(|n| n <= 16_000_000),
            "BNN exceeds the 16 million parameter resource limit"
        );
        Ok(())
    }
    fn widths(&self) -> Vec<usize> {
        let mut widths = vec![self.input_dim];
        widths.extend(std::iter::repeat_n(self.hidden_units, self.n_hidden_layers));
        widths.push(2);
        widths
    }
}

/// Gaussian heteroscedastic MLP with persistent Adam/RNG state.
///
/// Normalization is fixed on the first fit. Predictions use the law of total
/// variance: average aleatoric variance plus variance of dropout means.
#[derive(Clone, Serialize, Deserialize)]
pub struct Bnn {
    config: BnnConfig,
    net: Network,
    optimizer: Adam,
    train_rng: ChaCha8Rng,
    pub y_offset: Option<f64>,
    pub y_scale: Option<f64>,
}

pub type PopulationBnn = Bnn;
#[allow(clippy::upper_case_acronyms)]
pub type PopulationBNN = Bnn;

impl Bnn {
    pub fn new(config: BnnConfig) -> Result<Self> {
        config.validate()?;
        let mut init_rng = ChaCha8Rng::seed_from_u64(config.seed);
        let net = Network::new(&config.widths(), false, &mut init_rng);
        let optimizer = Adam::new(net.parameter_count());
        let train_rng = ChaCha8Rng::seed_from_u64(config.seed);
        Ok(Self {
            config,
            net,
            optimizer,
            train_rng,
            y_offset: None,
            y_scale: None,
        })
    }

    pub fn config(&self) -> &BnnConfig {
        &self.config
    }
    pub fn is_normalized(&self) -> bool {
        self.y_offset.is_some()
    }

    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        self.net.validate(&self.config.widths())?;
        self.optimizer.validate(self.net.parameter_count())?;
        ensure!(
            self.y_offset.is_some() == self.y_scale.is_some(),
            "normalization checkpoint is incomplete"
        );
        ensure!(
            self.y_offset.is_none_or(|v| v.is_finite())
                && self.y_scale.is_none_or(|v| v.is_finite() && v > 0.0),
            "normalization must be finite with positive scale"
        );
        Ok(())
    }

    pub fn reset_parameters(&mut self) -> Result<()> {
        let fresh = Self::new(self.config.clone())?;
        self.net = fresh.net;
        self.optimizer = fresh.optimizer;
        Ok(())
    }

    pub fn set_normalization(&mut self, y: &[f64]) -> Result<()> {
        ensure!(!y.is_empty(), "normalization requires observations");
        targets(y, y.len())?;
        let mean = y.iter().map(|v| v / y.len() as f64).sum::<f64>();
        let variance = y
            .iter()
            .map(|v| (v - mean).powi(2) / y.len() as f64)
            .sum::<f64>();
        ensure!(
            mean.is_finite() && variance.is_finite(),
            "normalization overflow; rescale targets"
        );
        self.y_offset = Some(mean);
        self.y_scale = Some(if variance.sqrt() > 1e-12 {
            variance.sqrt()
        } else {
            1.0
        });
        Ok(())
    }

    pub fn fit(&mut self, x: &Matrix, y: &[f64], epochs: usize) -> Result<Vec<f64>> {
        self.fit_nll(x, y, epochs)
    }

    pub fn fit_nll(&mut self, x: &Matrix, y: &[f64], epochs: usize) -> Result<Vec<f64>> {
        self.validate()?;
        inputs(x, self.config.input_dim, true)?;
        targets(y, x.len())?;
        let mut next = self.clone();
        if !next.is_normalized() {
            next.set_normalization(y)?;
        }
        let normalized = next.normalize(y);
        targets(&normalized, x.len())?;
        let history = next.train(x, &normalized, None, epochs)?;
        *self = next;
        Ok(history)
    }

    pub fn fit_replay(
        &mut self,
        x: &Matrix,
        mean: &[f64],
        variance: &[f64],
        epochs: usize,
        renormalize: bool,
    ) -> Result<Vec<f64>> {
        self.validate()?;
        inputs(x, self.config.input_dim, true)?;
        targets(mean, x.len())?;
        targets(variance, x.len())?;
        ensure!(
            variance.iter().all(|v| *v >= 0.0),
            "replay variances must be nonnegative"
        );
        let mut next = self.clone();
        if renormalize || !next.is_normalized() {
            next.set_normalization(mean)?;
        }
        let mu = next.normalize(mean);
        let scale = next.y_scale.unwrap();
        let var: Vec<f64> = variance.iter().map(|v| v / scale / scale).collect();
        targets(&mu, x.len())?;
        targets(&var, x.len())?;
        let history = next.train(x, &mu, Some(&var), epochs)?;
        *self = next;
        Ok(history)
    }

    fn normalize(&self, y: &[f64]) -> Vec<f64> {
        y.iter()
            .map(|v| (v - self.y_offset.unwrap()) / self.y_scale.unwrap())
            .collect()
    }

    fn train(
        &mut self,
        x: &Matrix,
        y: &[f64],
        variance: Option<&[f64]>,
        epochs: usize,
    ) -> Result<Vec<f64>> {
        let batch_size = self.config.batch_size.unwrap_or(x.len()).min(x.len());
        let mut order: Vec<usize> = (0..x.len()).collect();
        let mut history = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            order.shuffle(&mut self.train_rng);
            let mut total = 0.0;
            for batch in order.chunks(batch_size) {
                let mut grad = vec![0.0; self.net.parameter_count()];
                for &index in batch {
                    let (output, tape) =
                        self.net
                            .forward(&x[index], self.config.dropout, &mut self.train_rng);
                    let mean = output[0];
                    let logvar = output[1].clamp(LOGVAR_MIN, LOGVAR_MAX);
                    let var = logvar.exp();
                    let residual = mean - y[index];
                    let (loss, dmean, dlogvar) = if let Some(v) = variance {
                        (
                            residual.powi(2) + (var - v[index]).powi(2),
                            2.0 * residual,
                            2.0 * (var - v[index]) * var,
                        )
                    } else {
                        (
                            0.5 * (logvar + residual.powi(2) / var),
                            residual / var,
                            0.5 * (1.0 - residual.powi(2) / var),
                        )
                    };
                    ensure!(
                        loss.is_finite(),
                        "BNN loss overflow; rescale inputs/targets or reduce learning rate"
                    );
                    total += loss;
                    let dlogvar = if (LOGVAR_MIN..=LOGVAR_MAX).contains(&output[1]) {
                        dlogvar
                    } else {
                        0.0
                    };
                    self.net.backward(
                        &tape,
                        &[dmean / batch.len() as f64, dlogvar / batch.len() as f64],
                        &mut grad,
                    );
                }
                self.net
                    .apply(&mut self.optimizer, &grad, self.config.learning_rate)?;
            }
            let loss = total / x.len() as f64;
            ensure!(loss.is_finite(), "BNN epoch loss overflow");
            history.push(loss);
        }
        Ok(history)
    }

    pub fn predict(&self, x: &Matrix) -> Result<(Vec<f64>, Vec<f64>)> {
        self.predict_samples(x, self.config.n_predict_samples)
    }

    pub fn predict_samples(&self, x: &Matrix, n_samples: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        self.validate()?;
        inputs(x, self.config.input_dim, false)?;
        ensure!(n_samples > 0, "prediction sample count must be positive");
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed.wrapping_add(1));
        let (mut mean, mut m2, mut aleatoric) =
            (vec![0.0; x.len()], vec![0.0; x.len()], vec![0.0; x.len()]);
        for sample in 1..=n_samples {
            for (i, row) in x.iter().enumerate() {
                let output = self.net.forward(row, self.config.dropout, &mut rng).0;
                let delta = output[0] - mean[i];
                mean[i] += delta / sample as f64;
                m2[i] += delta * (output[0] - mean[i]);
                aleatoric[i] += output[1].clamp(LOGVAR_MIN, LOGVAR_MAX).exp() / n_samples as f64;
            }
        }
        let offset = self.y_offset.unwrap_or(0.0);
        let scale = self.y_scale.unwrap_or(1.0);
        let variance = aleatoric
            .iter()
            .zip(&m2)
            .map(|(a, e)| scale * scale * (a + e / n_samples as f64).max(0.0))
            .collect::<Vec<_>>();
        mean.iter_mut().for_each(|v| *v = offset + scale * *v);
        targets(&mean, x.len())?;
        targets(&variance, x.len())?;
        Ok((mean, variance))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        write_json(
            path.as_ref(),
            &serde_json::json!({"format":"openbo-bnn","version":1,"model":self}),
        )
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let payload: serde_json::Value = serde_json::from_slice(&std::fs::read(path.as_ref())?)
            .context("invalid native BNN JSON checkpoint")?;
        ensure!(
            payload["format"] == "openbo-bnn" && payload["version"] == 1,
            "unsupported BNN checkpoint format/version"
        );
        let model: Self = serde_json::from_value(payload["model"].clone())?;
        model.validate()?;
        Ok(model)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayTargets {
    pub x: Matrix,
    pub mean: Vec<f64>,
    pub var: Vec<f64>,
    pub n_users: Vec<usize>,
    pub kept_fraction: Vec<f64>,
    pub excluded_users: Vec<usize>,
}

/// Grid in 1D/2D, low-discrepancy Sobol design above 2D, plus uniform points.
pub fn replay_inputs(
    dim: usize,
    grid_points_per_dim: usize,
    n_random: usize,
    seed: u64,
) -> Result<Matrix> {
    ensure!(
        dim > 0 && grid_points_per_dim >= 2,
        "replay dimensions must be positive and grid width at least 2"
    );
    let n_grid = if dim == 1 {
        grid_points_per_dim
    } else {
        grid_points_per_dim
            .checked_mul(grid_points_per_dim)
            .context("replay grid size overflow")?
    };
    ensure!(
        n_grid
            .checked_add(n_random)
            .is_some_and(|n| n <= 10_000_000),
        "replay grid exceeds resource limit"
    );
    let mut x = if dim <= 2 {
        super::naf::regular_grid(grid_points_per_dim, dim)?
    } else {
        super::naf::sobol_points(n_grid, dim, seed)?
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    x.extend((0..n_random).map(|_| (0..dim).map(|_| rng.r#gen()).collect()));
    Ok(x)
}

pub fn replay_targets(
    gps: &[GaussianProcess],
    x: &Matrix,
    max_variance: Option<f64>,
) -> Result<ReplayTargets> {
    ensure!(!gps.is_empty(), "replay needs stored user GPs");
    let dim = x.first().context("replay inputs must be nonempty")?.len();
    ensure!(dim > 0, "replay dimension must be positive");
    inputs(x, dim, true)?;
    ensure!(
        max_variance.is_none_or(|v| v.is_finite() && v > 0.0),
        "variance threshold must be finite and positive"
    );
    let mut counts = vec![0usize; x.len()];
    let mut means = vec![0.0; x.len()];
    let mut variances = vec![0.0; x.len()];
    let mut kept_fraction = Vec::with_capacity(gps.len());
    let mut excluded_users = Vec::new();
    for (user, gp) in gps.iter().enumerate() {
        let (mean, var) = gp.predict(x)?;
        targets(&mean, x.len())?;
        targets(&var, x.len())?;
        ensure!(
            var.iter().all(|v| *v >= 0.0),
            "GP returned negative variance"
        );
        let mut kept = 0;
        for i in 0..x.len() {
            if max_variance.is_none_or(|threshold| var[i] < threshold) {
                counts[i] += 1;
                means[i] += (mean[i] - means[i]) / counts[i] as f64;
                variances[i] += (var[i] - variances[i]) / counts[i] as f64;
                kept += 1;
            }
        }
        kept_fraction.push(kept as f64 / x.len() as f64);
        if kept == 0 {
            excluded_users.push(user);
        }
    }
    let kept: Vec<usize> = (0..x.len()).filter(|i| counts[*i] > 0).collect();
    Ok(ReplayTargets {
        x: kept.iter().map(|i| x[*i].clone()).collect(),
        mean: kept.iter().map(|i| means[*i]).collect(),
        var: kept.iter().map(|i| variances[*i]).collect(),
        n_users: kept.iter().map(|i| counts[*i]).collect(),
        kept_fraction,
        excluded_users,
    })
}

pub fn standardized_variance_threshold(
    gps: &[GaussianProcess],
    x: &Matrix,
    fraction: f64,
) -> Result<f64> {
    ensure!(
        fraction.is_finite() && fraction > 0.0 && !gps.is_empty() && !x.is_empty(),
        "threshold needs positive fraction, GPs, and inputs"
    );
    let mut all = Vec::new();
    for gp in gps {
        all.extend(gp.predict(x)?.0);
    }
    targets(&all, all.len())?;
    let mean = all.iter().map(|v| v / all.len() as f64).sum::<f64>();
    let spread = all
        .iter()
        .map(|v| (v - mean).powi(2) / all.len() as f64)
        .sum::<f64>();
    let threshold = fraction * if spread > 1e-12 { spread } else { 1.0 };
    ensure!(
        threshold.is_finite() && threshold > 0.0,
        "variance threshold overflow/underflow"
    );
    Ok(threshold)
}
