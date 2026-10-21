use super::network::{Adam, Network, inputs, targets, write_json};
use crate::core::{GaussianProcess, GpConfig};
use crate::{Matrix, Point, Result};
use anyhow::{Context, ensure};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc};

/// MetaBO's per-candidate features `[mu, sigma, x..., t, T, weights...]`.
pub fn naf_features(
    mean: &[f64],
    std: &[f64],
    x: &Matrix,
    t: usize,
    budget: usize,
    use_x: bool,
    weights: Option<&[f64]>,
) -> Result<Matrix> {
    ensure!(
        budget > 0 && t <= budget,
        "step must be in [0,budget] with positive budget"
    );
    let dim = x.first().map_or(0, Vec::len);
    inputs(x, dim, false)?;
    targets(mean, x.len())?;
    targets(std, x.len())?;
    ensure!(
        std.iter().all(|v| *v >= 0.0),
        "posterior std must be nonnegative"
    );
    if let Some(w) = weights {
        validate_weights(w, w.len())?;
    }
    Ok(x.iter()
        .enumerate()
        .map(|(i, row)| {
            let mut f = vec![mean[i], std[i]];
            if use_x {
                f.extend(row);
            }
            f.extend([t as f64, budget as f64]);
            if let Some(w) = weights {
                f.extend(w);
            }
            f
        })
        .collect())
}

pub fn n_naf_features(dim: usize, use_x: bool, n_weights: usize) -> usize {
    4usize
        .saturating_add(if use_x { dim } else { 0 })
        .saturating_add(n_weights)
}

pub fn validate_weights(w: &[f64], n: usize) -> Result<()> {
    ensure!(
        n > 0 && w.len() == n && w.iter().all(|v| v.is_finite() && *v >= 0.0),
        "objective weights must be finite nonnegative values of expected length"
    );
    ensure!(
        (w.iter().sum::<f64>() - 1.0).abs() <= 1e-9,
        "objective weights must sum to one"
    );
    Ok(())
}

pub fn regular_grid(points_per_dim: usize, dim: usize) -> Result<Matrix> {
    ensure!(
        points_per_dim >= 2 && dim > 0,
        "grid width must be at least 2 and dimension positive"
    );
    let count = points_per_dim
        .checked_pow(dim.try_into().context("dimension too large")?)
        .context("grid size overflow")?;
    ensure!(
        count <= 2_000_000,
        "global grid too large; reduce points_per_dim"
    );
    Ok((0..count)
        .map(|mut index| {
            let mut row = vec![0.0; dim];
            for value in row.iter_mut().rev() {
                *value = (index % points_per_dim) as f64 / (points_per_dim - 1) as f64;
                index /= points_per_dim;
            }
            row
        })
        .collect())
}

/// Digitally shifted Sobol points. Different seed/float backends do not yield
/// SciPy's Owen-scrambled point sequence, but preserve the Sobol construction.
/// Supports dimensions 1..=16 with standard primitive-polynomial directions.
pub fn sobol_points(n: usize, dim: usize, seed: u64) -> Result<Matrix> {
    const TABLE: &[(usize, u32, &[u32])] = &[
        (1, 0, &[1]),
        (2, 1, &[1, 3]),
        (3, 1, &[1, 3, 1]),
        (3, 2, &[1, 1, 1]),
        (4, 1, &[1, 1, 3, 3]),
        (4, 4, &[1, 3, 5, 13]),
        (5, 2, &[1, 1, 5, 5, 17]),
        (5, 4, &[1, 1, 5, 5, 5]),
        (5, 7, &[1, 1, 7, 11, 19]),
        (5, 11, &[1, 1, 5, 1, 1]),
        (5, 13, &[1, 1, 1, 3, 11]),
        (5, 14, &[1, 3, 5, 5, 31]),
        (6, 1, &[1, 3, 3, 9, 7, 49]),
        (6, 13, &[1, 1, 1, 15, 21, 21]),
        (6, 16, &[1, 3, 1, 13, 27, 49]),
    ];
    ensure!(
        (1..=16).contains(&dim),
        "native Sobol supports dimensions 1..=16"
    );
    ensure!(n <= 10_000_000, "Sobol sample count exceeds resource limit");
    let mut directions = vec![[0u64; 64]; dim];
    for (bit, direction) in directions[0].iter_mut().enumerate() {
        *direction = 1u64 << (63 - bit);
    }
    for d in 1..dim {
        let (degree, a, m) = TABLE[d - 1];
        for (j, value) in m.iter().enumerate() {
            directions[d][j] = (*value as u64) << (63 - j);
        }
        for j in degree..64 {
            let mut v = directions[d][j - degree] ^ (directions[d][j - degree] >> degree);
            for k in 1..degree {
                if (a >> (degree - 1 - k)) & 1 != 0 {
                    v ^= directions[d][j - k];
                }
            }
            directions[d][j] = v;
        }
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let shifts: Vec<u64> = (0..dim).map(|_| rng.r#gen()).collect();
    let mut states = vec![0u64; dim];
    let mut points = Vec::with_capacity(n);
    for index in 0..n {
        if index > 0 {
            let bit = index.trailing_zeros() as usize;
            for d in 0..dim {
                states[d] ^= directions[d][bit];
            }
        }
        points.push(
            (0..dim)
                .map(|d| ((states[d] ^ shifts[d]) >> 11) as f64 / 9007199254740992.0)
                .collect(),
        );
    }
    Ok(points)
}

/// Fixed global grid plus the top `k` local acquisition maxima.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdaptiveSet {
    pub global_grid: Matrix,
    pub local_offsets: Matrix,
    pub k: usize,
}

impl AdaptiveSet {
    pub fn build(
        dim: usize,
        points_per_dim: usize,
        k: usize,
        n_local: usize,
        local_side: Option<f64>,
        seed: u64,
    ) -> Result<Self> {
        ensure!(
            k > 0 && n_local > 0,
            "local candidate counts must be positive"
        );
        let side = local_side.unwrap_or(2.0 / points_per_dim as f64);
        ensure!(
            side.is_finite() && side > 0.0,
            "local cube side must be finite and positive"
        );
        let global_grid = regular_grid(points_per_dim, dim)?;
        ensure!(
            k <= global_grid.len(),
            "local maxima count exceeds grid size"
        );
        let mut local_offsets = sobol_points(n_local, dim, seed)?;
        for row in &mut local_offsets {
            row.iter_mut().for_each(|v| *v = side * (*v - 0.5));
        }
        Ok(Self {
            global_grid,
            local_offsets,
            k,
        })
    }
    pub fn size(&self) -> usize {
        self.global_grid.len().saturating_add(self.k)
    }

    pub fn validate(&self) -> Result<usize> {
        let dim = self
            .global_grid
            .first()
            .context("candidate grid is empty")?
            .len();
        ensure!(
            dim > 0 && self.k > 0 && self.k <= self.global_grid.len(),
            "invalid candidate dimensions/count"
        );
        inputs(&self.global_grid, dim, true)?;
        inputs(&self.local_offsets, dim, true)?;
        ensure!(
            self.global_grid
                .iter()
                .flatten()
                .all(|v| (0.0..=1.0).contains(v)),
            "global grid must be in unit cube"
        );
        Ok(dim)
    }

    pub fn local_points(&self, centres: &Matrix) -> Result<Matrix> {
        let dim = self.validate()?;
        inputs(centres, dim, false)?;
        ensure!(
            centres
                .len()
                .checked_mul(self.local_offsets.len())
                .is_some_and(|n| n <= 10_000_000),
            "local candidate clouds exceed resource limit"
        );
        Ok(centres
            .iter()
            .flat_map(|c| {
                self.local_offsets.iter().map(move |off| {
                    c.iter()
                        .zip(off)
                        .map(|(a, b)| (a + b).clamp(0.0, 1.0))
                        .collect()
                })
            })
            .collect())
    }

    pub fn candidates<F>(&self, score: F) -> Result<(Matrix, Vec<f64>)>
    where
        F: Fn(&Matrix) -> Result<Vec<f64>>,
    {
        self.validate()?;
        let grid_scores = score(&self.global_grid)?;
        targets(&grid_scores, self.global_grid.len())?;
        let mut grid_order: Vec<usize> = (0..grid_scores.len()).collect();
        grid_order.sort_by(|a, b| grid_scores[*b].total_cmp(&grid_scores[*a]).then(a.cmp(b)));
        let centres = grid_order[..self.k]
            .iter()
            .map(|i| self.global_grid[*i].clone())
            .collect();
        let local = self.local_points(&centres)?;
        let local_scores = score(&local)?;
        targets(&local_scores, local.len())?;
        let mut local_order: Vec<usize> = (0..local_scores.len()).collect();
        local_order.sort_by(|a, b| local_scores[*b].total_cmp(&local_scores[*a]).then(a.cmp(b)));
        let mut points: Matrix = local_order[..self.k]
            .iter()
            .map(|i| local[*i].clone())
            .collect();
        let mut scores: Vec<f64> = local_order[..self.k]
            .iter()
            .map(|i| local_scores[*i])
            .collect();
        points.extend_from_slice(&self.global_grid);
        scores.extend(grid_scores);
        Ok((points, scores))
    }

    pub fn maximize<F>(&self, score: F) -> Result<Point>
    where
        F: Fn(&Matrix) -> Result<Vec<f64>>,
    {
        let (points, scores) = self.candidates(score)?;
        Ok(points[argmax(&scores)].clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct NafConfig {
    pub dim: usize,
    pub budget: usize,
    pub use_x: bool,
    pub hidden: Vec<usize>,
    pub points_per_dim: usize,
    pub n_local_maxima: usize,
    pub n_local: usize,
    pub n_iterations: usize,
    pub steps_per_iteration: usize,
    pub epochs: usize,
    pub minibatch_size: usize,
    pub learning_rate: f64,
    pub clip: f64,
    pub value_coefficient: f64,
    pub entropy_coefficient: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub seed: u64,
    pub n_weights: usize,
    /// Scoped native rollout workers. PPO updates use deterministic scalar kernels.
    pub n_workers: usize,
}

impl Default for NafConfig {
    fn default() -> Self {
        Self {
            dim: 2,
            budget: 30,
            use_x: true,
            hidden: vec![200; 4],
            points_per_dim: 31,
            n_local_maxima: 5,
            n_local: 1000,
            n_iterations: 2000,
            steps_per_iteration: 1200,
            epochs: 4,
            minibatch_size: 60,
            learning_rate: 1e-4,
            clip: 0.15,
            value_coefficient: 1.0,
            entropy_coefficient: 0.01,
            gamma: 0.98,
            gae_lambda: 0.98,
            seed: 0,
            n_weights: 0,
            n_workers: 1,
        }
    }
}

impl NafConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.dim > 0 && self.dim <= 16 && self.budget > 0,
            "NAF dimension must be 1..=16 and budget positive"
        );
        ensure!(
            self.points_per_dim >= 2 && self.n_local_maxima > 0 && self.n_local > 0,
            "candidate counts must be positive with grid width >=2"
        );
        ensure!(
            self.n_iterations > 0
                && self.steps_per_iteration > 0
                && self.steps_per_iteration % self.budget == 0,
            "training steps must contain at least one whole episode"
        );
        ensure!(
            self.epochs > 0 && self.minibatch_size > 0,
            "PPO epochs and minibatch size must be positive"
        );
        ensure!(
            self.hidden.len() <= 128 && self.hidden.iter().all(|n| *n > 0 && *n <= 100_000),
            "invalid network hidden widths"
        );
        ensure!(self.n_weights <= 10_000, "too many objective weights");
        ensure!(
            (1..=64).contains(&self.n_workers),
            "rollout worker count must be in 1..=64"
        );
        for widths in [self.policy_widths(), self.value_widths()] {
            let parameters = widths.windows(2).try_fold(0usize, |total, w| {
                w[0].checked_mul(w[1])
                    .and_then(|n| n.checked_add(w[1]))
                    .and_then(|n| total.checked_add(n))
            });
            ensure!(
                parameters.is_some_and(|n| n <= 16_000_000),
                "NAF exceeds the 16 million parameter resource limit"
            );
        }
        ensure!(
            self.learning_rate.is_finite() && self.learning_rate > 0.0,
            "learning rate must be finite and positive"
        );
        ensure!(
            self.clip.is_finite() && self.clip > 0.0 && self.clip < 1.0,
            "PPO clip must be in (0,1)"
        );
        ensure!(
            [self.value_coefficient, self.entropy_coefficient]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0),
            "PPO coefficients must be finite and nonnegative"
        );
        ensure!(
            self.gamma.is_finite()
                && self.gamma > 0.0
                && self.gamma <= 1.0
                && self.gae_lambda.is_finite()
                && (0.0..=1.0).contains(&self.gae_lambda),
            "invalid GAE discount parameters"
        );
        Ok(())
    }
    pub fn episodes_per_iteration(&self) -> usize {
        self.steps_per_iteration
            .checked_div(self.budget)
            .unwrap_or(0)
    }
    pub fn adaptive_set(&self) -> Result<AdaptiveSet> {
        self.validate()?;
        AdaptiveSet::build(
            self.dim,
            self.points_per_dim,
            self.n_local_maxima,
            self.n_local,
            None,
            self.seed,
        )
    }
    pub fn new_af(&self) -> Result<NeuralAf> {
        NeuralAf::new(self)
    }
    fn policy_widths(&self) -> Vec<usize> {
        let mut widths = vec![n_naf_features(self.dim, self.use_x, self.n_weights)];
        widths.extend(&self.hidden);
        widths.push(1);
        widths
    }
    fn value_widths(&self) -> Vec<usize> {
        let mut widths = vec![2 + self.n_weights];
        widths.extend(&self.hidden);
        widths.push(1);
        widths
    }
}

/// Neural acquisition policy, with optional simplex objective weights.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeuralAf {
    pub dim: usize,
    pub use_x: bool,
    pub hidden: Vec<usize>,
    pub n_weights: usize,
    net: Network,
}

pub type NeuralAF = NeuralAf;

impl NeuralAf {
    pub fn new(config: &NafConfig) -> Result<Self> {
        config.validate()?;
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        Ok(Self {
            dim: config.dim,
            use_x: config.use_x,
            hidden: config.hidden.clone(),
            n_weights: config.n_weights,
            net: Network::new(&config.policy_widths(), true, &mut rng),
        })
    }
    fn widths(&self) -> Vec<usize> {
        let mut w = vec![n_naf_features(self.dim, self.use_x, self.n_weights)];
        w.extend(&self.hidden);
        w.push(1);
        w
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.dim > 0
                && self.dim <= 16
                && self.hidden.len() <= 128
                && self.hidden.iter().all(|v| *v > 0 && *v <= 100_000),
            "invalid policy architecture"
        );
        ensure!(self.n_weights <= 10_000, "too many objective weights");
        let parameters = self.widths().windows(2).try_fold(0usize, |total, w| {
            w[0].checked_mul(w[1])
                .and_then(|n| n.checked_add(w[1]))
                .and_then(|n| total.checked_add(n))
        });
        ensure!(
            parameters.is_some_and(|n| n <= 16_000_000),
            "policy exceeds the 16 million parameter resource limit"
        );
        self.net.validate(&self.widths())
    }
    pub fn logits(&self, features: &Matrix) -> Result<Vec<f64>> {
        self.validate()?;
        inputs(
            features,
            n_naf_features(self.dim, self.use_x, self.n_weights),
            false,
        )?;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let logits = features
            .iter()
            .map(|row| self.net.forward(row, 0.0, &mut rng).0[0])
            .collect::<Vec<_>>();
        targets(&logits, features.len())?;
        Ok(logits)
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        write_json(
            path.as_ref(),
            &serde_json::json!({"format":"openbo-neural-af","version":1,"policy":self}),
        )
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            value["format"] == "openbo-neural-af" && value["version"] == 1,
            "unsupported policy checkpoint format/version"
        );
        let policy: Self = serde_json::from_value(value["policy"].clone())?;
        policy.validate()?;
        Ok(policy)
    }

    /// Export a portable JSON object with PyTorch `NeuralAF.state_dict()` keys.
    /// In Python, create tensors from this object's nested lists and use
    /// `af.load_state_dict(...)`. Native checkpoints include additional metadata.
    pub fn export_state_dict_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut state = serde_json::Map::new();
        for (i, layer) in self.net.layers.iter().enumerate() {
            let key = 2 * i;
            let matrix: Matrix = layer
                .weights
                .chunks(layer.inputs)
                .map(|row| row.to_vec())
                .collect();
            state.insert(format!("net.{key}.weight"), serde_json::to_value(matrix)?);
            state.insert(
                format!("net.{key}.bias"),
                serde_json::to_value(&layer.biases)?,
            );
        }
        write_json(path.as_ref(), &state)
    }

    /// Import the portable JSON conversion of a Python NeuralAF state dictionary.
    /// Tensor names, shapes and finite values are validated before mutation.
    pub fn import_state_dict_json(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let state: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&std::fs::read(path)?)?;
        self.import_state_dict(&state)
    }

    /// Import an in-memory state dictionary transactionally, without staging files.
    pub fn import_state_dict(
        &mut self,
        state: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<()> {
        ensure!(
            state.len() == 2 * self.net.layers.len(),
            "state dictionary tensor count mismatch"
        );
        let mut candidate = self.clone();
        for (i, layer) in candidate.net.layers.iter_mut().enumerate() {
            let key = 2 * i;
            let weight: Matrix = serde_json::from_value(
                state
                    .get(&format!("net.{key}.weight"))
                    .context("missing weight tensor")?
                    .clone(),
            )?;
            inputs(&weight, layer.inputs, true)?;
            ensure!(
                weight.len() == layer.outputs,
                "state dictionary weight shape mismatch"
            );
            let bias: Vec<f64> = serde_json::from_value(
                state
                    .get(&format!("net.{key}.bias"))
                    .context("missing bias tensor")?
                    .clone(),
            )?;
            targets(&bias, layer.outputs)?;
            layer.weights = weight.into_iter().flatten().collect();
            layer.biases = bias;
        }
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }
}

fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}

fn log_softmax(logits: &[f64]) -> Result<Vec<f64>> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
        "categorical logits must be finite and nonempty"
    );
    let max = logits[argmax(logits)];
    let logsum = logits.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
    let result: Vec<_> = logits.iter().map(|v| v - max - logsum).collect();
    ensure!(
        result.iter().all(|v| v.is_finite()),
        "categorical logit spread overflow"
    );
    Ok(result)
}

fn sample_action(log_probs: &[f64], rng: &mut ChaCha8Rng) -> usize {
    let draw: f64 = rng.r#gen();
    let mut cumulative = 0.0;
    for (i, lp) in log_probs.iter().enumerate() {
        cumulative += lp.exp();
        if draw < cumulative {
            return i;
        }
    }
    log_probs.len() - 1
}

fn policy_features(
    policy: &NeuralAf,
    gp: &GaussianProcess,
    fitted: bool,
    x: &Matrix,
    t: usize,
    budget: usize,
    weights: Option<&[f64]>,
) -> Result<Matrix> {
    let (mean, var) = if fitted {
        gp.predict(x)?
    } else {
        let scale = gp.config().fixed_target_std.unwrap_or(1.0);
        let mean = gp.config().fixed_target_mean.unwrap_or(0.0) + scale * gp.config().mean_constant;
        (
            vec![mean; x.len()],
            vec![gp.config().variance * scale * scale; x.len()],
        )
    };
    let std = var.iter().map(|v| v.max(0.0).sqrt()).collect::<Vec<_>>();
    naf_features(&mean, &std, x, t, budget, policy.use_x, weights)
}

/// Ask/tell maximization using a trained NAF and fixed GP hyperparameters.
pub struct NafOptimizer {
    pub config: NafConfig,
    pub policy: NeuralAf,
    pub gp: GaussianProcess,
    pub adaptive_set: AdaptiveSet,
    pub weights: Option<Vec<f64>>,
    pub x_obs: Matrix,
    pub y_obs: Vec<f64>,
}

pub type NAFSequentialOptimizer = NafOptimizer;

impl NafOptimizer {
    pub fn new(
        config: NafConfig,
        policy: NeuralAf,
        gp_config: GpConfig,
        weights: Option<Vec<f64>>,
    ) -> Result<Self> {
        config.validate()?;
        policy.validate()?;
        ensure!(
            policy.dim == config.dim
                && policy.use_x == config.use_x
                && policy.n_weights == config.n_weights
                && policy.hidden == config.hidden,
            "NAF policy/config mismatch"
        );
        let weights = if config.n_weights > 0 {
            let w = weights.context("conditioned policy requires objective weights")?;
            validate_weights(&w, config.n_weights)?;
            Some(w)
        } else {
            None
        };
        ensure!(
            gp_config.lengthscale.len() == 1 || gp_config.lengthscale.len() == config.dim,
            "GP lengthscale dimension mismatch"
        );
        let gp = GaussianProcess::new(gp_config)?;
        let adaptive_set = config.adaptive_set()?;
        Ok(Self {
            config,
            policy,
            gp,
            adaptive_set,
            weights,
            x_obs: Vec::new(),
            y_obs: Vec::new(),
        })
    }
    pub fn t(&self) -> usize {
        self.y_obs.len()
    }
    pub fn done(&self) -> bool {
        self.t() >= self.config.budget
    }
    pub fn observation(&self) -> Result<(Matrix, Matrix, Vec<f64>)> {
        ensure!(!self.done(), "NAF evaluation budget exhausted");
        let features = |x: &Matrix| {
            policy_features(
                &self.policy,
                &self.gp,
                self.t() > 0,
                x,
                self.t(),
                self.config.budget,
                self.weights.as_deref(),
            )
        };
        let (points, logits) = self
            .adaptive_set
            .candidates(|x| self.policy.logits(&features(x)?))?;
        let features = features(&points)?;
        Ok((points, features, logits))
    }
    pub fn suggest(&self) -> Result<Point> {
        let (points, _, logits) = self.observation()?;
        Ok(points[argmax(&logits)].clone())
    }
    pub fn sample(&self, rng: &mut ChaCha8Rng) -> Result<Point> {
        let (points, _, logits) = self.observation()?;
        Ok(points[sample_action(&log_softmax(&logits)?, rng)].clone())
    }
    pub fn observe(&mut self, x: &Point, y: f64) -> Result<()> {
        ensure!(!self.done(), "NAF evaluation budget exhausted");
        inputs(&vec![x.clone()], self.config.dim, true)?;
        ensure!(
            x.iter().all(|v| (0.0..=1.0).contains(v)) && y.is_finite(),
            "observation must have finite target and unit-cube input"
        );
        let mut new_x = self.x_obs.clone();
        new_x.push(x.clone());
        let mut new_y = self.y_obs.clone();
        new_y.push(y);
        let mut new_gp = GaussianProcess::new(self.gp.config().clone())?;
        new_gp.fit(&new_x, &new_y)?;
        self.gp = new_gp;
        self.x_obs = new_x;
        self.y_obs = new_y;
        Ok(())
    }
    pub fn best(&self) -> Option<(&Point, f64)> {
        if self.y_obs.is_empty() {
            None
        } else {
            let i = argmax(&self.y_obs);
            Some((&self.x_obs[i], self.y_obs[i]))
        }
    }
}

/// Whole-episode GAE; terminal states bootstrap with zero, never across episodes.
pub fn gae_advantages(
    rewards: &[f64],
    values: &[f64],
    budget: usize,
    gamma: f64,
    lambda: f64,
) -> Result<Vec<f64>> {
    ensure!(
        budget > 0 && !rewards.is_empty() && rewards.len() % budget == 0,
        "GAE requires complete nonempty episodes"
    );
    targets(rewards, rewards.len())?;
    targets(values, rewards.len())?;
    ensure!(
        gamma.is_finite()
            && gamma > 0.0
            && gamma <= 1.0
            && lambda.is_finite()
            && (0.0..=1.0).contains(&lambda),
        "invalid GAE parameters"
    );
    let mut advantages = vec![0.0; rewards.len()];
    for start in (0..rewards.len()).step_by(budget) {
        let mut running = 0.0;
        for i in (start..start + budget).rev() {
            let next = if i + 1 < start + budget {
                values[i + 1]
            } else {
                0.0
            };
            running = rewards[i] + gamma * next - values[i] + gamma * lambda * running;
            advantages[i] = running;
        }
    }
    targets(&advantages, rewards.len())?;
    Ok(advantages)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpoStats {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
}

#[derive(Debug, Clone)]
pub struct PpoLoss {
    pub loss: f64,
    pub stats: PpoStats,
    pub logit_gradients: Matrix,
    pub value_gradients: Vec<f64>,
}

/// Categorical clipped PPO loss and analytic gradients, including entropy.
/// Advantages use the sample standard deviation, matching Python/PyTorch.
#[allow(clippy::too_many_arguments)]
pub fn ppo_loss(
    logits: &Matrix,
    actions: &[usize],
    old_log_probs: &[f64],
    advantages: &[f64],
    values: &[f64],
    returns: &[f64],
    clip: f64,
    value_coefficient: f64,
    entropy_coefficient: f64,
) -> Result<PpoLoss> {
    let n = logits.len();
    ensure!(
        n > 0 && actions.len() == n,
        "PPO batch is empty or action shape mismatched"
    );
    for row in logits {
        ensure!(
            !row.is_empty() && row.iter().all(|v| v.is_finite()),
            "PPO logits must be finite nonempty rows"
        );
    }
    targets(old_log_probs, n)?;
    targets(advantages, n)?;
    targets(values, n)?;
    targets(returns, n)?;
    ensure!(
        old_log_probs.iter().all(|v| *v <= 1e-9),
        "old categorical log probabilities must be nonpositive"
    );
    ensure!(
        clip.is_finite()
            && clip > 0.0
            && clip < 1.0
            && [value_coefficient, entropy_coefficient]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0),
        "invalid PPO coefficients"
    );
    let mut adv = advantages.to_vec();
    if n > 1 {
        let mean = adv.iter().sum::<f64>() / n as f64;
        let std = (adv.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
        adv.iter_mut().for_each(|v| *v = (*v - mean) / (std + 1e-8));
    }
    let mut result = PpoLoss {
        loss: 0.0,
        stats: PpoStats {
            policy_loss: 0.0,
            value_loss: 0.0,
            entropy: 0.0,
        },
        logit_gradients: Vec::with_capacity(n),
        value_gradients: Vec::with_capacity(n),
    };
    for i in 0..n {
        ensure!(
            actions[i] < logits[i].len(),
            "PPO action index out of bounds"
        );
        let lp = log_softmax(&logits[i])?;
        let ratio = (lp[actions[i]] - old_log_probs[i]).exp();
        let a = adv[i];
        let surrogate = if a == 0.0 {
            0.0
        } else {
            (ratio * a).min(ratio.clamp(1.0 - clip, 1.0 + clip) * a)
        };
        let active = if a >= 0.0 {
            ratio <= 1.0 + clip
        } else {
            ratio >= 1.0 - clip
        };
        let dlp = if active && a != 0.0 {
            -a * ratio / n as f64
        } else {
            0.0
        };
        let entropy = -lp.iter().map(|v| v.exp() * v).sum::<f64>();
        result.stats.policy_loss -= surrogate / n as f64;
        result.stats.value_loss += (values[i] - returns[i]).powi(2) / n as f64;
        result.stats.entropy += entropy / n as f64;
        result.logit_gradients.push(
            lp.iter()
                .enumerate()
                .map(|(j, logp)| {
                    dlp * (if j == actions[i] { 1.0 } else { 0.0 } - logp.exp())
                        + entropy_coefficient * logp.exp() * (logp + entropy) / n as f64
                })
                .collect(),
        );
        result
            .value_gradients
            .push(2.0 * value_coefficient * (values[i] - returns[i]) / n as f64);
    }
    result.loss = result.stats.policy_loss + value_coefficient * result.stats.value_loss
        - entropy_coefficient * result.stats.entropy;
    ensure!(
        result.loss.is_finite()
            && result
                .logit_gradients
                .iter()
                .flatten()
                .all(|v| v.is_finite())
            && result.value_gradients.iter().all(|v| v.is_finite()),
        "PPO objective overflow"
    );
    Ok(result)
}

pub fn log_regret_reward(regret: f64) -> Result<f64> {
    ensure!(regret.is_finite(), "regret must be finite");
    Ok(-regret.max(1e-20).log10())
}

/// A native maximized task with a known optimum. Closures can capture a family
/// member's translation/scaling; no Python interpreter or tensor backend is used.
#[derive(Clone)]
pub struct NafTask {
    pub objective: Arc<NafObjective>,
    pub optimum: f64,
    pub weights: Option<Vec<f64>>,
}

pub type NafObjective = dyn Fn(&[f64]) -> f64 + Send + Sync;

impl NafTask {
    pub fn new<F>(objective: F, optimum: f64) -> Result<Self>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        ensure!(optimum.is_finite(), "task optimum must be finite");
        Ok(Self {
            objective: Arc::new(objective),
            optimum,
            weights: None,
        })
    }
    pub fn with_weights(mut self, weights: Vec<f64>) -> Result<Self> {
        validate_weights(&weights, weights.len())?;
        self.weights = Some(weights);
        Ok(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NafTrainingResult {
    pub af: NeuralAf,
    pub best_af: NeuralAf,
    pub best_iteration: Option<usize>,
    pub mean_return: Vec<f64>,
    pub mean_final_log_regret: Vec<f64>,
    pub seconds: Vec<f64>,
    pub stats: Vec<PpoStats>,
}

pub type NAFTrainingResult = NafTrainingResult;

#[derive(Clone, Serialize, Deserialize)]
struct TrainingState {
    format: String,
    version: u32,
    config: NafConfig,
    gp_config: GpConfig,
    value_net: Network,
    policy_adam: Adam,
    value_adam: Adam,
    rng: ChaCha8Rng,
    history: NafTrainingResult,
}

#[derive(Default)]
struct Rollout {
    features: Vec<Matrix>,
    value_features: Matrix,
    actions: Vec<usize>,
    log_probs: Vec<f64>,
    rewards: Vec<f64>,
    final_regrets: Vec<f64>,
}

fn collect(
    config: &NafConfig,
    policy: &NeuralAf,
    gp_config: &GpConfig,
    tasks: &[NafTask],
    iteration: usize,
) -> Result<Rollout> {
    let seed = config
        .seed
        .wrapping_add(1009u64.wrapping_mul((iteration as u64).wrapping_add(1)));
    if config.n_workers == 1 {
        return collect_serial(config, policy, gp_config, tasks, seed);
    }
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..config.n_workers.min(tasks.len()) {
            let assigned: Vec<_> = tasks
                .iter()
                .skip(worker)
                .step_by(config.n_workers)
                .cloned()
                .collect();
            handles.push(scope.spawn(move || {
                collect_serial(
                    config,
                    policy,
                    gp_config,
                    &assigned,
                    seed.wrapping_add(worker as u64),
                )
            }));
        }
        let mut combined = Rollout::default();
        // Join in worker order, independently of operating-system scheduling.
        for handle in handles {
            let part = handle
                .join()
                .map_err(|_| anyhow::anyhow!("NAF rollout worker panicked"))??;
            combined.features.extend(part.features);
            combined.value_features.extend(part.value_features);
            combined.actions.extend(part.actions);
            combined.log_probs.extend(part.log_probs);
            combined.rewards.extend(part.rewards);
            combined.final_regrets.extend(part.final_regrets);
        }
        Ok(combined)
    })
}

fn collect_serial(
    config: &NafConfig,
    policy: &NeuralAf,
    gp_config: &GpConfig,
    tasks: &[NafTask],
    seed: u64,
) -> Result<Rollout> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut rollout = Rollout::default();
    for task in tasks {
        ensure!(task.optimum.is_finite(), "task optimum must be finite");
        let mut optimizer = NafOptimizer::new(
            config.clone(),
            policy.clone(),
            gp_config.clone(),
            task.weights.clone(),
        )?;
        for t in 0..config.budget {
            let (points, features, logits) = optimizer.observation()?;
            let lp = log_softmax(&logits)?;
            let action = sample_action(&lp, &mut rng);
            let value = (task.objective)(&points[action]);
            ensure!(value.is_finite(), "task returned a nonfinite objective");
            optimizer.observe(&points[action], value)?;
            let regret = task.optimum - optimizer.best().unwrap().1;
            let mut vf = vec![t as f64, config.budget as f64];
            if let Some(weights) = optimizer.weights.as_deref() {
                vf.extend(weights);
            }
            rollout.features.push(features);
            rollout.value_features.push(vf);
            rollout.actions.push(action);
            rollout.log_probs.push(lp[action]);
            rollout.rewards.push(log_regret_reward(regret)?);
        }
        rollout
            .final_regrets
            .push(task.optimum - optimizer.best().unwrap().1);
    }
    Ok(rollout)
}

fn ppo_update(state: &mut TrainingState, rollout: &Rollout) -> Result<PpoStats> {
    let config = &state.config;
    let mut dummy = ChaCha8Rng::seed_from_u64(0);
    let values: Vec<f64> = rollout
        .value_features
        .iter()
        .map(|row| state.value_net.forward(row, 0.0, &mut dummy).0[0])
        .collect();
    let advantages = gae_advantages(
        &rollout.rewards,
        &values,
        config.budget,
        config.gamma,
        config.gae_lambda,
    )?;
    let returns: Vec<f64> = advantages.iter().zip(&values).map(|(a, v)| a + v).collect();
    let mut order: Vec<usize> = (0..rollout.actions.len()).collect();
    let mut total = PpoStats {
        policy_loss: 0.0,
        value_loss: 0.0,
        entropy: 0.0,
    };
    let mut count = 0;
    for _ in 0..config.epochs {
        order.shuffle(&mut state.rng);
        for indices in order.chunks(config.minibatch_size) {
            let mut logits = Vec::with_capacity(indices.len());
            let mut policy_tapes = Vec::with_capacity(indices.len());
            let mut value_tapes = Vec::with_capacity(indices.len());
            let mut new_values = Vec::with_capacity(indices.len());
            for &index in indices {
                let mut step_logits = Vec::with_capacity(rollout.features[index].len());
                let mut step_tapes = Vec::with_capacity(rollout.features[index].len());
                for row in &rollout.features[index] {
                    let (output, tape) = state.history.af.net.forward(row, 0.0, &mut dummy);
                    step_logits.push(output[0]);
                    step_tapes.push(tape);
                }
                logits.push(step_logits);
                policy_tapes.push(step_tapes);
                let (output, tape) =
                    state
                        .value_net
                        .forward(&rollout.value_features[index], 0.0, &mut dummy);
                new_values.push(output[0]);
                value_tapes.push(tape);
            }
            let actions = indices
                .iter()
                .map(|i| rollout.actions[*i])
                .collect::<Vec<_>>();
            let old_probs = indices
                .iter()
                .map(|i| rollout.log_probs[*i])
                .collect::<Vec<_>>();
            let advantages = indices.iter().map(|i| advantages[*i]).collect::<Vec<_>>();
            let returns = indices.iter().map(|i| returns[*i]).collect::<Vec<_>>();
            let loss = ppo_loss(
                &logits,
                &actions,
                &old_probs,
                &advantages,
                &new_values,
                &returns,
                config.clip,
                config.value_coefficient,
                config.entropy_coefficient,
            )?;
            let mut policy_gradient = vec![0.0; state.history.af.net.parameter_count()];
            let mut value_gradient = vec![0.0; state.value_net.parameter_count()];
            for i in 0..indices.len() {
                for (tape, gradient) in policy_tapes[i].iter().zip(&loss.logit_gradients[i]) {
                    state
                        .history
                        .af
                        .net
                        .backward(tape, &[*gradient], &mut policy_gradient);
                }
                state.value_net.backward(
                    &value_tapes[i],
                    &[loss.value_gradients[i]],
                    &mut value_gradient,
                );
            }
            state.history.af.net.apply(
                &mut state.policy_adam,
                &policy_gradient,
                config.learning_rate,
            )?;
            state
                .value_net
                .apply(&mut state.value_adam, &value_gradient, config.learning_rate)?;
            total.policy_loss += loss.stats.policy_loss;
            total.value_loss += loss.stats.value_loss;
            total.entropy += loss.stats.entropy;
            count += 1;
        }
    }
    total.policy_loss /= count as f64;
    total.value_loss /= count as f64;
    total.entropy /= count as f64;
    Ok(total)
}

/// Save policies and their shared training/GP configuration as native JSON.
pub fn save_naf(
    directory: impl AsRef<Path>,
    policy: &NeuralAf,
    config: &NafConfig,
    gp_config: &GpConfig,
    name: &str,
) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
        "policy name must be a plain filename stem"
    );
    config.validate()?;
    gp_config.validate()?;
    policy.validate()?;
    ensure!(
        policy.dim == config.dim
            && policy.use_x == config.use_x
            && policy.hidden == config.hidden
            && policy.n_weights == config.n_weights,
        "NAF policy/config mismatch"
    );
    policy.save(directory.as_ref().join(format!("{name}.json")))?;
    write_json(
        &directory.as_ref().join("naf_config.json"),
        &serde_json::json!({"format":"openbo-naf-config","version":1,"config":config,"gp_config":gp_config}),
    )
}

pub fn load_naf(
    directory: impl AsRef<Path>,
    name: &str,
) -> Result<(NeuralAf, NafConfig, GpConfig)> {
    ensure!(
        !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
        "policy name must be a plain filename stem"
    );
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.as_ref().join("naf_config.json"))?)?;
    ensure!(
        value["format"] == "openbo-naf-config" && value["version"] == 1,
        "unsupported NAF config format/version"
    );
    let config: NafConfig = serde_json::from_value(value["config"].clone())?;
    let gp_config: GpConfig = serde_json::from_value(value["gp_config"].clone())?;
    let policy = NeuralAf::load(directory.as_ref().join(format!("{name}.json")))?;
    config.validate()?;
    gp_config.validate()?;
    ensure!(
        policy.dim == config.dim
            && policy.use_x == config.use_x
            && policy.hidden == config.hidden
            && policy.n_weights == config.n_weights,
        "saved NAF policy/config mismatch"
    );
    Ok((policy, config, gp_config))
}

pub fn save_training(
    directory: impl AsRef<Path>,
    config: &NafConfig,
    gp_config: &GpConfig,
    history: &NafTrainingResult,
) -> Result<()> {
    save_naf(&directory, &history.af, config, gp_config, "af")?;
    save_naf(&directory, &history.best_af, config, gp_config, "best_af")?;
    let record = serde_json::json!({"best_iteration":history.best_iteration,"mean_return":history.mean_return,
        "mean_final_log_regret":history.mean_final_log_regret,"seconds":history.seconds,"stats":history.stats});
    write_json(&directory.as_ref().join("training_history.json"), &record)
}

/// PPO training on a fixed task pool. Resuming restores Adam, value network,
/// policy, best policy, RNG and history; only `n_iterations` may change.
pub fn train_naf(
    config: NafConfig,
    gp_config: GpConfig,
    task_pool: &[NafTask],
    checkpoint_dir: Option<&Path>,
    resume: bool,
) -> Result<NafTrainingResult> {
    ensure!(!task_pool.is_empty(), "training task pool must be nonempty");
    train_naf_inner(
        config,
        gp_config,
        |n, rng| {
            Ok((0..n)
                .map(|_| task_pool[rng.gen_range(0..task_pool.len())].clone())
                .collect())
        },
        checkpoint_dir,
        resume,
    )
}

/// Fresh-task variant. The sampler receives a deterministic seed each iteration;
/// identical samplers/inputs are required when continuing a saved run.
pub fn train_naf_with_sampler<F>(
    config: NafConfig,
    gp_config: GpConfig,
    mut sampler: F,
    checkpoint_dir: Option<&Path>,
    resume: bool,
) -> Result<NafTrainingResult>
where
    F: FnMut(usize, u64) -> Result<Vec<NafTask>>,
{
    train_naf_inner(
        config,
        gp_config,
        |n, rng| sampler(n, rng.r#gen()),
        checkpoint_dir,
        resume,
    )
}

fn train_naf_inner<F>(
    config: NafConfig,
    gp_config: GpConfig,
    mut tasks: F,
    checkpoint_dir: Option<&Path>,
    resume: bool,
) -> Result<NafTrainingResult>
where
    F: FnMut(usize, &mut ChaCha8Rng) -> Result<Vec<NafTask>>,
{
    config.validate()?;
    gp_config.validate()?;
    // Build once here to fail impossible grids before a long training run.
    config.adaptive_set()?;
    GaussianProcess::new(gp_config.clone())?;
    let policy = config.new_af()?;
    let mut value_rng = ChaCha8Rng::seed_from_u64(config.seed.wrapping_add(1));
    let value_net = Network::new(&config.value_widths(), true, &mut value_rng);
    let mut state = TrainingState {
        format: "openbo-naf-training".into(),
        version: 1,
        config: config.clone(),
        gp_config: gp_config.clone(),
        policy_adam: Adam::new(policy.net.parameter_count()),
        value_adam: Adam::new(value_net.parameter_count()),
        value_net,
        rng: ChaCha8Rng::seed_from_u64(config.seed),
        history: NafTrainingResult {
            af: policy.clone(),
            best_af: policy,
            best_iteration: None,
            mean_return: Vec::new(),
            mean_final_log_regret: Vec::new(),
            seconds: Vec::new(),
            stats: Vec::new(),
        },
    };
    if resume {
        let root = checkpoint_dir.context("resume requires a checkpoint directory")?;
        let mut saved: TrainingState =
            serde_json::from_slice(&std::fs::read(root.join("training_state.json"))?)?;
        ensure!(
            saved.format == state.format && saved.version == 1,
            "unsupported training checkpoint format/version"
        );
        saved.config.n_iterations = config.n_iterations;
        ensure!(
            saved.config == config && saved.gp_config == gp_config,
            "only n_iterations may change when resuming; GP configuration must match"
        );
        saved.history.af.validate()?;
        saved.history.best_af.validate()?;
        ensure!(
            [&saved.history.af, &saved.history.best_af]
                .iter()
                .all(|policy| policy.dim == config.dim
                    && policy.use_x == config.use_x
                    && policy.hidden == config.hidden
                    && policy.n_weights == config.n_weights),
            "training checkpoint policy/config mismatch"
        );
        ensure!(
            saved.history.af.widths() == config.policy_widths()
                && saved.history.best_af.widths() == config.policy_widths(),
            "training checkpoint policy architecture mismatch"
        );
        saved.value_net.validate(&config.value_widths())?;
        saved
            .policy_adam
            .validate(saved.history.af.net.parameter_count())?;
        saved
            .value_adam
            .validate(saved.value_net.parameter_count())?;
        let n = saved.history.mean_return.len();
        ensure!(
            n <= config.n_iterations
                && saved.history.mean_final_log_regret.len() == n
                && saved.history.seconds.len() == n
                && saved.history.stats.len() == n,
            "training history lengths are inconsistent"
        );
        ensure!(
            saved.history.best_iteration.is_none_or(|i| i < n)
                && (n == 0) == saved.history.best_iteration.is_none(),
            "invalid best checkpoint iteration"
        );
        targets(&saved.history.mean_return, n)?;
        targets(&saved.history.mean_final_log_regret, n)?;
        targets(&saved.history.seconds, n)?;
        ensure!(
            saved
                .history
                .stats
                .iter()
                .all(|s| [s.policy_loss, s.value_loss, s.entropy]
                    .iter()
                    .all(|v| v.is_finite())),
            "nonfinite training statistics"
        );
        state = saved;
    }
    for iteration in state.history.mean_return.len()..config.n_iterations {
        let start = std::time::Instant::now();
        let sampled = tasks(config.episodes_per_iteration(), &mut state.rng)?;
        ensure!(
            sampled.len() == config.episodes_per_iteration(),
            "task sampler returned an incorrect number of episodes"
        );
        let rollout = collect(&config, &state.history.af, &gp_config, &sampled, iteration)?;
        let mean_return = rollout.rewards.iter().sum::<f64>() / sampled.len() as f64;
        let mean_regret = rollout
            .final_regrets
            .iter()
            .map(|v| v.max(1e-20).log10())
            .sum::<f64>()
            / sampled.len() as f64;
        ensure!(
            mean_return.is_finite() && mean_regret.is_finite(),
            "rollout statistics overflow"
        );
        let best_return = state
            .history
            .best_iteration
            .map(|i| state.history.mean_return[i])
            .unwrap_or(f64::NEG_INFINITY);
        if mean_return > best_return {
            // Returns were collected with this policy, before the PPO update.
            state.history.best_af = state.history.af.clone();
            state.history.best_iteration = Some(iteration);
        }
        let stats = ppo_update(&mut state, &rollout)?;
        state.history.mean_return.push(mean_return);
        state.history.mean_final_log_regret.push(mean_regret);
        state.history.seconds.push(start.elapsed().as_secs_f64());
        state.history.stats.push(stats);
        if let Some(root) = checkpoint_dir {
            save_training(root, &config, &gp_config, &state.history)?;
            write_json(&root.join("training_state.json"), &state)?;
        }
    }
    Ok(state.history)
}
