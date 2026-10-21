//! Sequential ask/tell optimizers and reusable source populations.

mod population;
mod workflow;
pub use population::*;
pub use workflow::*;

use crate::{
    Matrix, Point, Result, acquisition as acq,
    core::{self, GaussianProcess, GpConfig},
};
use anyhow::ensure;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizerConfig {
    pub method: String,
    pub bounds: Vec<(f64, f64)>,
    pub n_init: usize,
    pub n_iter: usize,
    pub seed: u64,
    pub n_candidates: usize,
    pub n_starts: usize,
    pub mc_samples: usize,
    pub refine: bool,
    pub gp: GpConfig,
    #[serde(alias = "v_ref")]
    pub reference_point: Option<Vec<f64>>,
    #[serde(alias = "w_ref")]
    pub upper_reference: Option<Vec<f64>>,
    pub taf_run_dir: Option<PathBuf>,
    #[serde(alias = "weight_mode")]
    pub taf_weight_mode: String,
    pub rho: f64,
    pub target_weight: f64,
    pub source_only_warmup_iters: usize,
    pub source_reference_quantile: f64,
    pub source_improvement_temperature: f64,
    pub objective_weights: Option<Vec<f64>>,
    pub decay: bool,
    #[serde(alias = "d1")]
    pub decay_start_iter: usize,
    #[serde(alias = "d2")]
    pub decay_rate: f64,
    pub model: String,
    pub hyperparameters: String,
    /// Target acquisition for native TAF-EIPV and its `ieipv` baseline.
    pub acquisition: String,
    pub eipv_form: String,
    /// Screen a CEIPV pool with IEIPV, then rescore its best candidates.
    pub n_rescore: Option<usize>,
    pub n_hyper_samples: usize,
    pub slice_burn_in: usize,
    pub slice_thin: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            method: "bo_scratch".into(),
            bounds: vec![(0.0, 1.0); 2],
            n_init: 5,
            n_iter: 25,
            seed: 0,
            n_candidates: 128,
            n_starts: 8,
            mc_samples: 64,
            refine: true,
            gp: GpConfig {
                optimize_hyperparameters: true,
                ..Default::default()
            },
            reference_point: None,
            upper_reference: None,
            taf_run_dir: None,
            taf_weight_mode: "taf_r".into(),
            rho: 1.0,
            target_weight: 1.0,
            source_only_warmup_iters: 3,
            source_reference_quantile: 0.9,
            source_improvement_temperature: 0.05,
            objective_weights: None,
            decay: false,
            decay_start_iter: 2,
            decay_rate: 0.3,
            model: "independent".into(),
            hyperparameters: "map".into(),
            acquisition: "ieipv".into(),
            eipv_form: "exact".into(),
            n_rescore: None,
            n_hyper_samples: 10,
            slice_burn_in: 10,
            slice_thin: 2,
        }
    }
}

pub const METHODS: &[&str] = &[
    "random",
    "bo_scratch",
    "bo_scratch_grid",
    "bo_scratch_multistart",
    "bo_botorch",
    "bo_taf",
    "bo_taf_m",
    "bo_taf_r",
    "mobo_botorch",
    "mobo_ehvi",
    "mobo_ceipv",
    "mobo_ceipv_slf",
    "mobo_ceipv_slf_slice",
    "mobo_ceipv_mt",
    "mobo_ieipv",
    "mobo_taf",
    "taf_ehvi",
    "taf_plus",
    "taf",
    "taf_eipv",
    "mobo_taf_eipv",
    "ieipv",
];

impl OptimizerConfig {
    /// TAF-EIPV method defaults, preserving defaults of older methods.
    /// Set both objective-space references before constructing the optimizer.
    pub fn taf_eipv() -> Self {
        Self {
            method: "taf_eipv".into(),
            n_init: 0,
            n_iter: 10,
            target_weight: 0.75,
            ..Self::default()
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.gp.validate()?;
        ensure!(
            METHODS.contains(&self.method.as_str()),
            "unknown method {:?}; available: {METHODS:?}",
            self.method
        );
        ensure!(
            !self.bounds.is_empty()
                && self
                    .bounds
                    .iter()
                    .all(|(a, b)| a.is_finite() && b.is_finite() && b > a && (b - a).is_finite()),
            "bounds must be finite increasing intervals with a finite span"
        );
        ensure!(
            self.method == "random" || self.bounds.len() <= 32,
            "native acquisition search supports at most 32 input dimensions"
        );
        ensure!(
            ["independent", "mt", "slf"].contains(&self.model.as_str()),
            "unknown multi-output model"
        );
        ensure!(
            self.n_candidates > 0 && self.n_starts > 0 && self.mc_samples > 0,
            "candidate, start and Monte Carlo counts must be positive"
        );
        ensure!(
            self.n_init.checked_add(self.n_iter).is_some_and(|n| n > 0),
            "evaluation budget must be positive and not overflow"
        );
        ensure!(
            self.rho.is_finite()
                && self.rho > 0.0
                && self.target_weight.is_finite()
                && self.target_weight >= 0.0,
            "rho must be positive and target_weight non-negative"
        );
        ensure!(
            (0.0..=1.0).contains(&self.source_reference_quantile),
            "reference quantile must be in [0,1]"
        );
        ensure!(
            self.source_improvement_temperature.is_finite()
                && self.source_improvement_temperature > 0.0,
            "source temperature must be positive"
        );
        ensure!(
            self.decay_rate.is_finite() && (0.0..=1.0).contains(&self.decay_rate),
            "decay rate must be in [0,1]"
        );
        ensure!(
            ["taf_m", "taf_r", "taf_r_pareto", "poe", "uniform"]
                .contains(&self.taf_weight_mode.as_str()),
            "unknown transfer weight mode"
        );
        ensure!(
            ["map", "slice"].contains(&self.hyperparameters.as_str()),
            "hyperparameters must be map or slice"
        );
        if let Some(r) = &self.reference_point {
            ensure!(
                r.len() >= 2 && r.iter().all(|v| v.is_finite()),
                "reference point must contain at least two finite objectives"
            );
        }
        if let Some(upper) = &self.upper_reference {
            let lower = self
                .reference_point
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("upper_reference needs reference_point"))?;
            ensure!(
                upper.len() == lower.len()
                    && upper.iter().zip(lower).all(|(a, b)| a.is_finite() && a > b),
                "upper reference must be finite, match the lower reference, and exceed it"
            );
        }
        if let Some(w) = &self.objective_weights {
            validate_weights(w)?;
        }
        if let Some(w) = &self.objective_weights {
            ensure!(
                w.len() == self.n_objectives(),
                "objective weight count must match objectives"
            );
        }
        ensure!(
            self.taf_weight_mode != "poe" || self.is_plus(),
            "PoE weighting applies only to TAF+"
        );
        if self.is_multi() {
            ensure!(
                self.reference_point.is_some(),
                "multi-objective methods require reference_point"
            );
        }
        if self.is_plus() {
            ensure!(
                self.objective_weights.is_some(),
                "TAF+ requires objective_weights"
            );
            ensure!(
                ["poe", "taf_r"].contains(&self.taf_weight_mode.as_str()),
                "TAF+ supports only poe or taf_r weighting"
            );
        }
        ensure!(
            self.taf_weight_mode != "uniform" || self.is_taf_eipv(),
            "uniform transfer weights apply only to TAF-EIPV"
        );
        if self.is_native_eipv() {
            ensure!(
                self.upper_reference.is_some(),
                "native EIPV requires an upper_reference objective-space corner"
            );
            ensure!(
                ["ieipv", "ceipv"].contains(&self.acquisition.as_str()),
                "EIPV acquisition must be ieipv or ceipv"
            );
            ensure!(
                ["exact", "paper"].contains(&self.eipv_form.as_str()),
                "eipv_form must be exact or paper"
            );
            ensure!(
                self.n_rescore.is_none_or(|n| n > 0),
                "n_rescore must be positive when supplied"
            );
            ensure!(
                self.n_hyper_samples > 0 && self.slice_thin > 0,
                "hyperparameter sample count and thinning must be positive"
            );
            ensure!(
                self.gp.noise > 0.0,
                "multi-output GP noise must be positive"
            );
            if self.is_taf_eipv() {
                ensure!(
                    ["taf_r", "uniform"].contains(&self.taf_weight_mode.as_str()),
                    "TAF-EIPV supports taf_r or uniform weights"
                );
                ensure!(
                    self.target_weight > 0.0,
                    "TAF-EIPV target_weight must be positive"
                );
                ensure!(
                    !self.decay || self.decay_rate > 0.0,
                    "enabled TAF-EIPV decay must have a positive rate"
                );
            }
        }
        Ok(())
    }
    pub fn is_multi(&self) -> bool {
        self.method.starts_with("mobo_")
            || self.method == "taf_ehvi"
            || self.is_native_eipv()
            || (self.method == "random" && self.reference_point.is_some())
    }
    pub fn is_plus(&self) -> bool {
        ["taf_plus", "taf"].contains(&self.method.as_str())
    }
    pub fn n_objectives(&self) -> usize {
        if self.is_plus() {
            self.objective_weights.as_ref().map_or(1, Vec::len)
        } else if self.is_multi() {
            self.reference_point.as_ref().map_or(1, Vec::len)
        } else {
            1
        }
    }
    pub fn is_transfer(&self) -> bool {
        self.method.contains("taf") || self.is_plus()
    }
    pub fn is_taf_eipv(&self) -> bool {
        matches!(self.method.as_str(), "taf_eipv" | "mobo_taf_eipv")
    }
    pub fn is_native_eipv(&self) -> bool {
        self.is_taf_eipv() || self.method == "ieipv"
    }
}

pub fn validate_weights(weights: &[f64]) -> Result<()> {
    ensure!(
        !weights.is_empty()
            && weights.iter().all(|v| v.is_finite() && *v >= 0.0)
            && (weights.iter().sum::<f64>() - 1.0).abs() <= 1e-9,
        "objective weights must be finite, non-negative, and sum to one"
    );
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunResult {
    pub method: String,
    pub x_values: Matrix,
    pub y_values: Matrix,
    pub best_so_far: Vec<f64>,
    pub hypervolume_history: Vec<f64>,
    pub pareto_front: Matrix,
    pub metadata: Value,
}

pub struct SequentialOptimizer {
    pub config: OptimizerConfig,
    pub x_obs: Matrix,
    pub y_obs: Matrix,
    pub sources: Vec<Source>,
    pub warnings: Vec<String>,
    pub last_source_weights: Vec<f64>,
    pending: Option<Point>,
    bo_suggestions: usize,
    suggestion_seconds: Vec<f64>,
    eipv_gp: Option<acq::MultiOutputGP>,
    eipv_chain_state: Option<Vec<f64>>,
}

struct NativeEipvAcquisition {
    models: Vec<acq::MultiOutputGP>,
    target_cells: (Matrix, Matrix),
    target_terms: Option<acq::CeipvTerms>,
    source_cells: Vec<(Matrix, Matrix)>,
    source_weights: Vec<f64>,
}

impl SequentialOptimizer {
    pub fn new(mut config: OptimizerConfig) -> Result<Self> {
        if config.method == "bo_taf_m" {
            config.taf_weight_mode = "taf_m".into();
        }
        if config.method == "bo_taf_r" {
            config.taf_weight_mode = "taf_r".into();
        }
        config.validate()?;
        let (mut sources, mut warnings) = if let Some(dir) = &config.taf_run_dir {
            load_sources(dir, config.bounds.len(), config.n_objectives())?
        } else {
            (vec![], vec![])
        };
        if config.is_multi() && !config.is_native_eipv() {
            let reference = config.reference_point.as_ref().unwrap();
            sources.retain(|s| {
                let accepted =
                    s.y.iter()
                        .any(|r| r.iter().zip(reference).all(|(v, ref_v)| v > ref_v));
                if !accepted {
                    warnings.push(format!(
                        "Skipping source {}: front does not dominate reference point",
                        s.name
                    ));
                }
                accepted
            });
        }
        Ok(Self {
            config,
            x_obs: vec![],
            y_obs: vec![],
            sources,
            warnings,
            last_source_weights: vec![],
            pending: None,
            bo_suggestions: 0,
            suggestion_seconds: vec![],
            eipv_gp: None,
            eipv_chain_state: None,
        })
    }
    pub fn budget(&self) -> usize {
        self.config.n_init + self.config.n_iter
    }
    pub fn is_finished(&self) -> bool {
        self.y_obs.len() >= self.budget()
    }
    pub fn pending(&self) -> Option<&Point> {
        self.pending.as_ref()
    }
    pub fn phase(&self) -> &'static str {
        if self.y_obs.len() < self.config.n_init {
            "init"
        } else {
            "bo"
        }
    }
    /// Population influence for the next suggestion. TAF+ counts observations;
    /// TAF-EHVI counts BO suggestions so preloaded data do not consume warmup.
    pub fn population_decay(&self) -> f64 {
        if !self.config.decay {
            return 1.0;
        }
        let iteration = if self.config.is_plus() || self.config.is_taf_eipv() {
            self.y_obs.len().saturating_add(1)
        } else {
            self.bo_suggestions.saturating_add(1)
        };
        (1.0 - iteration.saturating_sub(self.config.decay_start_iter) as f64
            * self.config.decay_rate)
            .clamp(0.0, 1.0)
    }
    pub fn suggest(&mut self) -> Result<Point> {
        ensure!(
            self.pending.is_none(),
            "observe the pending suggestion before asking again"
        );
        ensure!(!self.is_finished(), "no evaluations remaining");
        let started = Instant::now();
        let seed = self
            .config
            .seed
            .wrapping_add((self.y_obs.len() as u64).wrapping_mul(1009));
        let point = if self.config.method == "random" || self.y_obs.len() < self.config.n_init {
            core::sample(&self.config.bounds, 1, seed)?.remove(0)
        } else {
            let pool: Matrix =
                core::low_discrepancy(self.config.bounds.len(), self.config.n_candidates, seed)?
                    .into_iter()
                    .map(|row| {
                        row.iter()
                            .zip(&self.config.bounds)
                            .map(|(v, (lo, hi))| lo + (hi - lo) * v)
                            .collect()
                    })
                    .collect();
            if self.config.is_native_eipv() {
                self.suggest_eipv(&pool, seed)?
            } else if self.y_obs.is_empty() {
                if self.sources.is_empty() {
                    if self.config.is_plus() {
                        self.config
                            .bounds
                            .iter()
                            .map(|(a, b)| a + (b - a) * 0.5)
                            .collect()
                    } else {
                        pool[0].clone()
                    }
                } else {
                    let values = self.population_means(&pool)?;
                    let best = pool[argmax(&values)?].clone();
                    if self.config.refine {
                        refine(&best, &self.config.bounds, |x| self.population_means(x))?
                    } else {
                        best
                    }
                }
            } else {
                let gps = fit_gps(&self.config.gp, &self.x_obs, &self.y_obs)?;
                let source_weights = if self.config.is_plus() {
                    vec![]
                } else {
                    self.compute_source_weights()?
                };
                let scores = self.score(&pool, &gps, &source_weights)?;
                let best_index = argmax(&scores)?;
                let mut best = pool[best_index].clone();
                if self.config.refine
                    && !self.config.is_multi()
                    && self.config.method != "bo_scratch_grid"
                {
                    let mut order = (0..pool.len()).collect::<Vec<_>>();
                    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
                    let mut best_value = scores[best_index];
                    for &i in order.iter().take(self.config.n_starts) {
                        let point = refine(&pool[i], &self.config.bounds, |points| {
                            self.score(points, &gps, &source_weights)
                        })?;
                        let value = self.score(&vec![point.clone()], &gps, &source_weights)?[0];
                        if value > best_value {
                            best = point;
                            best_value = value;
                        }
                    }
                }
                self.last_source_weights = source_weights;
                best
            }
        };
        if self.phase() == "bo" {
            self.bo_suggestions = self.bo_suggestions.saturating_add(1);
        }
        self.pending = Some(point.clone());
        self.suggestion_seconds
            .push(started.elapsed().as_secs_f64());
        Ok(point)
    }
    /// Record one experiment. Invalid data leave both observations and pending state unchanged.
    pub fn observe(&mut self, x: &[f64], y: &[f64]) -> Result<()> {
        validate_observation(x, y, &self.config.bounds, self.config.n_objectives())?;
        ensure!(!self.is_finished(), "evaluation budget exhausted");
        if let Some(p) = &self.pending {
            ensure!(
                p.iter().zip(x).all(|(a, b)| (a - b).abs() <= 1e-12),
                "observed x does not match pending suggestion"
            );
        }
        self.x_obs.push(x.to_vec());
        self.y_obs.push(y.to_vec());
        self.pending = None;
        Ok(())
    }
    pub fn bootstrap<F>(&mut self, objective: &F) -> Result<()>
    where
        F: Fn(&Point) -> Result<Vec<f64>>,
    {
        for _ in 0..self.config.n_init {
            let x = self.suggest()?;
            let y = objective(&x)?;
            self.observe(&x, &y)?;
        }
        Ok(())
    }
    pub fn result(&self) -> Result<RunResult> {
        let weights = self.config.objective_weights.clone().unwrap_or_else(|| {
            vec![1.0 / self.config.n_objectives() as f64; self.config.n_objectives()]
        });
        let values: Vec<f64> = self.y_obs.iter().map(|row| dot(row, &weights)).collect();
        let mut best = f64::NEG_INFINITY;
        let best_so_far = values
            .iter()
            .map(|&v| {
                best = best.max(v);
                best
            })
            .collect();
        let mut history = vec![];
        if self.config.is_multi() {
            for i in 1..=self.y_obs.len() {
                history.push(acq::hypervolume(
                    &self.y_obs[..i].to_vec(),
                    self.config.reference_point.as_ref().unwrap(),
                )?);
            }
        }
        let front = if self.config.is_multi() {
            acq::pareto_front(&self.y_obs)?
        } else {
            vec![]
        };
        Ok(RunResult {
            method: self.config.method.clone(),
            x_values: self.x_obs.clone(),
            y_values: self.y_obs.clone(),
            best_so_far,
            hypervolume_history: history,
            pareto_front: front,
            metadata: json!({"backend":"native_rust","upstream_revision":crate::UPSTREAM_REVISION,"config":self.config,"seed":self.config.seed,"total_evaluations":self.y_obs.len(),"n_init":self.config.n_init,"n_iter":self.config.n_iter,"history":"one entry per evaluation","sources":self.sources.iter().map(|s|&s.name).collect::<Vec<_>>(),"source_weights":self.last_source_weights,"last_source_weights":self.last_source_weights,"target_weight":self.config.target_weight,"weight_mode":self.config.taf_weight_mode,"warnings":self.warnings,"suggestion_seconds":self.suggestion_seconds,"total_suggestion_seconds":self.suggestion_seconds.iter().sum::<f64>()}),
        })
    }

    fn eipv_source_weights(&self) -> Result<Vec<f64>> {
        if !self.config.is_taf_eipv() || self.sources.is_empty() {
            return Ok(vec![]);
        }
        let mut weights = if self.config.taf_weight_mode == "uniform" || self.y_obs.len() < 2 {
            vec![0.75; self.sources.len()]
        } else {
            let predictions = self
                .sources
                .iter()
                .map(|s| predict_gps(&s.gps, &self.x_obs).map(|p| p.0))
                .collect::<Result<Vec<_>>>()?;
            acq::taf_r_source_weights(&self.y_obs, &predictions, self.config.rho)?
        };
        let decay = self.population_decay();
        for weight in &mut weights {
            *weight *= decay;
        }
        Ok(weights)
    }

    fn suggest_eipv(&mut self, pool: &Matrix, seed: u64) -> Result<Point> {
        ensure!(
            !self.y_obs.is_empty() || (self.config.is_taf_eipv() && !self.sources.is_empty()),
            "cannot suggest without observations or sources; bootstrap the target first"
        );
        let reference = self
            .config
            .reference_point
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("EIPV requires reference_point"))?;
        let upper = self
            .config
            .upper_reference
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("EIPV requires upper_reference"))?;
        let form = if self.config.eipv_form == "paper" {
            acq::EipvForm::Paper
        } else {
            acq::EipvForm::Exact
        };
        let target_cells = acq::reference_cells(&self.y_obs, reference, upper)?;
        let source_weights = self.eipv_source_weights()?;
        ensure!(
            !self.y_obs.is_empty() || source_weights.iter().any(|w| *w > 0.0),
            "source-only TAF-EIPV needs a source with positive weight after decay"
        );
        let source_cells = if self.config.is_taf_eipv() && source_weights.iter().any(|w| *w > 0.0) {
            self.sources
                .iter()
                .map(|s| {
                    let means = predict_gps(&s.gps, &self.x_obs)?.0;
                    acq::reference_cells(&means, reference, upper)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![]
        };
        let mut fitted = None;
        let mut chain_state = self.eipv_chain_state.clone();
        let models = if self.y_obs.is_empty() {
            vec![]
        } else {
            let warm = self
                .eipv_gp
                .as_ref()
                .filter(|gp| gp.mode == self.config.model);
            let mut gp = match warm {
                Some(model) => model.clone(),
                None => acq::MultiOutputGP::new(&self.config.model, self.config.gp.noise)?,
            };
            gp.noise = self.config.gp.noise;
            gp.optimize_hyperparameters = self.config.gp.optimize_hyperparameters;
            gp.optimize_noise = self.config.gp.optimize_noise;
            gp.max_iterations = self.config.gp.optimizer_maxiter;
            gp.optimizer_restarts = if warm.is_some() {
                1
            } else {
                self.config.gp.optimizer_restarts.max(1)
            };
            gp.seed = self.config.seed;
            gp.fit(&self.x_obs, &self.y_obs)?;
            let models = if self.config.hyperparameters == "slice" {
                let sampler = chain_state
                    .as_ref()
                    .and_then(|p| gp.with_hyperparameters(p).ok())
                    .unwrap_or_else(|| gp.clone());
                let samples = sampler.sample_hyperparameters(
                    self.config.n_hyper_samples,
                    self.config.slice_burn_in,
                    self.config.slice_thin,
                    seed ^ 0x454950565f534c49,
                )?;
                chain_state = samples.last().cloned();
                samples
                    .iter()
                    .map(|p| gp.with_hyperparameters(p))
                    .collect::<Result<Vec<_>>>()?
            } else {
                vec![gp.clone()]
            };
            fitted = Some(gp);
            models
        };
        let target_terms = if !models.is_empty() && self.config.acquisition == "ceipv" {
            Some(acq::ceipv_terms(
                &acq::pareto_front(&self.y_obs)?,
                reference,
                upper,
                form,
                &[0.1, 0.25, 0.5],
            )?)
        } else {
            None
        };
        let acquisition = NativeEipvAcquisition {
            models,
            target_cells,
            target_terms,
            source_cells,
            source_weights,
        };
        let cheap_screen = self.config.acquisition == "ceipv" && self.config.n_rescore.is_some();
        let screen = if cheap_screen {
            "ieipv"
        } else {
            self.config.acquisition.as_str()
        };
        let initial = self.eipv_scores(pool, &acquisition, screen)?;
        let mut order = (0..pool.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| initial[b].total_cmp(&initial[a]).then(a.cmp(&b)));
        if cheap_screen {
            order.truncate(self.config.n_rescore.unwrap().min(order.len()));
        }
        let points: Matrix = order.iter().map(|&i| pool[i].clone()).collect();
        let values = if cheap_screen {
            self.eipv_scores(&points, &acquisition, &self.config.acquisition)?
        } else {
            order.iter().map(|&i| initial[i]).collect()
        };
        let index = argmax(&values)?;
        let mut best = points[index].clone();
        let mut best_value = values[index];
        if self.config.refine {
            let mut starts = (0..points.len()).collect::<Vec<_>>();
            starts.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
            for &i in starts.iter().take(self.config.n_starts) {
                let point = refine(&points[i], &self.config.bounds, |x| {
                    self.eipv_scores(x, &acquisition, &self.config.acquisition)
                })?;
                let value =
                    self.eipv_scores(&vec![point.clone()], &acquisition, &self.config.acquisition)?
                        [0];
                if value > best_value {
                    best = point;
                    best_value = value;
                }
            }
        }
        // Publish fitted state only after scoring/search succeeds.
        self.eipv_gp = fitted;
        self.eipv_chain_state = chain_state;
        self.last_source_weights = acquisition.source_weights;
        Ok(best)
    }

    fn eipv_scores(
        &self,
        x: &Matrix,
        acquisition: &NativeEipvAcquisition,
        kind: &str,
    ) -> Result<Vec<f64>> {
        let form = if self.config.eipv_form == "paper" {
            acq::EipvForm::Paper
        } else {
            acq::EipvForm::Exact
        };
        let target = if acquisition.models.is_empty() {
            None
        } else {
            let predictions = acquisition
                .models
                .iter()
                .map(|gp| gp.predict(x))
                .collect::<Result<Vec<_>>>()?;
            let mut values = Vec::with_capacity(x.len());
            for i in 0..x.len() {
                let logs = predictions
                    .iter()
                    .map(|(mean, cov)| {
                        if kind == "ieipv" {
                            Ok(acq::ieipv(
                                &mean[i],
                                &cov[i],
                                &acquisition.target_cells.0,
                                &acquisition.target_cells.1,
                                form,
                            )?
                            .ln())
                        } else {
                            acq::log_ceipv(
                                &mean[i],
                                &cov[i],
                                acquisition
                                    .target_terms
                                    .as_ref()
                                    .ok_or_else(|| anyhow::anyhow!("CEIPV terms missing"))?,
                            )
                        }
                    })
                    .collect::<Result<Vec<f64>>>()?;
                let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                values.push(if maximum == f64::NEG_INFINITY {
                    acq::LOG_FLOOR
                } else {
                    (maximum + logs.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln()
                        - (logs.len() as f64).ln())
                    .max(acq::LOG_FLOOR)
                });
            }
            Some(values)
        };
        if acquisition.source_weights.iter().all(|w| *w == 0.0) {
            if let Some(values) = target {
                return Ok(values);
            }
        }
        let source_terms = self
            .sources
            .iter()
            .zip(&acquisition.source_cells)
            .map(|(s, cells)| {
                let means = predict_gps(&s.gps, x)?.0;
                acq::source_improvement(&means, cells)
            })
            .collect::<Result<Matrix>>()?;
        acq::log_taf_eipv(
            target.as_deref(),
            &source_terms,
            &acquisition.source_weights,
            self.config.target_weight,
        )
    }

    fn compute_source_weights(&self) -> Result<Vec<f64>> {
        if self.sources.is_empty() {
            return Ok(vec![]);
        }
        if self.config.taf_weight_mode == "taf_m" {
            let source_meta = self
                .sources
                .iter()
                .map(|s| s.meta_features.clone())
                .collect();
            return acq::meta_weights(
                &source_meta,
                &meta_features(&self.x_obs, &self.y_obs),
                self.config.rho,
            );
        }
        let predicted = self
            .sources
            .iter()
            .map(|s| predict_gps(&s.gps, &self.x_obs).map(|p| p.0))
            .collect::<Result<Vec<_>>>()?;
        let mode = if self.config.n_objectives() == 1 {
            "scalar"
        } else {
            self.config.taf_weight_mode.as_str()
        };
        acq::ranking_weights(&self.y_obs, &predicted, self.config.rho, mode)
    }
    fn population_means(&self, x: &Matrix) -> Result<Vec<f64>> {
        let weights = self.config.objective_weights.clone().unwrap_or_else(|| {
            vec![1.0 / self.config.n_objectives() as f64; self.config.n_objectives()]
        });
        let predictions = self
            .sources
            .iter()
            .map(|s| predict_gps(&s.gps, x))
            .collect::<Result<Vec<_>>>()?;
        let means = predictions.iter().map(|p| p.0.clone()).collect::<Vec<_>>();
        let variances = predictions.iter().map(|p| p.1.clone()).collect::<Vec<_>>();
        acq::population_mean(
            &means,
            &variances,
            &weights,
            if self.config.is_plus() && self.config.taf_weight_mode == "poe" {
                "poe"
            } else {
                "taf_r"
            },
        )
    }
    fn score(
        &self,
        x: &Matrix,
        gps: &[GaussianProcess],
        source_weights: &[f64],
    ) -> Result<Vec<f64>> {
        if self.config.is_plus() {
            return self.plus_scores(x, gps);
        }
        if self.config.is_multi() {
            return self.multi_scores(x, gps, source_weights);
        }
        let (mean, var) = gps[0].predict(x)?;
        let best = self
            .y_obs
            .iter()
            .map(|r| r[0])
            .fold(f64::NEG_INFINITY, f64::max);
        let target = mean
            .iter()
            .zip(&var)
            .map(|(&m, &v)| acq::expected_improvement(m, v, best))
            .collect::<Result<Vec<_>>>()?;
        if !self.config.is_transfer() || self.sources.is_empty() {
            return Ok(target);
        }
        let warmup = self.bo_suggestions < self.config.source_only_warmup_iters;
        let target_weight = if warmup {
            0.0
        } else {
            self.config.target_weight
        };
        let mut result = target.iter().map(|v| target_weight * v).collect::<Vec<_>>();
        let mut denominator = target_weight;
        for (s, &w) in self.sources.iter().zip(source_weights) {
            if w <= 0.0 {
                continue;
            }
            let (mu, _) = s.gps[0].predict(x)?;
            let mut ys = s.y.iter().map(|r| r[0]).collect::<Vec<_>>();
            ys.sort_by(f64::total_cmp);
            let q = self.config.source_reference_quantile * (ys.len() - 1) as f64;
            let reference =
                ys[q.floor() as usize] * (1.0 - q.fract()) + ys[q.ceil() as usize] * q.fract();
            let tau = self.config.source_improvement_temperature;
            for (r, m) in result.iter_mut().zip(mu) {
                let z = (m - reference) / tau;
                *r += w * tau * (z.max(0.0) + (-z.abs()).exp().ln_1p());
            }
            denominator += w;
        }
        if denominator <= 0.0 {
            return Ok(target);
        }
        Ok(result.iter().map(|v| v / denominator).collect())
    }
    fn plus_scores(&self, x: &Matrix, gps: &[GaussianProcess]) -> Result<Vec<f64>> {
        let (mean, var) = predict_gps(gps, x)?;
        let mut source_mean = vec![];
        let mut source_variance = vec![];
        let mut references = vec![];
        for s in &self.sources {
            let (m, v) = predict_gps(&s.gps, x)?;
            source_mean.push(m);
            source_variance.push(v);
            let (at_target, _) = predict_gps(&s.gps, &self.x_obs)?;
            references.push(column_max(&at_target));
        }
        let decay = self.population_decay();
        let mut ranking = vec![];
        for s in &self.sources {
            let (prediction, _) = predict_gps(&s.gps, &self.x_obs)?;
            let mut row = vec![];
            for j in 0..gps.len() {
                let y = self.y_obs.iter().map(|r| r[j]).collect::<Vec<_>>();
                let mu = prediction.iter().map(|r| r[j]).collect::<Vec<_>>();
                row.push(acq::epanechnikov_weight(
                    acq::ranking_distance(&y, &mu)?,
                    self.config.rho,
                )?);
            }
            ranking.push(row);
        }
        acq::taf_plus_acquisition(acq::TafPlusInput {
            target_mean: &mean,
            target_variance: &var,
            target_best: &column_max(&self.y_obs),
            source_mean: &source_mean,
            source_variance: &source_variance,
            source_reference: &references,
            objective_weights: self.config.objective_weights.as_ref().unwrap(),
            decay,
            weight_mode: if self.config.taf_weight_mode == "poe" {
                "poe"
            } else {
                "taf_r"
            },
            ranking: Some((&ranking, acq::epanechnikov_weight(0.0, self.config.rho)?)),
        })
    }
    fn multi_scores(
        &self,
        x: &Matrix,
        gps: &[GaussianProcess],
        source_weights: &[f64],
    ) -> Result<Vec<f64>> {
        let reference = self.config.reference_point.as_ref().unwrap();
        let front = acq::pareto_front(&self.y_obs)?;
        let is_correlated =
            self.config.method.contains("ceipv") || self.config.method == "mobo_ieipv";
        let correlated = if is_correlated {
            let mode = match self.config.method.as_str() {
                "mobo_ceipv_slf" | "mobo_ceipv_slf_slice" => "slf",
                "mobo_ceipv_mt" => "mt",
                _ => &self.config.model,
            };
            let mut model = acq::MultiOutputGP::new(mode, self.config.gp.noise)?;
            model.optimize_hyperparameters = self.config.gp.optimize_hyperparameters;
            model.optimize_noise = self.config.gp.optimize_noise;
            model.seed = self.config.seed;
            model.fit(&self.x_obs, &self.y_obs)?;
            let models = if self.config.hyperparameters == "slice"
                || self.config.method == "mobo_ceipv_slf_slice"
            {
                model
                    .sample_hyperparameters(4, 4, 1, self.config.seed)?
                    .iter()
                    .map(|p| model.with_hyperparameters(p))
                    .collect::<Result<Vec<_>>>()?
            } else {
                vec![model]
            };
            Some(
                models
                    .iter()
                    .map(|model| model.predict(x))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        let upper = self.config.upper_reference.clone().unwrap_or_else(|| {
            column_max(&self.y_obs)
                .iter()
                .zip(reference)
                .map(|(hi, lo)| hi + (hi - lo).abs().max(1.0))
                .collect()
        });
        let (lower_cells, upper_cells) = if is_correlated {
            acq::nondominated_cells(&front, reference, &upper)?
        } else {
            (vec![], vec![])
        };
        let terms = if is_correlated {
            Some(acq::ceipv_terms(
                &front,
                reference,
                &upper,
                acq::EipvForm::Exact,
                &[0.01, 0.1, 1.0],
            )?)
        } else {
            None
        };
        let (mean, var) = predict_gps(gps, x)?;
        let mut scores = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let target = if let Some(predictions) = &correlated {
                let mut total = 0.0;
                for (mu, cov) in predictions {
                    total += if self.config.method == "mobo_ieipv" {
                        acq::ieipv(
                            &mu[i],
                            &cov[i],
                            &lower_cells,
                            &upper_cells,
                            acq::EipvForm::Exact,
                        )?
                    } else {
                        acq::ceipv(&mu[i], &cov[i], terms.as_ref().unwrap())?
                    };
                }
                total / predictions.len() as f64
            } else if self.config.method == "mobo_ehvi" {
                let mut cov = vec![vec![0.0; gps.len()]; gps.len()];
                for (j, row) in cov.iter_mut().enumerate() {
                    row[j] = var[i][j];
                }
                acq::ehvi(
                    &mean[i],
                    &cov,
                    &front,
                    reference,
                    self.config.mc_samples,
                    self.config.seed,
                )?
            } else {
                let mut inputs = self.x_obs.clone();
                inputs.push(x[i].clone());
                let m = gps.len();
                let n = inputs.len();
                let mut joint_mean = vec![vec![0.0; m]; n];
                let mut joint_cov = vec![vec![0.0; m * n]; m * n];
                for (o, gp) in gps.iter().enumerate() {
                    let (mu, cov) = gp.predict_joint(&inputs)?;
                    for a in 0..n {
                        joint_mean[a][o] = mu[a];
                        for b in 0..n {
                            joint_cov[a * m + o][b * m + o] = cov[a][b];
                        }
                    }
                }
                acq::nehvi(
                    &joint_mean,
                    &joint_cov,
                    self.x_obs.len(),
                    reference,
                    self.config.mc_samples,
                    self.config.seed,
                )?
            };
            if ["mobo_taf", "taf_ehvi"].contains(&self.config.method.as_str())
                && !self.sources.is_empty()
            {
                let mut terms = vec![];
                let mut weights = vec![];
                let decay = self.population_decay();
                for (s, &w) in self.sources.iter().zip(source_weights) {
                    if w <= 0.0 || decay == 0.0 {
                        continue;
                    }
                    let (mu, _) = predict_gps(&s.gps, &vec![x[i].clone()])?;
                    let base = acq::hypervolume(&s.y, reference)?;
                    let mut augmented = s.y.clone();
                    augmented.push(mu[0].clone());
                    terms.push((acq::hypervolume(&augmented, reference)? - base).max(0.0));
                    weights.push(w * decay);
                }
                let warmup = self.bo_suggestions < self.config.source_only_warmup_iters;
                let tw = if warmup && !weights.is_empty() {
                    0.0
                } else {
                    self.config.target_weight
                };
                let denominator = weights.iter().sum::<f64>() + tw;
                scores.push(if denominator > 0.0 {
                    (tw * target + dot(&terms, &weights)) / denominator
                } else {
                    target
                });
            } else {
                scores.push(target);
            }
        }
        Ok(scores)
    }
    pub fn save_sources(&self, directory: &std::path::Path, name: &str) -> Result<()> {
        ensure!(!self.x_obs.is_empty(), "cannot save an empty run");
        let gps = fit_gps(&self.config.gp, &self.x_obs, &self.y_obs)?;
        Source {
            name: name.to_string(),
            gps,
            x: self.x_obs.clone(),
            y: self.y_obs.clone(),
            meta_features: meta_features(&self.x_obs, &self.y_obs),
        }
        .save(directory)
    }
}

pub fn run<F>(config: OptimizerConfig, mut objective: F) -> Result<RunResult>
where
    F: FnMut(&Point) -> Result<Vec<f64>>,
{
    let mut optimizer = SequentialOptimizer::new(config)?;
    while !optimizer.is_finished() {
        let x = optimizer.suggest()?;
        let y = objective(&x)?;
        optimizer.observe(&x, &y)?;
    }
    optimizer.result()
}

pub fn fit_gps(config: &GpConfig, x: &Matrix, y: &Matrix) -> Result<Vec<GaussianProcess>> {
    ensure!(!y.is_empty(), "cannot fit without observations");
    core::validate_matrix(x, None, false)?;
    core::validate_matrix(y, None, false)?;
    ensure!(
        x.len() == y.len(),
        "input and objective row counts must match"
    );
    let mut models = vec![];
    for col in 0..y[0].len() {
        let mut cfg = config.clone();
        cfg.optimize_hyperparameters &= x.len() >= 2;
        let mut gp = GaussianProcess::new(cfg)?;
        gp.fit(x, &y.iter().map(|r| r[col]).collect::<Vec<_>>())?;
        models.push(gp);
    }
    Ok(models)
}
pub fn predict_gps(gps: &[GaussianProcess], x: &Matrix) -> Result<(Matrix, Matrix)> {
    let mut mean = vec![vec![0.0; gps.len()]; x.len()];
    let mut var = mean.clone();
    for (o, gp) in gps.iter().enumerate() {
        let (mu, v) = gp.predict(x)?;
        for i in 0..x.len() {
            mean[i][o] = mu[i];
            var[i][o] = v[i];
        }
    }
    Ok((mean, var))
}
pub fn validate_observation(
    x: &[f64],
    y: &[f64],
    bounds: &[(f64, f64)],
    n_objectives: usize,
) -> Result<()> {
    ensure!(
        x.len() == bounds.len()
            && x.iter()
                .zip(bounds)
                .all(|(v, (a, b))| v.is_finite() && v >= a && v <= b),
        "x must be finite, match the input dimension and lie in bounds"
    );
    ensure!(
        y.len() == n_objectives && y.iter().all(|v| v.is_finite()),
        "y must contain {n_objectives} finite objective values"
    );
    Ok(())
}
pub fn column_max(y: &Matrix) -> Vec<f64> {
    if y.is_empty() {
        return vec![];
    }
    let mut best = vec![f64::NEG_INFINITY; y[0].len()];
    for row in y {
        for (b, &v) in best.iter_mut().zip(row) {
            *b = b.max(v);
        }
    }
    best
}
pub fn meta_features(x: &Matrix, y: &Matrix) -> Vec<f64> {
    let mut result = vec![x.first().map_or(0, Vec::len) as f64];
    let m = y.first().map_or(1, Vec::len);
    for j in 0..m {
        let mean = if y.is_empty() {
            0.0
        } else {
            y.iter().map(|r| r[j]).sum::<f64>() / y.len() as f64
        };
        let std = if y.is_empty() {
            0.0
        } else {
            (y.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / y.len() as f64).sqrt()
        };
        result.push(mean);
        result.push(std);
    }
    result
}
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(v, w)| v * w).sum()
}
pub fn argmax(values: &[f64]) -> Result<usize> {
    let mut best = None;
    for (i, &value) in values.iter().enumerate() {
        if value.is_finite() && best.is_none_or(|(_, old)| value > old) {
            best = Some((i, value));
        }
    }
    best.map(|(i, _)| i)
        .ok_or_else(|| anyhow::anyhow!("no finite acquisition values"))
}

pub fn refine<F>(initial: &Point, bounds: &[(f64, f64)], score: F) -> Result<Point>
where
    F: Fn(&Matrix) -> Result<Vec<f64>>,
{
    let mut best = initial.clone();
    let mut value = score(&vec![best.clone()])?[0];
    for level in 0..12 {
        let step = 0.2 * 0.5_f64.powi(level / 2);
        let mut pool = vec![];
        for d in 0..bounds.len() {
            for sign in [-1.0, 1.0] {
                let mut p = best.clone();
                p[d] = (p[d] + sign * step * (bounds[d].1 - bounds[d].0))
                    .clamp(bounds[d].0, bounds[d].1);
                pool.push(p);
            }
        }
        let scores = score(&pool)?;
        let i = argmax(&scores)?;
        if scores[i] > value {
            value = scores[i];
            best = pool[i].clone();
        }
    }
    Ok(best)
}
