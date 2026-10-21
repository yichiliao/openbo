//! WebSocket JSON ask/tell sessions with explicit validation and isolated state.
use crate::{
    Point, Result,
    neural::{NafOptimizer, load_naf},
    optimizers::*,
};
use anyhow::{Context, bail, ensure};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub optimizer: OptimizerConfig,
    pub y_range: Vec<(f64, f64)>,
    pub memory_dir: Option<PathBuf>,
    pub naf_dir: Option<PathBuf>,
    pub naf_name: String,
    pub naf_n_iter: Option<usize>,
    /// Use the TAF-MOBO protocol, where n_iter is the total evaluation budget.
    pub taf_mobo: bool,
    pub taf_mobo_n_init: Option<usize>,
    pub taf_mobo_raw_samples: usize,
    pub conbo: ConboConfig,
    pub auto_save_dir: Option<PathBuf>,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            optimizer: OptimizerConfig::default(),
            y_range: vec![(-1e300, 1e300)],
            memory_dir: None,
            naf_dir: None,
            naf_name: "best_af".into(),
            naf_n_iter: None,
            taf_mobo: false,
            taf_mobo_n_init: None,
            taf_mobo_raw_samples: 256,
            conbo: ConboConfig::default(),
            auto_save_dir: None,
        }
    }
}
impl ServerConfig {
    /// Read native config or the Python project's YAML fields.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let raw: Value = if path.extension().is_some_and(|v| v == "json") {
            serde_json::from_str(&text)?
        } else {
            serde_yaml::from_str(&text)?
        };
        if raw["optimizer"].is_object() {
            let mut cfg: Self = serde_json::from_value(raw.clone())?;
            if cfg.optimizer.method == "mobo_taf_eipv" {
                cfg.optimizer.method = "taf_eipv".into();
            }
            if cfg.optimizer.method == "taf_eipv" && raw["optimizer"].get("target_weight").is_none()
            {
                cfg.optimizer.target_weight = 0.75;
            }
            if raw.get("taf_mobo").is_none()
                && ["taf_eipv", "mobo_taf_eipv", "taf_ehvi"]
                    .contains(&cfg.optimizer.method.as_str())
            {
                cfg.taf_mobo = true;
                cfg.taf_mobo_n_init = Some(cfg.optimizer.n_init);
                if raw.get("taf_mobo_raw_samples").is_none() {
                    cfg.taf_mobo_raw_samples = cfg.optimizer.n_candidates;
                }
            }
            cfg.validate()?;
            return Ok(cfg);
        }
        ensure!(raw.is_object(), "server config must be a JSON/YAML object");
        let mut cfg = Self::default();
        if let Some(range) = raw.get("y_range") {
            cfg.y_range = if range[0].is_array() {
                serde_json::from_value(range.clone()).context("invalid y_range")?
            } else {
                vec![serde_json::from_value(range.clone()).context("invalid y_range")?]
            };
        }
        let explicit: Option<String> = config_field(&raw, "optimizer")?;
        let method_default: Option<String> = config_field(&raw, "method_default")?;
        let has_path = |key: &str| raw.get(key).is_some_and(|value| !value.is_null());
        let transfer_mo = method_default.is_some()
            || explicit.as_deref().is_some_and(|name| {
                ["taf_mobo", "taf_eipv", "taf_ehvi", "mobo_taf_eipv"].contains(&name)
            })
            || ((has_path("population_dir") || has_path("taf_run_dir"))
                && cfg.y_range.len() > 1
                && raw.get("objective_weights_default").is_none()
                && raw.get("objective_weights").is_none()
                && (raw.get("v_ref").is_some()
                    || raw.get("w_ref").is_some()
                    || has_path("taf_run_dir")));
        let engines = [
            has_path("af_dir"),
            raw.get("conbo").is_some() || has_path("memory_dir"),
            has_path("population_dir"),
            has_path("taf_run_dir"),
        ];
        ensure!(
            engines.iter().filter(|present| **present).count() <= 1,
            "ambiguous server config: choose exactly one of NAF, ConBO, TAF+, or TAF source settings"
        );
        let inferred = if transfer_mo {
            "taf_mobo"
        } else if engines[0] {
            "naf"
        } else if engines[1] {
            "conbo"
        } else if engines[2] {
            "taf_plus"
        } else if engines[3] {
            "bo_taf"
        } else if cfg.y_range.len() > 1
            || raw.get("model").is_some()
            || raw.get("acquisition").is_some()
        {
            "mobo"
        } else {
            "bo_scratch"
        };
        let name = explicit.as_deref().unwrap_or(inferred);
        cfg.taf_mobo = transfer_mo;
        cfg.optimizer.method = if transfer_mo {
            match name {
                "taf_eipv" | "taf_ehvi" => name.to_string(),
                "mobo_taf_eipv" => "taf_eipv".to_string(),
                "taf_mobo" => method_default.unwrap_or_else(|| "taf_eipv".into()),
                _ => bail!("TAF-MOBO method_default conflicts with optimizer {name:?}"),
            }
        } else {
            name.to_string()
        };
        let dim: usize = config_field(&raw, "input_dim")?.unwrap_or(2);
        ensure!(dim > 0 && dim <= 10000, "input_dim must be in 1..=10000");
        cfg.optimizer.bounds = vec![(0.0, 1.0); dim];
        let mut settings = serde_json::to_value(&cfg.optimizer)?;
        for (target, sources) in [
            ("n_init", &["n_init_default"][..]),
            ("n_iter", &["n_iter_default"][..]),
            ("n_candidates", &["n_candidates_default"][..]),
            ("n_starts", &["n_starts_default"][..]),
            ("seed", &["seed_default", "seed"][..]),
            (
                "taf_weight_mode",
                &["taf_weight_mode_default", "weight_mode_default"][..],
            ),
            ("rho", &["taf_rho_default", "rho_default"][..]),
            (
                "source_only_warmup_iters",
                &["source_only_warmup_iters_default"][..],
            ),
            (
                "objective_weights",
                &["objective_weights", "objective_weights_default"][..],
            ),
            ("decay", &["decay_default"][..]),
            ("decay_start_iter", &["d1_default"][..]),
            ("decay_rate", &["d2_default"][..]),
            ("refine", &["refine_suggestion_default"][..]),
            ("model", &["model"][..]),
            ("hyperparameters", &["hyperparameters"][..]),
        ] {
            if let Some(value) = sources.iter().find_map(|key| raw.get(*key)) {
                // Python NAF uses null to mean its trained episode budget.
                if target == "n_iter" && value.is_null() && name == "naf" {
                    continue;
                }
                if target == "n_init" && value.is_null() && transfer_mo {
                    continue;
                }
                settings[target] = value.clone();
            }
        }
        for (target, source) in [
            ("kernel_type", "kernel_type_default"),
            (
                "optimize_hyperparameters",
                "optimize_hyperparameters_default",
            ),
        ] {
            if let Some(value) = raw.get(source) {
                settings["gp"][target] = value.clone();
            }
        }
        cfg.optimizer = serde_json::from_value(settings)
            .context("invalid optimizer defaults in server config")?;
        if let Some(strategy) = config_field::<String>(&raw, "search_strategy_default")? {
            ensure!(
                ["grid", "multistart"].contains(&strategy.as_str()),
                "search strategy must be grid or multistart"
            );
            cfg.optimizer.refine = strategy == "multistart";
        }
        cfg.naf_n_iter = match raw.get("n_iter_default") {
            None | Some(Value::Null) => None,
            Some(value) => {
                Some(serde_json::from_value(value.clone()).context("invalid n_iter_default")?)
            }
        };
        cfg.naf_name = config_field(&raw, "af_name")?.unwrap_or_else(|| "best_af".into());
        cfg.naf_dir = if has_path("af_dir") {
            config_field(&raw, "af_dir")?
        } else {
            None
        };
        cfg.memory_dir = if has_path("memory_dir") {
            config_field(&raw, "memory_dir")?
        } else {
            None
        };
        cfg.optimizer.taf_run_dir = if has_path("taf_run_dir") {
            config_field(&raw, "taf_run_dir")?
        } else {
            None
        };
        if has_path("population_dir") {
            cfg.optimizer.taf_run_dir = config_field(&raw, "population_dir")?;
        }

        if transfer_mo {
            ensure!(
                cfg.y_range.len() >= 2,
                "TAF-MOBO requires at least two objectives"
            );
            ensure!(
                (has_path("population_dir") as u8 + has_path("taf_run_dir") as u8) == 1,
                "TAF-MOBO requires exactly one of population_dir and taf_run_dir"
            );
            cfg.taf_mobo_n_init = match raw.get("n_init_default") {
                None | Some(Value::Null) => None,
                Some(value) => {
                    Some(serde_json::from_value(value.clone()).context("invalid n_init_default")?)
                }
            };
            cfg.taf_mobo_raw_samples = config_field(&raw, "raw_samples_default")?.unwrap_or(256);
            if let Some(restarts) = config_field::<usize>(&raw, "num_restarts_default")? {
                ensure!(restarts > 0, "num_restarts_default must be positive");
            }
            let budget: usize = config_field(&raw, "n_iter_default")?.unwrap_or(10);
            cfg.optimizer.n_init =
                cfg.taf_mobo_n_init
                    .unwrap_or(if cfg.optimizer.method == "taf_ehvi" {
                        1
                    } else {
                        0
                    });
            ensure!(
                budget > 0 && cfg.optimizer.n_init <= budget,
                "TAF-MOBO needs n_iter>=1 and 0<=n_init<=n_iter"
            );
            cfg.optimizer.n_iter = budget - cfg.optimizer.n_init;
            if cfg.optimizer.method == "taf_eipv" {
                cfg.optimizer.target_weight = 0.75;
                cfg.optimizer.source_only_warmup_iters = 0;
            }
            if cfg.optimizer.method == "taf_ehvi" {
                cfg.optimizer.n_candidates = cfg.taf_mobo_raw_samples;
            }
        }

        if name == "mobo" {
            let acquisition: String =
                config_field(&raw, "acquisition")?.unwrap_or_else(|| "ceipv".into());
            cfg.optimizer.model = config_field(&raw, "model")?.unwrap_or_else(|| "slf".into());
            cfg.optimizer.method = match acquisition.as_str() {
                "ceipv" => "mobo_ceipv",
                "ieipv" => "mobo_ieipv",
                other => bail!(
                    "unsupported multi-objective acquisition {other:?}; choose ceipv or ieipv"
                ),
            }
            .into();
        } else if let Some(acquisition) = config_field::<String>(&raw, "acquisition")? {
            let matches = match acquisition.as_str() {
                "ceipv" => name.starts_with("mobo_ceipv"),
                "ieipv" => name == "mobo_ieipv",
                _ => false,
            };
            ensure!(
                matches,
                "acquisition {acquisition:?} conflicts with optimizer {name:?}"
            );
        }
        if cfg.optimizer.is_multi() || (name == "random" && cfg.y_range.len() > 1) {
            cfg.optimizer.reference_point = Some(
                config_field(&raw, "v_ref")?
                    .unwrap_or_else(|| cfg.y_range.iter().map(|(lo, _)| *lo).collect()),
            );
            cfg.optimizer.upper_reference = Some(
                config_field(&raw, "w_ref")?
                    .unwrap_or_else(|| cfg.y_range.iter().map(|(_, hi)| *hi).collect()),
            );
            let reference = cfg.optimizer.reference_point.as_ref().unwrap();
            let upper = cfg.optimizer.upper_reference.as_ref().unwrap();
            ensure!(
                reference.len() == cfg.y_range.len()
                    && upper.len() == reference.len()
                    && reference
                        .iter()
                        .zip(upper)
                        .all(|(lo, hi)| lo.is_finite() && hi.is_finite() && hi > lo),
                "v_ref and w_ref must match objective ranges and be finite increasing intervals"
            );
        }
        if cfg.optimizer.is_plus() {
            cfg.optimizer.n_init = 0;
            if raw.get("objective_weights").is_none()
                && raw.get("objective_weights_default").is_none()
            {
                cfg.optimizer.objective_weights =
                    Some(vec![1.0 / cfg.y_range.len() as f64; cfg.y_range.len()]);
            }
            if raw.get("weight_mode_default").is_none()
                && raw.get("taf_weight_mode_default").is_none()
            {
                cfg.optimizer.taf_weight_mode = "poe".into();
            }
            cfg.optimizer.decay =
                config_field(&raw, "decay_default")?.unwrap_or(name == "taf_plus");
        }
        if config_field::<bool>(&raw, "auto_save_scratch_artifacts")?.unwrap_or(false) {
            cfg.auto_save_dir = Some(
                config_field::<PathBuf>(&raw, "scratch_artifacts_dir")?
                    .unwrap_or_else(|| PathBuf::from("meta-bo-training/server-scratch-artifacts")),
            );
        }

        cfg.conbo.input_dim = dim;
        if let Some(v) = raw.get("conbo") {
            let mut merged = serde_json::to_value(&cfg.conbo)?;
            for (k, v) in v.as_object().context("conbo must be an object")? {
                merged[k] = v.clone();
            }
            if let Some(value) = v.get("r0") {
                merged["n_random"] = value.clone();
            }
            if let Some(width) = config_field::<usize>(v, "candidate_grid_points")? {
                ensure!(width >= 2, "candidate_grid_points must be at least two");
                let count = width
                    .checked_pow(dim.try_into().context("input_dim too large")?)
                    .context("candidate grid size overflow")?;
                ensure!(count <= 2_000_000, "candidate grid exceeds resource limit");
                merged["n_candidates"] = json!(count);
            }
            cfg.conbo = serde_json::from_value(merged)?;
        }
        cfg.conbo.n_evaluations =
            config_field(&raw, "n_evaluations_default")?.unwrap_or(cfg.conbo.n_evaluations);
        ensure!(
            cfg.conbo.input_dim == dim,
            "conbo.input_dim conflicts with input_dim"
        );
        cfg.validate()?;
        Ok(cfg)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.y_range.is_empty()
                && self
                    .y_range
                    .iter()
                    .all(|(lo, hi)| lo.is_finite() && hi.is_finite() && hi > lo),
            "y_range must contain finite increasing intervals"
        );
        ensure!(
            !self.optimizer.bounds.is_empty()
                && self.optimizer.bounds.iter().all(|(lo, hi)| lo.is_finite()
                    && hi.is_finite()
                    && hi > lo
                    && (hi - lo).is_finite()),
            "server bounds must be finite increasing intervals"
        );
        match self.optimizer.method.as_str() {
            "conbo" => {
                ensure!(self.memory_dir.is_some(), "ConBO requires memory_dir");
                ensure!(
                    self.conbo.input_dim == self.optimizer.bounds.len(),
                    "ConBO input dimension does not match server bounds"
                );
                ConboPopulation::new(self.conbo.clone())?;
            }
            "naf" => {
                ensure!(self.naf_dir.is_some(), "NAF requires af_dir/naf_dir");
                ensure!(
                    !self.naf_name.is_empty()
                        && self
                            .naf_name
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                    "af_name must be a plain filename stem"
                );
                ensure!(
                    self.naf_n_iter.is_none_or(|n| n > 0),
                    "NAF iteration limit must be positive"
                );
            }
            _ => self.optimizer.validate()?,
        }
        let objectives = self.optimizer.n_objectives();
        ensure!(
            self.y_range.len() == 1 || self.y_range.len() == objectives,
            "y_range count must match objectives"
        );
        if self.taf_mobo {
            validate_taf_mobo(&self.optimizer)?;
            ensure!(
                self.y_range.len() >= 2 && self.y_range.len() == objectives,
                "TAF-MOBO requires one range per objective and at least two objectives"
            );
            ensure!(
                self.taf_mobo_raw_samples > 0,
                "raw_samples must be positive"
            );
            ensure!(
                self.auto_save_dir.is_none(),
                "TAF-MOBO source populations are read-only; disable auto_save_dir"
            );
        }
        Ok(())
    }
}

fn config_field<T: serde::de::DeserializeOwned>(raw: &Value, key: &str) -> Result<Option<T>> {
    raw.get(key)
        .map(|value| {
            serde_json::from_value(value.clone())
                .with_context(|| format!("invalid server config field {key}"))
        })
        .transpose()
}

fn validate_taf_mobo(config: &OptimizerConfig) -> Result<()> {
    ensure!(
        ["taf_eipv", "taf_ehvi"].contains(&config.method.as_str()),
        "TAF-MOBO method must be taf_eipv or taf_ehvi"
    );
    ensure!(
        config.taf_run_dir.is_some(),
        "TAF-MOBO requires source population_dir/taf_run_dir"
    );
    ensure!(
        config.method != "taf_ehvi" || config.n_init >= 1,
        "taf_ehvi needs n_init>=1"
    );
    let modes: &[&str] = if config.method == "taf_eipv" {
        &["taf_r", "uniform"]
    } else {
        &["taf_r", "taf_r_pareto", "taf_m"]
    };
    ensure!(
        modes.contains(&config.taf_weight_mode.as_str()),
        "weight_mode is unsupported for {}",
        config.method
    );
    ensure!(
        config.rho.is_finite() && config.rho > 0.0,
        "rho must be positive and finite"
    );
    Ok(())
}

enum Engine {
    Standard(Box<SequentialOptimizer>),
    Conbo {
        population: Box<ConboPopulation>,
        user: Box<ConboUser>,
    },
    Naf(Box<NafOptimizer>),
}
pub struct Session {
    engine: Engine,
    pending: Option<Point>,
    ranges: Vec<(f64, f64)>,
    closed: bool,
    config: ServerConfig,
    name: String,
    limit: usize,
}
impl Session {
    pub fn start(payload: &Value, runtime: &ServerConfig) -> Result<Self> {
        ensure!(
            payload.is_object() && payload["type"] == "start",
            "first message must have type='start'"
        );
        runtime.validate()?;
        let mut cfg = runtime.clone();
        let requested: Option<String> = config_field(payload, "optimizer")?;
        let taf_mobo = cfg.taf_mobo
            || requested.as_deref().is_some_and(|m| {
                ["taf_mobo", "taf_eipv", "taf_ehvi", "mobo_taf_eipv"].contains(&m)
            });
        let mut method = requested
            .as_deref()
            .filter(|m| *m != "taf_mobo")
            .unwrap_or(&cfg.optimizer.method)
            .to_string();
        if taf_mobo {
            method = config_field(payload, "method")?.unwrap_or(method);
            if method == "mobo_taf_eipv" {
                method = "taf_eipv".into();
            }
            let total = config_field(payload, "n_iter")?.unwrap_or(
                cfg.optimizer
                    .n_init
                    .checked_add(cfg.optimizer.n_iter)
                    .context("evaluation budget overflow")?,
            );
            let default_init = cfg
                .taf_mobo_n_init
                .or_else(|| {
                    if !cfg.taf_mobo && method == cfg.optimizer.method {
                        Some(cfg.optimizer.n_init)
                    } else {
                        None
                    }
                })
                .unwrap_or(if method == "taf_ehvi" { 1 } else { 0 });
            let n_init: usize = config_field(payload, "n_init")?.unwrap_or(default_init);
            ensure!(
                total > 0 && n_init <= total,
                "TAF-MOBO needs n_iter>=1 and 0<=n_init<=n_iter"
            );
            cfg.optimizer.n_init = n_init;
            cfg.optimizer.n_iter = total - n_init;
            if method == "taf_eipv" && cfg.optimizer.method != "taf_eipv" {
                cfg.optimizer.target_weight = 0.75;
            }
            if method == "taf_ehvi" {
                cfg.optimizer.n_candidates =
                    config_field(payload, "raw_samples")?.unwrap_or(cfg.taf_mobo_raw_samples);
            }
            if let Some(restarts) = config_field::<usize>(payload, "num_restarts")? {
                ensure!(restarts > 0, "num_restarts must be positive");
            }
            if let Some(mode) = config_field::<String>(payload, "weight_mode")? {
                cfg.optimizer.taf_weight_mode = mode;
            }
            cfg.taf_mobo = true;
        } else {
            ensure!(
                payload.get("method").is_none(),
                "method override is only supported by TAF-MOBO sessions"
            );
        }
        cfg.optimizer.method = method.clone();
        let name = payload["task_name"]
            .as_str()
            .or_else(|| payload["user_name"].as_str())
            .unwrap_or("task")
            .to_string();
        safe_name(&name)?;
        for key in [
            "n_init",
            "n_iter",
            "seed",
            "n_candidates",
            "rho",
            "target_weight",
            "taf_weight_mode",
            "objective_weights",
            "source_only_warmup_iters",
        ] {
            if taf_mobo && ["n_init", "n_iter"].contains(&key) {
                continue;
            }
            if let Some(value) = payload.get(key) {
                let mut settings = serde_json::to_value(&cfg.optimizer)?;
                settings[key] = value.clone();
                cfg.optimizer = serde_json::from_value(settings)?;
            }
        }
        if taf_mobo {
            validate_taf_mobo(&cfg.optimizer)?;
        }
        let engine = match method.as_str() {
            "conbo" => {
                let dir = cfg
                    .memory_dir
                    .as_ref()
                    .context("ConBO requires memory_dir")?;
                let population = if dir.join("conbo_rust_state.json").exists() {
                    ConboPopulation::load(dir)?
                } else {
                    ConboPopulation::new(cfg.conbo.clone())?
                };
                ensure!(
                    population.config.input_dim == cfg.optimizer.bounds.len(),
                    "ConBO saved population input dimension does not match server input_dim"
                );
                let mut user = population.new_user(&name)?;
                if let Some(value) = payload.get("n_evaluations") {
                    user.config.n_evaluations = usize::try_from(
                        value
                            .as_u64()
                            .context("n_evaluations must be a positive integer")?,
                    )?;
                    ensure!(
                        user.config.n_evaluations > 0,
                        "n_evaluations must be positive"
                    );
                }
                Engine::Conbo {
                    population: Box::new(population),
                    user: Box::new(user),
                }
            }
            "naf" => {
                let dir = cfg.naf_dir.as_ref().context("NAF requires naf_dir")?;
                let (policy, naf_config, gp) = load_naf(dir, &cfg.naf_name)?;
                ensure!(
                    naf_config.dim == cfg.optimizer.bounds.len(),
                    "NAF input dimension does not match server input_dim"
                );
                Engine::Naf(Box::new(NafOptimizer::new(
                    naf_config,
                    policy,
                    gp,
                    cfg.optimizer.objective_weights.clone(),
                )?))
            }
            _ => {
                let optimizer = SequentialOptimizer::new(cfg.optimizer.clone())?;
                if taf_mobo {
                    ensure!(
                        !optimizer.sources.is_empty(),
                        "No sources found for TAF-MOBO"
                    );
                }
                Engine::Standard(Box::new(optimizer))
            }
        };
        let n = cfg.optimizer.n_objectives();
        let limit = match &engine {
            Engine::Naf(o) => {
                let limit = payload["n_iter"]
                    .as_u64()
                    .map(|v| v as usize)
                    .or(cfg.naf_n_iter)
                    .unwrap_or(o.config.budget);
                ensure!(
                    limit > 0 && limit <= o.config.budget,
                    "NAF n_iter must lie within its training budget"
                );
                limit
            }
            Engine::Standard(o) => o.budget(),
            Engine::Conbo { user, .. } => user.config.n_evaluations,
        };
        ensure!(
            cfg.y_range.len() == 1 || cfg.y_range.len() == n,
            "y_range count must match objectives"
        );
        Ok(Self {
            engine,
            pending: None,
            ranges: cfg.y_range.clone(),
            closed: false,
            config: cfg,
            name,
            limit,
        })
    }
    fn count(&self) -> usize {
        match &self.engine {
            Engine::Standard(o) => o.y_obs.len(),
            Engine::Conbo { user, .. } => user.y.len(),
            Engine::Naf(o) => o.y_obs.len(),
        }
    }
    fn budget(&self) -> usize {
        self.limit
    }
    fn ask(&mut self) -> Result<Value> {
        ensure!(
            self.pending.is_none(),
            "observe the pending suggestion first"
        );
        if self.count() >= self.budget() {
            return self.finish("done", false);
        }
        let x = match &mut self.engine {
            Engine::Standard(o) => o.suggest()?,
            Engine::Conbo { user, .. } => user.suggest()?,
            Engine::Naf(o) => o.suggest()?,
        };
        self.pending = Some(x.clone());
        if self.config.taf_mobo {
            let weights = if self.count() < self.config.optimizer.n_init {
                Value::Null
            } else {
                match &self.engine {
                    Engine::Standard(o) => json!(o.last_source_weights),
                    _ => Value::Null,
                }
            };
            let phase = if self.count() < self.config.optimizer.n_init {
                "random"
            } else if self.count() == 0 {
                "sources"
            } else {
                "taf"
            };
            return Ok(
                json!({"type":"suggest","optimizer":"taf_mobo","method":self.config.optimizer.method,"x":x,"iteration":self.count(),"n_observations":self.count(),"n_iter":self.budget(),"n_objectives":self.config.optimizer.n_objectives(),"phase":phase,"source_weights":weights}),
            );
        }
        Ok(
            json!({"type":"suggest","optimizer":self.config.optimizer.method,"x":x,"iteration":self.count(),"n_observations":self.count(),"phase":if self.count()<self.config.optimizer.n_init{"init"}else{"bo"}}),
        )
    }
    fn finish(&mut self, kind: &str, store: bool) -> Result<Value> {
        let mut payload = match &mut self.engine {
            Engine::Standard(o) => {
                if let Some(dir) = &self.config.auto_save_dir {
                    if !self.config.taf_mobo && !o.y_obs.is_empty() {
                        o.save_sources(dir, &self.name)?;
                    }
                }
                let result = o.result()?;
                let mut payload = serde_json::to_value(&result)?;
                if self.config.taf_mobo {
                    let indices: Vec<_> = result
                        .pareto_front
                        .iter()
                        .filter_map(|row| o.y_obs.iter().position(|y| y == row))
                        .collect();
                    payload["pareto_x"] = json!(
                        indices
                            .iter()
                            .map(|index| o.x_obs[*index].clone())
                            .collect::<Vec<_>>()
                    );
                    payload["pareto_y"] = json!(result.pareto_front);
                    payload["hv_history"] = json!(result.hypervolume_history);
                    payload["v_ref"] = json!(o.config.reference_point);
                    payload["sources"] =
                        json!(o.sources.iter().map(|s| s.name.clone()).collect::<Vec<_>>());
                }
                payload
            }
            Engine::Conbo { population, user } => {
                let mut result = user.result();
                if kind == "done" || store {
                    // Persist a complete candidate before advancing the live
                    // population, so an IO failure can be retried in-session.
                    let mut candidate = ConboPopulation {
                        config: population.config.clone(),
                        users: population.users.clone(),
                        bnn: population.bnn.clone(),
                    };
                    let report = candidate.finish_user(user)?;
                    candidate.save(self.config.memory_dir.as_ref().unwrap())?;
                    **population = candidate;
                    result["replay"] = report;
                    result["stored"] = json!(true);
                } else {
                    result["stored"] = json!(false);
                }
                result
            }
            Engine::Naf(o) => {
                let mut best = f64::NEG_INFINITY;
                let history = o
                    .y_obs
                    .iter()
                    .map(|v| {
                        best = best.max(*v);
                        best
                    })
                    .collect::<Vec<_>>();
                json!({"x_values":o.x_obs,"y_values":o.y_obs,"best_so_far":history,"budget":o.config.budget,"af_dir":self.config.naf_dir})
            }
        };
        if let Some(ys) = payload.get("y_values") {
            if self.config.optimizer.n_objectives() == 1 && ys[0].is_array() {
                payload["y_values"] = json!(
                    ys.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v[0].clone())
                        .collect::<Vec<_>>()
                );
            }
        }
        payload["type"] = json!(kind);
        payload["optimizer"] = json!(if self.config.taf_mobo {
            "taf_mobo"
        } else {
            self.config.optimizer.method.as_str()
        });
        if self.config.taf_mobo {
            payload["method"] = json!(self.config.optimizer.method);
        }
        payload["total_observations"] = json!(self.count());
        self.closed = true;
        Ok(payload)
    }
    pub fn handle(&mut self, payload: &Value) -> Result<Value> {
        ensure!(!self.closed, "session is finished");
        ensure!(payload.is_object(), "message must be a JSON object");
        match payload["type"]
            .as_str()
            .context("message needs string field type")?
        {
            "suggest" => self.ask(),
            "observe" => {
                let pending = self
                    .pending
                    .as_ref()
                    .context("no pending suggestion")?
                    .clone();
                if let Some(value) = payload.get("x") {
                    let x: Point = serde_json::from_value(value.clone())?;
                    ensure!(
                        x.len() == pending.len()
                            && x.iter()
                                .zip(&pending)
                                .all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-12),
                        "observed x does not match pending suggestion"
                    );
                }
                let y = if payload["y"].is_array() {
                    serde_json::from_value::<Vec<f64>>(payload["y"].clone())?
                } else {
                    vec![
                        payload["y"]
                            .as_f64()
                            .context("observe requires numeric y")?,
                    ]
                };
                ensure!(
                    y.len() == self.config.optimizer.n_objectives(),
                    "observed objective count mismatch"
                );
                for (i, &v) in y.iter().enumerate() {
                    let (lo, hi) = self.ranges[if self.ranges.len() == 1 { 0 } else { i }];
                    ensure!(
                        v.is_finite() && v >= lo && v <= hi,
                        "observed y outside finite configured range"
                    );
                }
                match &mut self.engine {
                    Engine::Standard(o) => o.observe(&pending, &y)?,
                    Engine::Conbo { user, .. } => user.observe(&pending, y[0])?,
                    Engine::Naf(o) => o.observe(&pending, y[0])?,
                }
                self.pending = None;
                if self.count() >= self.budget() {
                    self.finish("done", false)
                } else {
                    self.ask()
                }
            }
            "status" => {
                if self.config.taf_mobo {
                    if let Engine::Standard(o) = &self.engine {
                        let hv = crate::acquisition::hypervolume(
                            &o.y_obs,
                            o.config.reference_point.as_ref().unwrap(),
                        )?;
                        return Ok(
                            json!({"type":"status","optimizer":"taf_mobo","method":self.config.optimizer.method,"n_observations":self.count(),"n_iter":self.budget(),"has_pending":self.pending.is_some(),"hypervolume":hv,"n_sources":o.sources.len()}),
                        );
                    }
                }
                Ok(
                    json!({"type":"status","n_observations":self.count(),"n_evaluations":self.budget(),"has_pending":self.pending.is_some()}),
                )
            }
            "stop" => {
                let store = match payload.get("store") {
                    Some(value) => value.as_bool().context("store must be boolean")?,
                    None => false,
                };
                let reason: Option<String> = config_field(payload, "reason")?;
                let mut stopped = self.finish("stopped", store)?;
                if self.config.taf_mobo {
                    stopped["reason"] = json!(reason.unwrap_or_else(|| "client_stop".into()));
                }
                Ok(stopped)
            }
            "start" => bail!("session already started"),
            other => bail!("unsupported message type {other:?}"),
        }
    }
}

struct ConboGuard(Option<Arc<AtomicBool>>);
impl Drop for ConboGuard {
    fn drop(&mut self) {
        if let Some(active) = &self.0 {
            active.store(false, Ordering::SeqCst);
        }
    }
}

pub async fn serve(listener: TcpListener, config: ServerConfig) -> Result<()> {
    config.validate()?;
    let active = Arc::new(AtomicBool::new(false));
    loop {
        let (stream, _) = listener.accept().await?;
        let cfg = config.clone();
        let active = active.clone();
        tokio::spawn(async move {
            let Ok(socket) = accept_async(stream).await else {
                return;
            };
            let (mut sink, mut incoming) = socket.split();
            let mut session: Option<Session> = None;
            let mut guard = ConboGuard(None);
            while let Some(Ok(message)) = incoming.next().await {
                if message.is_close() {
                    break;
                }
                if let Message::Ping(data) = message {
                    if sink.send(Message::Pong(data)).await.is_err() {
                        break;
                    }
                    continue;
                }
                let Message::Text(raw) = message else {
                    continue;
                };
                let raw = raw.to_string();
                let cfg = cfg.clone();
                let active_for_start = active.clone();
                let (returned, response, acquired) = match tokio::task::spawn_blocking(move || {
                    let mut acquired = false;
                    let result = (|| -> Result<Value> {
                        let payload: Value = serde_json::from_str(&raw)?;
                        if let Some(existing) = session.as_mut() {
                            existing.handle(&payload)
                        } else {
                            let method = payload["optimizer"]
                                .as_str()
                                .unwrap_or(&cfg.optimizer.method);
                            if method == "conbo" {
                                ensure!(
                                    active_for_start
                                        .compare_exchange(
                                            false,
                                            true,
                                            Ordering::SeqCst,
                                            Ordering::SeqCst
                                        )
                                        .is_ok(),
                                    "another ConBO user is active"
                                );
                                acquired = true;
                            }
                            match Session::start(&payload, &cfg) {
                                Ok(s) => session = Some(s),
                                Err(e) => {
                                    if acquired {
                                        active_for_start.store(false, Ordering::SeqCst);
                                        acquired = false;
                                    }
                                    return Err(e);
                                }
                            }
                            session.as_mut().unwrap().handle(&json!({"type":"suggest"}))
                        }
                    })();
                    let response = match result {
                        Ok(v) => v,
                        Err(e) => json!({"type":"error","message":format!("{e:#}")}),
                    };
                    (session, response, acquired)
                })
                .await
                {
                    Ok(v) => v,
                    Err(_) => break,
                };
                session = returned;
                if acquired {
                    guard.0 = Some(active.clone());
                }
                let finished = response["type"] == "done" || response["type"] == "stopped";
                if sink
                    .send(Message::Text(response.to_string().into()))
                    .await
                    .is_err()
                    || finished
                {
                    break;
                }
            }
        });
    }
}
