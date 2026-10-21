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
            let cfg: Self = serde_json::from_value(raw)?;
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
        let engines = [
            raw.get("af_dir").is_some(),
            raw.get("conbo").is_some() || raw.get("memory_dir").is_some(),
            raw.get("population_dir").is_some(),
            raw.get("taf_run_dir").is_some(),
        ];
        ensure!(
            engines.iter().filter(|present| **present).count() <= 1,
            "ambiguous server config: choose exactly one of NAF, ConBO, TAF+, or TAF source settings"
        );
        let inferred = if engines[0] {
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
        cfg.optimizer.method = name.to_string();
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
        cfg.naf_dir = config_field(&raw, "af_dir")?;
        cfg.memory_dir = config_field(&raw, "memory_dir")?;
        cfg.optimizer.taf_run_dir = config_field(&raw, "taf_run_dir")?;
        if let Some(population) = config_field::<PathBuf>(&raw, "population_dir")? {
            cfg.optimizer.taf_run_dir = Some(population);
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
        let method = payload["optimizer"]
            .as_str()
            .unwrap_or(&cfg.optimizer.method)
            .to_string();
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
            if let Some(value) = payload.get(key) {
                let mut settings = serde_json::to_value(&cfg.optimizer)?;
                settings[key] = value.clone();
                cfg.optimizer = serde_json::from_value(settings)?;
            }
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
            _ => Engine::Standard(Box::new(SequentialOptimizer::new(cfg.optimizer.clone())?)),
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
        Ok(
            json!({"type":"suggest","optimizer":self.config.optimizer.method,"x":x,"iteration":self.count(),"n_observations":self.count(),"phase":if self.count()<self.config.optimizer.n_init{"init"}else{"bo"}}),
        )
    }
    fn finish(&mut self, kind: &str, store: bool) -> Result<Value> {
        let mut payload = match &mut self.engine {
            Engine::Standard(o) => {
                if let Some(dir) = &self.config.auto_save_dir {
                    if !o.y_obs.is_empty() {
                        o.save_sources(dir, &self.name)?;
                    }
                }
                serde_json::to_value(o.result()?)?
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
        payload["optimizer"] = json!(self.config.optimizer.method);
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
            "status" => Ok(
                json!({"type":"status","n_observations":self.count(),"n_evaluations":self.budget(),"has_pending":self.pending.is_some()}),
            ),
            "stop" => {
                let store = match payload.get("store") {
                    Some(value) => value.as_bool().context("store must be boolean")?,
                    None => false,
                };
                self.finish("stopped", store)
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
