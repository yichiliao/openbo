use super::*;
use crate::neural::{
    Bnn, BnnConfig, replay_inputs, replay_targets, standardized_variance_threshold,
};
use anyhow::Context;
use std::{fs, path::Path};

#[derive(Clone)]
pub struct Source {
    pub name: String,
    pub gps: Vec<GaussianProcess>,
    pub x: Matrix,
    pub y: Matrix,
    pub meta_features: Vec<f64>,
}

impl Source {
    pub fn save(&self, directory: &Path) -> Result<()> {
        safe_name(&self.name)?;
        ensure!(!self.gps.is_empty(), "source must contain at least one GP");
        core::validate_matrix(&self.x, None, false)?;
        core::validate_matrix(&self.y, Some(self.gps.len()), false)?;
        ensure!(
            self.x.len() == self.y.len(),
            "source observation counts must match"
        );
        for (j, gp) in self.gps.iter().enumerate() {
            let ys = self.y.iter().map(|r| r[j]).collect::<Vec<_>>();
            ensure!(
                gp.training_inputs() == Some(&self.x)
                    && gp.training_targets() == Some(ys.as_slice()),
                "source trajectory must match fitted GP training data"
            );
        }
        fs::create_dir_all(directory.join("gp_states"))?;
        fs::create_dir_all(directory.join("trajectories"))?;
        let states = self
            .gps
            .iter()
            .map(|gp| Ok(gp.source_state()?["gp_state"].clone()))
            .collect::<Result<Vec<Value>>>()?;
        let state = if states.len() == 1 {
            states[0].clone()
        } else {
            json!({"objectives":states})
        };
        write_json(
            &directory
                .join("gp_states")
                .join(format!("{}.json", self.name)),
            &json!({"gp_state":state}),
        )?;
        let y = if self.gps.len() == 1 {
            json!(self.y.iter().map(|r| r[0]).collect::<Vec<_>>())
        } else {
            json!(self.y)
        };
        write_json(
            &directory
                .join("trajectories")
                .join(format!("{}.json", self.name)),
            &json!({"x_values":self.x,"y_values":y}),
        )
    }
}

pub fn safe_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name != "."
            && name != ".."
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c)),
        "name must be a non-empty safe filename (letters, digits, '_', '-', '.')"
    );
    Ok(())
}

/// Atomic replacement prevents an interrupted write from exposing partial JSON.
pub fn write_json<T: Serialize>(path: &Path, payload: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(payload)?)?;
    // rename replaces an existing regular file on supported Rust platforms.
    fs::rename(&temporary, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

fn collect_json(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect_json(&entry.path(), files)?;
        } else if kind.is_file() && entry.path().extension().is_some_and(|e| e == "json") {
            files.push(entry.path());
        }
    }
    Ok(())
}

pub fn load_sources(
    directory: &Path,
    dim: usize,
    objectives: usize,
) -> Result<(Vec<Source>, Vec<String>)> {
    ensure!(
        dim > 0 && objectives > 0,
        "source input and objective dimensions must be positive"
    );
    let gp_dir = directory.join("gp_states");
    ensure!(
        gp_dir.is_dir(),
        "missing source directory {}",
        gp_dir.display()
    );
    let mut files = vec![];
    collect_json(&gp_dir, &mut files)?;
    files.sort();
    let mut sources = vec![];
    let mut warnings = vec![];
    for path in files {
        let relative = path.strip_prefix(&gp_dir)?;
        let loaded = (|| -> Result<Source> {
            let trajectory: Value =
                serde_json::from_slice(&fs::read(directory.join("trajectories").join(relative))?)?;
            let payload: Value = serde_json::from_slice(&fs::read(&path)?)?;
            let x: Matrix = serde_json::from_value(trajectory["x_values"].clone())?;
            ensure!(
                !x.is_empty()
                    && x.iter()
                        .all(|r| r.len() == dim && r.iter().all(|v| v.is_finite())),
                "source input dimension/values invalid"
            );
            let y: Matrix = if trajectory["y_values"][0].is_array() {
                serde_json::from_value(trajectory["y_values"].clone())?
            } else {
                serde_json::from_value::<Vec<f64>>(trajectory["y_values"].clone())?
                    .into_iter()
                    .map(|v| vec![v])
                    .collect()
            };
            ensure!(
                x.len() == y.len()
                    && y.iter()
                        .all(|r| r.len() == objectives && r.iter().all(|v| v.is_finite())),
                "source objectives/rows invalid"
            );
            let base = payload.get("gp_state").unwrap_or(&payload);
            let entries = base
                .get("objectives")
                .or_else(|| base.get("gp_states"))
                .or_else(|| payload.get("gp_states"));
            let mut gps = vec![];
            for j in 0..objectives {
                let state = if let Some(entries) = entries {
                    ensure!(
                        entries.as_array().is_some_and(|v| v.len() == objectives),
                        "source GP objective count mismatch"
                    );
                    &entries[j]
                } else {
                    base
                };
                let ys = y.iter().map(|r| r[j]).collect::<Vec<_>>();
                gps.push(GaussianProcess::from_source_state(state, &x, &ys)?);
            }
            Ok(Source {
                name: path.file_stem().unwrap().to_string_lossy().into_owned(),
                meta_features: meta_features(&x, &y),
                gps,
                x,
                y,
            })
        })();
        match loaded {
            Ok(source) => sources.push(source),
            Err(e) => warnings.push(format!("Skipping source {}: {e:#}", relative.display())),
        }
    }
    Ok((sources, warnings))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ConboConfig {
    pub input_dim: usize,
    pub n_random: usize,
    pub dr: usize,
    pub n_evaluations: usize,
    pub n_candidates: usize,
    pub alpha1: f64,
    pub alpha2: f64,
    pub weight_clock: String,
    pub use_user_gp: bool,
    pub online_update: bool,
    pub online_epochs: usize,
    pub retrain: String,
    pub seed: u64,
    pub population_epochs: usize,
    pub replay_grid_points: usize,
    pub replay_random_points: usize,
    pub replay_max_variance: Option<f64>,
    pub replay_variance_fraction: f64,
    pub replay_filter: bool,
    pub gp: GpConfig,
}
impl Default for ConboConfig {
    fn default() -> Self {
        Self {
            input_dim: 2,
            n_random: 5,
            dr: 1,
            n_evaluations: 20,
            n_candidates: 128,
            alpha1: 5.0,
            alpha2: 0.2,
            weight_clock: "evaluations".into(),
            use_user_gp: true,
            online_update: true,
            online_epochs: 15,
            retrain: "reset".into(),
            seed: 0,
            population_epochs: 100,
            replay_grid_points: 15,
            replay_random_points: 50,
            replay_max_variance: None,
            replay_variance_fraction: 0.05,
            replay_filter: true,
            gp: GpConfig {
                optimize_hyperparameters: true,
                ..Default::default()
            },
        }
    }
}
pub struct ConboPopulation {
    pub config: ConboConfig,
    pub users: Vec<Source>,
    pub bnn: Option<Bnn>,
}
impl ConboPopulation {
    pub fn new(config: ConboConfig) -> Result<Self> {
        config.gp.validate()?;
        ensure!(
            config.input_dim > 0 && config.n_candidates > 0 && config.n_evaluations > 0,
            "dimension, candidates and evaluations must be positive"
        );
        ensure!(
            config.alpha1.is_finite()
                && config.alpha1 >= 0.0
                && config.alpha2.is_finite()
                && config.alpha2 > 0.0,
            "invalid population weight schedule"
        );
        ensure!(
            config.replay_variance_fraction.is_finite() && config.replay_variance_fraction > 0.0,
            "replay variance fraction must be positive"
        );
        ensure!(
            ["evaluations", "bo_steps"].contains(&config.weight_clock.as_str())
                && ["reset", "continue"].contains(&config.retrain.as_str()),
            "invalid weight clock or retraining mode"
        );
        ensure!(
            config.replay_grid_points >= 2,
            "replay grid must have at least two points per dimension"
        );
        if let Some(v) = config.replay_max_variance {
            ensure!(
                v.is_finite() && v > 0.0,
                "replay variance threshold must be positive"
            );
        }
        Ok(Self {
            config,
            users: vec![],
            bnn: None,
        })
    }
    pub fn new_user(&self, name: &str) -> Result<ConboUser> {
        safe_name(name)?;
        ensure!(
            !self.users.iter().any(|s| s.name == name),
            "user already exists"
        );
        let mut config = self.config.clone();
        config.n_random = config
            .n_random
            .saturating_sub(self.users.len().saturating_mul(config.dr));
        if self.bnn.is_none() {
            config.n_random = config.n_random.max(1);
        }
        Ok(ConboUser {
            name: name.into(),
            config,
            user_index: self.users.len(),
            x: vec![],
            y: vec![],
            bnn: self.bnn.clone(),
            pending: None,
        })
    }
    pub fn finish_user(&mut self, user: &ConboUser) -> Result<Value> {
        ensure!(
            user.user_index == self.users.len(),
            "users must finish in the order they were created"
        );
        ensure!(!user.y.is_empty(), "cannot store an unobserved user");
        ensure!(
            !self.users.iter().any(|s| s.name == user.name),
            "user already stored"
        );
        let y = user.y.iter().map(|v| vec![*v]).collect::<Matrix>();
        let gps = fit_gps(&self.config.gp, &user.x, &y)?;
        let source = Source {
            name: user.name.clone(),
            meta_features: meta_features(&user.x, &y),
            gps,
            x: user.x.clone(),
            y,
        };
        // Build and train everything before publishing the new population.
        let mut users = self.users.clone();
        users.push(source);
        let gps = users.iter().map(|s| s.gps[0].clone()).collect::<Vec<_>>();
        let inputs = replay_inputs(
            self.config.input_dim,
            self.config.replay_grid_points,
            self.config.replay_random_points,
            self.config.seed,
        )?;
        let threshold = if let Some(v) = self.config.replay_max_variance {
            v
        } else {
            standardized_variance_threshold(&gps, &inputs, self.config.replay_variance_fraction)?
        };
        let targets = replay_targets(
            &gps,
            &inputs,
            if self.config.replay_filter {
                Some(threshold)
            } else {
                None
            },
        )?;
        let mut bnn = if self.config.retrain == "continue" {
            self.bnn.clone()
        } else {
            None
        };
        if !targets.x.is_empty() {
            let model = bnn.get_or_insert(Bnn::new(BnnConfig {
                input_dim: self.config.input_dim,
                seed: self.config.seed,
                ..Default::default()
            })?);
            model.fit_replay(
                &targets.x,
                &targets.mean,
                &targets.var,
                self.config.population_epochs,
                true,
            )?;
        } else {
            bnn = self.bnn.clone();
        }
        let report = json!({"n_users":users.len(),"n_inputs_kept":targets.x.len(),"n_inputs_total":inputs.len(),"max_variance":threshold,"excluded_users":targets.excluded_users});
        self.users = users;
        self.bnn = bnn;
        Ok(report)
    }
    pub fn save(&self, directory: &Path) -> Result<()> {
        // Empty populations are valid checkpoints too. Materialize both source
        // directories before saving the manifest so loading a new population
        // follows the same path as loading a trained one.
        fs::create_dir_all(directory.join("gp_states"))?;
        fs::create_dir_all(directory.join("trajectories"))?;
        for user in &self.users {
            user.save(directory)?;
        }
        write_json(
            &directory.join("conbo_rust_state.json"),
            &json!({"format":"openbo-rust-conbo-v1","config":self.config,"users":self.users.iter().map(|s|s.name.clone()).collect::<Vec<_>>(),"bnn":self.bnn}),
        )
    }
    pub fn load(directory: &Path) -> Result<Self> {
        let value: Value =
            serde_json::from_slice(&fs::read(directory.join("conbo_rust_state.json"))?)?;
        ensure!(
            value["format"] == "openbo-rust-conbo-v1",
            "unsupported ConBO checkpoint"
        );
        let mut population = Self::new(serde_json::from_value(value["config"].clone())?)?;
        let (sources, warnings) = load_sources(directory, population.config.input_dim, 1)?;
        ensure!(
            warnings.is_empty(),
            "invalid checkpoint sources: {warnings:?}"
        );
        let names: Vec<String> = serde_json::from_value(value["users"].clone())?;
        let unique = names.iter().collect::<std::collections::HashSet<_>>();
        ensure!(
            unique.len() == names.len(),
            "duplicate users in ConBO checkpoint"
        );
        for name in names {
            safe_name(&name)?;
            let s = sources
                .iter()
                .find(|s| s.name == name)
                .context("checkpoint user missing")?;
            population.users.push(s.clone());
        }
        population.bnn = serde_json::from_value(value["bnn"].clone())?;
        if let Some(bnn) = &population.bnn {
            bnn.validate()?;
            ensure!(
                bnn.config().input_dim == population.config.input_dim,
                "population BNN input dimension mismatch"
            );
        }
        Ok(population)
    }
}

pub struct ConboUser {
    pub name: String,
    pub config: ConboConfig,
    pub user_index: usize,
    pub x: Matrix,
    pub y: Vec<f64>,
    bnn: Option<Bnn>,
    pending: Option<Point>,
}
impl ConboUser {
    pub fn suggest(&mut self) -> Result<Point> {
        ensure!(
            self.pending.is_none(),
            "observe the pending suggestion first"
        );
        ensure!(
            self.y.len() < self.config.n_evaluations,
            "no evaluations remaining"
        );
        let bounds = vec![(0.0, 1.0); self.config.input_dim];
        let seed = self
            .config
            .seed
            .wrapping_add(self.user_index as u64 * 100003)
            .wrapping_add(self.y.len() as u64 * 1009);
        let initial =
            self.y.len() < self.config.n_random || self.y.is_empty() && self.bnn.is_none();
        let point = if initial {
            core::sample(&bounds, 1, seed)?.remove(0)
        } else {
            let pool = core::sample(&bounds, self.config.n_candidates, seed)?;
            let step = if self.config.weight_clock == "bo_steps" {
                self.y.len().saturating_sub(self.config.n_random) + 1
            } else {
                self.y.len() + 1
            };
            let weight = if self.bnn.is_none() {
                0.0
            } else if !self.config.use_user_gp || self.y.is_empty() {
                1.0
            } else {
                acq::population_weight(step, self.config.alpha1, self.config.alpha2)?
            };
            let pop = if let Some(bnn) = &self.bnn {
                Some(bnn.predict(&pool)?)
            } else {
                None
            };
            let user = if weight < 1.0 {
                let mut gp = GaussianProcess::new(self.config.gp.clone())?;
                gp.fit(&self.x, &self.y)?;
                Some(gp.predict(&pool)?)
            } else {
                None
            };
            let best = if self.y.is_empty() {
                pop.as_ref()
                    .unwrap()
                    .0
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            } else {
                self.y.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            };
            let mut scores = vec![];
            for i in 0..pool.len() {
                let p = if let Some((mu, var)) = &pop {
                    weight * acq::expected_improvement(mu[i], var[i], best)?
                } else {
                    0.0
                };
                let u = if let Some((mu, var)) = &user {
                    (1.0 - weight) * acq::expected_improvement(mu[i], var[i], best)?
                } else {
                    0.0
                };
                scores.push(p + u);
            }
            pool[argmax(&scores)?].clone()
        };
        self.pending = Some(point.clone());
        Ok(point)
    }
    pub fn observe(&mut self, x: &Point, y: f64) -> Result<()> {
        validate_observation(x, &[y], &vec![(0.0, 1.0); self.config.input_dim], 1)?;
        ensure!(self.y.len() < self.config.n_evaluations, "budget exhausted");
        if let Some(p) = &self.pending {
            ensure!(p == x, "observed x does not match pending suggestion");
        }
        let mut xs = self.x.clone();
        xs.push(x.clone());
        let mut ys = self.y.clone();
        ys.push(y);
        let mut bnn = self.bnn.clone();
        if self.config.online_update && self.config.online_epochs > 0 {
            if let Some(model) = &mut bnn {
                model.fit_nll(&xs, &ys, self.config.online_epochs)?;
            }
        }
        self.x = xs;
        self.y = ys;
        self.bnn = bnn;
        self.pending = None;
        Ok(())
    }
    pub fn result(&self) -> Value {
        json!({"user_name":self.name,"user_index":self.user_index,"x_values":self.x,"y_values":self.y,"n_observations":self.y.len()})
    }
}
