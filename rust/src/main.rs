use anyhow::{Context, bail, ensure};
use clap::{Parser, Subcommand};
use openbo::{
    Matrix, Point, Result, acquisition as acq,
    core::{self, GaussianProcess, GpConfig},
    neural,
    optimizers::*,
    server,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};
use std::{
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(
    version,
    about = "Native Rust Bayesian optimization: all objectives are maximized"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run one scalar or multi-objective method; save a portable JSON trajectory.
    Benchmark {
        #[arg(long)]
        method: Option<String>,
        #[arg(long, default_value = "branin")]
        function: String,
        #[arg(long)]
        n_init: Option<usize>,
        #[arg(long)]
        n_iter: Option<usize>,
        #[arg(long, conflicts_with = "n_iter")]
        n_evals: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        n_candidates: Option<usize>,
        #[arg(long)]
        mc_samples: Option<usize>,
        #[arg(long)]
        sources: Option<PathBuf>,
        #[arg(long, value_delimiter = ',')]
        objective_weights: Option<Vec<f64>>,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        fixed_hyperparameters: bool,
        #[arg(long)]
        hyperparameters: Option<String>,
        #[arg(long, default_value = "results/trajectory.json")]
        output: PathBuf,
    },
    /// A small equal-budget comparison for students; writes SVG and JSON.
    Quickstart {
        #[arg(long, default_value_t = 12)]
        n_evals: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "results/quickstart")]
        output_dir: PathBuf,
    },
    /// Evaluate a normalized test function on JSON input rows.
    Evaluate {
        #[arg(long)]
        function: String,
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long)]
        multi: bool,
    },
    /// Fit/predict a GP from {config,x_train,y_train,x_test}; useful for parity checks.
    Gp {
        #[arg(long)]
        input: Option<PathBuf>,
    },
    /// Evaluate an acquisition/geometry operation from JSON on standard input.
    Acquisition {
        #[arg(long)]
        input: Option<PathBuf>,
    },
    /// Persist reproducible train/test task-family variants.
    CreateSplit {
        #[arg(long, default_value = "branin")]
        function: String,
        #[arg(long, default_value_t = 20)]
        n_tasks: usize,
        #[arg(long, default_value_t = 0.8)]
        train_ratio: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "results/split.json")]
        output: PathBuf,
    },
    /// Run family tasks and save source GP/trajectory pairs.
    TrainSources {
        #[arg(long)]
        split: PathBuf,
        #[arg(long, default_value = "train")]
        subset: String,
        #[arg(long, default_value_t = 5)]
        n_init: usize,
        #[arg(long, default_value_t = 10)]
        n_iter: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "results/sources")]
        output_dir: PathBuf,
    },
    /// Inspect and validate a saved Python or Rust source population.
    InspectSources {
        #[arg(long)]
        directory: PathBuf,
        #[arg(long)]
        dim: usize,
        #[arg(long, default_value_t = 1)]
        objectives: usize,
    },
    /// Optimize one method across a saved test/train family partition.
    FamilyBenchmark {
        #[arg(long)]
        split: PathBuf,
        #[arg(long, default_value = "test")]
        subset: String,
        #[arg(long, default_value = "bo_scratch")]
        method: String,
        #[arg(long)]
        sources: Option<PathBuf>,
        #[arg(long, default_value_t = 10)]
        n_evals: usize,
        #[arg(long, default_value = "results/family.json")]
        output: PathBuf,
    },
    /// Train a native MetaBO policy with PPO and resumable JSON checkpoints.
    TrainNaf {
        #[arg(long, default_value = "branin")]
        family: String,
        #[arg(long, default_value_t = 10)]
        n_iterations: usize,
        #[arg(long, default_value_t = 5)]
        budget: usize,
        #[arg(long, default_value_t = 4)]
        episodes: usize,
        #[arg(long, default_value_t = 32)]
        hidden: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value = "results/naf")]
        output_dir: PathBuf,
    },
    /// Run a trained neural acquisition function on a normalized objective.
    NafBenchmark {
        #[arg(long)]
        af_dir: PathBuf,
        #[arg(long, default_value = "branin")]
        function: String,
        #[arg(long, default_value = "results/naf_run.json")]
        output: PathBuf,
    },
    /// Import a Python NAF export and verify its recorded forward-pass parity cases.
    ImportNaf {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "results/imported_naf")]
        output_dir: PathBuf,
    },
    /// Run successive users with a shared, persisted ConBO population model.
    ConboBenchmark {
        #[arg(long, default_value_t = 3)]
        n_users: usize,
        #[arg(long, default_value_t = 10)]
        n_evals: usize,
        #[arg(long, default_value_t = 100)]
        epochs: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "results/conbo")]
        output_dir: PathBuf,
    },
    /// Fit objective weights from JSON {trajectories,ratings,levels}.
    LearnWeights {
        #[arg(long)]
        input: Option<PathBuf>,
    },
    /// Select TAF+ decay using leave-one-user-out simulation.
    SelectDecay {
        #[arg(long)]
        sources: PathBuf,
        #[arg(long)]
        dim: usize,
        #[arg(long, value_delimiter = ',')]
        weights: Vec<f64>,
        #[arg(long, default_value_t = 5)]
        n_evals: usize,
    },
    /// Start a JSON ask/tell WebSocket server (Python YAML configs supported).
    Serve {
        #[arg(long, default_value = "127.0.0.1:8765")]
        address: String,
        #[arg(long)]
        config: PathBuf,
    },
    /// A fake client that evaluates scalar/MO test objectives on server suggestions.
    Client {
        #[arg(long, default_value = "ws://127.0.0.1:8765")]
        url: String,
        #[arg(long, default_value = "branin")]
        function: String,
        #[arg(long)]
        start: Option<PathBuf>,
    },
    /// Export a trajectory's convergence plot without a plotting dependency.
    Plot {
        #[arg(required = true)]
        trajectories: Vec<PathBuf>,
        #[arg(long, default_value = "results/comparison.svg")]
        output: PathBuf,
    },
}

fn read_json(path: Option<&Path>) -> Result<Value> {
    let text = if let Some(path) = path {
        std::fs::read_to_string(path)?
    } else {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        s
    };
    Ok(serde_json::from_str(&text)?)
}
fn output(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}
fn function(function: &str, x: &Point, multi: bool) -> Result<Vec<f64>> {
    if multi {
        Ok(core::multi_objective(function, &vec![x.clone()])?.remove(0))
    } else {
        core::synthetic(function, &vec![x.clone()])
    }
}
fn naf_function(name: &str) -> &str {
    match name {
        "branin" => "normalized_branin",
        "goldstein_price" => "normalized_goldstein_price",
        "hartmann3" => "normalized_hartmann3",
        _ => name,
    }
}
fn scalar_config(
    function: &str,
    method: &str,
    n_init: usize,
    n_iter: usize,
    seed: u64,
) -> Result<OptimizerConfig> {
    let multi = method.starts_with("mobo_")
        || method == "taf_ehvi"
        || (method == "random" && core::mo_function_spec(function).is_ok());
    let mut config = OptimizerConfig {
        method: method.into(),
        n_init,
        n_iter,
        seed,
        ..Default::default()
    };
    if multi {
        let spec = core::mo_function_spec(function)?;
        config.bounds = spec.bounds;
        config.reference_point = Some(spec.v_ref);
        config.upper_reference = Some(spec.w_ref);
        config.source_only_warmup_iters = 0;
    } else if ["taf_plus", "taf"].contains(&method) {
        let spec = core::mo_function_spec(function)?;
        config.bounds = spec.bounds;
        config.objective_weights = Some(vec![1.0 / spec.n_objectives as f64; spec.n_objectives]);
        config.n_init = 0;
        config.n_iter = n_init
            .checked_add(n_iter)
            .context("evaluation budget overflow")?;
        config.taf_weight_mode = "poe".into();
        config.decay = method == "taf_plus";
    } else {
        config.bounds = core::function_spec(function)?.bounds;
    }
    Ok(config)
}
fn svg_plot(runs: &[(String, Vec<f64>)], path: &Path) -> Result<()> {
    ensure!(
        !runs.is_empty()
            && runs
                .iter()
                .all(|(_, ys)| !ys.is_empty() && ys.iter().all(|v| v.is_finite())),
        "plot needs finite non-empty trajectories"
    );
    let n = runs.iter().map(|(_, v)| v.len()).max().unwrap();
    let lo = runs
        .iter()
        .flat_map(|(_, v)| v)
        .copied()
        .fold(f64::INFINITY, f64::min);
    let hi = runs
        .iter()
        .flat_map(|(_, v)| v)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let scale = hi.abs().max(lo.abs()).max(1.0);
    let lower_scaled = lo / scale;
    let range = (hi / scale - lower_scaled).max(1e-12);
    let mut svg = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"500\" viewBox=\"0 0 900 500\"><rect width=\"900\" height=\"500\" fill=\"white\"/><g font-family=\"sans-serif\" font-size=\"15\" fill=\"#17212b\"><path d=\"M80 35V425H860\" fill=\"none\" stroke=\"#17212b\"/><text x=\"350\" y=\"480\">Objective evaluations (including initialization)</text><text transform=\"translate(22 330) rotate(-90)\">Best score / hypervolume</text>",
    );
    for tick in 0..=5 {
        let fraction = tick as f64 / 5.0;
        let v = lo * (1.0 - fraction) + hi * fraction;
        let y = 425.0 - 370.0 * tick as f64 / 5.0;
        svg.push_str(&format!("<text x=\"30\" y=\"{y:.2}\">{v:.3}</text><path d=\"M80 {y:.2}H860\" stroke=\"#e8ecf0\"/>"));
    }
    let colors = ["#1768ac", "#e77728", "#24996b", "#9a47bd", "#d44455"];
    for (j, (name, ys)) in runs.iter().enumerate() {
        let color = colors[j % colors.len()];
        let points = ys
            .iter()
            .enumerate()
            .map(|(i, v)| {
                format!(
                    "{:.2},{:.2}",
                    80.0 + 780.0 * i as f64 / (n - 1).max(1) as f64,
                    425.0 - 370.0 * (v / scale - lower_scaled) / range
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let safe = name
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");
        svg.push_str(&format!("<polyline points=\"{points}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"3\"/><text x=\"110\" y=\"{}\" fill=\"{color}\">{safe}</text>",58+24*j));
    }
    svg.push_str("</g></svg>");
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::write(path, svg)?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Benchmark {
            method,
            function: fun,
            n_init,
            n_iter,
            n_evals,
            seed,
            n_candidates,
            mc_samples,
            sources,
            objective_weights,
            config,
            fixed_hyperparameters,
            hyperparameters,
            output: destination,
        } => {
            let mut cfg = if let Some(path) = config {
                let text = std::fs::read_to_string(&path)?;
                let mut config: OptimizerConfig = if path.extension().is_some_and(|v| v == "json") {
                    serde_json::from_str(&text)?
                } else {
                    serde_yaml::from_str(&text)?
                };
                if let Some(method) = method {
                    config.method = method;
                }
                if let Some(n) = n_init {
                    config.n_init = n;
                }
                if let Some(n) = n_iter {
                    config.n_iter = n;
                }
                if let Some(n) = n_evals {
                    ensure!(n >= config.n_init, "n-evals cannot be below n-init");
                    config.n_iter = n - config.n_init;
                }
                if let Some(seed) = seed {
                    config.seed = seed;
                }
                config
            } else {
                let n_init = n_init.unwrap_or(5);
                let n_iter = if let Some(n) = n_evals {
                    ensure!(n >= n_init, "n-evals cannot be below n-init");
                    n - n_init
                } else {
                    n_iter.unwrap_or(25)
                };
                scalar_config(
                    &fun,
                    method.as_deref().unwrap_or("bo_scratch"),
                    n_init,
                    n_iter,
                    seed.unwrap_or(0),
                )?
            };
            if let Some(n) = n_candidates {
                cfg.n_candidates = n;
            }
            if let Some(n) = mc_samples {
                cfg.mc_samples = n;
            }
            if sources.is_some() {
                cfg.taf_run_dir = sources;
            }
            if objective_weights.is_some() {
                cfg.objective_weights = objective_weights;
            }
            if fixed_hyperparameters {
                cfg.gp.optimize_hyperparameters = false;
            }
            if let Some(h) = hyperparameters {
                cfg.hyperparameters = h;
            }
            let multi = cfg.n_objectives() > 1;
            let result = run(cfg, |x| function(&fun, x, multi))?;
            write_json(&destination, &result)?;
            output(
                &json!({"saved":destination,"evaluations":result.y_values.len(),"best":result.best_so_far.last(),"hypervolume":result.hypervolume_history.last()}),
            )?;
        }
        Command::Quickstart {
            n_evals,
            seed,
            output_dir,
        } => {
            ensure!(n_evals >= 3, "n-evals must be at least 3");
            let mut curves = vec![];
            for method in ["random", "bo_scratch"] {
                let cfg = scalar_config("branin", method, 3, n_evals - 3, seed)?;
                let result = run(cfg, |x| function("branin", x, false))?;
                curves.push((method.into(), result.best_so_far.clone()));
                write_json(&output_dir.join(format!("{method}.json")), &result)?;
                println!(
                    "{method}: {} evaluations; best={:.6}",
                    result.y_values.len(),
                    result.best_so_far.last().unwrap()
                );
            }
            svg_plot(&curves, &output_dir.join("comparison.svg"))?;
            println!("Results: {}", output_dir.display());
        }
        Command::Evaluate {
            function: fun,
            input,
            multi,
        } => {
            let data = read_json(input.as_deref())?;
            let x: Matrix = serde_json::from_value(data.get("x").unwrap_or(&data).clone())?;
            if multi {
                output(&core::multi_objective(&fun, &x)?)?;
            } else {
                output(&core::synthetic(&fun, &x)?)?;
            }
        }
        Command::Gp { input } => {
            let data = read_json(input.as_deref())?;
            let config = serde_json::from_value(data.get("config").cloned().unwrap_or(json!({})))?;
            let mut gp = GaussianProcess::new(config)?;
            let x = serde_json::from_value(data["x_train"].clone())?;
            let y = serde_json::from_value::<Vec<f64>>(data["y_train"].clone())?;
            gp.fit(&x, &y)?;
            let query = serde_json::from_value(data["x_test"].clone())?;
            let (mean, variance) = gp.predict(&query)?;
            let (_, covariance) = gp.predict_joint(&query)?;
            output(
                &json!({"mean":mean,"variance":variance,"covariance":covariance,"state":gp.source_state()?}),
            )?;
        }
        Command::Acquisition { input } => {
            let data = read_json(input.as_deref())?;
            let result = match data["operation"].as_str().context("need operation")? {
                "ei" => {
                    let mean: Vec<f64> = serde_json::from_value(data["mean"].clone())?;
                    let variance: Vec<f64> = serde_json::from_value(data["variance"].clone())?;
                    ensure!(mean.len() == variance.len(), "shape mismatch");
                    json!(
                        mean.iter()
                            .zip(variance)
                            .map(|(&m, v)| acq::expected_improvement(
                                m,
                                v,
                                data["best"].as_f64().unwrap_or(0.0)
                            ))
                            .collect::<Result<Vec<_>>>()?
                    )
                }
                "hypervolume" => json!(acq::hypervolume(
                    &serde_json::from_value(data["y"].clone())?,
                    &serde_json::from_value::<Vec<f64>>(data["reference"].clone())?
                )?),
                "pareto" => json!(acq::pareto_front(&serde_json::from_value(
                    data["y"].clone()
                )?)?),
                "ranking" => json!(acq::ranking_weights(
                    &serde_json::from_value(data["target"].clone())?,
                    &serde_json::from_value::<Vec<Matrix>>(data["sources"].clone())?,
                    data["rho"].as_f64().unwrap_or(1.0),
                    data["mode"].as_str().unwrap_or("taf_r")
                )?),
                "meta" => json!(acq::meta_weights(
                    &serde_json::from_value(data["sources"].clone())?,
                    &serde_json::from_value::<Vec<f64>>(data["target"].clone())?,
                    data["rho"].as_f64().unwrap_or(1.0)
                )?),
                op => bail!("unknown acquisition operation {op}"),
            };
            output(&result)?;
        }
        Command::CreateSplit {
            function: fun,
            n_tasks,
            train_ratio,
            seed,
            output: destination,
        } => {
            let variants =
                core::generate_variants(&fun, n_tasks, seed, &core::FamilyConfig::default())?;
            let split = core::split_variants(&fun, &variants, train_ratio, seed)?;
            core::save_family_split(&split, &destination)?;
            output(
                &json!({"saved":destination,"train":split.train_variants.len(),"test":split.test_variants.len()}),
            )?;
        }
        Command::TrainSources {
            split,
            subset,
            n_init,
            n_iter,
            seed,
            output_dir,
        } => {
            let split = core::load_family_split(split)?;
            let tasks = if subset == "train" {
                &split.train_variants
            } else if subset == "test" {
                &split.test_variants
            } else {
                bail!("subset must be train or test")
            };
            for (i, task) in tasks.iter().enumerate() {
                let mut cfg = scalar_config(
                    &split.base_name,
                    "bo_scratch",
                    n_init,
                    n_iter,
                    seed.wrapping_add(i as u64),
                )?;
                cfg.gp.optimizer_restarts = 2;
                let mut optimizer = SequentialOptimizer::new(cfg)?;
                let mut rng = ChaCha8Rng::seed_from_u64(task.seed.unwrap_or(seed));
                while !optimizer.is_finished() {
                    let x = optimizer.suggest()?;
                    let y = task.evaluate(&split.base_name, &vec![x.clone()], &mut rng, None)?;
                    optimizer.observe(&x, &y)?;
                }
                optimizer.save_sources(&output_dir, &format!("{subset}_task_{i:03}"))?;
            }
            output(&json!({"saved":output_dir,"tasks":tasks.len()}))?;
        }
        Command::InspectSources {
            directory,
            dim,
            objectives,
        } => {
            let (sources, warnings) = load_sources(&directory, dim, objectives)?;
            output(
                &json!({"sources":sources.iter().map(|s|json!({"name":s.name,"observations":s.x.len(),"objectives":s.gps.len()})).collect::<Vec<_>>(),"warnings":warnings}),
            )?;
        }
        Command::FamilyBenchmark {
            split,
            subset,
            method,
            sources,
            n_evals,
            output: destination,
        } => {
            let split = core::load_family_split(split)?;
            let tasks = match subset.as_str() {
                "train" => split.train_variants,
                "test" => split.test_variants,
                _ => bail!("subset must be train or test"),
            };
            ensure!(n_evals > 0, "n-evals must be positive");
            let mut runs = vec![];
            for (i, task) in tasks.iter().enumerate() {
                let init = 3.min(n_evals);
                let mut cfg =
                    scalar_config(&split.base_name, &method, init, n_evals - init, i as u64)?;
                cfg.taf_run_dir = sources.clone();
                let mut optimizer = SequentialOptimizer::new(cfg)?;
                let mut rng = ChaCha8Rng::seed_from_u64(task.seed.unwrap_or(i as u64));
                while !optimizer.is_finished() {
                    let x = optimizer.suggest()?;
                    let y = task.evaluate(&split.base_name, &vec![x.clone()], &mut rng, None)?;
                    optimizer.observe(&x, &y)?;
                }
                runs.push(optimizer.result()?);
            }
            write_json(&destination, &runs)?;
            output(&json!({"saved":destination,"tasks":runs.len()}))?;
        }
        Command::TrainNaf {
            family,
            n_iterations,
            budget,
            episodes,
            hidden,
            seed,
            resume,
            output_dir,
        } => {
            ensure!(
                episodes > 0 && hidden > 0,
                "episodes and hidden width must be positive"
            );
            let dim = core::function_spec(&family)?.dim;
            let config = neural::NafConfig {
                dim,
                budget,
                n_iterations,
                steps_per_iteration: budget
                    .checked_mul(episodes)
                    .context("episode count overflow")?,
                hidden: vec![hidden, hidden],
                seed,
                points_per_dim: if dim == 3 { 8 } else { 15 },
                n_local: 64,
                ..Default::default()
            };
            let tasks = core::sample_metabo_tasks(&family, 10, seed)?;
            let pool = tasks
                .into_iter()
                .map(|task| {
                    let optimum = task.optimum()?;
                    neural::NafTask::new(
                        move |x| task.evaluate(&vec![x.to_vec()]).expect("validated task")[0],
                        optimum,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let gp = GpConfig {
                kernel_type: "rbf".into(),
                lengthscale: vec![0.3],
                standardize_targets: false,
                ..Default::default()
            };
            let result = neural::train_naf(config, gp, &pool, Some(&output_dir), resume)?;
            output(
                &json!({"saved":output_dir,"iterations":result.mean_return.len(),"best_iteration":result.best_iteration,"mean_return":result.mean_return}),
            )?;
        }
        Command::NafBenchmark {
            af_dir,
            function: fun,
            output: destination,
        } => {
            let (policy, config, gp) = neural::load_naf(af_dir, "best_af")?;
            let mut optimizer = neural::NafOptimizer::new(config, policy, gp, None)?;
            while !optimizer.done() {
                let x = optimizer.suggest()?;
                let y = function(naf_function(&fun), &x, false)?[0];
                optimizer.observe(&x, y)?;
            }
            let mut best = f64::NEG_INFINITY;
            let history = optimizer
                .y_obs
                .iter()
                .map(|v| {
                    best = best.max(*v);
                    best
                })
                .collect();
            let result = RunResult {
                method: "naf".into(),
                x_values: optimizer.x_obs,
                y_values: optimizer.y_obs.iter().map(|y| vec![*y]).collect(),
                best_so_far: history,
                hypervolume_history: vec![],
                pareto_front: vec![],
                metadata: json!({"backend":"native_rust","function":naf_function(&fun)}),
            };
            write_json(&destination, &result)?;
            output(&json!({"saved":destination}))?;
        }
        Command::ImportNaf { input, output_dir } => {
            let payload = read_json(Some(&input))?;
            ensure!(
                payload["format"] == "openbo-python-naf-export" && payload["version"] == 1,
                "unsupported Python NAF export"
            );
            let config: neural::NafConfig = serde_json::from_value(payload["config"].clone())?;
            let gp: GpConfig = serde_json::from_value(payload["gp_config"].clone())?;
            let mut policy = neural::NeuralAf::new(&config)?;
            policy.import_state_dict(
                payload["state_dict"]
                    .as_object()
                    .context("state_dict must be an object")?,
            )?;
            let absolute = payload["parity_tolerance"]["absolute"]
                .as_f64()
                .unwrap_or(5e-4);
            let relative = payload["parity_tolerance"]["relative"]
                .as_f64()
                .unwrap_or(2e-5);
            ensure!(
                absolute.is_finite() && relative.is_finite() && absolute >= 0.0 && relative >= 0.0,
                "invalid parity tolerance"
            );
            let mut maximum_error = 0.0_f64;
            let cases = payload["parity_cases"]
                .as_array()
                .context("missing recorded parity cases")?;
            ensure!(!cases.is_empty(), "parity cases cannot be empty");
            for case in cases {
                let features: Matrix = serde_json::from_value(case["features"].clone())?;
                let expected: Vec<f64> = serde_json::from_value(case["logits"].clone())?;
                let actual = policy.logits(&features)?;
                ensure!(
                    actual.len() == expected.len(),
                    "parity case output shape mismatch"
                );
                for (a, b) in actual.iter().zip(expected) {
                    let error = (a - b).abs();
                    maximum_error = maximum_error.max(error);
                    ensure!(
                        b.is_finite() && error <= absolute + relative * b.abs(),
                        "imported NAF forward pass mismatch ({a} vs {b})"
                    );
                }
            }
            neural::save_naf(&output_dir, &policy, &config, &gp, "af")?;
            neural::save_naf(&output_dir, &policy, &config, &gp, "best_af")?;
            write_json(&output_dir.join("state_dict.json"), &payload["state_dict"])?;
            output(
                &json!({"saved":output_dir,"parity_cases":cases.len(),"maximum_error":maximum_error}),
            )?;
        }
        Command::ConboBenchmark {
            n_users,
            n_evals,
            epochs,
            seed,
            output_dir,
        } => {
            ensure!(n_users > 0, "n-users must be positive");
            let cfg = ConboConfig {
                n_evaluations: n_evals,
                n_random: 3.min(n_evals),
                population_epochs: epochs,
                seed,
                ..Default::default()
            };
            let mut population = ConboPopulation::new(cfg)?;
            let mut results = vec![];
            for i in 0..n_users {
                let mut user = population.new_user(&format!("user_{i:03}"))?;
                while user.y.len() < n_evals {
                    let x = user.suggest()?;
                    let y = core::synthetic("branin", &vec![x.clone()])?[0];
                    user.observe(&x, y)?;
                }
                let report = population.finish_user(&user)?;
                results.push(json!({"trajectory":user.result(),"replay":report}));
                population.save(&output_dir)?;
            }
            write_json(&output_dir.join("runs.json"), &results)?;
            output(&json!({"saved":output_dir,"users":results.len()}))?;
        }
        Command::LearnWeights { input } => {
            let data = read_json(input.as_deref())?;
            let ys = serde_json::from_value::<Vec<Matrix>>(data["trajectories"].clone())?;
            ensure!(!ys.is_empty() && !ys[0].is_empty(), "empty trajectories");
            let ratings = serde_json::from_value::<Vec<Vec<f64>>>(data["ratings"].clone())?;
            let grid = simplex_grid(
                ys[0][0].len(),
                data["levels"].as_u64().unwrap_or(10) as usize,
            )?;
            let (best, scores) = optimize_objective_weights(&ys, &ratings, &grid)?;
            output(&json!({"best":best,"candidates":grid,"scores":scores}))?;
        }
        Command::SelectDecay {
            sources,
            dim,
            weights,
            n_evals,
        } => {
            let (users, warnings) = load_sources(&sources, dim, weights.len())?;
            ensure!(warnings.is_empty(), "invalid sources: {warnings:?}");
            let cfg = OptimizerConfig {
                method: "taf_plus".into(),
                bounds: vec![(0.0, 1.0); dim],
                n_init: 0,
                n_iter: n_evals,
                objective_weights: Some(weights),
                taf_weight_mode: "poe".into(),
                ..Default::default()
            };
            let pairs = [(0, 0.1), (2, 0.1), (2, 0.3), (5, 0.2)];
            let (best, scores) = select_decay(&users, &cfg, &pairs, n_evals)?;
            output(&json!({"best":best,"pairs":pairs,"scores":scores}))?;
        }
        Command::Serve { address, config } => {
            let cfg = server::ServerConfig::from_file(&config)?;
            let listener = tokio::net::TcpListener::bind(&address).await?;
            println!("OpenBO listening on ws://{}", listener.local_addr()?);
            tokio::select! {result=server::serve(listener,cfg)=>result?,_ = tokio::signal::ctrl_c()=>{}}
        }
        Command::Client {
            url,
            function: fun,
            start,
        } => {
            use futures_util::{SinkExt, StreamExt};
            use tokio_tungstenite::{connect_async, tungstenite::Message};
            let (mut socket, _) = connect_async(&url).await?;
            let payload = if let Some(path) = start {
                read_json(Some(&path))?
            } else {
                json!({"type":"start","n_init":3,"n_iter":5,"seed":0})
            };
            socket
                .send(Message::Text(payload.to_string().into()))
                .await?;
            while let Some(message) = socket.next().await {
                let message = message?;
                if !message.is_text() {
                    continue;
                }
                let value: Value = serde_json::from_str(message.to_text()?)?;
                match value["type"].as_str() {
                    Some("suggest") => {
                        let x = serde_json::from_value::<Point>(value["x"].clone())?;
                        let objective = if value["optimizer"] == "naf" {
                            naf_function(&fun)
                        } else {
                            &fun
                        };
                        let multi = core::mo_function_spec(objective).is_ok();
                        let y = function(objective, &x, multi)?;
                        let observation = if multi { json!(y) } else { json!(y[0]) };
                        socket
                            .send(Message::Text(
                                json!({"type":"observe","x":x,"y":observation})
                                    .to_string()
                                    .into(),
                            ))
                            .await?;
                    }
                    Some("error") => bail!("server error: {}", value["message"]),
                    _ => {
                        output(&value)?;
                        break;
                    }
                }
            }
        }
        Command::Plot {
            trajectories,
            output: destination,
        } => {
            let mut curves = vec![];
            for path in trajectories {
                let value: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
                let runs: Vec<Value> = if let Value::Array(rows) = value {
                    rows
                } else {
                    vec![value]
                };
                let label = path.file_stem().and_then(|s| s.to_str()).unwrap_or("run");
                let multiple = runs.len() > 1;
                for (i, entry) in runs.into_iter().enumerate() {
                    let value = entry.get("trajectory").unwrap_or(&entry);
                    let hypervolume: Vec<f64> = serde_json::from_value(
                        value
                            .get("hypervolume_history")
                            .cloned()
                            .unwrap_or(json!([])),
                    )?;
                    let best: Vec<f64> = serde_json::from_value(
                        value.get("best_so_far").cloned().unwrap_or(json!([])),
                    )?;
                    let values = if !hypervolume.is_empty() {
                        hypervolume
                    } else if !best.is_empty() {
                        best
                    } else {
                        let y: Vec<f64> = serde_json::from_value(value["y_values"].clone())
                            .context("trajectory needs hypervolume_history, best_so_far or scalar y_values")?;
                        let mut best = f64::NEG_INFINITY;
                        y.into_iter()
                            .map(|v| {
                                best = best.max(v);
                                best
                            })
                            .collect()
                    };
                    let method = value["method"]
                        .as_str()
                        .or_else(|| value["user_name"].as_str())
                        .unwrap_or("conbo");
                    let name = if multiple {
                        format!("{label}: {method} {}", i + 1)
                    } else {
                        format!("{label}: {method}")
                    };
                    curves.push((name, values));
                }
            }
            svg_plot(&curves, &destination)?;
            output(&json!({"saved":destination}))?;
        }
    }
    Ok(())
}
