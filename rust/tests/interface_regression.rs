//! Regressions for boundary values and configuration crossing public interfaces.
use openbo::{Result, core, optimizers::*};
use serde_json::json;
use std::{fs, process::Command};

#[test]
fn random_baseline_handles_multi_objective_data_with_a_fixed_reference() -> Result<()> {
    let result = run(
        OptimizerConfig {
            method: "random".into(),
            bounds: vec![(0., 1.); 3],
            n_init: 2,
            n_iter: 2,
            reference_point: Some(vec![-5., -5.]),
            ..Default::default()
        },
        |x| Ok(core::multi_objective("oka2", &vec![x.clone()])?.remove(0)),
    )?;
    assert_eq!(result.hypervolume_history.len(), 4);
    assert!(result.hypervolume_history.windows(2).all(|p| p[1] >= p[0]));
    assert_eq!(
        result.y_values,
        core::multi_objective("oka2", &result.x_values)?
    );
    Ok(())
}

#[test]
fn cold_start_midpoint_remains_finite_for_large_valid_bounds() -> Result<()> {
    let mut optimizer = SequentialOptimizer::new(OptimizerConfig {
        method: "taf_plus".into(),
        bounds: vec![(1e308, 1.1e308)],
        n_init: 0,
        n_iter: 1,
        objective_weights: Some(vec![1.]),
        ..Default::default()
    })?;
    let x = optimizer.suggest()?;
    assert!(x[0].is_finite() && (1e308..=1.1e308).contains(&x[0]));
    optimizer.observe(&x, &[0.])?;
    Ok(())
}

#[test]
fn benchmark_keeps_config_values_until_an_explicit_cli_override() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("config.json");
    fs::write(
        &config,
        serde_json::to_vec(&json!({
            "method":"random", "bounds":[[0.,1.],[0.,1.]],
            "n_init":1,"n_iter":2,"n_candidates":7,"mc_samples":3,
            "gp":{"optimize_hyperparameters":false}
        }))?,
    )?;
    let output = directory.path().join("run.json");
    for explicit in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_openbo"));
        command
            .args(["benchmark", "--config"])
            .arg(&config)
            .arg("--output")
            .arg(&output);
        if explicit {
            command.args(["--n-candidates", "9"]);
        }
        let response = command.output()?;
        assert!(
            response.status.success(),
            "{}",
            String::from_utf8_lossy(&response.stderr)
        );
        let run: RunResult = serde_json::from_slice(&fs::read(&output)?)?;
        assert_eq!(run.y_values.len(), 3);
        let saved = &run.metadata["config"];
        assert_eq!(saved["n_candidates"], if explicit { 9 } else { 7 });
        assert_eq!(saved["mc_samples"], 3);
        assert_eq!(saved["gp"]["optimize_hyperparameters"], false);
    }
    let response = Command::new(env!("CARGO_BIN_EXE_openbo"))
        .args(["benchmark", "--config"])
        .arg(&config)
        .args([
            "--n-init",
            "2",
            "--n-evals",
            "4",
            "--seed",
            "81",
            "--method",
            "bo_scratch",
            "--n-candidates",
            "3",
        ])
        .arg("--output")
        .arg(&output)
        .output()?;
    assert!(
        response.status.success(),
        "{}",
        String::from_utf8_lossy(&response.stderr)
    );
    let run: RunResult = serde_json::from_slice(&fs::read(&output)?)?;
    assert_eq!(run.y_values.len(), 4);
    assert_eq!(run.method, "bo_scratch");
    assert_eq!(run.metadata["config"]["seed"], 81);
    assert_eq!(run.metadata["config"]["n_init"], 2);
    Ok(())
}

#[test]
fn plot_reads_family_arrays_and_conbo_sequence_wrappers() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let input = directory.path().join("runs.json");
    let output = directory.path().join("comparison.svg");
    let run = run(
        OptimizerConfig {
            method: "random".into(),
            n_init: 1,
            n_iter: 1,
            ..Default::default()
        },
        |x| core::synthetic("branin", &vec![x.clone()]),
    )?;
    let fixtures = [
        json!([run]),
        json!([
            {"trajectory":{"user_name":"first","x_values":[[0.,0.],[1.,1.]],"y_values":[-3.,-1.]},"replay":{}},
            {"trajectory":{"user_name":"second","x_values":[[0.,0.],[1.,1.]],"y_values":[-2.,-1.]},"replay":{}}
        ]),
        json!([{"trajectory":{"user_name":"extreme","y_values":[-1e308,1e308]}}]),
    ];
    for fixture in fixtures {
        fs::write(&input, serde_json::to_vec(&fixture)?)?;
        let response = Command::new(env!("CARGO_BIN_EXE_openbo"))
            .arg("plot")
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .output()?;
        assert!(
            response.status.success(),
            "{}",
            String::from_utf8_lossy(&response.stderr)
        );
        let svg = fs::read_to_string(&output)?;
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
        assert!(!svg.contains("NaN") && !svg.contains("inf"));
    }
    Ok(())
}

#[test]
fn run_accepts_a_stateful_objective() -> Result<()> {
    let mut observations = 0;
    let result = run(
        OptimizerConfig {
            method: "random".into(),
            n_init: 1,
            n_iter: 2,
            ..Default::default()
        },
        |x| {
            observations += 1;
            core::synthetic("branin", &vec![x.clone()])
        },
    )?;
    assert_eq!(observations, 3);
    assert_eq!(result.y_values.len(), observations);
    Ok(())
}

#[test]
fn transfer_decay_counts_bo_suggestions_and_preserves_preloaded_warmup() -> Result<()> {
    let mut optimizer = SequentialOptimizer::new(OptimizerConfig {
        method: "mobo_taf".into(),
        n_init: 2,
        n_iter: 4,
        reference_point: Some(vec![-5., -5.]),
        decay: true,
        decay_start_iter: 2,
        decay_rate: 0.3,
        n_candidates: 2,
        mc_samples: 2,
        gp: core::GpConfig {
            optimize_hyperparameters: false,
            ..Default::default()
        },
        ..Default::default()
    })?;
    optimizer.observe(&[0.1, 0.2], &[-1., -2.])?;
    optimizer.observe(&[0.7, 0.8], &[-2., -1.])?;
    assert_eq!(optimizer.population_decay(), 1.);
    for expected in [1., 0.7, 0.4] {
        let x = optimizer.suggest()?;
        assert!(optimizer.suggest().is_err());
        optimizer.observe(&x, &[-1., -1.])?;
        assert!((optimizer.population_decay() - expected).abs() < 1e-12);
    }
    optimizer.config.decay_start_iter = usize::MAX;
    assert_eq!(optimizer.population_decay(), 1.);
    Ok(())
}

#[test]
fn taf_plus_rejects_weight_modes_it_cannot_execute() {
    for mode in ["taf_m", "taf_r_pareto"] {
        assert!(
            SequentialOptimizer::new(OptimizerConfig {
                method: "taf_plus".into(),
                taf_weight_mode: mode.into(),
                objective_weights: Some(vec![0.5, 0.5]),
                ..Default::default()
            })
            .is_err()
        );
    }
}
