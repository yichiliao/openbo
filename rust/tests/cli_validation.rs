//! End-to-end checks of the workflows documented for students.
use openbo::{
    Matrix,
    core::{self, GpConfig},
    neural::{NafConfig, load_naf},
    optimizers::{ConboPopulation, RunResult},
};
use serde_json::{Value, json};
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_openbo"))
}

fn success(command: &mut Command) -> Output {
    let output = command.output().expect("execute native openbo CLI");
    assert!(
        output.status.success(),
        "command {command:?} failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("read generated JSON"))
        .expect("valid generated JSON")
}

fn stdout_json(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).expect("CLI prints JSON")
}

fn trajectory(path: &Path, n: usize, dim: usize, objectives: usize) -> RunResult {
    let run: RunResult =
        serde_json::from_value(read_json(path)).expect("portable RunResult schema");
    assert_eq!(run.x_values.len(), n);
    assert_eq!(run.y_values.len(), n);
    assert!(run.x_values.iter().all(|row| {
        row.len() == dim
            && row
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    }));
    assert!(
        run.y_values
            .iter()
            .all(|row| row.len() == objectives && row.iter().all(|v| v.is_finite()))
    );
    if objectives == 1 {
        assert_eq!(run.best_so_far.len(), n);
        assert!(run.best_so_far.windows(2).all(|pair| pair[1] >= pair[0]));
    } else {
        assert_eq!(run.hypervolume_history.len(), n);
        assert!(
            run.hypervolume_history
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
        );
        assert!(
            run.hypervolume_history
                .windows(2)
                .all(|pair| pair[1] + 1e-10 >= pair[0])
        );
    }
    run
}

#[test]
fn cli_help_and_invalid_arguments_report_useful_errors() {
    let help = success(cli().arg("--help"));
    let text = String::from_utf8(help.stdout).unwrap();
    for command in [
        "quickstart",
        "benchmark",
        "train-naf",
        "serve",
        "import-naf",
    ] {
        assert!(
            text.contains(command),
            "missing documented command {command}"
        );
    }
    let invalid = cli().arg("no-such-command").output().unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("unrecognized"));
    let invalid = cli()
        .args(["benchmark", "--n-init", "4", "--n-evals", "2"])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("n-evals cannot be below n-init"));
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("invalid-gp.json");
    fs::write(&input, serde_json::to_vec(&json!({"config":{"lengthscale":-1.0},"x_train":[[0.0]],"y_train":[0.0],"x_test":[[0.5]]})).unwrap()).unwrap();
    let invalid = cli().args(["gp", "--input"]).arg(&input).output().unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("lengthscale"));
}

#[test]
fn student_quickstart_writes_equal_budget_trajectories_and_svg() {
    let directory = tempfile::tempdir().unwrap();
    success(
        cli()
            .args([
                "quickstart",
                "--n-evals",
                "5",
                "--seed",
                "3",
                "--output-dir",
            ])
            .arg(directory.path()),
    );
    for method in ["random", "bo_scratch"] {
        let run = trajectory(&directory.path().join(format!("{method}.json")), 5, 2, 1);
        assert_eq!(run.method, method);
        let expected = core::synthetic("branin", &run.x_values).unwrap();
        for (row, expected) in run.y_values.iter().zip(expected) {
            assert!((row[0] - expected).abs() < 1e-10);
        }
    }
    let svg = fs::read_to_string(directory.path().join("comparison.svg")).unwrap();
    assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
    assert!(svg.contains("random") && svg.contains("bo_scratch"));
}

#[test]
fn scalar_and_multi_objective_benchmark_commands_produce_valid_results() {
    let directory = tempfile::tempdir().unwrap();
    let scalar = directory.path().join("scalar.json");
    success(
        cli()
            .args([
                "benchmark",
                "--method",
                "bo_scratch",
                "--function",
                "branin",
                "--n-init",
                "2",
                "--n-iter",
                "1",
                "--n-candidates",
                "8",
                "--fixed-hyperparameters",
                "--output",
            ])
            .arg(&scalar),
    );
    let run = trajectory(&scalar, 3, 2, 1);
    assert_eq!(run.method, "bo_scratch");
    let multi = directory.path().join("multi.json");
    success(
        cli()
            .args([
                "benchmark",
                "--method",
                "mobo_ehvi",
                "--function",
                "oka2",
                "--n-init",
                "2",
                "--n-iter",
                "1",
                "--n-candidates",
                "8",
                "--mc-samples",
                "4",
                "--fixed-hyperparameters",
                "--output",
            ])
            .arg(&multi)
            .arg("--save-source-dir")
            .arg(directory.path().join("sources"))
            .args(["--source-name", "earlier_user"]),
    );
    let run = trajectory(&multi, 3, 3, 2);
    let expected = core::multi_objective("oka2", &run.x_values).unwrap();
    assert_eq!(run.y_values, expected);
    let sources = directory.path().join("sources");
    assert!(sources.join("gp_states/earlier_user.json").exists());
    let transfer = directory.path().join("transfer.json");
    success(
        cli()
            .args([
                "benchmark",
                "--method",
                "taf_eipv",
                "--function",
                "oka2",
                "--n-evals",
                "3",
                "--n-candidates",
                "8",
                "--fixed-hyperparameters",
                "--sources",
            ])
            .arg(&sources)
            .arg("--output")
            .arg(&transfer),
    );
    let run = trajectory(&transfer, 3, 3, 2);
    assert_eq!(
        run.y_values,
        core::multi_objective("oka2", &run.x_values).unwrap()
    );
    assert_eq!(run.metadata["sources"], json!(["earlier_user"]));
    assert!(run.metadata["elapsed_seconds"].as_f64().unwrap() >= 0.);
}

#[test]
fn family_split_source_training_inspection_and_transfer_pipeline_runs() {
    let directory = tempfile::tempdir().unwrap();
    let split = directory.path().join("split.json");
    success(
        cli()
            .args([
                "create-split",
                "--function",
                "branin",
                "--n-tasks",
                "3",
                "--train-ratio",
                "0.67",
                "--seed",
                "2",
                "--output",
            ])
            .arg(&split),
    );
    let parsed = core::load_family_split(&split).unwrap();
    assert_eq!(parsed.train_variants.len(), 2);
    assert_eq!(parsed.test_variants.len(), 1);
    let sources = directory.path().join("sources");
    success(
        cli()
            .args(["train-sources", "--split"])
            .arg(&split)
            .args(["--n-init", "2", "--n-iter", "1", "--output-dir"])
            .arg(&sources),
    );
    let inspected = stdout_json(&success(
        cli()
            .args(["inspect-sources", "--directory"])
            .arg(&sources)
            .args(["--dim", "2"]),
    ));
    assert_eq!(inspected["sources"].as_array().unwrap().len(), 2);
    assert!(inspected["warnings"].as_array().unwrap().is_empty());
    assert!(
        inspected["sources"]
            .as_array()
            .unwrap()
            .iter()
            .all(|source| source["observations"] == 3 && source["objectives"] == 1)
    );
    let output = directory.path().join("family.json");
    success(
        cli()
            .args(["family-benchmark", "--split"])
            .arg(&split)
            .args(["--method", "bo_taf", "--sources"])
            .arg(&sources)
            .args(["--n-evals", "3", "--output"])
            .arg(&output),
    );
    let runs: Vec<RunResult> = serde_json::from_value(read_json(&output)).unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].method, "bo_taf");
    assert_eq!(runs[0].x_values.len(), 3);
    assert_eq!(runs[0].y_values.len(), 3);
    assert!(runs[0].y_values.iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn native_naf_training_resumes_and_benchmark_uses_the_normalized_family_scale() {
    let directory = tempfile::tempdir().unwrap();
    let policy = directory.path().join("policy");
    success(
        cli()
            .args([
                "train-naf",
                "--n-iterations",
                "1",
                "--budget",
                "2",
                "--episodes",
                "2",
                "--hidden",
                "4",
                "--output-dir",
            ])
            .arg(&policy),
    );
    assert_eq!(
        read_json(&policy.join("training_history.json"))["mean_return"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    success(
        cli()
            .args([
                "train-naf",
                "--n-iterations",
                "2",
                "--budget",
                "2",
                "--episodes",
                "2",
                "--hidden",
                "4",
                "--resume",
                "--output-dir",
            ])
            .arg(&policy),
    );
    assert_eq!(
        read_json(&policy.join("training_history.json"))["mean_return"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    for name in [
        "af.json",
        "best_af.json",
        "naf_config.json",
        "training_state.json",
    ] {
        assert!(policy.join(name).is_file());
    }
    let output = directory.path().join("naf.json");
    success(
        cli()
            .args(["naf-benchmark", "--af-dir"])
            .arg(&policy)
            .args(["--output"])
            .arg(&output),
    );
    let run = trajectory(&output, 2, 2, 1);
    let expected = core::synthetic("normalized_branin", &run.x_values).unwrap();
    for (actual, expected) in run.y_values.iter().zip(expected) {
        assert!((actual[0] - expected).abs() < 1e-12);
    }
    assert_eq!(run.metadata["function"], "normalized_branin");
}

#[test]
fn conbo_command_persists_a_reloadable_two_user_population() {
    let directory = tempfile::tempdir().unwrap();
    success(
        cli()
            .args([
                "conbo-benchmark",
                "--n-users",
                "2",
                "--n-evals",
                "2",
                "--epochs",
                "1",
                "--output-dir",
            ])
            .arg(directory.path()),
    );
    let runs = read_json(&directory.path().join("runs.json"));
    assert_eq!(runs.as_array().unwrap().len(), 2);
    for (i, run) in runs.as_array().unwrap().iter().enumerate() {
        assert_eq!(run["trajectory"]["x_values"].as_array().unwrap().len(), 2);
        assert_eq!(run["trajectory"]["y_values"].as_array().unwrap().len(), 2);
        assert_eq!(run["replay"]["n_users"], i + 1);
    }
    let population = ConboPopulation::load(directory.path()).unwrap();
    assert_eq!(population.users.len(), 2);
    assert!(population.bnn.is_some());
    population.bnn.as_ref().unwrap().validate().unwrap();
}

#[test]
fn import_naf_checks_portable_tensor_parity_before_saving_policies() {
    let directory = tempfile::tempdir().unwrap();
    let config = NafConfig {
        dim: 1,
        budget: 2,
        hidden: vec![4],
        points_per_dim: 5,
        n_local_maxima: 2,
        n_local: 8,
        steps_per_iteration: 4,
        n_iterations: 1,
        ..Default::default()
    };
    let policy = config.new_af().unwrap();
    let state_dict_path = directory.path().join("state_dict.json");
    policy.export_state_dict_json(&state_dict_path).unwrap();
    let features: Matrix = vec![
        vec![0.0, 1.0, 0.25, 0.0, 2.0],
        vec![0.4, 0.6, 0.75, 1.0, 2.0],
    ];
    let logits = policy.logits(&features).unwrap();
    let gp = GpConfig {
        kernel_type: "rbf".into(),
        standardize_targets: false,
        ..Default::default()
    };
    let input = directory.path().join("export.json");
    let mut exported = json!({"format":"openbo-python-naf-export","version":1,"config":config,"gp_config":gp,"state_dict":read_json(&state_dict_path),"parity_cases":[{"features":features,"logits":logits}],"parity_tolerance":{"absolute":1e-14,"relative":1e-14}});
    fs::write(&input, serde_json::to_vec(&exported).unwrap()).unwrap();
    let output_dir = directory.path().join("imported");
    let report = stdout_json(&success(
        cli()
            .args(["import-naf", "--input"])
            .arg(&input)
            .arg("--output-dir")
            .arg(&output_dir),
    ));
    assert_eq!(report["parity_cases"], 1);
    assert_eq!(report["maximum_error"], 0.0);
    let (loaded, loaded_config, loaded_gp) = load_naf(&output_dir, "best_af").unwrap();
    assert_eq!(loaded, policy);
    assert_eq!(loaded_config, config);
    assert_eq!(loaded_gp, gp);
    assert_eq!(loaded.logits(&features).unwrap(), logits);
    exported["parity_cases"][0]["logits"][0] = json!(100.0);
    fs::write(&input, serde_json::to_vec(&exported).unwrap()).unwrap();
    let invalid_dir = directory.path().join("invalid-import");
    let rejected = cli()
        .args(["import-naf", "--input"])
        .arg(&input)
        .arg("--output-dir")
        .arg(&invalid_dir)
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("mismatch"));
    assert!(!invalid_dir.join("af.json").exists());
    let original_state = fs::read(output_dir.join("state_dict.json")).unwrap();
    let original_policy = fs::read(output_dir.join("af.json")).unwrap();
    exported["state_dict"]["net.0.bias"][0] = json!(300.0);
    fs::write(&input, serde_json::to_vec(&exported).unwrap()).unwrap();
    let rejected = cli()
        .args(["import-naf", "--input"])
        .arg(&input)
        .arg("--output-dir")
        .arg(&output_dir)
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert_eq!(
        fs::read(output_dir.join("state_dict.json")).unwrap(),
        original_state
    );
    assert_eq!(
        fs::read(output_dir.join("af.json")).unwrap(),
        original_policy
    );
}
