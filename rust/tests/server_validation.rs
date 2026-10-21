use futures_util::{SinkExt, StreamExt};
use openbo::{Result, core::GpConfig, neural, optimizers::*, server::*};
use serde_json::{Value, json};
use tokio_tungstenite::{connect_async, tungstenite::Message};

fn runtime() -> ServerConfig {
    ServerConfig {
        optimizer: OptimizerConfig {
            bounds: vec![(0.0, 1.0); 2],
            n_init: 2,
            n_iter: 1,
            n_candidates: 8,
            refine: false,
            gp: GpConfig {
                optimize_hyperparameters: false,
                ..Default::default()
            },
            ..Default::default()
        },
        y_range: vec![(-200.0, 10.0)],
        ..Default::default()
    }
}

#[test]
fn scalar_protocol_validates_without_consuming_pending_state() -> Result<()> {
    let mut session = Session::start(&json!({"type":"start"}), &runtime())?;
    assert!(session.handle(&json!({"type":"observe","y":1})).is_err());
    let mut message = session.handle(&json!({"type":"suggest"}))?;
    assert!(session.handle(&json!({"type":"suggest"})).is_err());
    for bad in [json!(true), json!("bad"), json!([1, 2]), json!(-201)] {
        assert!(session.handle(&json!({"type":"observe","y":bad})).is_err());
        assert_eq!(
            session.handle(&json!({"type":"status"}))?["n_observations"],
            0
        );
    }
    assert!(
        session
            .handle(&json!({"type":"observe","x":[0,0],"y":0}))
            .is_err()
    );
    for i in 0..3 {
        message = session.handle(&json!({"type":"observe","x":message["x"],"y":-(i as f64)}))?;
    }
    assert_eq!(message["type"], "done");
    assert_eq!(message["total_observations"], 3);
    assert_eq!(message["y_values"], json!([0.0, -1.0, -2.0]));
    assert!(session.handle(&json!({"type":"status"})).is_err());
    Ok(())
}

#[test]
fn vector_protocol_retains_all_objectives() -> Result<()> {
    let mut cfg = runtime();
    cfg.optimizer.method = "mobo_ehvi".into();
    cfg.optimizer.reference_point = Some(vec![-5.0, -5.0]);
    cfg.optimizer.n_init = 1;
    cfg.optimizer.n_iter = 1;
    cfg.optimizer.mc_samples = 8;
    let mut session = Session::start(&json!({"type":"start"}), &cfg)?;
    let mut message = session.handle(&json!({"type":"suggest"}))?;
    for _ in 0..2 {
        message = session.handle(&json!({"type":"observe","x":message["x"],"y":[-1.0,-2.0]}))?;
    }
    assert_eq!(message["type"], "done");
    assert_eq!(message["y_values"][0], json!([-1.0, -2.0]));
    assert_eq!(message["hypervolume_history"], json!([12.0, 12.0]));
    Ok(())
}

#[test]
fn conbo_server_stores_and_reloads_population() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut cfg = runtime();
    cfg.optimizer.method = "conbo".into();
    cfg.memory_dir = Some(dir.path().into());
    cfg.conbo.population_epochs = 1;
    cfg.conbo.replay_grid_points = 3;
    cfg.conbo.replay_random_points = 2;
    cfg.conbo.n_evaluations = 2;
    cfg.conbo.gp.optimize_hyperparameters = false;
    for index in 0..2 {
        let mut session = Session::start(
            &json!({"type":"start","user_name":format!("user_{index}")}),
            &cfg,
        )?;
        let mut message = session.handle(&json!({"type":"suggest"}))?;
        for _ in 0..2 {
            message = session.handle(&json!({"type":"observe","x":message["x"],"y":-1.0}))?;
        }
        assert_eq!(message["stored"], true);
    }
    assert_eq!(ConboPopulation::load(dir.path())?.users.len(), 2);
    cfg.optimizer.bounds = vec![(0.0, 1.0); 3];
    cfg.conbo.input_dim = 3;
    assert!(Session::start(&json!({"type":"start","user_name":"wrong_dimension"}), &cfg).is_err());
    Ok(())
}

#[test]
fn conbo_persistence_failure_keeps_the_session_retryable() -> Result<()> {
    let root = tempfile::tempdir()?;
    let memory = root.path().join("memory");
    std::fs::write(&memory, "a file deliberately blocks directory creation")?;
    let mut cfg = runtime();
    cfg.optimizer.method = "conbo".into();
    cfg.memory_dir = Some(memory.clone());
    cfg.conbo.n_evaluations = 1;
    cfg.conbo.population_epochs = 1;
    cfg.conbo.replay_grid_points = 3;
    cfg.conbo.replay_random_points = 2;
    cfg.conbo.gp.optimize_hyperparameters = false;
    let mut session = Session::start(&json!({"type":"start","user_name":"retryable_user"}), &cfg)?;
    let suggestion = session.handle(&json!({"type":"suggest"}))?;
    assert!(
        session
            .handle(&json!({"type":"observe","x":suggestion["x"],"y":-1.0}))
            .is_err()
    );
    assert_eq!(
        session.handle(&json!({"type":"status"}))?["n_observations"],
        1
    );
    std::fs::remove_file(&memory)?;
    let finished = session.handle(&json!({"type":"suggest"}))?;
    assert_eq!(finished["type"], "done");
    assert_eq!(finished["stored"], true);
    assert_eq!(ConboPopulation::load(&memory)?.users.len(), 1);
    Ok(())
}

#[test]
fn naf_server_honors_best_checkpoint_and_early_stop_without_changing_training_budget() -> Result<()>
{
    let dir = tempfile::tempdir()?;
    let config = neural::NafConfig {
        dim: 2,
        budget: 5,
        points_per_dim: 3,
        n_local: 2,
        n_local_maxima: 1,
        hidden: vec![4],
        ..Default::default()
    };
    let gp = GpConfig::default();
    let policy = config.new_af()?;
    neural::save_naf(dir.path(), &policy, &config, &gp, "best_af")?;
    let mut cfg = runtime();
    cfg.optimizer.method = "naf".into();
    cfg.naf_dir = Some(dir.path().into());
    cfg.naf_n_iter = Some(2);
    let mut session = Session::start(&json!({"type":"start"}), &cfg)?;
    let mut message = session.handle(&json!({"type":"suggest"}))?;
    for _ in 0..2 {
        message = session.handle(&json!({"type":"observe","x":message["x"],"y":-1.0}))?;
    }
    assert_eq!(message["type"], "done");
    assert_eq!(message["budget"], 5);
    assert_eq!(message["total_observations"], 2);
    assert!(Session::start(&json!({"type":"start","n_iter":6}), &cfg).is_err());
    cfg.optimizer.bounds = vec![(0.0, 1.0)];
    assert!(Session::start(&json!({"type":"start"}), &cfg).is_err());
    Ok(())
}

#[test]
fn python_yaml_runtime_config_loads_and_native_roundtrips() -> Result<()> {
    let cfg = python_yaml(
        "optimizer: bo_scratch\ninput_dim: 2\ny_range: [-350.0, 1.0]\nn_init_default: 2\nn_iter_default: 8\nn_candidates_default: 512\nkernel_type_default: rbf\noptimize_hyperparameters_default: false\nauto_save_scratch_artifacts: true\nscratch_artifacts_dir: saved/scratch\n",
    )?;
    assert_eq!(cfg.optimizer.method, "bo_scratch");
    assert_eq!(cfg.optimizer.n_init, 2);
    assert_eq!(cfg.optimizer.n_iter, 8);
    assert_eq!(cfg.optimizer.n_candidates, 512);
    assert_eq!(cfg.optimizer.gp.kernel_type, "rbf");
    assert!(!cfg.optimizer.gp.optimize_hyperparameters);
    assert_eq!(cfg.auto_save_dir, Some("saved/scratch".into()));
    let serialized = serde_json::to_string(&cfg)?;
    let recovered: ServerConfig = serde_json::from_str(&serialized)?;
    recovered.validate()?;
    let root = tempfile::tempdir()?;
    let path = root.path().join("native.json");
    std::fs::write(&path, &serialized)?;
    assert_eq!(
        serde_json::to_value(ServerConfig::from_file(&path)?)?,
        serde_json::to_value(recovered)?
    );
    Ok(())
}

fn python_yaml(text: &str) -> Result<ServerConfig> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("server.yaml");
    std::fs::write(&path, text)?;
    ServerConfig::from_file(&path)
}

#[test]
fn python_yaml_infers_naf_and_conbo_launchers_and_maps_their_defaults() -> Result<()> {
    let naf = python_yaml(
        "input_dim: 2\ny_range: [-12.0, 2.0]\naf_dir: saved/policy\naf_name: af\nn_iter_default: null\n",
    )?;
    assert_eq!(naf.optimizer.method, "naf");
    assert_eq!(naf.naf_dir, Some("saved/policy".into()));
    assert_eq!(naf.naf_name, "af");
    assert_eq!(naf.naf_n_iter, None);
    let conbo = python_yaml(
        "input_dim: 2\ny_range: [-350.0, 1.0]\nmemory_dir: saved/memory\nn_evaluations_default: 8\nconbo:\n  r0: 6\n  dr: 2\n  candidate_grid_points: 4\n  population_epochs: 3\n",
    )?;
    assert_eq!(conbo.optimizer.method, "conbo");
    assert_eq!(conbo.memory_dir, Some("saved/memory".into()));
    assert_eq!(conbo.conbo.n_random, 6);
    assert_eq!(conbo.conbo.dr, 2);
    assert_eq!(conbo.conbo.n_candidates, 16);
    assert_eq!(conbo.conbo.n_evaluations, 8);
    assert_eq!(conbo.conbo.population_epochs, 3);
    Ok(())
}

#[test]
fn python_yaml_infers_taf_and_weighted_population_sessions() -> Result<()> {
    let taf = python_yaml(
        "input_dim: 2\ny_range: [-350.0, 1.0]\ntaf_run_dir: saved/taf\nn_init_default: 0\nn_iter_default: 7\ntaf_weight_mode_default: taf_m\ntaf_rho_default: 0.5\nsource_only_warmup_iters_default: 2\n",
    )?;
    assert_eq!(taf.optimizer.method, "bo_taf");
    assert_eq!(taf.optimizer.taf_run_dir, Some("saved/taf".into()));
    assert_eq!(taf.optimizer.n_init, 0);
    assert_eq!(taf.optimizer.n_iter, 7);
    assert_eq!(taf.optimizer.taf_weight_mode, "taf_m");
    assert_eq!(taf.optimizer.rho, 0.5);
    assert_eq!(taf.optimizer.source_only_warmup_iters, 2);
    let plus = python_yaml(
        "input_dim: 4\ny_range: [[-12.0, 2.0], [-12.0, 2.0], [-12.0, 2.0]]\npopulation_dir: saved/population\nobjective_weights_default: [0.3, 0.5, 0.2]\ndecay_default: false\nd1_default: 8\nd2_default: 0.1\nweight_mode_default: poe\nrho_default: 0.7\nn_iter_default: 10\nn_candidates_default: 32\nrefine_suggestion_default: false\n",
    )?;
    assert_eq!(plus.optimizer.method, "taf_plus");
    assert_eq!(plus.optimizer.n_objectives(), 3);
    assert_eq!(plus.optimizer.taf_run_dir, Some("saved/population".into()));
    assert!(plus.memory_dir.is_none());
    assert_eq!(plus.optimizer.objective_weights, Some(vec![0.3, 0.5, 0.2]));
    assert_eq!(plus.optimizer.n_init, 0);
    assert!(!plus.optimizer.decay);
    assert_eq!(plus.optimizer.decay_start_iter, 8);
    assert_eq!(plus.optimizer.decay_rate, 0.1);
    assert_eq!(plus.optimizer.taf_weight_mode, "poe");
    assert_eq!(plus.optimizer.rho, 0.7);
    assert_eq!(plus.optimizer.n_candidates, 32);
    assert!(!plus.optimizer.refine);
    Ok(())
}

#[test]
fn python_yaml_preserves_mobo_model_acquisition_and_reference_box() -> Result<()> {
    let cfg = python_yaml(
        "input_dim: 3\ny_range: [[-3.2, 3.2], [-5.1, 0.0]]\nmodel: mt\nacquisition: ceipv\nhyperparameters: slice\nn_candidates_default: 17\n",
    )?;
    assert_eq!(cfg.optimizer.method, "mobo_ceipv");
    assert_eq!(cfg.optimizer.model, "mt");
    assert_eq!(cfg.optimizer.hyperparameters, "slice");
    assert_eq!(cfg.optimizer.n_candidates, 17);
    assert_eq!(cfg.optimizer.reference_point, Some(vec![-3.2, -5.1]));
    assert_eq!(cfg.optimizer.upper_reference, Some(vec![3.2, 0.0]));
    let cfg = python_yaml(
        "optimizer: mobo\ninput_dim: 3\ny_range: [[-3.2, 3.1416], [-5.1, 0.0]]\nmodel: slf\nacquisition: ieipv\nv_ref: [-7.0, -8.0]\nw_ref: [4.0, 3.0]\n",
    )?;
    assert_eq!(cfg.optimizer.method, "mobo_ieipv");
    assert_eq!(cfg.optimizer.model, "slf");
    assert_eq!(cfg.optimizer.reference_point, Some(vec![-7.0, -8.0]));
    assert_eq!(cfg.optimizer.upper_reference, Some(vec![4.0, 3.0]));
    let baseline =
        python_yaml("optimizer: random\ninput_dim: 3\ny_range: [[-3.2, 3.2], [-5.1, 0.0]]\n")?;
    assert_eq!(baseline.optimizer.method, "random");
    assert_eq!(baseline.optimizer.n_objectives(), 2);
    assert_eq!(baseline.optimizer.reference_point, Some(vec![-3.2, -5.1]));
    Ok(())
}

#[test]
fn python_yaml_rejects_ambiguous_engines_and_invalid_numeric_or_choice_fields() {
    for invalid in [
        "[]",
        "input_dim: -1\ny_range: [-1.0, 1.0]",
        "input_dim: 0\ny_range: [-1.0, 1.0]",
        "n_candidates_default: -1",
        "input_dim: 2\naf_dir: saved/policy\nmemory_dir: saved/memory",
        "y_range: [[-1.0, 1.0], [-2.0, 2.0]]\nacquisition: imaginary",
        "y_range: [[-1.0, 1.0], [-2.0, 2.0]]\nmodel: imaginary",
        "y_range: [[-1.0, 1.0], [-2.0, 2.0]]\nv_ref: [-1.0]",
        "y_range: [[-1.0, 1.0], [-2.0, 2.0]]\nw_ref: [-2.0, -3.0]",
        "optimizer: bo_scratch\nacquisition: ceipv",
    ] {
        assert!(
            python_yaml(invalid).is_err(),
            "accepted invalid YAML: {invalid}"
        );
    }
}

#[tokio::test]
async fn actual_websocket_rejects_bad_start_then_completes_and_isolates_sessions() -> Result<()> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let handle = tokio::spawn(serve(listener, runtime()));
    let (mut a, _) = connect_async(format!("ws://{addr}")).await?;
    let (mut b, _) = connect_async(format!("ws://{addr}")).await?;
    a.send(Message::Text("[]".into())).await?;
    let error: Value = serde_json::from_str(a.next().await.unwrap()?.to_text()?)?;
    assert_eq!(error["type"], "error");
    let start = json!({"type":"start","seed":91,"n_init":1,"n_iter":1});
    a.send(Message::Text(start.to_string().into())).await?;
    b.send(Message::Text(start.to_string().into())).await?;
    let mut left: Value = serde_json::from_str(a.next().await.unwrap()?.to_text()?)?;
    let right: Value = serde_json::from_str(b.next().await.unwrap()?.to_text()?)?;
    assert_eq!(left["x"], right["x"]);
    for _ in 0..2 {
        a.send(Message::Text(
            json!({"type":"observe","x":left["x"],"y":-1.0})
                .to_string()
                .into(),
        ))
        .await?;
        left = serde_json::from_str(a.next().await.unwrap()?.to_text()?)?;
    }
    assert_eq!(left["type"], "done");
    b.send(Message::Text(json!({"type":"status"}).to_string().into()))
        .await?;
    let status: Value = serde_json::from_str(b.next().await.unwrap()?.to_text()?)?;
    assert_eq!(status["n_observations"], 0);
    b.close(None).await?;
    handle.abort();
    Ok(())
}

#[test]
fn standard_stop_does_not_serialize_negative_infinity() -> Result<()> {
    let mut session = Session::start(&json!({"type":"start"}), &runtime())?;
    let stopped = session.handle(&json!({"type":"stop"}))?;
    serde_json::to_string(&stopped)?;
    assert_eq!(stopped["total_observations"], 0);
    Ok(())
}
