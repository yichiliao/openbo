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

fn transfer_population(directory: &std::path::Path) -> Result<()> {
    use openbo::{Matrix, core::GaussianProcess};
    let x: Matrix = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    for user in 0..2 {
        let y: Matrix = x
            .iter()
            .map(|x| {
                vec![
                    1.0 - (x[0] - 0.2 - 0.1 * user as f64).powi(2) - (x[1] - 0.3).powi(2),
                    1.0 - (x[0] - 0.7).powi(2) - (x[1] - 0.6 + 0.1 * user as f64).powi(2),
                ]
            })
            .collect();
        let mut gps = Vec::new();
        for column in 0..2 {
            let mut gp = GaussianProcess::new(GpConfig {
                kernel_type: "rbf".into(),
                lengthscale: vec![0.4],
                noise: 1e-5,
                optimize_hyperparameters: false,
                ..Default::default()
            })?;
            gp.fit(&x, &y.iter().map(|row| row[column]).collect::<Vec<_>>())?;
            gps.push(gp);
        }
        Source {
            name: format!("source_{user}"),
            gps,
            x: x.clone(),
            y,
            meta_features: vec![],
        }
        .save(directory)?;
    }
    Ok(())
}

fn transfer_runtime(directory: &std::path::Path) -> Result<ServerConfig> {
    python_yaml(&format!(
        "input_dim: 2\ny_range: [[-2.0, 2.0], [-2.0, 2.0]]\nv_ref: [-1.0, -1.0]\nw_ref: [1.2, 1.2]\ntaf_run_dir: '{}'\nmethod_default: taf_eipv\nn_iter_default: 3\nn_init_default: null\nweight_mode_default: taf_r\nrho_default: 1.0\nn_candidates_default: 12\nnum_restarts_default: 1\nraw_samples_default: 8\noptimize_hyperparameters_default: false\nrefine_suggestion_default: false\n",
        directory.display().to_string().replace('\'', "''")
    ))
}

#[test]
fn taf_mobo_yaml_dispatches_methods_and_preserves_total_budget_semantics() -> Result<()> {
    let cfg = python_yaml(
        "input_dim: 4\ny_range: [[-12.0, 2.0], [-12.0, 2.0], [-12.0, 2.0]]\nv_ref: [-1.0, -1.0, -1.0]\nw_ref: [1.2, 1.2, 1.2]\npopulation_dir: saved/population\nmethod_default: taf_eipv\nn_iter_default: 10\nn_init_default: null\nweight_mode_default: uniform\n",
    )?;
    assert!(cfg.taf_mobo);
    assert_eq!(cfg.optimizer.method, "taf_eipv");
    assert_eq!(cfg.optimizer.n_init, 0);
    assert_eq!(cfg.optimizer.n_iter, 10);
    assert_eq!(cfg.optimizer.target_weight, 0.75);
    assert_eq!(cfg.optimizer.taf_weight_mode, "uniform");
    assert_eq!(cfg.optimizer.n_objectives(), 3);
    assert_eq!(cfg.optimizer.upper_reference, Some(vec![1.2; 3]));
    let cfg = python_yaml(
        "input_dim: 2\ny_range: [[-2.0, 2.0], [-2.0, 2.0]]\ntaf_run_dir: saved/sources\nmethod_default: taf_ehvi\nn_iter_default: 5\nraw_samples_default: 32\n",
    )?;
    assert!(cfg.taf_mobo);
    assert_eq!(cfg.optimizer.method, "taf_ehvi");
    assert_eq!(cfg.optimizer.n_init, 1);
    assert_eq!(cfg.optimizer.n_iter, 4);
    assert_eq!(cfg.optimizer.n_candidates, 32);
    let nullable = python_yaml(
        "input_dim: 2\ny_range: [[-2.0, 2.0], [-2.0, 2.0]]\npopulation_dir: null\ntaf_run_dir: saved/sources\nmethod_default: taf_eipv\n",
    )?;
    assert_eq!(nullable.optimizer.taf_run_dir, Some("saved/sources".into()));
    let root = tempfile::tempdir()?;
    let path = root.path().join("native.json");
    std::fs::write(
        &path,
        serde_json::to_vec(
            &json!({"optimizer":{"method":"taf_eipv","bounds":[[0.0,1.0],[0.0,1.0]],"n_init":0,"n_iter":3,"taf_run_dir":"saved/sources","reference_point":[-1.0,-1.0],"upper_reference":[1.2,1.2]},"y_range":[[-2.0,2.0],[-2.0,2.0]]}),
        )?,
    )?;
    let native = ServerConfig::from_file(&path)?;
    assert!(native.taf_mobo);
    assert_eq!(native.taf_mobo_n_init, Some(0));
    assert_eq!(native.optimizer.target_weight, 0.75);
    Ok(())
}

#[tokio::test]
async fn taf_mobo_actual_websocket_dispatches_method_and_total_budget() -> Result<()> {
    let root = tempfile::tempdir()?;
    transfer_population(root.path())?;
    let runtime = transfer_runtime(root.path())?;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let handle = tokio::spawn(serve(listener, runtime));
    let (mut socket, _) = connect_async(format!("ws://{address}")).await?;
    socket
        .send(Message::Text(
            json!({"type":"start","method":"invalid"})
                .to_string()
                .into(),
        ))
        .await?;
    let rejected: Value = serde_json::from_str(socket.next().await.unwrap()?.to_text()?)?;
    assert_eq!(rejected["type"], "error");
    socket
        .send(Message::Text(
            json!({"type":"start","n_iter":2,"weight_mode":"uniform"})
                .to_string()
                .into(),
        ))
        .await?;
    let mut response: Value = serde_json::from_str(socket.next().await.unwrap()?.to_text()?)?;
    assert_eq!(response["method"], "taf_eipv");
    assert_eq!(response["phase"], "sources");
    assert_eq!(response["n_iter"], 2);
    for _ in 0..2 {
        socket
            .send(Message::Text(
                json!({"type":"observe","y":[0.1,0.2]}).to_string().into(),
            ))
            .await?;
        response = serde_json::from_str(socket.next().await.unwrap()?.to_text()?)?;
    }
    assert_eq!(response["type"], "done");
    assert_eq!(response["total_observations"], 2);
    assert_eq!(response["sources"].as_array().unwrap().len(), 2);
    handle.abort();
    Ok(())
}

#[test]
fn taf_mobo_sessions_report_sources_phases_pareto_and_hypervolume_without_writing_sources()
-> Result<()> {
    use openbo::{Matrix, acquisition};
    let root = tempfile::tempdir()?;
    transfer_population(root.path())?;
    let state = root.path().join("gp_states/source_0.json");
    let before = std::fs::read(&state)?;
    let runtime = transfer_runtime(root.path())?;
    let mut cfg = runtime.optimizer.clone();
    cfg.seed = 3;
    let mut local = SequentialOptimizer::new(cfg)?;
    let first = local.suggest()?;
    let mut session = Session::start(&json!({"type":"start","seed":3}), &runtime)?;
    let mut response = session.handle(&json!({"type":"suggest"}))?;
    assert_eq!(response["x"], json!(first));
    assert_eq!(response["optimizer"], "taf_mobo");
    assert_eq!(response["method"], "taf_eipv");
    assert_eq!(response["phase"], "sources");
    assert_eq!(response["source_weights"].as_array().unwrap().len(), 2);
    assert_eq!(response["n_iter"], 3);
    for i in 0..3 {
        assert!(
            session
                .handle(&json!({"type":"observe","y":[0.0]}))
                .is_err()
        );
        response = session.handle(
            &json!({"type":"observe","x":response["x"],"y":[0.1*i as f64,0.2+0.1*i as f64]}),
        )?;
        if i < 2 {
            assert_eq!(response["phase"], "taf");
        }
    }
    assert_eq!(response["type"], "done");
    assert_eq!(response["total_observations"], 3);
    assert_eq!(response["sources"], json!(["source_0", "source_1"]));
    let y: Matrix = serde_json::from_value(response["y_values"].clone())?;
    for i in 0..3 {
        assert!(
            (response["hv_history"][i].as_f64().unwrap()
                - acquisition::hypervolume(&y[..=i].to_vec(), &[-1.0, -1.0])?)
            .abs()
                < 1e-12
        );
    }
    assert_eq!(response["pareto_y"], json!(acquisition::pareto_front(&y)?));
    assert_eq!(response["pareto_x"].as_array().unwrap().len(), 1);
    assert_eq!(std::fs::read(&state)?, before);
    assert_eq!(std::fs::read_dir(root.path().join("gp_states"))?.count(), 2);
    Ok(())
}

#[test]
fn taf_mobo_ehvi_override_uses_one_random_start_and_validates_overrides() -> Result<()> {
    let root = tempfile::tempdir()?;
    transfer_population(root.path())?;
    let runtime = transfer_runtime(root.path())?;
    for invalid in [
        json!({"method":"taf_ehvi","n_init":0}),
        json!({"n_iter":2,"n_init":3}),
        json!({"method":"other"}),
        json!({"weight_mode":"taf_m"}),
        json!({"method":"taf_ehvi","weight_mode":"uniform"}),
        json!({"rho":0.0}),
        json!({"num_restarts":0}),
    ] {
        let mut payload = invalid;
        payload["type"] = json!("start");
        assert!(
            Session::start(&payload, &runtime).is_err(),
            "accepted {payload}"
        );
    }
    let mut session = Session::start(
        &json!({"type":"start","method":"taf_ehvi","n_iter":2,"weight_mode":"taf_m","raw_samples":8}),
        &runtime,
    )?;
    let mut response = session.handle(&json!({"type":"suggest"}))?;
    assert_eq!(response["phase"], "random");
    assert!(response["source_weights"].is_null());
    response = session.handle(&json!({"type":"observe","y":[0.0,0.1]}))?;
    assert_eq!(response["phase"], "taf");
    assert_eq!(response["method"], "taf_ehvi");
    let status = session.handle(&json!({"type":"status"}))?;
    assert_eq!(status["n_observations"], 1);
    assert_eq!(status["n_sources"], 2);
    assert_eq!(status["n_iter"], 2);
    assert!(status["hypervolume"].as_f64().unwrap() > 0.0);
    let done = session.handle(&json!({"type":"observe","y":[0.2,0.3]}))?;
    assert_eq!(done["type"], "done");
    assert_eq!(done["total_observations"], 2);
    let mut session = Session::start(&json!({"type":"start"}), &runtime)?;
    let stopped = session.handle(&json!({"type":"stop","reason":"test"}))?;
    assert_eq!(stopped["reason"], "test");
    assert_eq!(stopped["hv_history"], json!([]));
    Ok(())
}

#[test]
fn taf_mobo_runtime_rejects_invalid_new_yaml_settings() {
    for text in [
        "input_dim: 2\ny_range: [[0.0, 1.0], [0.0, 1.0]]\nmethod_default: taf_eipv",
        "input_dim: 2\ny_range: [[0.0, 1.0]]\npopulation_dir: saved/population\nmethod_default: taf_eipv",
        "input_dim: 2\ny_range: [[0.0, 1.0], [0.0, 1.0]]\npopulation_dir: saved/population\nmethod_default: qehvi",
        "input_dim: 2\ny_range: [[0.0, 1.0], [0.0, 1.0]]\npopulation_dir: saved/population\nmethod_default: taf_ehvi\nweight_mode_default: uniform",
        "input_dim: 2\ny_range: [[0.0, 1.0], [0.0, 1.0]]\npopulation_dir: saved/population\nmethod_default: taf_ehvi\nn_init_default: 0",
    ] {
        assert!(python_yaml(text).is_err(), "accepted {text}");
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
