use openbo::Matrix;
use openbo::core::{GaussianProcess, GpConfig};
use openbo::neural::*;

fn gp_config() -> GpConfig {
    GpConfig {
        lengthscale: vec![0.25],
        variance: 1.0,
        noise: 1e-8,
        kernel_type: "rbf".into(),
        standardize_targets: false,
        ..Default::default()
    }
}

fn tiny_config() -> NafConfig {
    NafConfig {
        dim: 1,
        budget: 3,
        hidden: vec![8],
        points_per_dim: 5,
        n_local_maxima: 2,
        n_local: 8,
        n_iterations: 2,
        steps_per_iteration: 6,
        epochs: 2,
        minibatch_size: 3,
        learning_rate: 0.003,
        ..Default::default()
    }
}

#[test]
fn bnn_trains_and_has_repeatable_positive_predictions() {
    let mut bnn = Bnn::new(BnnConfig {
        input_dim: 1,
        hidden_units: 16,
        n_hidden_layers: 2,
        dropout: 0.0,
        learning_rate: 0.01,
        n_predict_samples: 3,
        ..Default::default()
    })
    .unwrap();
    let x: Matrix = (0..21).map(|i| vec![i as f64 / 20.0]).collect();
    let y: Vec<f64> = x.iter().map(|x| 2.0 * x[0] - 0.7).collect();
    let losses = bnn.fit_nll(&x, &y, 200).unwrap();
    assert!(losses[199] < losses[0] - 1.0, "{losses:?}");
    let first = bnn.predict(&x).unwrap();
    assert_eq!(first, bnn.predict(&x).unwrap());
    assert!(first.1.iter().all(|v| v.is_finite() && *v > 0.0));
    let mse = first
        .0
        .iter()
        .zip(&y)
        .map(|(m, y)| (m - y).powi(2))
        .sum::<f64>()
        / y.len() as f64;
    assert!(mse < 0.03, "mse={mse}");
}

#[test]
fn bnn_replay_loss_decreases_and_keeps_normalization() {
    let mut bnn = Bnn::new(BnnConfig {
        input_dim: 1,
        hidden_units: 12,
        n_hidden_layers: 1,
        dropout: 0.0,
        learning_rate: 0.01,
        ..Default::default()
    })
    .unwrap();
    let x = vec![vec![0.0], vec![0.25], vec![0.5], vec![0.75], vec![1.0]];
    let y = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let v = vec![0.1; 5];
    let losses = bnn.fit_replay(&x, &y, &v, 150, true).unwrap();
    assert!(losses[149] < losses[0] * 0.05);
    let normalization = (bnn.y_offset, bnn.y_scale);
    bnn.fit_nll(&x, &[4.0; 5], 1).unwrap();
    assert_eq!(normalization, (bnn.y_offset, bnn.y_scale));
}

#[test]
fn bnn_checkpoint_restores_adam_and_rng() {
    let mut first = Bnn::new(BnnConfig {
        input_dim: 1,
        hidden_units: 6,
        n_hidden_layers: 2,
        dropout: 0.3,
        batch_size: Some(2),
        ..Default::default()
    })
    .unwrap();
    let x = vec![vec![0.1], vec![0.5], vec![0.8]];
    let y = [-0.7, 0.2, 0.9];
    first.fit_nll(&x, &y, 5).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("bnn.json");
    first.save(&file).unwrap();
    let mut restored = Bnn::load(&file).unwrap();
    assert_eq!(first.predict(&x).unwrap(), restored.predict(&x).unwrap());
    assert_eq!(
        first.fit_nll(&x, &y, 5).unwrap(),
        restored.fit_nll(&x, &y, 5).unwrap()
    );
    assert_eq!(first.predict(&x).unwrap(), restored.predict(&x).unwrap());
}

#[test]
fn bnn_rejects_invalid_data_without_poisoning_state() {
    let mut model = Bnn::new(BnnConfig {
        input_dim: 1,
        hidden_units: 4,
        n_hidden_layers: 1,
        ..Default::default()
    })
    .unwrap();
    let before = serde_json::to_value(&model).unwrap();
    assert!(model.fit_nll(&vec![vec![f64::NAN]], &[0.0], 1).is_err());
    assert!(
        model
            .fit_nll(&vec![vec![0.0]], &[f64::INFINITY], 1)
            .is_err()
    );
    assert!(model.fit_nll(&vec![], &[], 1).is_err());
    assert!(
        model
            .fit_replay(&vec![vec![0.0]], &[0.0], &[-1.0], 1, true)
            .is_err()
    );
    assert!(
        model
            .fit_nll(&vec![vec![f64::MAX]], &[f64::MAX], 2)
            .is_err()
    );
    assert_eq!(before, serde_json::to_value(&model).unwrap());
    assert!(model.predict_samples(&vec![vec![0.0]], 0).is_err());
}

#[test]
fn bnn_load_rejects_bad_checkpoint_shapes() {
    let model = Bnn::new(BnnConfig {
        input_dim: 1,
        hidden_units: 4,
        n_hidden_layers: 1,
        ..Default::default()
    })
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.json");
    model.save(&path).unwrap();
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["model"]["net"]["layers"][0]["weights"] = serde_json::json!([]);
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(Bnn::load(&path).is_err());
}

#[test]
fn grid_and_sobol_are_deterministic_and_well_distributed() {
    assert_eq!(
        regular_grid(2, 2).unwrap(),
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0]
        ]
    );
    for dim in 1..=16 {
        let x = sobol_points(256, dim, 3).unwrap();
        assert_eq!(x, sobol_points(256, dim, 3).unwrap());
        for d in 0..dim {
            assert!(x.iter().all(|row| (0.0..1.0).contains(&row[d])));
            let mean = x.iter().map(|row| row[d]).sum::<f64>() / x.len() as f64;
            assert!((mean - 0.5).abs() < 0.003);
            let mut bins = vec![false; 256];
            for row in &x {
                bins[(row[d] * 256.0) as usize] = true;
            }
            assert!(bins.iter().all(|v| *v), "dim={dim} d={d}");
        }
    }
    assert!(regular_grid(1000, 10).is_err());
    assert!(sobol_points(1, 17, 0).is_err());
}

#[test]
fn adaptive_candidates_find_peak_and_validate_scores() {
    let set = AdaptiveSet::build(1, 5, 2, 32, None, 11).unwrap();
    let best = set
        .maximize(|x| Ok(x.iter().map(|p| -(p[0] - 0.37).powi(2)).collect()))
        .unwrap();
    assert!((best[0] - 0.37).abs() < 0.02);
    let (points, scores) = set.candidates(|x| Ok(vec![0.0; x.len()])).unwrap();
    assert_eq!(points.len(), set.size());
    assert_eq!(scores.len(), set.size());
    assert!(set.candidates(|_| Ok(vec![0.0])).is_err());
    assert!(set.candidates(|x| Ok(vec![f64::NAN; x.len()])).is_err());
    assert!(AdaptiveSet::build(1, 2, 3, 2, None, 0).is_err());
}

#[test]
fn features_and_weight_validation() {
    assert_eq!(
        naf_features(
            &[1.0],
            &[2.0],
            &vec![vec![0.3, 0.4]],
            3,
            10,
            true,
            Some(&[0.25, 0.75])
        )
        .unwrap(),
        vec![vec![1.0, 2.0, 0.3, 0.4, 3.0, 10.0, 0.25, 0.75]]
    );
    assert_eq!(n_naf_features(2, true, 2), 8);
    assert_eq!(n_naf_features(2, false, 2), 6);
    assert!(validate_weights(&[f64::NAN, 1.0], 2).is_err());
    assert!(validate_weights(&[-0.2, 1.2], 2).is_err());
    assert!(naf_features(&[1.0], &[-2.0], &vec![vec![0.0]], 0, 1, true, None).is_err());
    assert!(naf_features(&[1.0], &[2.0], &vec![vec![0.0]], 2, 1, true, None).is_err());
}

#[test]
fn gae_has_correct_terminal_boundaries() {
    let result = gae_advantages(&[1.0, 2.0, 3.0, 4.0], &[0.2, 0.3, 0.4, 0.5], 2, 0.9, 0.8).unwrap();
    assert!((result[0] - (1.0 + 0.9 * 0.3 - 0.2 + 0.9 * 0.8 * (2.0 - 0.3))).abs() < 1e-12);
    assert_eq!(result[1], 1.7);
    assert_eq!(result[3], 3.5);
    assert!(gae_advantages(&[], &[], 0, 1.0, 1.0).is_err());
    assert!(gae_advantages(&[1.0], &[1.0], 2, 1.0, 1.0).is_err());
}

#[test]
fn ppo_gradients_match_finite_differences() {
    let logits = vec![vec![0.1, 0.3, -0.2], vec![0.0, -0.3, 0.4]];
    let actions = [1, 0];
    let old = [-1.1, -1.0];
    let adv = [2.0, -1.0];
    let values = [0.3, -0.5];
    let returns = [1.2, 0.5];
    let result = ppo_loss(
        &logits, &actions, &old, &adv, &values, &returns, 0.2, 0.7, 0.03,
    )
    .unwrap();
    for i in 0..2 {
        for j in 0..3 {
            let mut plus = logits.clone();
            let mut minus = logits.clone();
            plus[i][j] += 1e-6;
            minus[i][j] -= 1e-6;
            let lp = ppo_loss(
                &plus, &actions, &old, &adv, &values, &returns, 0.2, 0.7, 0.03,
            )
            .unwrap()
            .loss;
            let lm = ppo_loss(
                &minus, &actions, &old, &adv, &values, &returns, 0.2, 0.7, 0.03,
            )
            .unwrap()
            .loss;
            let numeric = (lp - lm) / 2e-6;
            assert!(
                (numeric - result.logit_gradients[i][j]).abs() < 1e-8,
                "{i}/{j}: numeric={numeric} analytic={}",
                result.logit_gradients[i][j]
            );
        }
        let mut plus = values;
        let mut minus = values;
        plus[i] += 1e-6;
        minus[i] -= 1e-6;
        let lp = ppo_loss(
            &logits, &actions, &old, &adv, &plus, &returns, 0.2, 0.7, 0.03,
        )
        .unwrap()
        .loss;
        let lm = ppo_loss(
            &logits, &actions, &old, &adv, &minus, &returns, 0.2, 0.7, 0.03,
        )
        .unwrap()
        .loss;
        assert!(((lp - lm) / 2e-6 - result.value_gradients[i]).abs() < 1e-8);
    }
}

#[test]
fn ppo_clipping_freezes_saturated_policy_gradient() {
    let result = ppo_loss(
        &vec![vec![0.0, 0.0]],
        &[0],
        &[-3.0],
        &[1.0],
        &[0.0],
        &[0.0],
        0.2,
        0.0,
        0.0,
    )
    .unwrap();
    assert_eq!(result.logit_gradients[0], vec![0.0, 0.0]);
    assert!((result.loss + 1.2).abs() < 1e-12);
    let result = ppo_loss(
        &vec![vec![0.0, -5.0]],
        &[1],
        &[-1.0],
        &[-1.0],
        &[0.0],
        &[0.0],
        0.2,
        0.0,
        0.0,
    )
    .unwrap();
    assert_eq!(result.logit_gradients[0], vec![0.0, 0.0]);
    assert!((result.loss - 0.8).abs() < 1e-12);
}

#[test]
fn policy_checkpoint_and_portable_state_dict_roundtrip() {
    let config = tiny_config();
    let policy = config.new_af().unwrap();
    let dir = tempfile::tempdir().unwrap();
    save_naf(dir.path(), &policy, &config, &gp_config(), "af").unwrap();
    let (restored, restored_config, restored_gp) = load_naf(dir.path(), "af").unwrap();
    assert_eq!(policy, restored);
    assert_eq!(config, restored_config);
    assert_eq!(gp_config(), restored_gp);
    let path = dir.path().join("state_dict.json");
    policy.export_state_dict_json(&path).unwrap();
    let mut other = NafConfig { seed: 8, ..config }.new_af().unwrap();
    assert_ne!(policy, other);
    other.import_state_dict_json(&path).unwrap();
    assert_eq!(policy, other);
    assert!(
        save_naf(
            dir.path(),
            &policy,
            &restored_config,
            &restored_gp,
            "../escape"
        )
        .is_err()
    );
}

#[test]
fn naf_observe_is_transactional_and_enforces_budget() {
    let config = tiny_config();
    let policy = config.new_af().unwrap();
    let mut optimizer = NafOptimizer::new(config, policy, gp_config(), None).unwrap();
    let first = optimizer.suggest().unwrap();
    assert!(optimizer.observe(&vec![f64::NAN], 0.0).is_err());
    assert!(optimizer.observe(&vec![0.5], f64::NAN).is_err());
    assert!(optimizer.observe(&vec![1.1], 0.0).is_err());
    assert_eq!(optimizer.t(), 0);
    assert_eq!(first, optimizer.suggest().unwrap());
    for i in 0..3 {
        optimizer.observe(&vec![i as f64 * 0.3], i as f64).unwrap();
    }
    assert_eq!(optimizer.best().unwrap().1, 2.0);
    assert!(optimizer.suggest().is_err());
    assert!(optimizer.observe(&vec![0.5], 2.0).is_err());
    assert_eq!(optimizer.t(), 3);
}

#[test]
fn native_ppo_updates_policy_and_resume_is_exact() {
    let config = tiny_config();
    let initial = config.new_af().unwrap();
    let tasks = vec![NafTask::new(|x| -(x[0] - 0.37).powi(2), 0.0).unwrap()];
    let full = train_naf(config.clone(), gp_config(), &tasks, None, false).unwrap();
    assert_ne!(full.af, initial);
    assert_eq!(full.mean_return.len(), 2);
    assert!(full.mean_return.iter().all(|v| v.is_finite()));
    let dir = tempfile::tempdir().unwrap();
    let first = train_naf(
        NafConfig {
            n_iterations: 1,
            ..config.clone()
        },
        gp_config(),
        &tasks,
        Some(dir.path()),
        false,
    )
    .unwrap();
    // Best checkpoint must be the evaluated policy, before its first update.
    assert_eq!(first.best_af, initial);
    let resumed = train_naf(config.clone(), gp_config(), &tasks, Some(dir.path()), true).unwrap();
    assert_eq!(full.af, resumed.af);
    assert_eq!(full.best_af, resumed.best_af);
    assert_eq!(full.mean_return, resumed.mean_return);
    assert_eq!(full.stats, resumed.stats);
    assert!(
        train_naf(
            NafConfig { seed: 99, ..config },
            gp_config(),
            &tasks,
            Some(dir.path()),
            true
        )
        .is_err()
    );
}

#[test]
fn short_native_training_learns_a_shared_family_peak() {
    let config = NafConfig {
        budget: 3,
        hidden: vec![16, 16],
        points_per_dim: 11,
        n_local: 24,
        n_local_maxima: 2,
        n_iterations: 100,
        steps_per_iteration: 120,
        minibatch_size: 30,
        epochs: 4,
        learning_rate: 0.003,
        ..tiny_config()
    };
    let tasks: Vec<_> = [0.9, 1.0, 1.1]
        .iter()
        .map(|&scale| {
            NafTask::new(move |x| scale * (1.0 - 20.0 * (x[0] - 0.8).powi(2)), scale).unwrap()
        })
        .collect();
    let initial = NafOptimizer::new(config.clone(), config.new_af().unwrap(), gp_config(), None)
        .unwrap()
        .suggest()
        .unwrap()[0];
    let result = train_naf(config.clone(), gp_config(), &tasks, None, false).unwrap();
    let learned = NafOptimizer::new(config, result.af, gp_config(), None)
        .unwrap()
        .suggest()
        .unwrap()[0];
    assert!(
        (learned - 0.8).abs() < 0.08,
        "initial={initial}, learned={learned}"
    );
    let early = result.mean_final_log_regret[..20].iter().sum::<f64>() / 20.0;
    let late = result.mean_final_log_regret[80..].iter().sum::<f64>() / 20.0;
    assert!(
        late < early - 0.5,
        "initial={initial}, learned={learned}, early={early}, late={late}"
    );
}

#[test]
fn weighted_policy_and_fresh_task_sampler_train() {
    let config = NafConfig {
        n_weights: 2,
        n_iterations: 1,
        ..tiny_config()
    };
    let tasks = [NafTask::new(|x| -(x[0] - 0.4).powi(2), 0.0)
        .unwrap()
        .with_weights(vec![0.3, 0.7])
        .unwrap()];
    let result = train_naf_with_sampler(
        config.clone(),
        gp_config(),
        |n, _| Ok(vec![tasks[0].clone(); n]),
        None,
        false,
    )
    .unwrap();
    let mut optimizer =
        NafOptimizer::new(config, result.af, gp_config(), Some(vec![0.3, 0.7])).unwrap();
    let point = optimizer.suggest().unwrap();
    optimizer.observe(&point, 0.0).unwrap();
    assert_eq!(optimizer.t(), 1);
}

#[test]
fn parallel_rollouts_are_complete_deterministic_and_resumable() {
    let config = NafConfig {
        n_workers: 3,
        steps_per_iteration: 12,
        ..tiny_config()
    };
    let tasks = vec![NafTask::new(|x| -(x[0] - 0.7).powi(2), 0.0).unwrap()];
    let first = train_naf(config.clone(), gp_config(), &tasks, None, false).unwrap();
    let second = train_naf(config.clone(), gp_config(), &tasks, None, false).unwrap();
    assert_eq!(first.af, second.af);
    assert_eq!(first.mean_return, second.mean_return);
    let dir = tempfile::tempdir().unwrap();
    train_naf(
        NafConfig {
            n_iterations: 1,
            ..config.clone()
        },
        gp_config(),
        &tasks,
        Some(dir.path()),
        false,
    )
    .unwrap();
    let resumed = train_naf(config, gp_config(), &tasks, Some(dir.path()), true).unwrap();
    assert_eq!(first.af, resumed.af);
    assert_eq!(first.stats, resumed.stats);
}

#[test]
fn replay_filters_and_averages_qualifying_users() {
    let x = vec![vec![0.0], vec![0.5], vec![1.0]];
    let mut gp1 = GaussianProcess::new(gp_config()).unwrap();
    gp1.fit(&x, &[0.0, 1.0, 0.0]).unwrap();
    let mut gp2 = GaussianProcess::new(gp_config()).unwrap();
    gp2.fit(&x, &[2.0, 3.0, 2.0]).unwrap();
    let targets = replay_targets(&[gp1.clone(), gp2.clone()], &x, None).unwrap();
    assert_eq!(targets.n_users, vec![2; 3]);
    assert_eq!(targets.kept_fraction, vec![1.0; 2]);
    assert!(targets.excluded_users.is_empty());
    assert!((targets.mean[1] - 2.0).abs() < 1e-6);
    let filtered = replay_targets(
        &[gp1.clone(), gp2.clone()],
        &vec![vec![0.0], vec![0.25]],
        Some(1e-5),
    )
    .unwrap();
    assert_eq!(filtered.x, vec![vec![0.0]]);
    assert_eq!(filtered.n_users, vec![2]);
    let excluded = replay_targets(&[gp1.clone()], &vec![vec![0.25]], Some(1e-20)).unwrap();
    assert!(excluded.x.is_empty());
    assert_eq!(excluded.excluded_users, vec![0]);
    assert!(standardized_variance_threshold(&[gp1, gp2], &x, 0.5).unwrap() > 0.0);
    assert!(replay_targets(&[], &x, None).is_err());
    assert!(replay_targets(&[], &x, Some(f64::NAN)).is_err());
    assert_eq!(replay_inputs(2, 3, 2, 0).unwrap().len(), 11);
    assert_eq!(replay_inputs(4, 3, 2, 0).unwrap().len(), 11);
}
