//! Integration validation of ask/tell, transfer fallback and persisted populations.
use openbo::{
    Matrix,
    core::{self, GaussianProcess, GpConfig},
    optimizers::*,
};
use serde_json::json;

fn config(method: &str) -> OptimizerConfig {
    OptimizerConfig {
        method: method.into(),
        n_init: 2,
        n_iter: 3,
        n_candidates: 16,
        mc_samples: 8,
        refine: false,
        gp: GpConfig {
            optimize_hyperparameters: false,
            ..Default::default()
        },
        ..Default::default()
    }
}
fn source(name: &str, n_objectives: usize) -> Source {
    let x = vec![vec![0., 0.], vec![0.25, 0.8], vec![0.7, 0.2], vec![1., 1.]];
    let y: Matrix = x
        .iter()
        .map(|p| {
            (0..n_objectives)
                .map(|j| {
                    10. * j as f64 - (p[0] - 0.2 * (j + 1) as f64).powi(2) - (p[1] - 0.7).powi(2)
                })
                .collect()
        })
        .collect();
    let gps = (0..n_objectives)
        .map(|j| {
            let mut gp = GaussianProcess::new(GpConfig {
                lengthscale: vec![0.3, 0.6],
                variance: 1.7,
                noise: 1e-4,
                mean_constant: 0.25 + j as f64,
                fixed_target_mean: Some(j as f64 + 0.3),
                fixed_target_std: Some(j as f64 + 1.1),
                ..Default::default()
            })
            .unwrap();
            gp.fit(&x, &y.iter().map(|r| r[j]).collect::<Vec<_>>())
                .unwrap();
            gp
        })
        .collect();
    Source {
        name: name.into(),
        meta_features: meta_features(&x, &y),
        gps,
        x,
        y,
    }
}
fn feed(optimizer: &mut SequentialOptimizer) {
    while !optimizer.is_finished() {
        let x = optimizer.suggest().unwrap();
        let y = if optimizer.config.n_objectives() == 1 {
            core::synthetic("sphere", &vec![x.clone()]).unwrap()
        } else {
            let w = optimizer.config.n_objectives();
            (0..w)
                .map(|j| -(x[0] - 0.2 * j as f64).powi(2) - (x[1] - 0.7).powi(2))
                .collect()
        };
        optimizer.observe(&x, &y).unwrap();
    }
}

#[test]
fn scratch_run_is_reproducible_and_has_one_history_entry_per_evaluation() {
    let mut first = SequentialOptimizer::new(config("bo_scratch")).unwrap();
    feed(&mut first);
    let result = first.result().unwrap();
    assert_eq!(result.x_values.len(), 5);
    assert_eq!(result.y_values.len(), 5);
    assert_eq!(result.best_so_far.len(), 5);
    assert!(result.best_so_far.windows(2).all(|v| v[1] >= v[0]));
    assert!(result.y_values.iter().flatten().all(|v| v.is_finite()));
    assert!(first.suggest().is_err());
    let mut duplicate = SequentialOptimizer::new(config("bo_scratch")).unwrap();
    feed(&mut duplicate);
    assert_eq!(duplicate.x_obs, first.x_obs);
    assert_eq!(duplicate.y_obs, first.y_obs);
    for method in [
        "random",
        "bo_scratch_grid",
        "bo_scratch_multistart",
        "bo_botorch",
    ] {
        let mut opt = SequentialOptimizer::new(config(method)).unwrap();
        feed(&mut opt);
        assert_eq!(opt.y_obs.len(), 5);
    }
}

#[test]
fn pending_observation_validation_is_atomic_and_recoverable() {
    let mut optimizer = SequentialOptimizer::new(config("bo_scratch")).unwrap();
    let x = optimizer.suggest().unwrap();
    assert!(optimizer.suggest().is_err());
    assert_eq!(optimizer.pending(), Some(&x));
    for (bad_x, bad_y) in [
        (x.clone(), vec![f64::NAN]),
        (vec![0.5], vec![1.]),
        (vec![2., 2.], vec![1.]),
        (x.clone(), vec![1., 2.]),
    ] {
        assert!(optimizer.observe(&bad_x, &bad_y).is_err());
        assert!(optimizer.y_obs.is_empty());
        assert_eq!(optimizer.pending(), Some(&x));
    }
    let different = vec![(x[0] + 0.5) % 1., x[1]];
    assert!(optimizer.observe(&different, &[1.]).is_err());
    optimizer.observe(&x, &[1.]).unwrap();
    assert!(optimizer.pending().is_none());
    assert_eq!(optimizer.y_obs, vec![vec![1.]]);
}

#[test]
fn invalid_configuration_is_rejected_at_construction() {
    for bad in [
        OptimizerConfig {
            gp: GpConfig {
                noise: -1.,
                ..Default::default()
            },
            ..config("bo_scratch")
        },
        OptimizerConfig {
            bounds: vec![(0., f64::INFINITY)],
            ..config("bo_scratch")
        },
        OptimizerConfig {
            n_candidates: 0,
            ..config("bo_scratch")
        },
        OptimizerConfig {
            n_init: 0,
            n_iter: 0,
            ..config("bo_scratch")
        },
        OptimizerConfig {
            source_reference_quantile: f64::NAN,
            ..config("bo_scratch")
        },
        OptimizerConfig {
            reference_point: Some(vec![f64::NAN; 2]),
            ..config("mobo_ehvi")
        },
        OptimizerConfig {
            objective_weights: Some(vec![0.5, f64::NAN]),
            ..config("taf_plus")
        },
    ] {
        assert!(SequentialOptimizer::new(bad).is_err());
    }
    assert!(
        fit_gps(
            &GpConfig::default(),
            &vec![vec![0.], vec![1.]],
            &vec![vec![1., 2.], vec![3.]]
        )
        .is_err()
    );
}

#[test]
fn source_roundtrip_replays_outcome_scaling_prior_mean_and_each_objective() {
    for objectives in [1, 3] {
        let directory = tempfile::tempdir().unwrap();
        let src = source("user_1", objectives);
        src.save(directory.path()).unwrap();
        src.save(directory.path()).unwrap();
        let (loaded, warnings) = load_sources(directory.path(), 2, objectives).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].x, src.x);
        assert_eq!(loaded[0].y, src.y);
        let q = core::sample(&[(0., 1.); 2], 20, 42).unwrap();
        for j in 0..objectives {
            assert_eq!(
                loaded[0].gps[j].predict_joint(&q).unwrap(),
                src.gps[j].predict_joint(&q).unwrap()
            );
        }
        let invalid = directory.path().join("gp_states/corrupt.json");
        std::fs::write(invalid, "{}").unwrap();
        let (loaded, warnings) = load_sources(directory.path(), 2, objectives).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(warnings.len(), 1);
        let (loaded, warnings) = load_sources(directory.path(), 3, objectives).unwrap();
        assert!(loaded.is_empty());
        assert!(!warnings.is_empty());
    }
}

#[test]
fn invalid_public_source_cannot_panic_or_save_mismatched_trajectory() {
    let directory = tempfile::tempdir().unwrap();
    let mut src = source("valid", 1);
    src.y[0].clear();
    assert!(src.save(directory.path()).is_err());
    let mut src = source("valid", 1);
    src.x[0][0] = 0.1;
    assert!(src.save(directory.path()).is_err());
    let mut src = source("valid", 1);
    src.gps.clear();
    assert!(src.save(directory.path()).is_err());
    let src = source("../unsafe", 1);
    assert!(src.save(directory.path()).is_err());
}

#[test]
fn transfer_without_sources_reduces_to_target_search_and_supports_zero_initial_points() {
    let mut baseline = SequentialOptimizer::new(config("bo_scratch")).unwrap();
    feed(&mut baseline);
    for method in ["bo_taf", "bo_taf_m", "bo_taf_r"] {
        let mut transferred = SequentialOptimizer::new(config(method)).unwrap();
        feed(&mut transferred);
        assert_eq!(transferred.x_obs, baseline.x_obs);
        assert_eq!(transferred.y_obs, baseline.y_obs);
    }
    let mut zero = SequentialOptimizer::new(OptimizerConfig {
        n_init: 0,
        n_iter: 3,
        ..config("bo_taf")
    })
    .unwrap();
    feed(&mut zero);
    assert_eq!(zero.y_obs.len(), 3);
}

#[test]
fn transfer_warmup_uses_source_even_when_target_scale_changes() {
    let mut a = SequentialOptimizer::new(OptimizerConfig {
        source_only_warmup_iters: 2,
        target_weight: 100.,
        ..config("bo_taf")
    })
    .unwrap();
    let mut b = SequentialOptimizer::new(a.config.clone()).unwrap();
    a.sources = vec![source("source", 1)];
    b.sources = a.sources.clone();
    for (p, y) in [(vec![0.1, 0.1], -2.), (vec![0.9, 0.9], 1.)] {
        a.observe(&p, &[y]).unwrap();
        b.observe(&p, &[100. * y + 27.]).unwrap();
    }
    assert_eq!(a.suggest().unwrap(), b.suggest().unwrap());
    assert!(!a.last_source_weights.is_empty());
}

#[test]
fn multiobjective_fallback_and_correlated_methods_produce_finite_hypervolume() {
    let reference = Some(vec![-4.; 2]);
    let mut baseline = SequentialOptimizer::new(OptimizerConfig {
        reference_point: reference.clone(),
        n_init: 1,
        n_iter: 2,
        ..config("mobo_botorch")
    })
    .unwrap();
    feed(&mut baseline);
    for method in ["mobo_taf", "taf_ehvi"] {
        let mut fallback = SequentialOptimizer::new(OptimizerConfig {
            reference_point: reference.clone(),
            n_init: 1,
            n_iter: 2,
            ..config(method)
        })
        .unwrap();
        feed(&mut fallback);
        assert_eq!(fallback.x_obs, baseline.x_obs);
        let report = fallback.result().unwrap();
        assert_eq!(report.hypervolume_history.len(), 3);
        assert!(report.hypervolume_history.windows(2).all(|h| h[1] >= h[0]));
    }
    for method in ["mobo_ehvi", "mobo_ceipv_slf", "mobo_ceipv_mt", "mobo_ieipv"] {
        let mut optimizer = SequentialOptimizer::new(OptimizerConfig {
            reference_point: reference.clone(),
            upper_reference: Some(vec![1.; 2]),
            n_init: 1,
            n_iter: 1,
            n_candidates: 4,
            mc_samples: 4,
            ..config(method)
        })
        .unwrap();
        feed(&mut optimizer);
        assert!(
            optimizer
                .result()
                .unwrap()
                .hypervolume_history
                .iter()
                .all(|v| v.is_finite() && *v >= 0.)
        );
    }
}

#[test]
fn taf_plus_handles_empty_and_loaded_populations() {
    let weights = Some(vec![0.2, 0.3, 0.5]);
    for mode in ["taf_r", "poe"] {
        for has_population in [false, true] {
            let mut optimizer = SequentialOptimizer::new(OptimizerConfig {
                objective_weights: weights.clone(),
                taf_weight_mode: mode.into(),
                n_init: 0,
                n_iter: 3,
                ..config("taf_plus")
            })
            .unwrap();
            if has_population {
                optimizer.sources = vec![source("one", 3), source("two", 3)];
            }
            feed(&mut optimizer);
            let report = optimizer.result().unwrap();
            assert_eq!(report.y_values.len(), 3);
            assert!(report.best_so_far.iter().all(|v| v.is_finite()));
        }
    }
}

#[test]
fn conbo_two_user_checkpoint_replays_gp_and_bnn_and_can_continue() {
    let cfg = ConboConfig {
        n_random: 1,
        n_evaluations: 3,
        n_candidates: 8,
        population_epochs: 2,
        replay_grid_points: 3,
        replay_random_points: 2,
        replay_max_variance: Some(1e9),
        gp: GpConfig {
            optimize_hyperparameters: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut population = ConboPopulation::new(cfg).unwrap();
    for user_name in ["first", "second"] {
        let mut user = population.new_user(user_name).unwrap();
        for _ in 0..3 {
            let x = user.suggest().unwrap();
            let y = core::synthetic("sphere", &vec![x.clone()]).unwrap()[0];
            user.observe(&x, y).unwrap();
        }
        let report = population.finish_user(&user).unwrap();
        assert!(report["n_inputs_kept"].as_u64().unwrap() > 0);
    }
    let dir = tempfile::tempdir().unwrap();
    population.save(dir.path()).unwrap();
    population.save(dir.path()).unwrap();
    let mut restored = ConboPopulation::load(dir.path()).unwrap();
    assert_eq!(restored.users.len(), 2);
    let q = core::sample(&[(0., 1.); 2], 8, 42).unwrap();
    for i in 0..2 {
        assert_eq!(
            population.users[i].gps[0].predict(&q).unwrap(),
            restored.users[i].gps[0].predict(&q).unwrap()
        );
    }
    assert_eq!(
        population.bnn.as_ref().unwrap().predict(&q).unwrap(),
        restored.bnn.as_ref().unwrap().predict(&q).unwrap()
    );
    assert_eq!(
        population.new_user("third").unwrap().suggest().unwrap(),
        restored.new_user("third").unwrap().suggest().unwrap()
    );
    restored.config.n_random = 0;
    restored.config.alpha1 = 0.;
    let mut cold = restored.new_user("cold").unwrap();
    let x = cold.suggest().unwrap();
    assert!(x.iter().all(|v| v.is_finite()));
    cold.observe(&x, 1.).unwrap();
    assert!(cold.suggest().is_ok());
    let path = dir.path().join("conbo_rust_state.json");
    let mut payload: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    payload["users"] = json!(["first", "first"]);
    std::fs::write(path, serde_json::to_vec(&payload).unwrap()).unwrap();
    assert!(ConboPopulation::load(dir.path()).is_err());
}

#[test]
fn objective_weight_workflow_matches_paper_grid_and_table_one() {
    let grid = simplex_grid(3, 10).unwrap();
    assert_eq!(grid.len(), 36);
    assert!(grid.iter().all(
        |r| r.iter().all(|v| *v >= 0.1 - 1e-12) && (r.iter().sum::<f64>() - 1.0).abs() < 1e-12
    ));
    assert_eq!(simplex_grid(1, 10).unwrap(), vec![vec![1.0]]);
    assert_eq!(simplex_grid(3, 3).unwrap(), vec![vec![1.0 / 3.0; 3]]);
    assert!(simplex_grid(3, 2).is_err());
    let trajectories = vec![vec![
        vec![7.0, 2.0, 5.0],
        vec![5.0, 4.0, 8.0],
        vec![3.0, 9.0, 2.0],
    ]];
    let candidates = vec![
        vec![0.7, 0.2, 0.1],
        vec![0.2, 0.3, 0.5],
        vec![0.2, 0.6, 0.2],
    ];
    let (best, scores) =
        optimize_objective_weights(&trajectories, &[vec![75.0, 100.0, 1.0]], &candidates).unwrap();
    assert_eq!(best, vec![0.2, 0.3, 0.5]);
    assert_eq!(scores, vec![75.0, 100.0, 1.0]);
}

#[test]
fn objective_weight_workflow_rejects_ragged_candidates_and_nonfinite_ratings() {
    let trajectories = vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]];
    assert!(
        optimize_objective_weights(
            &trajectories,
            &[vec![100.0, 1.0]],
            &vec![vec![0.5, 0.5], vec![1.0]]
        )
        .is_err()
    );
    assert!(
        optimize_objective_weights(&trajectories, &[vec![f64::NAN, 1.0]], &vec![vec![0.5, 0.5]])
            .is_err()
    );
    assert!(
        optimize_objective_weights(&trajectories, &[vec![1.0]], &vec![vec![0.5, 0.5]]).is_err()
    );
    assert!(rescale_ratings(&[]).is_err());
    assert_eq!(
        rescale_ratings(&[-1e308, 0.0, 1e308]).unwrap(),
        vec![1.0, 50.5, 100.0]
    );
    assert_eq!(rescale_ratings(&[-7.0, -7.0]).unwrap(), vec![100.0, 100.0]);
}

#[test]
fn empty_conbo_population_checkpoint_roundtrips_and_accepts_first_user() {
    let population = ConboPopulation::new(ConboConfig::default()).unwrap();
    let directory = tempfile::tempdir().unwrap();
    population.save(directory.path()).unwrap();
    let restored = ConboPopulation::load(directory.path()).unwrap();
    assert!(restored.users.is_empty());
    assert!(restored.bnn.is_none());
    let mut original_user = population.new_user("first").unwrap();
    let mut restored_user = restored.new_user("first").unwrap();
    assert_eq!(
        original_user.suggest().unwrap(),
        restored_user.suggest().unwrap()
    );
    assert!(load_sources(directory.path(), 0, 1).is_err());
    assert!(load_sources(directory.path(), 2, 0).is_err());
}
