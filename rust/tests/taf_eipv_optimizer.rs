//! TAF-EIPV optimizer integration against independently assembled native terms.
use openbo::{
    Matrix, Point, acquisition as acq,
    core::{self, GpConfig},
    optimizers::*,
};

fn config() -> OptimizerConfig {
    OptimizerConfig {
        bounds: vec![(0.0, 1.0); 2],
        reference_point: Some(vec![-2.0; 2]),
        upper_reference: Some(vec![2.0; 2]),
        n_init: 0,
        n_iter: 5,
        n_candidates: 32,
        n_starts: 2,
        refine: false,
        gp: GpConfig {
            optimize_hyperparameters: false,
            optimize_noise: false,
            ..Default::default()
        },
        ..OptimizerConfig::taf_eipv()
    }
}
fn source(name: &str, opposite: bool) -> Source {
    let x: Matrix = vec![
        vec![0., 0.],
        vec![0.2, 0.7],
        vec![0.4, 0.5],
        vec![0.8, 0.3],
        vec![1., 1.],
        vec![1., 0.],
    ];
    let sign = if opposite { -1.0 } else { 1.0 };
    let y: Matrix = x
        .iter()
        .map(|p| {
            vec![
                sign * (1.0 - (p[0] - 0.2_f64).powi(2) - (p[1] - 0.7_f64).powi(2)),
                sign * (1.0 - (p[0] - 0.8_f64).powi(2) - (p[1] - 0.3_f64).powi(2)),
            ]
        })
        .collect();
    let gps = fit_gps(
        &GpConfig {
            lengthscale: vec![0.35, 0.6],
            optimize_hyperparameters: false,
            ..Default::default()
        },
        &x,
        &y,
    )
    .unwrap();
    Source {
        name: name.into(),
        gps,
        meta_features: meta_features(&x, &y),
        x,
        y,
    }
}
fn objective(x: &Point) -> Vec<f64> {
    vec![
        -(x[0] - 0.3).powi(2) - (x[1] - 0.7).powi(2),
        -(x[0] - 0.8).powi(2) - (x[1] - 0.2).powi(2),
    ]
}
fn pool(config: &OptimizerConfig, n_observed: usize) -> Matrix {
    let seed = config
        .seed
        .wrapping_add((n_observed as u64).wrapping_mul(1009));
    core::low_discrepancy(config.bounds.len(), config.n_candidates, seed)
        .unwrap()
        .iter()
        .map(|p| {
            p.iter()
                .zip(&config.bounds)
                .map(|(v, (lo, hi))| lo + (hi - lo) * v)
                .collect()
        })
        .collect()
}

#[test]
fn source_only_first_point_maximizes_predicted_hypervolume_in_the_pool() {
    let mut optimizer = SequentialOptimizer::new(config()).unwrap();
    optimizer.sources = vec![source("aligned", false), source("opposite", true)];
    let candidates = pool(&optimizer.config, 0);
    let cells = acq::reference_cells(&vec![], &[-2.; 2], &[2.; 2]).unwrap();
    let terms = optimizer
        .sources
        .iter()
        .map(|s| {
            acq::source_improvement(&predict_gps(&s.gps, &candidates).unwrap().0, &cells).unwrap()
        })
        .collect();
    let values = acq::log_taf_eipv(None, &terms, &[0.75, 0.75], 0.75).unwrap();
    let expected = candidates[argmax(&values).unwrap()].clone();
    let suggested = optimizer.suggest().unwrap();
    assert_eq!(suggested, expected);
    assert_eq!(optimizer.last_source_weights, vec![0.75, 0.75]);
    optimizer
        .observe(&suggested, &objective(&suggested))
        .unwrap();
    let report = optimizer.result().unwrap();
    assert_eq!(
        report.metadata["sources"],
        serde_json::json!(["aligned", "opposite"])
    );
    assert_eq!(
        report.metadata["last_source_weights"],
        serde_json::json!([0.75, 0.75])
    );
    assert_eq!(
        report.metadata["upstream_revision"],
        openbo::UPSTREAM_REVISION
    );
    assert_eq!(
        report.metadata["suggestion_seconds"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(
        report.metadata["total_suggestion_seconds"]
            .as_f64()
            .unwrap()
            >= 0.
    );
}

#[test]
fn no_source_taf_eipv_is_exactly_native_baseline_including_ceipv_screening() {
    for (acquisition, model, rescore) in [
        ("ieipv", "independent", None),
        ("ieipv", "mt", None),
        ("ceipv", "slf", Some(4)),
    ] {
        let cfg = OptimizerConfig {
            n_init: 2,
            n_iter: 2,
            n_candidates: 12,
            acquisition: acquisition.into(),
            model: model.into(),
            n_rescore: rescore,
            ..config()
        };
        let mut transfer = SequentialOptimizer::new(cfg.clone()).unwrap();
        let mut baseline = SequentialOptimizer::new(OptimizerConfig {
            method: "ieipv".into(),
            ..cfg
        })
        .unwrap();
        while !baseline.is_finished() {
            let x = baseline.suggest().unwrap();
            let transferred = transfer.suggest().unwrap();
            assert_eq!(x, transferred);
            let y = objective(&x);
            baseline.observe(&x, &y).unwrap();
            transfer.observe(&transferred, &y).unwrap();
            assert!(transfer.last_source_weights.is_empty());
        }
        assert_eq!(baseline.y_obs, transfer.y_obs);
        assert_eq!(
            baseline.result().unwrap().hypervolume_history,
            transfer.result().unwrap().hypervolume_history
        );
    }
}

#[test]
fn blend_uses_source_predictions_at_target_designs_as_reference_front() {
    let mut optimizer = SequentialOptimizer::new(OptimizerConfig {
        taf_weight_mode: "uniform".into(),
        ..config()
    })
    .unwrap();
    optimizer.sources = vec![source("previous", false)];
    for x in [vec![0., 0.], vec![1., 1.]] {
        optimizer.observe(&x, &objective(&x)).unwrap();
    }
    let candidates = pool(&optimizer.config, optimizer.x_obs.len());
    let mut target = acq::MultiOutputGP::new("independent", optimizer.config.gp.noise).unwrap();
    target.optimize_hyperparameters = false;
    target.optimize_noise = false;
    target.fit(&optimizer.x_obs, &optimizer.y_obs).unwrap();
    let (mean, cov) = target.predict(&candidates).unwrap();
    let cells = acq::reference_cells(&optimizer.y_obs, &[-2.; 2], &[2.; 2]).unwrap();
    let log_target: Vec<f64> = mean
        .iter()
        .zip(&cov)
        .map(|(m, c)| {
            acq::ieipv(m, c, &cells.0, &cells.1, acq::EipvForm::Exact)
                .unwrap()
                .ln()
                .max(acq::LOG_FLOOR)
        })
        .collect();
    let source = &optimizer.sources[0];
    let predicted_reference = predict_gps(&source.gps, &optimizer.x_obs).unwrap().0;
    let source_cells = acq::reference_cells(&predicted_reference, &[-2.; 2], &[2.; 2]).unwrap();
    let mean = predict_gps(&source.gps, &candidates).unwrap().0;
    let terms = acq::source_improvement(&mean, &source_cells).unwrap();
    let wrong_cells = acq::reference_cells(&source.y, &[-2.; 2], &[2.; 2]).unwrap();
    let wrong = acq::source_improvement(&mean, &wrong_cells).unwrap();
    assert!(terms.iter().zip(wrong).any(|(a, b)| (a - b).abs() > 1e-4));
    let scores = acq::log_taf_eipv(Some(&log_target), &vec![terms], &[0.75], 0.75).unwrap();
    let expected = candidates[argmax(&scores).unwrap()].clone();
    assert_eq!(optimizer.suggest().unwrap(), expected);
    assert_eq!(optimizer.last_source_weights, vec![0.75]);
}

#[test]
fn pooled_ranking_weights_are_raw_and_uniform_mode_keeps_every_source() {
    for mode in ["taf_r", "uniform"] {
        let mut optimizer = SequentialOptimizer::new(OptimizerConfig {
            taf_weight_mode: mode.into(),
            ..config()
        })
        .unwrap();
        optimizer.sources = vec![source("aligned", false), source("reversed", true)];
        for x in [vec![0.1, 0.2], vec![0.3, 0.6], vec![0.9, 0.1]] {
            let y = predict_gps(&optimizer.sources[0].gps, &vec![x.clone()])
                .unwrap()
                .0
                .remove(0);
            optimizer.observe(&x, &y).unwrap();
        }
        optimizer.suggest().unwrap();
        assert_eq!(
            optimizer.last_source_weights,
            if mode == "taf_r" {
                vec![0.75, 0.0]
            } else {
                vec![0.75, 0.75]
            }
        );
    }
}

#[test]
fn observed_count_decay_reaches_target_only_fallback() {
    let mut optimizer = SequentialOptimizer::new(OptimizerConfig {
        decay: true,
        decay_start_iter: 1,
        decay_rate: 0.5,
        ..config()
    })
    .unwrap();
    optimizer.sources = vec![source("earlier", false)];
    for expected in [0.75, 0.375, 0.0] {
        let x = optimizer.suggest().unwrap();
        assert_eq!(optimizer.last_source_weights, vec![expected]);
        optimizer.observe(&x, &objective(&x)).unwrap();
    }
    assert_eq!(optimizer.result().unwrap().hypervolume_history.len(), 3);
}

#[test]
fn native_eipv_refines_sources_and_averages_slice_samples() {
    let mut optimizer = SequentialOptimizer::new(OptimizerConfig {
        refine: true,
        n_starts: 1,
        n_candidates: 8,
        n_iter: 3,
        hyperparameters: "slice".into(),
        n_hyper_samples: 2,
        slice_burn_in: 0,
        slice_thin: 1,
        ..config()
    })
    .unwrap();
    optimizer.sources = vec![source("earlier", false)];
    for _ in 0..3 {
        let x = optimizer.suggest().unwrap();
        assert!(x.iter().all(|v| (0.0..=1.0).contains(v)));
        optimizer.observe(&x, &objective(&x)).unwrap();
    }
    assert!(
        optimizer
            .result()
            .unwrap()
            .hypervolume_history
            .windows(2)
            .all(|v| v[1] >= v[0])
    );
}

#[test]
fn both_source_json_layouts_load_and_original_source_front_is_not_filtered() {
    let directory = tempfile::tempdir().unwrap();
    let src = source("previous", false);
    src.save(directory.path()).unwrap();
    let mut optimizer = SequentialOptimizer::new(OptimizerConfig {
        taf_run_dir: Some(directory.path().into()),
        reference_point: Some(vec![10.; 2]),
        upper_reference: Some(vec![20.; 2]),
        ..config()
    })
    .unwrap();
    assert_eq!(optimizer.sources.len(), 1);
    assert!(optimizer.suggest().is_ok());
    assert_eq!(optimizer.last_source_weights, vec![0.75]);
    let state = serde_json::json!({"gp_states":src.gps.iter().map(|gp|gp.source_state().unwrap()["gp_state"].clone()).collect::<Vec<_>>()});
    std::fs::write(
        directory.path().join("gp_states/previous.json"),
        serde_json::to_vec(&state).unwrap(),
    )
    .unwrap();
    let mut loaded = SequentialOptimizer::new(OptimizerConfig {
        taf_run_dir: Some(directory.path().into()),
        ..config()
    })
    .unwrap();
    assert_eq!(loaded.sources.len(), 1);
    assert!(loaded.suggest().is_ok());
}

#[test]
fn eipv_configuration_and_empty_target_state_are_validated() {
    assert!(METHODS.contains(&"taf_eipv"));
    assert!(METHODS.contains(&"mobo_taf_eipv"));
    assert!(METHODS.contains(&"ieipv"));
    assert_eq!(OptimizerConfig::taf_eipv().target_weight, 0.75);
    for bad in [
        OptimizerConfig {
            upper_reference: None,
            ..config()
        },
        OptimizerConfig {
            taf_weight_mode: "taf_m".into(),
            ..config()
        },
        OptimizerConfig {
            target_weight: 0.,
            ..config()
        },
        OptimizerConfig {
            acquisition: "ehvi".into(),
            ..config()
        },
        OptimizerConfig {
            eipv_form: "unknown".into(),
            ..config()
        },
        OptimizerConfig {
            n_rescore: Some(0),
            ..config()
        },
        OptimizerConfig {
            n_hyper_samples: 0,
            ..config()
        },
        OptimizerConfig {
            slice_thin: 0,
            ..config()
        },
        OptimizerConfig {
            decay: true,
            decay_rate: 0.0,
            ..config()
        },
        OptimizerConfig {
            bounds: vec![(0., 1.); 33],
            ..config()
        },
    ] {
        assert!(SequentialOptimizer::new(bad).is_err());
    }
    let mut empty = SequentialOptimizer::new(config()).unwrap();
    assert!(empty.suggest().is_err());
    assert!(empty.pending().is_none());
    assert_eq!(
        empty.result().unwrap().metadata["suggestion_seconds"],
        serde_json::json!([])
    );
    empty.observe(&[0.5, 0.5], &[0., 0.]).unwrap();
    assert!(empty.suggest().is_ok());
    let mut decayed = SequentialOptimizer::new(OptimizerConfig {
        decay: true,
        decay_start_iter: 0,
        decay_rate: 1.0,
        ..config()
    })
    .unwrap();
    decayed.sources = vec![source("earlier", false)];
    assert!(
        decayed
            .suggest()
            .unwrap_err()
            .to_string()
            .contains("positive weight")
    );
    assert!(decayed.pending().is_none());
}
