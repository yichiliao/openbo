use openbo::Matrix;
use openbo::acquisition::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} != {b} within {tol}");
}
fn m(rows: &[&[f64]]) -> Matrix {
    rows.iter().map(|r| r.to_vec()).collect()
}

#[test]
fn ei_closed_form_zero_variance_and_stable_log_tail() {
    close(
        expected_improvement(0.0, 1.0, 0.0).unwrap(),
        1.0 / (2.0 * std::f64::consts::PI).sqrt(),
        1e-14,
    );
    close(
        expected_improvement(1.0, 4.0, 0.5).unwrap(),
        1.0726893964471604,
        1e-13,
    );
    close(
        expected_improvement_minimization(-1.0, 4.0, -0.5).unwrap(),
        expected_improvement(1.0, 4.0, 0.5).unwrap(),
        1e-14,
    );
    assert_eq!(expected_improvement(1.0, 0.0, 0.0).unwrap(), 1.0);
    assert_eq!(expected_improvement(0.0, 0.0, 1.0).unwrap(), 0.0);
    assert_eq!(
        log_expected_improvement(0.0, 0.0, 1.0).unwrap(),
        f64::NEG_INFINITY
    );
    close(
        log_expected_improvement(-20.0, 1.0, 0.0).unwrap(),
        -206.9178385094251,
        1e-10,
    );
    close(
        log_expected_improvement(-100.0, 1.0, 0.0).unwrap(),
        -5010.12957880025,
        1e-9,
    );
    assert!(
        log_expected_improvement(-40.0, 1.0, 0.0)
            .unwrap()
            .is_finite()
    );
    assert!(expected_improvement(1.0, -1.0, 0.0).is_err());
    assert!(expected_improvement(f64::NAN, 1.0, 0.0).is_err());
    assert!(expected_improvement(1e308, 1.0, -1e308).is_err());
    assert!(
        log_expected_improvement(1e308, 1.0, -1e308)
            .unwrap()
            .is_finite()
    );
}

#[test]
fn probability_and_confidence_acquisitions_have_real_validated_formulas() {
    close(
        probability_of_improvement(0.0, 1.0, 0.0).unwrap(),
        0.5,
        1e-14,
    );
    assert_eq!(probability_of_improvement(0.0, 0.0, 0.0).unwrap(), 0.0);
    assert_eq!(probability_of_improvement(1.0, 0.0, 0.0).unwrap(), 1.0);
    close(upper_confidence_bound(1.0, 4.0, 9.0).unwrap(), 7.0, 1e-14);
    assert!(probability_of_improvement(0.0, -1.0, 0.0).is_err());
    assert!(upper_confidence_bound(0.0, 1.0, -1.0).is_err());
}

fn hv_inclusion(y: &Matrix, r: &[f64]) -> f64 {
    let mut out = 0.0;
    for mask in 1usize..1usize << y.len() {
        let corner: Vec<f64> = (0..r.len())
            .map(|k| {
                y.iter()
                    .enumerate()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, row)| row[k])
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let volume = corner
            .iter()
            .zip(r)
            .map(|(a, b)| (a - b).max(0.0))
            .product::<f64>();
        out += if mask.count_ones() % 2 == 1 {
            volume
        } else {
            -volume
        };
    }
    out
}

#[test]
fn exact_hypervolume_matches_independent_inclusion_exclusion_in_one_to_four_dimensions() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    for d in 1..=4 {
        for _ in 0..20 {
            let y: Matrix = (0..7)
                .map(|_| (0..d).map(|_| rng.gen_range(-1.0..2.0)).collect())
                .collect();
            let r = vec![0.0; d];
            close(hypervolume(&y, &r).unwrap(), hv_inclusion(&y, &r), 1e-11);
        }
    }
    let front = pareto_front(&m(&[&[1.0, 2.0], &[2.0, 1.0], &[1.0, 2.0], &[0.0, 0.0]])).unwrap();
    assert_eq!(front, m(&[&[1.0, 2.0], &[2.0, 1.0]]));
    assert!(hypervolume(&m(&[&[f64::NAN, 1.0]]), &[0.0, 0.0]).is_err());
    assert!(hypervolume(&m(&[&[1.0, 1.0]]), &[f64::NEG_INFINITY, 0.0]).is_err());
    assert!(hypervolume(&m(&[&[1.0, 1.0]]), &[0.0]).is_err());
}

#[test]
fn nondominated_cells_conserve_volume_and_exact_candidate_gain() {
    let mut rng = ChaCha8Rng::seed_from_u64(15);
    for d in 1..=3 {
        for _ in 0..15 {
            let y: Matrix = (0..5)
                .map(|_| (0..d).map(|_| rng.gen_range(-0.5..1.0)).collect())
                .collect();
            let r = vec![0.0; d];
            let w = vec![1.0; d];
            let hv = hypervolume(&y, &r).unwrap();
            let (lo, hi) = nondominated_cells(&y, &r, &w).unwrap();
            let free: f64 = lo
                .iter()
                .zip(&hi)
                .map(|(a, b)| a.iter().zip(b).map(|(x, z)| z - x).product::<f64>())
                .sum();
            close(hv + free, 1.0, 1e-12);
            let candidate: Vec<f64> = (0..d).map(|_| rng.gen_range(-0.2..1.0)).collect();
            let mut union = y.clone();
            union.push(candidate.clone());
            close(
                hypervolume_gain(&candidate, &lo, &hi).unwrap(),
                hypervolume(&union, &r).unwrap() - hv,
                1e-12,
            );
        }
    }
    assert!(nondominated_cells(&Vec::new(), &[1.0], &[0.0]).is_err());
}

#[test]
fn ranking_variants_handle_tradeoffs_inversion_ties_and_invalid_modes() {
    let target = m(&[&[0.1, 0.9], &[0.5, 0.5], &[0.9, 0.1]]);
    let inverted: Matrix = target
        .iter()
        .map(|r| r.iter().map(|v| -v).collect())
        .collect();
    let flat = vec![vec![1.0; 2]; 3];
    assert_eq!(
        ranking_weights(
            &target,
            &[target.clone(), inverted.clone(), flat],
            1.0,
            "taf_r"
        )
        .unwrap(),
        vec![1.0, 0.0, 0.0]
    );
    assert_eq!(
        ranking_weights(&target, &[target.clone(), inverted], 1.0, "taf_r_pareto").unwrap(),
        vec![0.0, 0.0]
    );
    assert_eq!(
        ranking_weights(
            &m(&[&[0.0], &[1.0]]),
            &[m(&[&[2.0], &[2.0]])],
            1.0,
            "scalar"
        )
        .unwrap(),
        vec![0.75]
    );
    close(
        ranking_distance(&[0.0, 1.0], &[1.0, 1.0]).unwrap(),
        0.5,
        1e-14,
    );
    assert!(ranking_weights(&target, &[], 1.0, "typo").is_err());
    assert!(ranking_weights(&target, &[m(&[&[1.0]])], 1.0, "taf_r").is_err());
    assert!(ranking_weights(&target, &[], f64::NAN, "taf_r").is_err());
}

#[test]
fn metadata_weights_normalize_and_empty_source_is_well_defined() {
    let w = meta_weights(&m(&[&[0.0, 1.0], &[1.0, 0.0]]), &[0.0, 1.0], 1.5).unwrap();
    assert_eq!(w, vec![1.0, 0.0]);
    assert_eq!(
        meta_weights(&Vec::new(), &[1.0], 1.0).unwrap(),
        Vec::<f64>::new()
    );
    assert_eq!(
        meta_weights(&m(&[&[1.0], &[2.0]]), &[0.0], 0.01).unwrap(),
        vec![0.5, 0.5]
    );
    assert!(meta_weights(&m(&[&[1.0, 2.0]]), &[0.0], 1.0).is_err());
}

#[test]
fn transfer_blends_keep_only_active_sources_and_stable_softplus() {
    close(
        taf_acquisition(
            Some(2.0),
            &[4.0, 100.0],
            &[1.0, 0.0],
            &[1.0, 0.0],
            1.0,
            None,
        )
        .unwrap(),
        2.5,
        1e-14,
    );
    close(
        taf_acquisition(None, &[100.0], &[0.0], &[1.0], 0.0, Some(0.05)).unwrap(),
        100.0,
        1e-14,
    );
    close(
        weighted_log_blend(
            Some(2.0_f64.ln()),
            &[4.0_f64.ln(), 100.0_f64.ln()],
            &[1.0, 0.0],
            1.0,
        )
        .unwrap(),
        3.0_f64.ln(),
        1e-14,
    );
    assert_eq!(
        weighted_log_blend(Some(f64::NEG_INFINITY), &[], &[], 1.0).unwrap(),
        f64::NEG_INFINITY
    );
    assert!(weighted_log_blend(None, &[], &[], 0.0).is_err());
    close(
        weighted_log_blend(Some(2.0_f64.ln()), &[2.0_f64.ln()], &[1e308], 1e308).unwrap(),
        2.0_f64.ln(),
        1e-12,
    );
    assert!(taf_acquisition(None, &[], &[], &[], 0.0, None).is_err());
    close(population_weight(10, 5.0, 0.2).unwrap(), 0.0, 1e-14);
    close(population_weight(1, 5.0, 0.2).unwrap(), 1.0, 1e-14);
    close(decay_factor(3, 2, 0.3).unwrap(), 0.7, 1e-14);
    close(
        mixed_expected_improvement(1.0, Some((0.0, 1.0)), None, 0.0, f64::NAN).unwrap(),
        expected_improvement(0.0, 1.0, 0.0).unwrap(),
        1e-14,
    );
    assert!(population_weight(0, 5.0, 0.2).is_err());
    assert!(population_weight(1, 5.0, f64::NAN).is_err());
}

#[test]
fn taf_plus_equations_and_extreme_precisions_are_finite() {
    let target = m(&[&[1.0, 2.0]]);
    let tv = m(&[&[0.0, 0.0]]);
    let sm = vec![m(&[&[4.0, 4.0]])];
    let sv = vec![m(&[&[0.0, 1e-310]])];
    let reference = m(&[&[1.0, 1.0]]);
    let result = taf_plus_acquisition(TafPlusInput {
        target_mean: &target,
        target_variance: &tv,
        target_best: &[0.0, 0.0],
        source_mean: &sm,
        source_variance: &sv,
        source_reference: &reference,
        objective_weights: &[0.5, 0.5],
        decay: 1.0,
        weight_mode: "poe",
        ranking: None,
    })
    .unwrap();
    close(result[0], 2.25, 1e-14);
    let result = taf_plus_acquisition(TafPlusInput {
        target_mean: &target,
        target_variance: &tv,
        target_best: &[0.0, 0.0],
        source_mean: &sm,
        source_variance: &sv,
        source_reference: &reference,
        objective_weights: &[0.5, 0.5],
        decay: 0.0,
        weight_mode: "poe",
        ranking: None,
    })
    .unwrap();
    close(result[0], 1.5, 1e-14);
    let weights = learn_objective_weights(
        &m(&[&[1.0, 0.0], &[0.0, 1.0], &[2.0, 1.0]]),
        &[0.8, 0.2, 1.8],
        200,
    )
    .unwrap();
    close(weights[0], 0.8, 1e-10);
    close(weights.iter().sum(), 1.0, 1e-14);
}

#[test]
fn taf_plus_population_and_source_reference_helpers_follow_paper_axes() {
    let means = vec![
        m(&[&[1.0, 5.0], &[2.0, 4.0]]),
        m(&[&[4.0, 0.0], &[3.0, 2.0]]),
    ];
    let variances = vec![
        m(&[&[1.0, 1.0], &[1.0, 1.0]]),
        m(&[&[4.0, 4.0], &[4.0, 4.0]]),
    ];
    assert_eq!(
        source_references(&means).unwrap(),
        m(&[&[2.0, 5.0], &[4.0, 2.0]])
    );
    let mean = population_mean(&means, &variances, &[0.5, 0.5], "poe").unwrap();
    close(mean[0], 2.8, 1e-14);
    close(mean[1], 2.9, 1e-14);
    let mean = population_mean(&means, &variances, &[0.5, 0.5], "taf_r").unwrap();
    close(mean[0], 2.5, 1e-14);
    close(mean[1], 2.75, 1e-14);
    let weights = poe_weights(&variances).unwrap();
    close(weights[0][0][0], 0.5, 1e-14);
    close(weights[1][0][0], 0.125, 1e-14);
    let (ranking, target) = taf_plus_ranking_weights(&means[0], &means, 1.0).unwrap();
    assert_eq!(target, 0.75);
    assert_eq!(ranking, m(&[&[0.75, 0.75], &[0.0, 0.0]]));
    assert_eq!(
        default_incumbents(&[1.0, 2.0], &[100.0]).unwrap(),
        (2.0, 2.0)
    );
    assert_eq!(default_incumbents(&[], &[100.0]).unwrap(), (100.0, 100.0));
    assert!(default_incumbents(&[], &[]).is_err());
}

#[test]
fn moment_matched_ceipv_matches_python_golden_values() {
    // Generated from current upstream openbo.acquisition.mobo with prune_z=None.
    let front = m(&[&[0.4, 0.8], &[0.7, 0.5]]);
    let mean = [0.6, 0.7];
    let cov = m(&[&[0.04, 0.015], &[0.015, 0.09]]);
    let (lo, hi) = nondominated_cells(&front, &[0.0, 0.0], &[1.0, 1.0]).unwrap();
    let terms = ceipv_terms(
        &front,
        &[0.0, 0.0],
        &[1.0, 1.0],
        EipvForm::Exact,
        &[0.1, 0.25, 0.5],
    )
    .unwrap();
    close(
        ceipv(&mean, &cov, &terms).unwrap(),
        0.09174161239239248,
        1e-13,
    );
    close(
        ieipv(&mean, &cov, &lo, &hi, EipvForm::Exact).unwrap(),
        0.08427744249978031,
        1e-10,
    );
    let terms = ceipv_terms(
        &front,
        &[0.0, 0.0],
        &[1.0, 1.0],
        EipvForm::Paper,
        &[0.1, 0.25, 0.5],
    )
    .unwrap();
    close(
        ceipv(&mean, &cov, &terms).unwrap(),
        0.01155793071346884,
        1e-13,
    );
    close(
        ieipv(&mean, &cov, &lo, &hi, EipvForm::Paper).unwrap(),
        0.012447692373203176,
        1e-10,
    );
    assert!(
        log_ceipv(&[-100.0, -100.0], &cov, &terms)
            .unwrap()
            .is_finite()
    );
}

#[test]
fn independent_eipv_is_analytic_and_matches_monte_carlo_with_error_bound() {
    let front = m(&[&[0.4, 0.8], &[0.7, 0.5]]);
    let mean = [0.6, 0.7];
    let cov = m(&[&[0.04, 0.0], &[0.0, 0.09]]);
    let (lo, hi) = nondominated_cells(&front, &[0.0, 0.0], &[1.0, 1.0]).unwrap();
    for form in [EipvForm::Exact, EipvForm::Paper] {
        let analytic = ieipv(&mean, &cov, &lo, &hi, form).unwrap();
        let (mc, se) = eipv_mc(&mean, &cov, &lo, &hi, 15000, 19, form).unwrap();
        close(analytic, mc, 6.0 * se);
    }
    let singular = m(&[&[0.0, 0.0], &[0.0, 0.0]]);
    close(
        ieipv(&mean, &singular, &lo, &hi, EipvForm::Exact).unwrap(),
        hypervolume_gain(&mean, &lo, &hi).unwrap(),
        1e-13,
    );
    assert!(eipv_mc(&mean, &cov, &lo, &hi, 0, 0, EipvForm::Exact).is_err());
    let (huge, _) = eipv_mc(
        &[0.0],
        &m(&[&[1e308]]),
        &m(&[&[0.0]]),
        &m(&[&[1.0]]),
        8,
        0,
        EipvForm::Exact,
    )
    .unwrap();
    assert!((0.0..=1.0).contains(&huge));
    close(
        ieipv(
            &[1e20],
            &m(&[&[1.0]]),
            &m(&[&[0.0]]),
            &m(&[&[1.0]]),
            EipvForm::Exact,
        )
        .unwrap(),
        1.0,
        1e-14,
    );
    assert!(
        eipv_mc(
            &mean,
            &m(&[&[1.0, 2.0], &[2.0, 1.0]]),
            &lo,
            &hi,
            1,
            0,
            EipvForm::Exact
        )
        .is_err()
    );
}

#[test]
fn ehvi_and_joint_noisy_ehvi_preserve_deterministic_and_correlated_limits() {
    let front = m(&[&[1.0, 1.0]]);
    let zero = m(&[&[0.0, 0.0], &[0.0, 0.0]]);
    close(
        ehvi(&[2.0, 2.0], &zero, &front, &[0.0, 0.0], 16, 0).unwrap(),
        3.0,
        1e-14,
    );
    close(
        ehvi(&[0.5, 0.5], &zero, &front, &[0.0, 0.0], 16, 0).unwrap(),
        0.0,
        1e-14,
    );
    let joint = vec![vec![1.0; 4]; 4];
    let identical = m(&[&[1.0, 1.0], &[1.0, 1.0]]);
    close(
        nehvi(&identical, &joint, 1, &[0.0, 0.0], 128, 2).unwrap(),
        0.0,
        1e-6,
    );
    assert!(ehvi(&[1.0, 1.0], &zero, &front, &[0.0, 0.0], 0, 0).is_err());
    assert!(nehvi(&identical, &joint, 2, &[0.0, 0.0], 8, 0).is_err());
}

#[test]
fn multi_output_fixed_hyperparameter_posterior_matches_python_in_all_modes() {
    // Golden values from Python MultiOutputGP(optimize_hyperparameters=False).
    let x = m(&[&[0.0], &[0.3], &[0.65], &[1.0]]);
    let y = m(&[&[0.0, 1.0], &[0.8, 0.2], &[0.1, 0.7], &[-0.4, 1.4]]);
    let q = m(&[&[0.12], &[0.5], &[0.87]]);
    for mode in ["independent", "mt", "slf"] {
        let mut gp = MultiOutputGP::new(mode, 1e-4).unwrap();
        gp.optimize_hyperparameters = false;
        gp.fit(&x, &y).unwrap();
        let (mean, cov) = gp.predict(&q).unwrap();
        let (mj, joint) = gp.predict_joint(&q).unwrap();
        assert_eq!(mean, mj);
        if mode == "slf" {
            close(mean[0][0], 0.3893821227731735, 1e-11);
            close(mean[2][1], 1.225316548711136, 1e-11);
            close(cov[1][0][1], 0.0003816352715918834, 1e-11);
        } else {
            close(mean[0][0], 0.3893859877454329, 1e-11);
            close(mean[2][1], 1.2253326374490254, 1e-11);
            close(cov[1][0][1], 0.0, 1e-13);
        }
        close(cov[0][0][0], 0.0027921711791382264, 1e-10);
        for i in 0..q.len() {
            for a in 0..2 {
                for b in 0..2 {
                    close(cov[i][a][b], joint[2 * i + a][2 * i + b], 1e-14);
                }
            }
        }
        assert!(gp.predict(&m(&[&[0.1, 0.2]])).is_err());
        let before = gp.predict(&q).unwrap();
        assert!(gp.fit(&x, &m(&[&[f64::NAN]])).is_err());
        assert_eq!(before, gp.predict(&q).unwrap());
    }
}

#[test]
fn correlated_models_learn_nonzero_correlation_and_map_improves_objective() {
    let x: Matrix = (0..8).map(|i| vec![i as f64 / 7.0]).collect();
    let y: Matrix = x
        .iter()
        .map(|r| {
            let v = (r[0] * 6.0).sin();
            vec![v, -v]
        })
        .collect();
    for mode in ["mt", "slf"] {
        let mut gp = MultiOutputGP::new(mode, 1e-4).unwrap();
        gp.optimize_hyperparameters = false;
        gp.fit(&x, &y).unwrap();
        let initial = gp.log_posterior(gp.hyperparameters()).unwrap();
        gp.optimize_hyperparameters = true;
        gp.max_iterations = 18;
        gp.fit(&x, &y).unwrap();
        assert!(gp.log_posterior(gp.hyperparameters()).unwrap() > initial);
        assert!(gp.output_correlation().unwrap()[0][1] < -0.1);
        let draws = gp.sample_hyperparameters(2, 1, 1, 5).unwrap();
        assert_eq!(draws.len(), 2);
        let bounds = gp.hyperparameter_bounds();
        for draw in draws {
            assert!(draw.iter().zip(&bounds).all(|(v, (l, u))| v >= l && v <= u));
        }
    }
}

#[test]
fn multi_output_handles_finite_extremes_without_silent_nan() {
    let mut gp = MultiOutputGP::new("slf", 1e-4).unwrap();
    gp.optimize_hyperparameters = false;
    gp.fit(&m(&[&[1e300], &[-1e300]]), &m(&[&[0.0, 0.0], &[1.0, 1.0]]))
        .unwrap();
    let (mean, cov) = gp.predict(&m(&[&[0.0]])).unwrap();
    assert!(
        mean.iter()
            .flatten()
            .chain(cov.iter().flatten().flatten())
            .all(|v| v.is_finite())
    );
    gp.fit(
        &m(&[&[0.0], &[1.0]]),
        &m(&[&[1e300, 1e300], &[-1e300, -1e300]]),
    )
    .unwrap();
    // The true output covariance exceeds f64, so fail explicitly.
    assert!(gp.predict(&m(&[&[0.5]])).is_err());
}

#[test]
fn slice_sampler_reproduces_normal_and_bounded_uniform_moments() {
    let options = SliceOptions {
        burn_in: 100,
        ..Default::default()
    };
    let draws = slice_sample(
        |x| -0.5 * x.iter().map(|v| v * v).sum::<f64>(),
        &[0.0, 0.0],
        6000,
        17,
        &options,
    )
    .unwrap();
    for d in 0..2 {
        let mean = draws.iter().map(|r| r[d]).sum::<f64>() / draws.len() as f64;
        let var = draws.iter().map(|r| (r[d] - mean).powi(2)).sum::<f64>() / draws.len() as f64;
        close(mean, 0.0, 0.06);
        close(var, 1.0, 0.1);
    }
    let options = SliceOptions {
        lower: vec![-1.0],
        upper: vec![2.0],
        burn_in: 50,
        thin: 2,
        ..Default::default()
    };
    let draws = slice_sample(|_| 0.0, &[0.0], 5000, 19, &options).unwrap();
    assert!(draws.iter().all(|r| r[0] >= -1.0 && r[0] <= 2.0));
    close(
        draws.iter().map(|r| r[0]).sum::<f64>() / draws.len() as f64,
        0.5,
        0.06,
    );
    assert_eq!(
        draws,
        slice_sample(|_| 0.0, &[0.0], 5000, 19, &options).unwrap()
    );
    assert!(
        slice_sample(
            |_| 0.0,
            &[0.0],
            1,
            0,
            &SliceOptions {
                widths: vec![0.0],
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(slice_sample(|_| f64::NAN, &[0.0], 1, 0, &Default::default()).is_err());
    assert!(
        slice_sample(
            |_| 0.0,
            &[0.0],
            1,
            0,
            &SliceOptions {
                widths: vec![1e308],
                lower: vec![-1e308],
                upper: vec![1e308],
                ..Default::default()
            }
        )
        .is_err()
    );
}
