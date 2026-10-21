//! Numerical fixtures evaluated with the upstream Python GPScratch implementation.
use openbo::Matrix;
use openbo::core::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::json;

fn assert_near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a:.16} != {b:.16} (tolerance {tol})");
}
fn data() -> (Matrix, Vec<f64>, Matrix) {
    (
        vec![vec![0., 0.], vec![0.25, 0.7], vec![0.7, 0.1], vec![1., 1.]],
        vec![1.2, -0.7, 2.3, 0.4],
        vec![vec![0.1, 0.2], vec![0.6, 0.8], vec![0.9, 0.3]],
    )
}

#[test]
fn python_gp_posterior_and_likelihood_parity() {
    let cases = [
        (
            "rbf",
            false,
            [0.618225040240572, 0.64507374631676, 1.7830182839116318],
            [0.09213390352007522, 0.6607936083121582, 0.36531046796502786],
            7.7026564220817475,
        ),
        (
            "rbf",
            true,
            [0.5396875237507313, 0.6548156362366458, 1.7753550482901148],
            [0.11102135374169064, 0.7962562980161506, 0.4401991138978586],
            6.702699406957889,
        ),
        (
            "matern52",
            false,
            [0.6468667328808706, 0.7199565326430948, 1.5772578100999317],
            [0.23358293452597167, 0.9245022868216235, 0.6173433493853904],
            7.41170723716474,
        ),
        (
            "matern52",
            true,
            [0.5900252637487418, 0.798700805373268, 1.6284870250381043],
            [0.2814674361037959, 1.1140252556200563, 0.7438987360093955],
            6.477738807983062,
        ),
    ];
    let (x, y, q) = data();
    for (kernel, std, means, variances, nll) in cases {
        let mut gp = GaussianProcess::new(GpConfig {
            lengthscale: vec![0.3, 0.8],
            variance: 1.7,
            noise: 1e-4,
            kernel_type: kernel.into(),
            standardize_targets: std,
            ..Default::default()
        })
        .unwrap();
        gp.fit(&x, &y).unwrap();
        let (m, v) = gp.predict(&q).unwrap();
        let (mj, c) = gp.predict_joint(&q).unwrap();
        for i in 0..q.len() {
            assert_near(m[i], means[i], 2e-12);
            assert_near(v[i], variances[i], 2e-12);
            assert_near(m[i], mj[i], 1e-13);
            assert_near(v[i], c[i][i], 1e-13);
            for (j, row) in c.iter().enumerate() {
                assert_near(c[i][j], row[i], 1e-13);
            }
        }
        assert_near(gp.negative_log_marginal_likelihood().unwrap(), nll, 2e-12);
    }
}

#[test]
fn fixed_scaling_and_constant_mean_replay_round_trip() {
    let (x, y, q) = data();
    let mut gp = GaussianProcess::new(GpConfig {
        lengthscale: vec![0.2, 0.8],
        variance: 2.2,
        noise: 0.001,
        mean_constant: 0.37,
        fixed_target_mean: Some(4.2),
        fixed_target_std: Some(3.1),
        ..Default::default()
    })
    .unwrap();
    gp.fit(&x, &y).unwrap();
    let expected = gp.predict_joint(&q).unwrap();
    let state = gp.source_state().unwrap();
    let replay = GaussianProcess::from_state(&state).unwrap();
    assert_eq!(replay.predict_joint(&q).unwrap(), expected);
    let replay = GaussianProcess::from_source_state(&state["gp_state"], &x, &y).unwrap();
    assert_eq!(replay.predict_joint(&q).unwrap(), expected);
    let mut plain = GaussianProcess::new(GpConfig {
        mean_constant: 0.5,
        standardize_targets: false,
        ..Default::default()
    })
    .unwrap();
    plain.fit(&vec![vec![0.]], &[2.]).unwrap();
    assert_near(plain.predict(&vec![vec![100.]]).unwrap().0[0], 0.5, 1e-12);
}

#[test]
fn source_state_accepts_python_scalar_lengthscale_and_rejects_incomplete_scaling() {
    let gp=GaussianProcess::from_source_state(&json!({"lengthscale":0.4,"variance":1.2,"noise":0.,"kernel_type":"rbf","standardize_targets":true,"y_mean":7.,"y_std":2.,"mean_constant":0.3}),&vec![vec![0.,0.],vec![1.,1.]],&[1.,2.]).unwrap();
    assert_eq!(gp.config().lengthscale, vec![0.4, 0.4]);
    assert!(
        GaussianProcess::from_source_state(&json!({"y_mean":1.}), &vec![vec![0.]], &[1.]).is_err()
    );
    assert!(GaussianProcess::from_state(&json!({"gp_state":{}})).is_err());
    assert!(GaussianProcess::from_source_state(&json!({}), &vec![vec![0.]], &[1.]).is_err());
    assert!(
        GaussianProcess::from_source_state(
            &json!({"lengthscale":1.,"variance":1.,"noise":0.,"dim":2}),
            &vec![vec![0.]],
            &[1.]
        )
        .is_err()
    );
}

#[test]
fn constant_and_single_target_keep_uncertainty() {
    let mut gp = GaussianProcess::new(GpConfig::default()).unwrap();
    gp.fit(&vec![vec![0.3, 0.7]], &[2.5]).unwrap();
    let (m, v) = gp.predict(&vec![vec![0.9, 0.1]]).unwrap();
    assert_eq!(m[0], 2.5);
    assert!(v[0] > 0.1);
    gp.fit(&vec![vec![0., 0.], vec![0.5, 0.5], vec![1., 1.]], &[3.; 3])
        .unwrap();
    assert!(gp.predict(&vec![vec![0.9, 0.1]]).unwrap().1[0] > 0.1);
}

#[test]
fn duplicate_noiseless_inputs_are_stable_and_atomic_failed_refit() {
    let mut gp = GaussianProcess::new(GpConfig {
        noise: 0.,
        ..Default::default()
    })
    .unwrap();
    gp.fit(&vec![vec![0.], vec![0.], vec![1.]], &[1., 1., 2.])
        .unwrap();
    let query = vec![vec![0.], vec![0.5], vec![1.]];
    let before = gp.predict(&query).unwrap();
    assert_near(before.0[0], 1., 1e-8);
    assert_near(before.0[2], 2., 1e-8);
    assert!(before.1.iter().all(|v| v.is_finite() && *v > 0.));
    assert!(gp.fit(&vec![vec![0.], vec![f64::NAN]], &[1., 2.]).is_err());
    assert_eq!(gp.predict(&query).unwrap(), before);
}

#[test]
fn scalar_configuration_can_refit_on_different_dimension() {
    let mut gp = GaussianProcess::new(GpConfig::default()).unwrap();
    gp.fit(&vec![vec![0., 0.], vec![1., 1.]], &[1., 2.])
        .unwrap();
    gp.fit(&vec![vec![0., 0., 0.], vec![1., 1., 1.]], &[1., 2.])
        .unwrap();
    assert_eq!(gp.config().lengthscale, vec![1.; 3]);
    let mut ard = GaussianProcess::new(GpConfig {
        lengthscale: vec![0.4, 0.8],
        ..Default::default()
    })
    .unwrap();
    assert!(ard.fit(&vec![vec![0., 0., 0.]], &[1.]).is_err());
}

#[test]
fn fit_and_predict_validation() {
    let mut gp = GaussianProcess::new(GpConfig::default()).unwrap();
    assert!(gp.predict(&vec![vec![0.]]).is_err());
    assert!(gp.fit(&vec![], &[]).is_err());
    assert!(gp.fit(&vec![vec![]], &[1.]).is_err());
    assert!(gp.fit(&vec![vec![0.]], &[]).is_err());
    assert!(gp.fit(&vec![vec![0.]], &[f64::INFINITY]).is_err());
    gp.fit(&vec![vec![0.]], &[1.]).unwrap();
    assert_eq!(gp.predict(&vec![]).unwrap(), (vec![], vec![]));
    assert_eq!(gp.predict_joint(&vec![]).unwrap(), (vec![], vec![]));
    assert!(gp.predict(&vec![vec![0., 1.]]).is_err());
    assert!(gp.predict(&vec![vec![f64::NAN]]).is_err());
    assert!(gp.predict_joint(&vec![vec![0.], vec![]]).is_err());
    for c in [
        GpConfig {
            variance: 0.,
            ..Default::default()
        },
        GpConfig {
            variance: f64::INFINITY,
            ..Default::default()
        },
        GpConfig {
            noise: -1.,
            ..Default::default()
        },
        GpConfig {
            lengthscale: vec![f64::NAN],
            ..Default::default()
        },
        GpConfig {
            kernel_type: "oops".into(),
            ..Default::default()
        },
        GpConfig {
            fixed_target_mean: Some(0.),
            ..Default::default()
        },
    ] {
        assert!(GaussianProcess::new(c).is_err());
    }
}

#[test]
fn marginal_likelihood_fitting_improves_and_is_reproducible() {
    let x = sample(&[(0., 1.); 2], 12, 0).unwrap();
    let y: Vec<f64> = x
        .iter()
        .map(|r| (4. * r[0]).sin() + 0.3 * (3. * r[1]).cos())
        .collect();
    for kernel in ["rbf", "matern52"] {
        for noise in [0., 1e-6] {
            let config = GpConfig {
                lengthscale: vec![1.5; 2],
                noise,
                kernel_type: kernel.into(),
                optimizer_restarts: 3,
                optimizer_maxiter: 50,
                ..Default::default()
            };
            let mut baseline = GaussianProcess::new(config.clone()).unwrap();
            baseline.fit(&x, &y).unwrap();
            let nll = baseline.negative_log_marginal_likelihood().unwrap();
            let mut optimized = GaussianProcess::new(GpConfig {
                optimize_hyperparameters: true,
                ..config.clone()
            })
            .unwrap();
            optimized.fit(&x, &y).unwrap();
            assert!(optimized.negative_log_marginal_likelihood().unwrap() <= nll + 1e-8);
            assert!(optimized.negative_log_marginal_likelihood().unwrap() < nll - 0.1);
            let mut rerun = GaussianProcess::new(GpConfig {
                optimize_hyperparameters: true,
                ..config
            })
            .unwrap();
            rerun.fit(&x, &y).unwrap();
            assert_eq!(optimized.config(), rerun.config());
        }
    }
}

#[test]
fn optimize_noise_is_finite_when_initial_noise_is_zero() {
    let (x, y, _) = data();
    let mut gp = GaussianProcess::new(GpConfig {
        noise: 0.,
        optimize_noise: true,
        optimize_hyperparameters: true,
        optimizer_restarts: 2,
        optimizer_maxiter: 20,
        ..Default::default()
    })
    .unwrap();
    gp.fit(&x, &y).unwrap();
    assert!(gp.config().noise > 0.);
    assert!(gp.negative_log_marginal_likelihood().unwrap().is_finite());
}

#[test]
fn kernels_are_symmetric_and_ard_respects_individual_coordinates() {
    for kind in ["rbf", "matern52"] {
        let k = kernel_matrix(
            &vec![vec![0., 0.], vec![1., 0.], vec![0., 1.]],
            &vec![vec![0., 0.], vec![1., 0.], vec![0., 1.]],
            &[0.2, 0.8],
            2.,
            kind,
        )
        .unwrap();
        for (i, row) in k.iter().enumerate() {
            assert_near(row[i], 2., 1e-14);
            for (j, other) in k.iter().enumerate() {
                assert_near(row[j], other[i], 1e-14);
            }
        }
        assert!(k[0][1] < k[0][2]);
    }
    assert!(kernel_matrix(&vec![vec![0.]], &vec![vec![0., 0.]], &[1.], 1., "rbf").is_err());
    assert!(kernel_matrix(&vec![vec![0.]], &vec![vec![0.]], &[1.], f64::NAN, "rbf").is_err());
}

#[test]
fn analytical_scalar_maxima() {
    let optimums = [
        (
            "branin",
            vec![(-std::f64::consts::PI + 5.) / 15., 12.275 / 15.],
        ),
        ("sphere", vec![0.5; 2]),
        ("ackley", vec![0.5; 2]),
        ("rastrigin", vec![0.5; 2]),
        ("rosenbrock", vec![0.75; 2]),
        (
            "hartmann6",
            vec![0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
        ),
        (
            "mccormick",
            vec![
                ((1. - 2. * std::f64::consts::PI / 3.) / 2. + 1.5) / 5.5,
                ((-1. - 2. * std::f64::consts::PI / 3.) / 2. + 3.) / 7.,
            ],
        ),
        ("normalized_goldstein_price", vec![0.5, 0.25]),
        (
            "normalized_hartmann3",
            vec![0.11458887075439354, 0.5556488906008059, 0.8525469799283214],
        ),
    ];
    for (name, x) in optimums {
        let spec = function_spec(name).unwrap();
        assert_near(synthetic(name, &vec![x]).unwrap()[0], spec.optimum, 1e-8);
    }
}

#[test]
fn scalar_noise_is_decoupled_repeatable_and_capped() {
    let x = vec![vec![0.5; 2]; 100];
    let y = synthetic_noisy("sphere", &x, 3., 42, true).unwrap();
    assert!(y.iter().all(|v| *v <= 0.));
    assert_eq!(y, synthetic_noisy("sphere", &x, 3., 42, true).unwrap());
    assert!(y.iter().any(|v| *v < 0.));
    assert_eq!(
        synthetic_noisy("sphere", &x, 0., 42, false).unwrap(),
        vec![0.; 100]
    );
    assert!(synthetic_noisy("sphere", &x, f64::NAN, 42, false).is_err());
    assert!(synthetic("branin", &vec![vec![0.]]).is_err());
    assert!(synthetic("missing", &vec![]).is_err());
    assert!(synthetic("sphere", &vec![vec![f64::INFINITY; 2]]).is_err());
    assert_eq!(synthetic("sphere", &vec![]).unwrap(), Vec::<f64>::new());
}

#[test]
fn multiobjective_known_values_and_reference_bounds() {
    let p = std::f64::consts::PI;
    let y = multi_objective("oka2", &vec![vec![0.5, 1., 0.5]]).unwrap();
    assert_near(y[0][0], 0., 1e-14);
    assert_near(y[0][1], -0.75, 1e-14);
    let y = multi_objective("dtlz1a", &vec![vec![0.5; 6]]).unwrap();
    assert_eq!(y[0], vec![-0.25, -0.25]);
    let y = multi_objective("vlmop3", &vec![vec![0.5; 2]]).unwrap();
    assert_near(y[0][0], 0., 1e-14);
    assert_near(y[0][1], -17. - 1. / 27., 1e-14);
    assert_near(y[0][2], 0.1, 1e-14);
    assert!(mo_function_spec("oka2").unwrap().true_hypervolume.unwrap() > 2. * p * 4.);
    for name in ["oka2", "vlmop3", "dtlz1a"] {
        let spec = mo_function_spec(name).unwrap();
        let x = sample(&spec.bounds, 500, 12).unwrap();
        for row in multi_objective(name, &x).unwrap() {
            assert_eq!(row.len(), spec.n_objectives);
            for (i, v) in row.iter().enumerate() {
                assert!(*v >= spec.v_ref[i] - 1e-10);
                assert!(*v <= spec.w_ref[i] + 1e-10);
            }
        }
    }
}

#[test]
fn sobol_matches_scipy_and_metabo_grid_keeps_optimum() {
    let expected = vec![
        vec![0., 0., 0., 0.],
        vec![0.5, 0.5, 0.5, 0.5],
        vec![0.75, 0.25, 0.25, 0.25],
        vec![0.25, 0.75, 0.75, 0.75],
        vec![0.375, 0.375, 0.625, 0.875],
        vec![0.875, 0.875, 0.125, 0.375],
        vec![0.625, 0.125, 0.875, 0.625],
        vec![0.125, 0.625, 0.375, 0.125],
    ];
    assert_eq!(sobol(4, 8).unwrap(), expected);
    assert!(sobol(5, 1).is_err());
    assert!(sobol(1, 0).unwrap().is_empty());
    for name in ["branin", "goldstein_price", "hartmann3"] {
        let tasks = metabo_grid_tasks(name, 50).unwrap();
        assert!(tasks[0].translation.iter().all(|v| *v == 0.));
        assert_eq!(tasks[0].scaling, 1.);
        for task in tasks {
            let p = task.argmax().unwrap();
            assert!(p.iter().all(|v| (0.0..=1.).contains(v)));
            assert_near(
                task.evaluate(&vec![p]).unwrap()[0],
                task.optimum().unwrap(),
                1e-8,
            );
        }
    }
}

#[test]
fn low_discrepancy_and_random_sampling_validate_domains() {
    for d in [1, 2, 4, 6, 32] {
        let x = low_discrepancy(d, 64, 42).unwrap();
        assert_eq!(x, low_discrepancy(d, 64, 42).unwrap());
        assert_eq!(x.len(), 64);
        assert!(
            x.iter()
                .all(|r| r.len() == d && r.iter().all(|v| *v >= 0. && *v < 1.))
        );
    }
    assert!(low_discrepancy(0, 1, 0).is_err());
    assert!(sample(&[], 1, 0).is_err());
    assert!(sample(&[(1., 0.)], 1, 0).is_err());
    assert!(sample(&[(0., f64::INFINITY)], 1, 0).is_err());
    let x = sample(&[(-3., -2.), (1., 4.)], 50, 21).unwrap();
    assert_eq!(x, sample(&[(-3., -2.), (1., 4.)], 50, 21).unwrap());
    assert!(
        x.iter()
            .all(|r| r[0] >= -3. && r[0] < -2. && r[1] >= 1. && r[1] < 4.)
    );
}

#[test]
fn variants_identity_clipping_noise_and_validation() {
    let variant = TaskVariantSpec {
        input_shift: vec![0.; 2],
        input_scale: vec![1.; 2],
        ..Default::default()
    };
    let x = vec![vec![0.1, 0.8], vec![0.9, 0.4]];
    assert_eq!(
        variant
            .evaluate("sphere", &x, &mut ChaCha8Rng::seed_from_u64(0), None)
            .unwrap(),
        synthetic("sphere", &x).unwrap()
    );
    let shifted = TaskVariantSpec {
        input_shift: vec![2.; 2],
        ..variant.clone()
    };
    assert_eq!(
        shifted
            .evaluate("sphere", &x, &mut ChaCha8Rng::seed_from_u64(0), None)
            .unwrap(),
        vec![-50.; 2]
    );
    let noisy = TaskVariantSpec {
        noise_std: 3.,
        cap_at_optimum: true,
        ..variant
    };
    assert!(
        noisy
            .evaluate(
                "sphere",
                &vec![vec![0.5; 2]; 200],
                &mut ChaCha8Rng::seed_from_u64(0),
                Some(-1.)
            )
            .unwrap()
            .iter()
            .all(|v| *v <= -1.)
    );
    assert!(
        TaskVariantSpec {
            input_shift: vec![0.; 2],
            input_scale: vec![f64::NAN; 2],
            ..Default::default()
        }
        .validate_for_dim(2)
        .is_err()
    );
}

#[test]
fn family_generation_splitting_and_json_roundtrip() {
    let variants = generate_variants("branin", 20, 42, &FamilyConfig::default()).unwrap();
    assert_eq!(
        variants,
        generate_variants("branin", 20, 42, &FamilyConfig::default()).unwrap()
    );
    let split = split_variants("branin", &variants, 0.8, 0).unwrap();
    assert_eq!(split.train_variants.len(), 16);
    assert_eq!(split.test_variants.len(), 4);
    for v in &variants {
        assert!(split.train_variants.contains(v) ^ split.test_variants.contains(v));
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested/family.json");
    save_family_split(&split, &path).unwrap();
    assert_eq!(load_family_split(&path).unwrap(), split);
    assert!(split_variants("branin", &variants, 1., 0).is_err());
    assert!(
        generate_variants(
            "branin",
            1,
            0,
            &FamilyConfig {
                max_input_shift: f64::NAN,
                ..Default::default()
            }
        )
        .is_err()
    );
    let users = generate_user_variants("mccormick", 5, 0., 0., 0).unwrap();
    assert!(users.iter().all(|u| u.input_shift == vec![0.; 2]
        && u.input_scale == vec![1.; 2]
        && u.output_scale == 1.));
}

#[test]
fn sphere_users_have_exact_weighted_optimum_including_boundary_and_zero_weights() {
    let users = make_sphere_users(4, 1.2, 0.6, 42).unwrap();
    for user in users {
        for weights in [vec![1., 2., 3.], vec![0., 1., 0.], vec![1., 0., 0.]] {
            let (x, y) = user.weighted_optimum(&weights).unwrap();
            assert!(x.iter().all(|v| (0.0..=1.).contains(v)));
            for row in sample(&[(0., 1.); 4], 300, 42).unwrap() {
                let val: f64 = user.noise_free(&vec![row]).unwrap()[0]
                    .iter()
                    .zip(&weights)
                    .map(|(y, w)| y * w)
                    .sum();
                assert!(val <= y + 1e-12);
            }
        }
    }
    let base = SphereUser {
        shift: vec![0.; 4],
        scale: 1.,
        noise_std: 0.,
    };
    assert_eq!(
        base.noise_free(&vec![vec![0.5; 4]]).unwrap(),
        multi_objective("three_spheres", &vec![vec![0.5; 4]]).unwrap()
    );
    assert!(base.weighted_optimum(&[0.; 3]).is_err());
    assert!(base.weighted_optimum(&[-1., 1., 1.]).is_err());
    assert!(make_sphere_users(1, 0., 2., 0).is_err());
}

#[test]
fn python_all_objective_fixtures() {
    let values = [
        ("branin", -2.337292471983326),
        ("sphere", -25.0),
        ("ackley", -10.138626172095204),
        ("rastrigin", -28.985021508052014),
        ("rosenbrock", -191.72000000000008),
        ("mccormick", -22.52436502845392),
        ("hartmann6", 0.11838177451087073),
        ("normalized_branin", 1.0128831167965917),
        ("normalized_goldstein_price", -1.2486348704105843),
        ("normalized_hartmann3", -0.9032887936070843),
    ];
    let p = [0.1, 0.8, 0.2, 0.7, 0.3, 0.6];
    for (name, value) in values {
        let dim = function_spec(name).unwrap().dim;
        assert_near(
            synthetic(name, &vec![p[..dim].to_vec()]).unwrap()[0],
            value,
            2e-12,
        );
    }
    for (name, expected) in [
        ("oka2", vec![2.5132741228718345, -3.3008375483885115]),
        (
            "vlmop3",
            vec![-4.912118485241756, -21.15925925925926, -0.09986424921550463],
        ),
        ("dtlz1a", vec![-22.354915028125262, -201.19423525312735]),
    ] {
        let dim = mo_function_spec(name).unwrap().dim;
        let y = multi_objective(name, &vec![p[..dim].to_vec()]).unwrap();
        for (a, b) in y[0].iter().zip(&expected) {
            assert_near(*a, *b, 2e-11);
        }
    }
}

#[test]
fn slf_gp_correlated_draws_are_deterministic_and_handle_extreme_correlation() {
    let config = SlfGpConfig {
        n_support: 32,
        n_reference: 128,
        ..Default::default()
    };
    for rho in [-1., 0., 1.] {
        let problem = SlfGpProblem::with_config(rho, 42, config.clone()).unwrap();
        let duplicate = SlfGpProblem::with_config(rho, 42, config.clone()).unwrap();
        let q = sample(&[(0., 1.); 3], 20, 0).unwrap();
        let y = problem.evaluate(&q).unwrap();
        assert_eq!(y, duplicate.evaluate(&q).unwrap());
        assert!(problem.spec().true_hypervolume.unwrap() > 0.);
        if rho.abs() == 1. {
            for row in &y {
                assert_near(row[1] + 2., rho * (row[0] - 2.), 2e-13);
            }
        }
        assert_eq!(problem.spec().v_ref.len(), 2);
        assert_eq!(problem.spec().w_ref.len(), 2);
        assert!(problem.evaluate(&vec![vec![0.]]).is_err());
        assert!(problem.evaluate(&vec![vec![f64::NAN; 3]]).is_err());
        assert!(problem.evaluate(&vec![]).unwrap().is_empty());
    }
    assert!(SlfGpProblem::with_config(1.1, 0, config.clone()).is_err());
    assert!(
        SlfGpProblem::with_config(
            0.,
            0,
            SlfGpConfig {
                n_support: 0,
                ..config.clone()
            }
        )
        .is_err()
    );
    assert!(
        SlfGpProblem::with_config(
            0.,
            0,
            SlfGpConfig {
                lengthscales: vec![vec![1.], vec![1., 2.]],
                ..config
            }
        )
        .is_err()
    );
}

#[test]
fn reachable_variant_optimum_and_observation_cap_agree() {
    let variant = TaskVariantSpec {
        input_shift: vec![0.8; 2],
        input_scale: vec![0.1; 2],
        noise_std: 2.,
        cap_at_optimum: true,
        seed: Some(42),
        ..Default::default()
    };
    let mut objective =
        VariantFunction::new("sphere", variant.clone(), Some("shifted_sphere")).unwrap();
    assert_near(objective.optimum.unwrap(), -18., 1e-12);
    let q = sample(&[(0., 1.); 2], 200, 42).unwrap();
    let observations = objective.evaluate(&q).unwrap();
    assert!(
        observations
            .iter()
            .all(|v| *v <= objective.optimum.unwrap())
    );
    let mut same = VariantFunction::new("sphere", variant, None).unwrap();
    assert_eq!(observations, same.evaluate(&q).unwrap());
    assert_ne!(observations, objective.evaluate(&q).unwrap());
    let (p, y) = estimate_optimum(
        |x| {
            Ok(x.iter()
                .map(|r| -(r[0] - 0.271828).powi(2) - (r[1] - 0.618034).powi(2))
                .collect())
        },
        2,
        &OptimumConfig {
            grid_points_per_dim: 21,
            ..Default::default()
        },
    )
    .unwrap();
    assert_near(y, 0., 1e-14);
    assert_near(p[0], 0.271828, 1e-7);
    assert_near(p[1], 0.618034, 1e-7);
}

#[test]
fn optimizer_never_worsens_out_of_box_start_and_overflows_are_errors() {
    let x = vec![vec![0.], vec![1e-4], vec![1.]];
    let y = vec![0., 1., 0.];
    let config = GpConfig {
        lengthscale: vec![1e-5],
        standardize_targets: false,
        optimizer_restarts: 1,
        optimizer_maxiter: 5,
        ..Default::default()
    };
    let mut baseline = GaussianProcess::new(config.clone()).unwrap();
    baseline.fit(&x, &y).unwrap();
    let mut optimized = GaussianProcess::new(GpConfig {
        optimize_hyperparameters: true,
        ..config
    })
    .unwrap();
    optimized.fit(&x, &y).unwrap();
    assert!(
        optimized.negative_log_marginal_likelihood().unwrap()
            <= baseline.negative_log_marginal_likelihood().unwrap() + 1e-12
    );
    let user = SphereUser {
        shift: vec![1e200; 4],
        scale: 1.,
        noise_std: 0.,
    };
    assert!(user.noise_free(&vec![vec![0.; 4]]).is_err());
    let mut huge = GaussianProcess::new(GpConfig::default()).unwrap();
    huge.fit(&vec![vec![0.], vec![1.]], &[-1e200, 1e200])
        .unwrap();
    assert!(huge.predict(&vec![vec![0.5]]).is_err());
}

#[test]
fn extreme_finite_kernel_coordinates_use_the_stationary_limit_and_stable_scaling() {
    for kind in ["rbf", "matern52"] {
        let x = vec![vec![0.], vec![1e200], vec![-1e200], vec![f64::MAX]];
        let cov = kernel_matrix(&x, &x, &[1e-308], 1.7, kind).unwrap();
        for (i, row) in cov.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert_eq!(*value, if i == j { 1.7 } else { 0. });
            }
        }
        let x = vec![vec![1e308], vec![-1e308]];
        let extreme = kernel_matrix(&x, &x, &[1e308], 1.3, kind).unwrap();
        let expected = kernel_matrix(
            &vec![vec![1.], vec![-1.]],
            &vec![vec![1.], vec![-1.]],
            &[1.],
            1.3,
            kind,
        )
        .unwrap();
        assert_eq!(extreme, expected);
        assert!(extreme[0][1] > 0.);
    }
}
