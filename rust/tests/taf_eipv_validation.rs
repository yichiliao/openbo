//! TAF-EIPV mathematical invariants and fixed upstream Python reference values.
use openbo::{Matrix, acquisition::*};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn m(rows: &[&[f64]]) -> Matrix {
    rows.iter().map(|row| row.to_vec()).collect()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12, "{a} != {b}");
}

#[test]
fn source_hypervolume_matches_current_python_reference_and_caps_upper_corner() {
    // Python reference: upstream/main 3ceda2d, acquisition/taf_eipv.py.
    let front = m(&[&[-0.4, 0.5, 0.1], &[0.2, 0.1, 0.6], &[0.1, 0.6, -0.2]]);
    let points = m(&[
        &[0.0, 0.0, 0.0],
        &[0.4, 0.7, 0.8],
        &[1.5, 0.9, -0.1],
        &[-2.0, 0.7, 0.8],
    ]);
    let cells = reference_cells(&front, &[-1.0; 3], &[1.2; 3]).unwrap();
    let gains = source_improvement(&points, &cells).unwrap();
    for (gain, expected) in gains
        .iter()
        .zip([0.0, 1.6600000000000004, 2.1100000000000003, 0.0])
    {
        close(*gain, expected);
    }
    for gain in source_improvement(&front, &cells).unwrap() {
        close(gain, 0.0);
    }
    let empty = reference_cells(&vec![], &[0.0; 2], &[2.0; 2]).unwrap();
    assert_eq!(empty, (m(&[&[0.0, 0.0]]), m(&[&[2.0, 2.0]])));
    close(
        source_improvement(&m(&[&[3.0, 3.0]]), &empty).unwrap()[0],
        4.0,
    );
}

#[test]
fn source_improvement_agrees_with_independent_hypervolume_difference() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    for d in 1..=4 {
        for n_front in [0, 1, 6] {
            let front: Matrix = (0..n_front)
                .map(|_| (0..d).map(|_| rng.gen_range(-0.8..1.0)).collect())
                .collect();
            let points: Matrix = (0..25)
                .map(|_| (0..d).map(|_| rng.gen_range(-1.5..1.5)).collect())
                .collect();
            let lo = vec![-1.0; d];
            let hi = vec![1.2; d];
            let cells = reference_cells(&front, &lo, &hi).unwrap();
            let gains = source_improvement(&points, &cells).unwrap();
            let before = hypervolume(&front, &lo).unwrap();
            for (point, gain) in points.iter().zip(gains) {
                let mut union = front.clone();
                union.push(point.iter().map(|v| v.min(1.2)).collect());
                close(gain, hypervolume(&union, &lo).unwrap() - before);
            }
        }
    }
}

#[test]
fn pooled_original_ranking_weights_match_python_including_tied_sources() {
    let y = m(&[&[0.0, 0.0], &[1.0, 2.0], &[2.0, 1.0]]);
    let inverted = y
        .iter()
        .map(|row| row.iter().map(|v| -v).collect())
        .collect();
    let sources = vec![
        y.clone(),
        inverted,
        vec![vec![1.0; 2]; 3],
        m(&[&[0.0, 0.0], &[2.0, 1.0], &[1.0, 2.0]]),
    ];
    let weights = taf_r_source_weights(&y, &sources, 1.0).unwrap();
    for (weight, expected) in weights.iter().zip([0.75, 0.0, 0.5625, 0.6666666666666666]) {
        close(*weight, expected);
    }
    assert_eq!(
        taf_r_source_weights(&m(&[&[1.0, 1.0]]), &[m(&[&[4.0, 4.0]])], 1.0).unwrap(),
        vec![0.75]
    );
    assert_eq!(
        taf_r_source_weights(&vec![], &[vec![], vec![]], 1.0).unwrap(),
        vec![0.75, 0.75]
    );
    assert_eq!(
        taf_r_source_weights(&y, &[], 1.0).unwrap(),
        Vec::<f64>::new()
    );
}

#[test]
fn dominated_reference_observations_do_not_inflate_the_cell_grid() {
    let front: Matrix = (1..=130).map(|i| vec![i as f64 / 130.0; 3]).collect();
    let cells = reference_cells(&front, &[-1.0; 3], &[1.2; 3]).unwrap();
    assert_eq!(cells.0.len(), 7);
    close(
        source_improvement(&vec![vec![1.2; 3]], &cells).unwrap()[0],
        2.2_f64.powi(3) - 2.0_f64.powi(3),
    );
}

#[test]
fn log_blend_matches_python_and_omits_absent_target_and_rejected_sources() {
    let target = [0.25_f64.ln(), 0.0, 4.0_f64.ln(), f64::NEG_INFINITY];
    let terms = m(&[&[1.0, 2.0, 3.0, 0.0], &[100.0; 4], &[0.0, 4.0, 0.0, 0.0]]);
    let weights = [0.5, 0.0, 0.25];
    let blended = log_taf_eipv(Some(&target), &terms, &weights, 0.75).unwrap();
    let source_only = log_taf_eipv(None, &terms, &weights, 0.75).unwrap();
    for (actual, expected) in blended.iter().zip([
        -0.7801585575495751,
        0.6061358035703155,
        1.0986122886681096,
        LOG_FLOOR,
    ]) {
        close(*actual, expected);
    }
    for (actual, expected) in source_only.iter().zip([
        -0.4054651081081644,
        0.9808292530117262,
        0.6931471805599454,
        LOG_FLOOR,
    ]) {
        close(*actual, expected);
    }
    for (actual, expected) in log_taf_eipv(Some(&target), &vec![], &[], 0.75)
        .unwrap()
        .iter()
        .zip(target)
    {
        close(*actual, expected.max(LOG_FLOOR));
    }
    assert_eq!(
        log_taf_eipv(None, &vec![vec![0.0; 4]], &[0.75], 0.75).unwrap(),
        vec![LOG_FLOOR; 4]
    );
    assert!(log_taf_eipv(None, &terms, &[0.0; 3], 0.75).is_err());
    let extreme = log_taf_eipv(Some(&[1e308_f64.ln()]), &m(&[&[1e308]]), &[1e308], 1e308).unwrap();
    close(extreme[0], 1e308_f64.ln());
}

#[test]
fn malformed_nonfinite_and_negative_inputs_are_rejected() {
    assert!(reference_cells(&vec![], &[], &[]).is_err());
    assert!(reference_cells(&vec![], &[0.0], &[f64::INFINITY]).is_err());
    assert!(reference_cells(&vec![], &[1.0], &[0.0]).is_err());
    assert!(reference_cells(&m(&[&[0.0, 1.0]]), &[0.0], &[1.0]).is_err());
    assert!(source_improvement(&m(&[&[f64::NAN]]), &(m(&[&[0.0]]), m(&[&[1.0]]))).is_err());
    assert!(source_improvement(&m(&[&[0.5]]), &(m(&[&[0.0]]), vec![])).is_err());
    assert!(
        source_improvement(
            &m(&[&[1e300, 1e300]]),
            &(m(&[&[0.0, 0.0]]), m(&[&[1e300, 1e300]]))
        )
        .is_err()
    );
    assert!(taf_r_source_weights(&m(&[&[0.0]]), &[m(&[&[0.0, 1.0]])], 1.0).is_err());
    assert!(taf_r_source_weights(&vec![], &[], 0.0).is_err());
    assert!(log_taf_eipv(Some(&[0.0]), &m(&[&[-1.0]]), &[1.0], 1.0).is_err());
    assert!(log_taf_eipv(Some(&[0.0]), &m(&[&[1.0]]), &[f64::NAN], 1.0).is_err());
    assert!(log_taf_eipv(Some(&[f64::NAN]), &m(&[&[1.0]]), &[1.0], 1.0).is_err());
    assert!(log_taf_eipv(Some(&[f64::INFINITY]), &vec![], &[], 1.0).is_err());
    assert!(log_taf_eipv(Some(&[0.0, 0.0]), &m(&[&[1.0]]), &[1.0], 1.0).is_err());
    assert!(log_taf_eipv(None, &m(&[&[1.0], &[1.0, 2.0]]), &[1.0, 1.0], 1.0).is_err());
    assert!(log_taf_eipv(None, &m(&[&[1.0]]), &[-1.0], 1.0).is_err());
    assert!(log_taf_eipv(Some(&[0.0]), &vec![], &[], -1.0).is_err());
}
