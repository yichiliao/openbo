//! Rating-based objective weights and leave-one-user-out decay selection.
use super::*;
use anyhow::ensure;

/// Paper-compatible objective-weight grid: strictly positive multiples of
/// `1 / levels` summing to one (36 rows for three objectives and ten levels).
pub fn simplex_grid(n_objectives: usize, levels: usize) -> Result<Matrix> {
    ensure!(
        n_objectives > 0 && levels > 0,
        "objective count and grid levels must be positive"
    );
    ensure!(
        n_objectives <= 10 && levels <= 100,
        "simplex grid too large"
    );
    ensure!(
        levels >= n_objectives,
        "grid needs at least one level per objective"
    );
    let mut count = 1u128;
    for i in 1..n_objectives {
        count = count * (levels - i) as u128 / i as u128;
        ensure!(
            count <= 100_000,
            "simplex grid would exceed 100000 candidates"
        );
    }
    fn recurse(
        out: &mut Matrix,
        row: &mut Vec<f64>,
        remaining: usize,
        columns: usize,
        levels: usize,
    ) {
        if columns == 1 {
            row.push(remaining as f64 / levels as f64);
            out.push(row.clone());
            row.pop();
        } else {
            for n in 1..=remaining - (columns - 1) {
                row.push(n as f64 / levels as f64);
                recurse(out, row, remaining - n, columns - 1, levels);
                row.pop();
            }
        }
    }
    let mut out = vec![];
    recurse(&mut out, &mut vec![], levels, n_objectives, levels);
    Ok(out)
}
pub fn rescale_ratings(ratings: &[f64]) -> Result<Vec<f64>> {
    ensure!(
        !ratings.is_empty() && ratings.iter().all(|v| v.is_finite()),
        "ratings must be non-empty and finite"
    );
    let lo = ratings.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = ratings.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let magnitude = lo.abs().max(hi.abs());
    Ok(ratings
        .iter()
        .map(|v| {
            if hi == lo {
                100.0
            } else {
                // The ratio is unchanged by a common scale. Scaling first
                // keeps finite opposite-sign ratings from overflowing.
                1.0 + 99.0 * ((v / magnitude - lo / magnitude) / (hi / magnitude - lo / magnitude))
            }
        })
        .collect())
}
pub fn optimize_objective_weights(
    trajectories: &[Matrix],
    ratings: &[Vec<f64>],
    candidates: &Matrix,
) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure!(
        !trajectories.is_empty() && trajectories.len() == ratings.len() && !candidates.is_empty(),
        "need matching trajectories/ratings and candidate weights"
    );
    for weights in candidates {
        validate_weights(weights)?;
    }
    let n = candidates[0].len();
    ensure!(
        candidates.iter().all(|w| w.len() == n),
        "candidate weight dimensions must agree"
    );
    let mut scores = vec![0.0; candidates.len()];
    for (ys, ratings) in trajectories.iter().zip(ratings) {
        ensure!(
            !ys.is_empty()
                && ys.len() == ratings.len()
                && ys
                    .iter()
                    .all(|r| r.len() == n && r.iter().all(|v| v.is_finite())),
            "invalid trajectory/rating shape"
        );
        let scaled = rescale_ratings(ratings)?;
        for (j, w) in candidates.iter().enumerate() {
            let values = ys.iter().map(|r| dot(r, w)).collect::<Vec<_>>();
            scores[j] += scaled[argmax(&values)?];
        }
    }
    Ok((candidates[argmax(&scores)?].clone(), scores))
}
pub fn leave_one_out_curves(
    sources: &[Source],
    config: &OptimizerConfig,
    evaluations: usize,
) -> Result<Matrix> {
    ensure!(
        sources.len() >= 2 && evaluations > 0,
        "leave-one-out requires at least two users and a positive budget"
    );
    let mut curves = vec![];
    for (i, user) in sources.iter().enumerate() {
        let mut cfg = config.clone();
        cfg.taf_run_dir = None;
        cfg.n_init = 0;
        cfg.n_iter = evaluations;
        let mut optimizer = SequentialOptimizer::new(cfg)?;
        optimizer.sources = sources
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, s)| s.clone())
            .collect();
        while !optimizer.is_finished() {
            let x = optimizer.suggest()?;
            let y = predict_gps(&user.gps, &vec![x.clone()])?.0.remove(0);
            optimizer.observe(&x, &y)?;
        }
        let weights = config
            .objective_weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("objective weights required"))?;
        curves.push(optimizer.y_obs.iter().map(|r| dot(r, weights)).collect());
    }
    Ok(curves)
}
pub fn select_decay(
    sources: &[Source],
    config: &OptimizerConfig,
    pairs: &[(usize, f64)],
    evaluations: usize,
) -> Result<((usize, f64), Vec<f64>)> {
    ensure!(!pairs.is_empty(), "decay candidates cannot be empty");
    let mut scores = vec![];
    for &(start, rate) in pairs {
        let mut cfg = config.clone();
        cfg.decay = true;
        cfg.decay_start_iter = start;
        cfg.decay_rate = rate;
        let curves = leave_one_out_curves(sources, &cfg, evaluations)?;
        scores.push(curves.iter().map(|row| row.iter().sum::<f64>()).sum());
    }
    Ok((pairs[argmax(&scores)?], scores))
}
