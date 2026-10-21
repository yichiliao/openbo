//! Coordinate slice sampler with bounded stepping-out and interval shrinkage.

use crate::{Matrix, Result};
use anyhow::{bail, ensure};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone)]
pub struct SliceOptions {
    /// Empty means width 1 in every coordinate; otherwise one positive width per coordinate.
    pub widths: Vec<f64>,
    /// Empty means unbounded.
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub burn_in: usize,
    pub thin: usize,
    /// Total stepping-out budget, randomly allocated across the two interval ends.
    pub max_steps_out: usize,
    pub max_shrink: usize,
}

impl Default for SliceOptions {
    fn default() -> Self {
        Self {
            widths: Vec::new(),
            lower: Vec::new(),
            upper: Vec::new(),
            burn_in: 0,
            thin: 1,
            max_steps_out: 20,
            max_shrink: 10000,
        }
    }
}

pub fn slice_sample<F>(
    log_density: F,
    start: &[f64],
    n_samples: usize,
    seed: u64,
    options: &SliceOptions,
) -> Result<Matrix>
where
    F: Fn(&[f64]) -> f64,
{
    let d = start.len();
    ensure!(
        d > 0 && start.iter().all(|v| v.is_finite()),
        "slice start must be finite and nonempty"
    );
    let widths = if options.widths.is_empty() {
        vec![1.0; d]
    } else {
        options.widths.clone()
    };
    let lower = if options.lower.is_empty() {
        vec![f64::NEG_INFINITY; d]
    } else {
        options.lower.clone()
    };
    let upper = if options.upper.is_empty() {
        vec![f64::INFINITY; d]
    } else {
        options.upper.clone()
    };
    ensure!(
        widths.len() == d && lower.len() == d && upper.len() == d,
        "slice option dimensions disagree"
    );
    ensure!(
        widths.iter().all(|v| v.is_finite() && *v > 0.0)
            && lower
                .iter()
                .zip(&upper)
                .all(|(a, b)| !a.is_nan() && !b.is_nan() && a < b),
        "slice widths must be positive and bounds strictly increasing"
    );
    ensure!(
        options.thin > 0 && options.max_steps_out > 0 && options.max_shrink > 0,
        "slice thinning and iteration limits must be positive"
    );
    ensure!(
        start
            .iter()
            .enumerate()
            .all(|(i, v)| *v >= lower[i] && *v <= upper[i]),
        "slice start outside bounds"
    );
    let bounded = |x: &[f64]| {
        if x.iter()
            .enumerate()
            .any(|(i, v)| !v.is_finite() || *v < lower[i] || *v > upper[i])
        {
            return f64::NEG_INFINITY;
        }
        let value = log_density(x);
        if value.is_finite() {
            value
        } else {
            f64::NEG_INFINITY
        }
    };
    let mut x = start.to_vec();
    let mut current = bounded(&x);
    ensure!(
        current.is_finite(),
        "slice start must have finite log density"
    );
    let sweeps = n_samples
        .checked_mul(options.thin)
        .and_then(|n| n.checked_add(options.burn_in))
        .ok_or_else(|| anyhow::anyhow!("slice iteration count overflow"))?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(n_samples);
    let mut coordinates: Vec<usize> = (0..d).collect();
    for sweep in 0..sweeps {
        coordinates.shuffle(&mut rng);
        for &i in &coordinates {
            let u = rng.gen_range(f64::MIN_POSITIVE..1.0_f64);
            let level = current + u.ln();
            let origin = x[i];
            let mut left = origin - widths[i] * rng.gen_range(0.0..1.0);
            let mut right = left + widths[i];
            let mut proposal = x.clone();
            // Neal's random split is needed for detailed balance with a finite
            // stepping-out budget; simply truncating both ends independently
            // can bias the chain.
            let left_steps = rng.gen_range(0..options.max_steps_out);
            let right_steps = options.max_steps_out - 1 - left_steps;
            for _ in 0..left_steps {
                proposal[i] = left;
                if left <= lower[i] || bounded(&proposal) <= level {
                    break;
                }
                left -= widths[i];
            }
            for _ in 0..right_steps {
                proposal[i] = right;
                if right >= upper[i] || bounded(&proposal) <= level {
                    break;
                }
                right += widths[i];
            }
            left = left.max(lower[i]);
            right = right.min(upper[i]);
            ensure!(
                left.is_finite()
                    && right.is_finite()
                    && (right - left).is_finite()
                    && left <= origin
                    && right >= origin,
                "slice interval overflow"
            );
            let mut accepted = false;
            for _ in 0..options.max_shrink {
                if right <= left || right - left <= f64::EPSILON * origin.abs().max(1.0) {
                    proposal[i] = origin;
                    accepted = true;
                    break;
                }
                proposal[i] = rng.gen_range(left..right);
                let value = bounded(&proposal);
                if value >= level {
                    x = proposal.clone();
                    current = value;
                    accepted = true;
                    break;
                }
                if proposal[i] < origin {
                    left = proposal[i];
                } else {
                    right = proposal[i];
                }
            }
            if !accepted {
                bail!("slice shrinkage failed to converge");
            }
        }
        if sweep >= options.burn_in && (sweep - options.burn_in + 1) % options.thin == 0 {
            samples.push(x.clone());
        }
    }
    Ok(samples)
}
