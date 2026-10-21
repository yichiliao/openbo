//! Read alongside README.md: a GP, an acquisition and a real ask/tell loop.
use openbo::{
    Result, acquisition,
    core::{self, GaussianProcess, GpConfig},
    optimizers::{OptimizerConfig, SequentialOptimizer},
};

fn main() -> Result<()> {
    let x = vec![vec![0.05], vec![0.4], vec![0.85]];
    let y = x
        .iter()
        .map(|row| -(row[0] - 0.35_f64).powi(2))
        .collect::<Vec<_>>();
    let mut gp = GaussianProcess::new(GpConfig {
        kernel_type: "rbf".into(),
        lengthscale: vec![0.2],
        ..Default::default()
    })?;
    gp.fit(&x, &y)?;
    let (mean, variance) = gp.predict(&vec![vec![0.3], vec![0.7]])?;
    let best = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for i in 0..mean.len() {
        println!(
            "GP mean={}, variance={}, EI={}",
            mean[i],
            variance[i],
            acquisition::expected_improvement(mean[i], variance[i], best)?
        );
    }
    let config = OptimizerConfig {
        n_init: 3,
        n_iter: 3,
        ..Default::default()
    };
    let mut optimizer = SequentialOptimizer::new(config)?;
    while !optimizer.is_finished() {
        let point = optimizer.suggest()?;
        let value = core::synthetic("branin", &vec![point.clone()])?;
        optimizer.observe(&point, &value)?;
    }
    let result = optimizer.result()?;
    println!(
        "{} evaluations; best score={:?}",
        result.y_values.len(),
        result.best_so_far.last()
    );
    Ok(())
}
