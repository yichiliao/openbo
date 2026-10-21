//! Native, double precision Bayesian optimization. Every objective is maximized.
//!
//! ```
//! use openbo::{Result, core, optimizers::{OptimizerConfig, run}};
//!
//! # fn main() -> Result<()> {
//! let result = run(OptimizerConfig {
//!     method: "random".into(), n_init: 2, n_iter: 1,
//!     ..Default::default()
//! }, |x| core::synthetic("branin", &vec![x.clone()]))?;
//! assert_eq!(result.y_values.len(), 3);
//! # Ok(())
//! # }
//! ```

pub mod acquisition;
pub mod core;
pub mod neural;
pub mod optimizers;
pub mod server;

pub type Point = Vec<f64>;
pub type Matrix = Vec<Vec<f64>>;
pub type Result<T> = anyhow::Result<T>;

/// The upstream Python source revision covered by this port.
pub const UPSTREAM_REVISION: &str = "3ceda2d34225d2f1601fc57815cd821fbc78f37d";
