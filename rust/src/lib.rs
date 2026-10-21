//! Native, double precision Bayesian optimization. Every objective is maximized.

pub mod acquisition;
pub mod core;
pub mod neural;
pub mod optimizers;
pub mod server;

pub type Point = Vec<f64>;
pub type Matrix = Vec<Vec<f64>>;
pub type Result<T> = anyhow::Result<T>;
