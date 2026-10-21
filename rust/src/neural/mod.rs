//! Native neural population and acquisition models.
//!
//! Networks, reverse-mode differentiation, Adam, categorical PPO and GAE are
//! implemented in Rust using `f64`. Checkpoints are versioned JSON, including
//! optimizer and ChaCha RNG state; PyTorch `.pt` files are not Rust checkpoints.
mod naf;
mod network;
mod population;

pub use naf::*;
pub use population::*;
