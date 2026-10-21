use crate::{Matrix, Result};
use anyhow::ensure;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(super) struct Layer {
    pub inputs: usize,
    pub outputs: usize,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
}

/// A small dense ReLU network. The final layer has no activation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(super) struct Network {
    pub layers: Vec<Layer>,
}

#[derive(Clone)]
pub(super) struct Tape {
    activations: Matrix,
    preactivations: Matrix,
    mask: Vec<f64>,
}

impl Network {
    pub fn new(widths: &[usize], normal: bool, rng: &mut ChaCha8Rng) -> Self {
        let layers = widths
            .windows(2)
            .map(|width| {
                let (inputs, outputs) = (width[0], width[1]);
                let bound = 1.0 / (inputs as f64).sqrt();
                let weights = (0..inputs * outputs)
                    .map(|_| {
                        if normal {
                            let z: f64 = StandardNormal.sample(rng);
                            0.01 * z
                        } else {
                            rng.gen_range(-bound..bound)
                        }
                    })
                    .collect();
                let biases = (0..outputs)
                    .map(|_| {
                        if normal {
                            0.0
                        } else {
                            rng.gen_range(-bound..bound)
                        }
                    })
                    .collect();
                Layer {
                    inputs,
                    outputs,
                    weights,
                    biases,
                }
            })
            .collect();
        Self { layers }
    }

    pub fn validate(&self, widths: &[usize]) -> Result<()> {
        ensure!(
            self.layers.len() + 1 == widths.len(),
            "checkpoint network depth mismatch"
        );
        for (l, width) in self.layers.iter().zip(widths.windows(2)) {
            ensure!(
                l.inputs == width[0] && l.outputs == width[1],
                "checkpoint network width mismatch"
            );
            ensure!(
                l.weights.len() == l.inputs * l.outputs && l.biases.len() == l.outputs,
                "checkpoint tensor shape mismatch"
            );
            ensure!(
                l.weights.iter().chain(&l.biases).all(|v| v.is_finite()),
                "checkpoint contains nonfinite weights"
            );
        }
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() + l.biases.len())
            .sum()
    }

    pub fn forward(&self, x: &[f64], dropout: f64, rng: &mut ChaCha8Rng) -> (Vec<f64>, Tape) {
        let mut activations = vec![x.to_vec()];
        let mut preactivations = Vec::with_capacity(self.layers.len());
        let mut mask = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let input = activations.last().unwrap();
            let mut z = layer.biases.clone();
            for (j, zj) in z.iter_mut().enumerate() {
                *zj += layer.weights[j * layer.inputs..(j + 1) * layer.inputs]
                    .iter()
                    .zip(input)
                    .map(|(w, x)| w * x)
                    .sum::<f64>();
            }
            preactivations.push(z.clone());
            if i + 1 < self.layers.len() {
                z.iter_mut().for_each(|v| *v = v.max(0.0));
                if i + 2 == self.layers.len() && dropout > 0.0 {
                    let keep = 1.0 - dropout;
                    mask = z
                        .iter_mut()
                        .map(|v| {
                            let m = if rng.gen_bool(keep) { 1.0 / keep } else { 0.0 };
                            *v *= m;
                            m
                        })
                        .collect();
                }
            }
            activations.push(z);
        }
        (
            activations.last().unwrap().clone(),
            Tape {
                activations,
                preactivations,
                mask,
            },
        )
    }

    /// Accumulate the parameter gradient in layer order: weights, then biases.
    pub fn backward(&self, tape: &Tape, output_gradient: &[f64], gradient: &mut [f64]) {
        let mut delta = output_gradient.to_vec();
        let mut offset = gradient.len();
        for i in (0..self.layers.len()).rev() {
            let layer = &self.layers[i];
            offset -= layer.weights.len() + layer.biases.len();
            let input = &tape.activations[i];
            for (j, dj) in delta.iter().enumerate() {
                for (k, xk) in input.iter().enumerate() {
                    gradient[offset + j * layer.inputs + k] += dj * xk;
                }
                gradient[offset + layer.weights.len() + j] += dj;
            }
            if i > 0 {
                let mut previous = vec![0.0; layer.inputs];
                for (k, pk) in previous.iter_mut().enumerate() {
                    *pk = delta
                        .iter()
                        .enumerate()
                        .map(|(j, d)| d * layer.weights[j * layer.inputs + k])
                        .sum();
                    if tape.preactivations[i - 1][k] <= 0.0 {
                        *pk = 0.0;
                    }
                    if i + 1 == self.layers.len() && !tape.mask.is_empty() {
                        *pk *= tape.mask[k];
                    }
                }
                delta = previous;
            }
        }
    }

    pub fn apply(&mut self, adam: &mut Adam, gradient: &[f64], learning_rate: f64) -> Result<()> {
        ensure!(
            gradient.len() == self.parameter_count(),
            "gradient shape mismatch"
        );
        ensure!(
            gradient.iter().all(|v| v.is_finite()),
            "training produced a nonfinite gradient"
        );
        adam.step = adam
            .step
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("Adam step counter overflow"))?;
        let b1 = 1.0 - 0.9_f64.powf(adam.step as f64);
        let b2 = 1.0 - 0.999_f64.powf(adam.step as f64);
        let mut index = 0;
        for layer in &mut self.layers {
            for p in layer.weights.iter_mut().chain(&mut layer.biases) {
                let g = gradient[index];
                adam.first[index] = 0.9 * adam.first[index] + 0.1 * g;
                adam.second[index] = 0.999 * adam.second[index] + 0.001 * g * g;
                *p -= learning_rate * (adam.first[index] / b1)
                    / ((adam.second[index] / b2).sqrt() + 1e-8);
                ensure!(p.is_finite(), "training produced a nonfinite parameter");
                index += 1;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(super) struct Adam {
    pub first: Vec<f64>,
    pub second: Vec<f64>,
    pub step: u64,
}

impl Adam {
    pub fn new(n: usize) -> Self {
        Self {
            first: vec![0.0; n],
            second: vec![0.0; n],
            step: 0,
        }
    }
    pub fn validate(&self, n: usize) -> Result<()> {
        ensure!(
            self.step < u64::MAX,
            "optimizer step counter exceeds supported range"
        );
        ensure!(
            self.first.len() == n && self.second.len() == n,
            "checkpoint optimizer shape mismatch"
        );
        ensure!(
            self.first.iter().all(|v| v.is_finite())
                && self.second.iter().all(|v| v.is_finite() && *v >= 0.0),
            "invalid optimizer moments"
        );
        Ok(())
    }
}

pub(super) fn inputs(x: &Matrix, dim: usize, nonempty: bool) -> Result<()> {
    ensure!(!nonempty || !x.is_empty(), "at least one input is required");
    ensure!(
        x.iter()
            .all(|row| row.len() == dim && row.iter().all(|v| v.is_finite())),
        "inputs must be finite rows of dimension {dim}"
    );
    Ok(())
}

pub(super) fn targets(y: &[f64], n: usize) -> Result<()> {
    ensure!(
        y.len() == n && y.iter().all(|v| v.is_finite()),
        "targets must be finite and match inputs"
    );
    Ok(())
}

pub(super) fn write_json<T: Serialize>(path: &std::path::Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    // Write a complete JSON document before replacing an existing checkpoint.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn backprop_matches_central_differences() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let net = Network::new(&[2, 3, 2], false, &mut rng);
        let x = [0.7, -0.2];
        let (_, tape) = net.forward(&x, 0.0, &mut rng);
        let direction = [0.4, -0.8];
        let mut gradient = vec![0.0; net.parameter_count()];
        net.backward(&tape, &direction, &mut gradient);
        for (index, analytical) in gradient.iter().enumerate() {
            let eval = |eps: f64| {
                let mut copy = net.clone();
                let mut k = 0;
                for l in &mut copy.layers {
                    for p in l.weights.iter_mut().chain(&mut l.biases) {
                        if k == index {
                            *p += eps;
                        }
                        k += 1;
                    }
                }
                let out = copy.forward(&x, 0.0, &mut rng.clone()).0;
                out.iter().zip(direction).map(|(x, d)| x * d).sum::<f64>()
            };
            let numerical = (eval(1e-6) - eval(-1e-6)) / 2e-6;
            assert!(
                (numerical - analytical).abs() < 1e-8,
                "parameter {index}: {numerical} vs {analytical}"
            );
        }
    }
}
