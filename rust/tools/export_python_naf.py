"""Export an existing PyTorch NAF to the Rust port's portable JSON interchange.

Usage from the repository root::

    python rust/tools/export_python_naf.py meta-bo-training/naf/RUN policy.json
    cargo run --manifest-path rust/Cargo.toml -- import-naf --input policy.json --output-dir rust-policy

The conversion is a one-time Python operation. Rust inference and training use
native Rust kernels. This tool never claims that Rust directly reads ``.pt``.
Only tensor dictionaries are deserialized (``weights_only=True``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Keep the exporter runnable with a Python environment containing the project's
# dependencies, even when the project itself has not been installed as a wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from openbo.acquisition.naf import NeuralAF, naf_features

NATIVE_CONFIG_FIELDS = (
    "dim", "budget", "use_x", "hidden", "points_per_dim", "n_local_maxima",
    "n_local", "n_iterations", "steps_per_iteration", "epochs", "minibatch_size",
    "learning_rate", "clip", "value_coefficient", "entropy_coefficient", "gamma",
    "gae_lambda", "seed", "n_weights", "n_workers",
)


def export_naf(
    directory: Path, output: Path, *, name: str = "best_af", samples: int = 16, seed: int = 0,
) -> dict[str, Any]:
    """Validate/load a Python policy, then export weights and inference examples."""
    if not name or not all(c.isascii() and (c.isalnum() or c in "_-") for c in name):
        raise ValueError("name must be a plain filename stem")
    if samples < 1:
        raise ValueError("samples must be positive")
    metadata = json.loads((directory / "naf_config.json").read_text(encoding="utf-8"))
    raw = metadata["config"]
    dim, n_weights = int(raw["dim"]), int(raw.get("n_weights", 0))
    if not 1 <= dim <= 16 or n_weights < 0:
        raise ValueError("native policies require dimension 1..16 and nonnegative n_weights")
    af = NeuralAF(
        dim, use_x=bool(raw.get("use_x", True)), hidden=tuple(raw["hidden"]),
        seed=int(raw.get("seed", 0)), n_weights=n_weights,
    )
    state = torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("policy checkpoint must be a tensor-only state dictionary")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
        raise ValueError("policy checkpoint contains nonfinite tensor values")
    af.load_state_dict(state, strict=True)
    af.eval()
    state = af.state_dict()
    hp = metadata["gp_hyperparameters"]
    lengthscale = hp["lengthscale"]
    if isinstance(lengthscale, (int, float)):
        lengthscale = [lengthscale]
    lengthscale = [float(value) for value in lengthscale]
    variance, noise = float(hp["variance"]), float(hp.get("noise", 1e-10))
    if len(lengthscale) not in (1, dim) or not all(math.isfinite(v) and v > 0 for v in lengthscale):
        raise ValueError("GP lengthscales must be positive finite values matching the input dimension")
    if not math.isfinite(variance) or variance <= 0 or not math.isfinite(noise) or noise < 0:
        raise ValueError("GP variance/noise are invalid")

    rng = np.random.default_rng(seed)
    cases = []
    budget = int(raw["budget"])
    if budget < 1:
        raise ValueError("budget must be positive")
    for t in sorted({0, budget // 2, budget}):
        x = rng.uniform(0.0, 1.0, (samples, dim))
        mean = rng.normal(0.0, math.sqrt(variance), samples)
        std = rng.uniform(0.0, math.sqrt(variance), samples)
        weights = rng.dirichlet(np.ones(n_weights)) if n_weights else None
        # Persist the exact float32 inputs that Python evaluated. Rust's f64
        # kernels still differ slightly from PyTorch float32 accumulation.
        features = naf_features(mean, std, x, t, budget, af.use_x, weights).astype(np.float32).astype(np.float64)
        logits = af.logits(features)
        if not np.isfinite(logits).all():
            raise ValueError("policy inference produced nonfinite logits")
        cases.append({"features": features.tolist(), "logits": logits.tolist()})

    config = {key: raw[key] for key in NATIVE_CONFIG_FIELDS if key in raw}
    config.setdefault("n_weights", 0)
    payload = {
        "format": "openbo-python-naf-export", "version": 1,
        "source": {"checkpoint_name": name, "dtype": "float32", "update_threads": raw.get("update_threads", 1)},
        "config": config,
        "gp_config": {
            "lengthscale": lengthscale, "variance": variance, "noise": noise,
            "kernel_type": "rbf", "standardize_targets": False,
            "optimize_hyperparameters": False, "mean_constant": 0.0,
            "fixed_target_mean": None, "fixed_target_std": None,
        },
        "state_dict": {key: tensor.detach().cpu().double().tolist() for key, tensor in state.items()},
        "parity_cases": cases,
        "parity_tolerance": {"absolute": 5e-4, "relative": 2e-5},
    }
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Python scripts/train_naf.py output directory")
    parser.add_argument("output", type=Path, help="portable JSON output")
    parser.add_argument("--name", default="best_af", help="checkpoint filename stem (default: best_af)")
    parser.add_argument("--samples", type=int, default=16, help="candidates per parity example")
    parser.add_argument("--seed", type=int, default=0, help="parity-example RNG seed")
    args = parser.parse_args()
    result = export_naf(args.directory, args.output, name=args.name, samples=args.samples, seed=args.seed)
    print(f"Exported {len(result['state_dict'])} tensors and {len(result['parity_cases'])} parity cases to {args.output}")


if __name__ == "__main__":
    main()
